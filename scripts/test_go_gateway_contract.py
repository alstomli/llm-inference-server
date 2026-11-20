"""Real HTTP contract tests for the Go gateway, without a GPU.

Run with CHAT_GATEWAY_BINARY set to a compiled Go executable (default
/usr/local/bin/chat-gateway) and Pillow installed. Each case owns
fresh processes, credentials and an ephemeral fake vLLM listener. No deployment
configuration or account secrets are read.
"""
import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid


TOKEN = 'local-contract-test-token-' + 'x' * 40
UPSTREAM_KEY = 'local-contract-upstream-key'
DONE = b'data: [DONE]\n\n'
CONTENT = 'data: {"choices":[{"delta":{"content":"hello 世界"}}]}\n\n'.encode()
FINISH = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
INTERRUPTED = b'data: {"error":{"message":"The model response was interrupted. Please retry."}}\n\n'


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


class FakeModel(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, adapter=False):
        super().__init__(('127.0.0.1', 0), ModelHandler)
        self.adapter = adapter
        self.calls = []
        self.lock = threading.Lock()
        self.mode = None
        self.capacity = 40960
        self.closed = threading.Event()
        self.started = threading.Event()

    def record(self, path, headers, body):
        with self.lock:
            self.calls.append((path, dict(headers), body))

    def bodies(self, path):
        with self.lock:
            return [body for route, _, body in self.calls if route == path]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, OSError)):
            super().handle_error(request, client_address)


class ModelHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def reply(self, body, status=200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def wait_for_disconnect(self):
        self.server.started.set()
        self.connection.settimeout(4)
        try:
            self.connection.recv(1)
            self.server.closed.set()
        except ConnectionError:
            self.server.closed.set()
        except TimeoutError:
            pass
        self.close_connection = True

    def do_GET(self):
        self.server.record(self.path, self.headers, None)
        if self.path == '/health':
            return self.reply({})
        if self.path != '/v1/models':
            return self.reply({'error': 'private route'}, 404)
        if self.server.mode == 'models-wait':
            return self.wait_for_disconnect()
        if self.server.mode == 'models-error':
            return self.reply({'error': 'SECRET upstream detail'}, 500)
        cards = [{'id': 'qwen-bf16', 'max_model_len': self.server.capacity}]
        if self.server.adapter:
            cards = [{'id': 'qwen-bf16-base', 'max_model_len': self.server.capacity},
                     {'id': 'qwen-bf16', 'parent': 'wrong' if self.server.mode == 'bad-lineage' else 'qwen-bf16-base'}]
        self.reply({'data': cards})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.record(self.path, self.headers, body)
        prompt = body['messages'][-1]['content']
        if isinstance(prompt, list):
            prompt = prompt[0]['text']
        if self.path == '/tokenize':
            if self.server.mode == 'tokenizer-wait':
                return self.wait_for_disconnect()
            if self.server.mode == 'tokenizer-error':
                return self.reply({'error': 'SECRET tokenizer'}, 500)
            if self.server.mode == 'bad-count':
                return self.reply({'count': True})
            count = 50000 if len(body['messages']) > 2 or prompt == 'oversized' else 100
            return self.reply({'count': count})
        if self.path != '/v1/chat/completions':
            return self.reply({}, 404)
        self.server.started.set()
        if self.server.mode == 'prefill-wait':
            return self.wait_for_disconnect()
        if self.server.mode == 'generation-error':
            return self.reply({'error': 'SECRET model'}, 500)
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            if prompt in ('slow', 'timeout'):
                while True:
                    self.wfile.write(CONTENT)
                    self.wfile.flush()
                    time.sleep(.025)
            elif self.server.mode == 'image-error-event':
                self.wfile.write(b'data: {"error":"SECRET image data"}\n\n')
            elif self.server.mode == 'incomplete-event':
                self.wfile.write(b'data: {"choices":[]')
            else:
                # Split within a multibyte UTF-8 sequence and across SSE boundaries.
                content = CONTENT + FINISH
                if body.get('stream_options', {}).get('include_usage'):
                    content += b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102}}\n\n'
                content += DONE
                for part in (content[:45], content[45:49], content[49:]):
                    self.wfile.write(part)
                    self.wfile.flush()
                    time.sleep(.005)
        except (ConnectionError, OSError):
            pass
        finally:
            self.server.closed.set()
            self.close_connection = True


class RunningGateway:
    def __init__(self, kind, *, adapter=False, image=False, diagnostics=True, identity=True, user_rate=6, capacity=2, queue=6, rate=20):
        self.kind = kind
        self.model = FakeModel(adapter)
        self.thread = threading.Thread(target=self.model.serve_forever, daemon=True)
        self.thread.start()
        self.temp = tempfile.TemporaryDirectory(prefix='gateway-contract-')
        self.port = free_port()
        token_path = Path(self.temp.name) / 'token'
        token_path.write_text(TOKEN)
        env = dict(os.environ)
        # Override every gateway input so inherited deployment settings cannot
        # select real services or reveal credentials in this local harness.
        env.update(CHAT_TOKEN_FILE=str(token_path), CHAT_HOST='127.0.0.1', CHAT_PORT=str(self.port),
            VLLM_PORT=str(self.model.server_port), VLLM_API_KEY=UPSTREAM_KEY, SERVED_MODEL_NAME='qwen-bf16',
            MAX_MODEL_LEN='40960', MAX_NUM_SEQS=str(capacity), MODEL_PROFILE='bf16',
            CHAT_MAX_CONCURRENT=str(capacity), CHAT_MAX_WAITING=str(queue), CHAT_RATE_LIMIT_PER_MINUTE=str(rate),
            CHAT_IMAGE_INPUT_ENABLED=str(image).lower(), CHAT_DIAGNOSTICS_ENABLED=str(diagnostics).lower(),
            CHAT_REQUIRE_USER_ID=str(identity).lower(), CHAT_USER_MAX_CONCURRENT='1',
            CHAT_USER_MAX_WAITING='1', CHAT_USER_RATE_LIMIT_PER_MINUTE=str(user_rate),
            LORA_ADAPTER_PATH='local-fixture-adapter' if adapter else '', VLLM_BASE_MODEL_NAME='qwen-bf16-base' if adapter else '')
        command = [os.environ.get('CHAT_GATEWAY_BINARY', '/usr/local/bin/chat-gateway')]
        self.log = open(Path(self.temp.name) / 'gateway.log', 'w+b')
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=self.log)
            until = time.monotonic() + 20
            while time.monotonic() < until:
                if self.process.poll() is not None:
                    break
                try:
                    if self.request('/health', method='GET')[0] == 200:
                        return
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(.025)
            self.log.seek(0)
            raise AssertionError(f'{kind} gateway did not become ready: {self.log.read().decode(errors="replace")}')
        except BaseException:
            self.close()
            raise

    def request(self, path='/v1/chat/completions', body=None, headers=None, method='POST', raw=None):
        data = raw if raw is not None else json.dumps(body or request_body(), ensure_ascii=False).encode()
        values = [('Authorization', 'Bearer ' + TOKEN), ('Content-Type', 'application/json'),
                  ('X-Chat-User-Id', str(uuid.uuid4()))]
        if headers is not None:
            overridden = {name.lower() for name, _ in headers}
            values = [(name, value) for name, value in values if name.lower() not in overridden]
            values += [(name, value) for name, value in headers if value is not None]
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=8)
        try:
            connection.putrequest(method, path)
            for name, value in values:
                connection.putheader(name, value)
            connection.putheader('Content-Length', str(len(data)))
            connection.endheaders(data)
            response = connection.getresponse()
            return response.status, dict((name.lower(), value) for name, value in response.getheaders()), response.read()
        finally:
            connection.close()

    def stream(self, prompt='slow', user=None, settings=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        body = request_body(prompt)
        if settings:
            body['settings'] = settings
        connection.request('POST', '/v1/chat/completions', json.dumps(body), {
            'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json',
            'X-Chat-User-Id': user or str(uuid.uuid4())})
        return connection, connection.getresponse()

    def close(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.model.shutdown()
        self.model.server_close()
        self.thread.join(timeout=2)
        self.log.close()
        self.temp.cleanup()


def request_body(prompt='hello', **extras):
    return {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 128, 'stream': True, **extras}


def image_data(fmt='PNG', **kwargs):
    from PIL import Image
    output = BytesIO()
    Image.new('RGB', (2, 2), 'white').save(output, format=fmt, **kwargs)
    mime = {'PNG': 'png', 'JPEG': 'jpeg', 'WEBP': 'webp'}[fmt]
    return 'data:image/' + mime + ';base64,' + base64.b64encode(output.getvalue()).decode()


class GatewayContractTests(unittest.TestCase):
    @contextmanager
    def gateways(self, **config):
        running = []
        try:
            running.append(RunningGateway('go', **config))
            yield running
        finally:
            for gateway in reversed(running):
                gateway.close()

    def check(self, gateways, *, status=200, **request):
        results = [gateway.request(**request) for gateway in gateways]
        for gateway, result in zip(gateways, results):
            self.assertEqual(status, result[0], (gateway.kind, result))
            self.assertEqual('no-store', result[1].get('cache-control'), gateway.kind)
            if result[1].get('retry-after') is not None:
                self.assertTrue(1 <= int(result[1]['retry-after']) <= 60)
        return results

    def assert_idle(self, gateway):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            health = json.loads(gateway.request('/health', method='GET')[2])
            if health['admission']['activeRequests'] == health['admission']['waitingRequests'] == 0:
                return
            time.sleep(.025)
        self.fail(f'{gateway.kind} leaked admission: {health}')

    def test_health_auth_and_route_isolation(self):
        with self.gateways() as gateways:
            health = [gateway.request('/health', method='GET') for gateway in gateways]
            go_health = json.loads(health[0][2])
            self.assertTrue(go_health['accountQuotasEnforced'])
            self.assertEqual(go_health.pop('webSearchVersion'), 1)
            self.assertFalse(go_health.pop('webSearchAvailable'))
            web_capability = go_health.pop('webSearch')
            self.assertEqual(web_capability['reason'], 'disabled')
            self.assertTrue(all(not model['available'] for model in web_capability['models']))
            self.check(gateways, headers=[('Authorization', 'Bearer wrong')], status=401)
            self.check(gateways, headers=[('Authorization', None)], status=401)
            for path in ('/v1/models', '/tokenize', '/docs', '/openapi.json'):
                self.check(gateways, path=path, method='GET', status=404)
            self.check(gateways, method='GET', status=405)

    def test_identity_is_header_only_and_duplicates_fail(self):
        with self.gateways() as gateways:
            for headers in ([('X-Chat-User-Id', None)], [('X-Chat-User-Id', 'invalid')],
                            [('X-Chat-User-Id', str(uuid.uuid4())), ('X-Chat-User-Id', str(uuid.uuid4()))]):
                self.check(gateways, headers=headers, body=request_body(userId=str(uuid.uuid4())), status=400)
            self.assertTrue(all(not gateway.model.bodies('/tokenize') for gateway in gateways))

    def test_validation_and_json_types(self):
        with self.gateways() as gateways:
            cases = [request_body(stream=False), request_body(max_tokens=True), request_body(temperature=True),
                     request_body(settings=None), request_body(settings={'unknown': 1}), request_body(useAdapter=False),
                     request_body(settings={'outputTokens': 129}), request_body(settings={'inputCharacters': 1.0}),
                     request_body(messages=[{'role': 'system', 'content': 'injected'}]),
                     request_body(messages=[{'role': 'user', 'content': '\u2003'}]),
                     request_body(messages=[{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': 'y'}])]
            for body in cases:
                with self.subTest(body=body):
                    self.check(gateways, body=body, status=400)
            for raw in (b'null', b'[]', b'{} trailing', b'{bad', b'{"messages":null}',
                        json.dumps(request_body()).replace('128', '128.0').encode()):
                self.check(gateways, raw=raw, status=400)
            self.check(gateways, headers=[('Content-Type', 'text/plain')], status=415)
            self.check(gateways, raw=b' ' * 640001, status=413)

    def test_utf16_and_request_limits(self):
        with self.gateways() as gateways:
            for body in (request_body('😀', settings={'outputTokens': 128, 'inputCharacters': 1}),
                         request_body('x' * 120001), request_body('x' * 200, settings={'outputTokens': 128, 'historyBytes': 128})):
                self.check(gateways, body=body, status=400)
            self.check(gateways, body=request_body('x' * 1100, settings={'outputTokens': 128, 'requestBytes': 1024}), status=413)
            self.check(gateways, body=request_body('😀', settings={'outputTokens': 128, 'inputCharacters': 2}))

    def test_history_byte_boundary_uses_compact_unescaped_utf8(self):
        body = request_body('<>&\u2028😀' * 15)
        size = len(json.dumps(body['messages'], ensure_ascii=False, separators=(',', ':')).encode())
        with self.gateways() as gateways:
            self.check(gateways, body={**body, 'settings': {'outputTokens': 128, 'historyBytes': size}})
            self.check(gateways, body={**body, 'settings': {'outputTokens': 128, 'historyBytes': size - 1}}, status=400)

    def test_plain_stream_and_server_controlled_payload(self):
        with self.gateways() as gateways:
            self.check(gateways, body=request_body('hello <>&\u2028世界', model='attacker', tools=[{}], user='ignored'))
            payloads = [gateway.model.bodies('/v1/chat/completions')[-1] for gateway in gateways]
            self.assertEqual('qwen-bf16', payloads[0]['model'])
            self.assertNotIn('tools', payloads[0])
            self.assertEqual(CONTENT + FINISH + DONE, gateways[0].request()[2])
            for gateway in gateways:
                self.assertTrue(all(headers.get('Authorization') == 'Bearer ' + UPSTREAM_KEY
                                    for _, headers, _ in gateway.model.calls))

    def test_trimming_and_docs_prompt_match_exactly(self):
        history = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'answer'},
                   {'role': 'user', 'content': 'question 世界'}]
        with self.gateways() as gateways:
            for extra in ({}, {'docsPassages': [{'heading': 'Heading 世界', 'passage': 'Untrusted <>& evidence.'}]},
                          {'docsPassages': [{'passage': 'Evidence with reversed key order.', 'heading': 'Reversed'}]}):
                results = self.check(gateways, body=request_body(messages=history, **extra))
                self.assertEqual('2', results[0][1]['x-context-trimmed'])
                calls = [gateway.model.bodies('/v1/chat/completions')[-1] for gateway in gateways]
                for gateway in gateways:
                    self.assertEqual(gateway.model.bodies('/tokenize')[-1]['messages'], calls[0]['messages'])
            self.check(gateways, body=request_body('oversized'), status=400)

    def test_docs_validation_and_image_priority(self):
        with self.gateways() as gateways:
            for passages in ([], None, [{'heading': 'x', 'passage': 'y', 'extra': 1}],
                             [{'heading': '😀' * 81, 'passage': 'x'}], [{'heading': 'x', 'passage': ' '}]):
                self.check(gateways, body=request_body(docsPassages=passages), status=400)
            result = self.check(gateways, body=request_body(docsPassages=[], image='invalid'), status=400)
            self.assertEqual('docs_image', result[0][1]['x-chat-rejection'])

    def test_model_failures_are_sanitized_and_release_capacity(self):
        with self.gateways() as gateways:
            for mode in ('models-error', 'tokenizer-error', 'bad-count', 'generation-error'):
                for gateway in gateways:
                    gateway.model.mode = mode
                self.check(gateways, status=503)
                for gateway in gateways:
                    gateway.model.mode = None
                    self.assert_idle(gateway)
            for gateway in gateways:
                gateway.model.capacity = 4096
            self.check(gateways, status=400)

    def test_diagnostics_and_disabled_opt_in(self):
        with self.gateways(diagnostics=False) as gateways:
            self.check(gateways, headers=[('X-Chat-Diagnostics', '1')], status=403)
        with self.gateways() as gateways:
            results = [gateway.request(headers=[('X-Chat-Diagnostics', '1')]) for gateway in gateways]
            diagnostics = []
            for result in results:
                self.assertEqual(200, result[0])
                diagnostic = json.loads(result[1]['x-chat-diagnostics'])
                request_id = diagnostic.pop('requestId')
                self.assertEqual(32, len(request_id))
                diagnostics.append(diagnostic)
                self.assertEqual(1, result[2].count(b'event: chat-timing\n'))
                before, event = result[2].split(b'event: chat-timing\n')
                self.assertTrue(event.endswith(DONE))
                timing = json.loads(event.split(b'data: ', 1)[1].split(b'\n\n', 1)[0])
                self.assertEqual(request_id, timing['requestId'])
                self.assertTrue(all(value is not None and value >= 0 for value in timing['timings'].values()))
                # Go adds opt-in trace metadata; ordinary answer frames remain exact.
                frames = [frame + b'\n\n' for frame in before.split(b'\n\n') if frame]
                trace_frames = [frame for frame in frames if frame.startswith(b'event: chat-trace')]
                self.assertGreater(len(trace_frames), 1)
                pending = ''
                requests = []
                for sequence, frame in enumerate(trace_frames, 1):
                    item = json.loads(frame.split(b'data: ', 1)[1])
                    self.assertEqual(request_id, item['requestId'])
                    self.assertEqual(sequence, item['sequence'])
                    if frame.startswith(b'event: chat-trace-end\n'):
                        self.assertTrue(item['complete'])
                        self.assertEqual('', pending)
                    else:
                        pending += item['data']
                        if item['last']:
                            record = json.loads(pending)
                            if record['kind'] == 'request':
                                requests.append(record['payload']['sent'])
                            pending = ''
                self.assertEqual(gateways[0].model.bodies('/v1/chat/completions'), requests)
                ordinary = b''.join(frame for frame in frames if not frame.startswith(b'event: chat-trace'))
                self.assertEqual(CONTENT + FINISH, ordinary.split(b'data: {"choices":[],')[0])

    def test_prefix_reuse_disabled_has_distinct_server_salts(self):
        with self.gateways() as gateways:
            for gateway in gateways:
                for _ in range(2):
                    self.assertEqual(200, gateway.request(body=request_body(settings={'outputTokens': 128, 'reuseCachedPrefixes': False}))[0])
                first, second = gateway.model.bodies('/v1/chat/completions')
                self.assertEqual(64, len(first['cache_salt']))
                self.assertNotEqual(first['cache_salt'], second['cache_salt'])

    def test_adapter_routes_both_tokenization_and_generation(self):
        with self.gateways(adapter=True) as gateways:
            for enabled in (True, False):
                results = self.check(gateways, body=request_body(useAdapter=enabled))
                self.assertEqual(str(enabled).lower(), results[0][1]['x-chat-adapter-enabled'])
                for gateway in gateways:
                    self.assertEqual('qwen-bf16' if enabled else 'qwen-bf16-base', gateway.model.bodies('/tokenize')[-1]['model'])
                    self.assertEqual(gateway.model.bodies('/tokenize')[-1]['model'], gateway.model.bodies('/v1/chat/completions')[-1]['model'])
            for gateway in gateways:
                gateway.model.mode = 'bad-lineage'
            self.check(gateways, body=request_body(useAdapter=False), status=503)

    def test_images_require_opt_in_and_adapter_disabled(self):
        with self.gateways() as gateways:
            results = self.check(gateways, body=request_body(image='invalid'), status=400)
            self.assertEqual('image_unsupported', results[0][1]['x-chat-rejection'])
        with self.gateways(adapter=True, image=True) as gateways:
            results = self.check(gateways, body=request_body(image=image_data()), status=400)
            self.assertEqual('image_adapter', results[0][1]['x-chat-rejection'])
            for fmt in ('PNG', 'JPEG', 'WEBP'):
                self.check(gateways, body=request_body(image=image_data(fmt), useAdapter=False))

    def test_image_validation_rejects_malformed_and_animated_files(self):
        from PIL import Image
        animation = BytesIO()
        Image.new('RGB', (2, 2), 'white').save(animation, format='PNG', save_all=True,
            append_images=[Image.new('RGB', (2, 2), 'black')], duration=50, loop=0)
        default_animation = BytesIO()
        Image.new('RGB', (2, 2), 'white').save(default_animation, format='PNG', save_all=True,
            default_image=True, append_images=[Image.new('RGB', (2, 2), 'black')], duration=50, loop=0)
        webp_animation = BytesIO()
        Image.new('RGB', (2, 2), 'white').save(webp_animation, format='WEBP', save_all=True,
            append_images=[Image.new('RGB', (2, 2), 'black')], duration=50, loop=0)
        oversized = BytesIO()
        Image.new('RGB', (2049, 1), 'white').save(oversized, format='PNG')
        values = [None, 'https://private/image.png', 'data:image/png;base64,AAAA',
                  image_data().replace('image/png', 'image/jpeg'), image_data() + '\n',
                  'data:image/png;base64,' + base64.b64encode(animation.getvalue()).decode(),
                  'data:image/png;base64,' + base64.b64encode(default_animation.getvalue()).decode(),
                  'data:image/webp;base64,' + base64.b64encode(webp_animation.getvalue()).decode(),
                  'data:image/png;base64,' + base64.b64encode(oversized.getvalue()).decode()]
        with self.gateways(image=True) as gateways:
            for value in values:
                self.check(gateways, body=request_body(image=value), status=400)
            for gateway in gateways:
                self.assertFalse(gateway.model.bodies('/tokenize'))

    def test_image_upstream_event_is_sanitized(self):
        with self.gateways(image=True) as gateways:
            for gateway in gateways:
                gateway.model.mode = 'image-error-event'
            results = self.check(gateways, body=request_body(image=image_data()))
            self.assertEqual(INTERRUPTED + DONE, results[0][2])

    def test_account_rate_normalizes_uuid_and_compatibility_principal(self):
        user = str(uuid.uuid4())
        with self.gateways(user_rate=1) as gateways:
            self.check(gateways, headers=[('X-Chat-User-Id', user.upper())])
            results = self.check(gateways, headers=[('X-Chat-User-Id', user)], status=429)
            self.assertEqual('user_rate_limit', results[0][1]['x-chat-rejection'])
            self.check(gateways)
        with self.gateways(identity=False, user_rate=1) as gateways:
            self.check(gateways, headers=[('X-Chat-User-Id', None)])
            self.check(gateways, headers=[('X-Chat-User-Id', None)], status=429)

    def test_stream_disconnect_closes_upstream_and_releases_admission(self):
        with self.gateways() as gateways:
            for gateway in gateways:
                connection, response = gateway.stream()
                self.assertEqual(200, response.status)
                self.assertTrue(response.read(20))
                response.close()
                connection.close()
                self.assertTrue(gateway.model.closed.wait(3), gateway.kind)
                self.assert_idle(gateway)
                self.assertEqual(200, gateway.request()[0])

    def test_timeout_interrupts_stream_and_releases_admission(self):
        with self.gateways() as gateways:
            results = [gateway.request(body=request_body('timeout', settings={'outputTokens': 128, 'timeoutSeconds': 1}))
                       for gateway in gateways]
            for gateway, result in zip(gateways, results):
                self.assertEqual(200, result[0])
                self.assertIn(INTERRUPTED + DONE, result[2])
                self.assert_idle(gateway)

    def test_disconnect_during_context_tokenization_and_prefill(self):
        with self.gateways() as gateways:
            for gateway in gateways:
                for mode in ('models-wait', 'tokenizer-wait', 'prefill-wait'):
                    gateway.model.started.clear()
                    gateway.model.closed.clear()
                    gateway.model.mode = mode
                    connection = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                    connection.request('POST', '/v1/chat/completions', json.dumps(request_body()), {
                        'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json',
                        'X-Chat-User-Id': str(uuid.uuid4())})
                    self.assertTrue(gateway.model.started.wait(3), (gateway.kind, mode))
                    connection.sock.shutdown(socket.SHUT_RDWR)
                    connection.close()
                    self.assertTrue(gateway.model.closed.wait(3), (gateway.kind, mode))
                    gateway.model.mode = None
                    self.assert_idle(gateway)

    def test_queued_disconnect_and_account_capacity_do_not_block_peer(self):
        with self.gateways(capacity=8, queue=24, rate=80) as gateways:
            for gateway in gateways:
                user = str(uuid.uuid4())
                active, active_response = gateway.stream(user=user)
                pending = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                peer = peer_response = None
                try:
                    pending.request('POST', '/v1/chat/completions', json.dumps(request_body()), {
                        'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json', 'X-Chat-User-Id': user})
                    until = time.monotonic() + 3
                    while time.monotonic() < until:
                        health = json.loads(gateway.request('/health', method='GET')[2])
                        if health['admission']['waitingRequests'] == 1:
                            break
                        time.sleep(.025)
                    self.assertEqual(1, health['admission']['waitingRequests'], gateway.kind)
                    rejected = gateway.request(headers=[('X-Chat-User-Id', user)])
                    self.assertEqual((429, 'user_capacity'), (rejected[0], rejected[1].get('x-chat-rejection')))
                    peer, peer_response = gateway.stream()
                    self.assertEqual(200, peer_response.status, gateway.kind)
                    pending.sock.shutdown(socket.SHUT_RDWR)
                    pending.close()
                    until = time.monotonic() + 3
                    while time.monotonic() < until:
                        health = json.loads(gateway.request('/health', method='GET')[2])
                        if health['admission']['waitingRequests'] == 0:
                            break
                        time.sleep(.025)
                    self.assertEqual((2, 0), (health['admission']['activeRequests'], health['admission']['waitingRequests']))
                finally:
                    pending.close()
                    active_response.close()
                    active.close()
                    if peer_response:
                        peer_response.close()
                        peer.close()
                self.assert_idle(gateway)

    def test_global_queue_capacity_and_fifo_promotion(self):
        for capacity, waiting, rate in ((2, 6, 20), (8, 24, 80)):
            with self.subTest(capacity=capacity), self.gateways(capacity=capacity, queue=waiting, rate=rate) as gateways:
                for gateway in gateways:
                    active = [gateway.stream() for _ in range(capacity)]
                    queued = []
                    try:
                        for index in range(waiting):
                            connection = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                            connection.request('POST', '/v1/chat/completions', json.dumps(request_body(f'queued-{index}')), {
                                'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json',
                                'X-Chat-User-Id': str(uuid.uuid4())})
                            queued.append(connection)
                            until = time.monotonic() + 3
                            while time.monotonic() < until:
                                health = json.loads(gateway.request('/health', method='GET')[2])
                                if health['admission']['waitingRequests'] == index + 1:
                                    break
                                time.sleep(.01)
                            self.assertEqual(index + 1, health['admission']['waitingRequests'])
                        rejected = gateway.request()
                        self.assertEqual((429, 'queue_full'), (rejected[0], rejected[1].get('x-chat-rejection')))
                        active[0][1].close()
                        active[0][0].close()
                        for connection in queued:
                            response = connection.getresponse()
                            self.assertEqual(200, response.status)
                            response.read()
                        promoted = [body['messages'][-1]['content']
                                    for body in gateway.model.bodies('/v1/chat/completions')
                                    if body['messages'][-1]['content'].startswith('queued-')]
                        self.assertEqual([f'queued-{index}' for index in range(waiting)], promoted)
                    finally:
                        for connection in queued:
                            connection.close()
                        for connection, response in active:
                            response.close()
                            connection.close()
                    self.assert_idle(gateway)


    def test_queue_timeout_does_not_reach_upstream(self):
        with self.gateways(capacity=8, queue=24, rate=80) as gateways:
            for gateway in gateways:
                user = str(uuid.uuid4())
                active, response = gateway.stream(user=user)
                try:
                    result = gateway.request(body=request_body('queued-timeout', settings={
                        'outputTokens': 128, 'timeoutSeconds': 1}), headers=[('X-Chat-User-Id', user)])
                    self.assertEqual((504, 'queue_timeout'), (result[0], result[1].get('x-chat-rejection')))
                    self.assertEqual(1, len(gateway.model.bodies('/v1/chat/completions')))
                finally:
                    response.close()
                    active.close()
                self.assert_idle(gateway)

    def test_global_rate_applies_even_with_available_slots(self):
        with self.gateways() as gateways:
            for _ in range(20):
                self.check(gateways)
            results = self.check(gateways, status=429)
            self.assertEqual('rate_limit', results[0][1]['x-chat-rejection'])


if __name__ == '__main__':
    unittest.main()
