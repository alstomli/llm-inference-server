"""Exercise the real gateway with a controlled vLLM-compatible upstream."""
import asyncio
import base64
import importlib.util
from io import BytesIO
import json
import os
from unittest.mock import patch
import re
from pathlib import Path
import unittest
from uuid import uuid4
import httpx
from PIL import Image

spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('chat-gateway.py'))
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)
TOKEN = 'test-token-' + 'x' * 40
AUTH = {'Authorization': 'Bearer ' + TOKEN}
def user_auth(user=None):
    return {**AUTH, 'X-Chat-User-Id': str(uuid4()) if user is None else user}


SSE = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'


def image_uri(image_format='PNG', size=(32, 32), **save_options):
    buffer = BytesIO()
    with Image.new('RGB', size, color='red') as image:
        image.save(buffer, format=image_format, **save_options)
    return 'data:image/' + image_format.lower() + ';base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.forwarded = []
        self.tokenizer_requests = 0
        self.tokenizer_models = []
        self.tokenized = []
        self.model_requests = 0
        self.tokenizer_failure = False
        self.tokenizer_invalid = False
        self.model_failure = False
        self.oversized = False
        self.token_count = None
        self.model_context = 40960
        self.model_cards = None
        self.tokenize_delay = 0
        self.reply_delay = 0
        self.stream_parts = None
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.blocked_prompts = set()
        self.started_prompts = []
        self.finished_prompts = set()
        self.prompt_releases = {}
        owner = self

        class ControlledStream(httpx.AsyncByteStream):
            def __init__(self, prompt):
                self.prompt = prompt

            async def __aiter__(self):
                owner.started_prompts.append(self.prompt)
                await owner.prompt_releases.setdefault(self.prompt, asyncio.Event()).wait()
                yield SSE

            async def aclose(self):
                owner.finished_prompts.add(self.prompt)

        class BlockedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                owner.entered.set()
                if owner.reply_delay:
                    await asyncio.sleep(owner.reply_delay)
                    yield SSE
                    return
                await owner.release.wait()
                yield SSE

            async def aclose(self):
                owner.closed = True

        class FragmentedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for part in owner.stream_parts:
                    await asyncio.sleep(0.01)
                    yield part

        async def upstream(request):
            if request.url.path == '/health':
                return httpx.Response(503 if self.model_failure else 200)
            if request.url.path == '/v1/models':
                self.model_requests += 1
                return httpx.Response(200, json={'data': self.model_cards if self.model_cards is not None
                                                 else [{'id': 'qwen-bf16', 'max_model_len': self.model_context}]})
            body = json.loads(request.content)
            if request.url.path == '/tokenize':
                self.tokenizer_requests += 1
                self.tokenizer_models.append(body['model'])
                self.tokenized.append(body)
                await asyncio.sleep(self.tokenize_delay)
                if self.tokenizer_failure:
                    return httpx.Response(503)
                if self.tokenizer_invalid:
                    return httpx.Response(200, json={'count': 'invalid'})
                count = 50000 if self.oversized or len(body['messages']) > 2 else 150
                if self.token_count is not None:
                    count = self.token_count
                return httpx.Response(200, json={'count': count})
            self.forwarded.append(body)
            if self.model_failure:
                return httpx.Response(500, text='internal details must not be exposed')
            prompt = body['messages'][-1]['content']
            if isinstance(prompt, list):
                prompt = prompt[0]['text']
            stream = ControlledStream(prompt) if prompt in self.blocked_prompts else FragmentedStream() if self.stream_parts is not None else BlockedStream() if self.block or self.reply_delay else httpx.ByteStream(SSE)
            return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})

        self.transport = httpx.MockTransport(upstream)
        self.app = gateway.create_app(TOKEN, transport=self.transport, diagnostics_enabled=False)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.app.state.client.aclose()

    async def send(self, messages=None, **extra):
        return await self.client.post('/v1/chat/completions', headers=user_auth(), json={
            'messages': messages or [{'role': 'user', 'content': 'Hi'}], 'stream': True, **extra,
        })

    async def enable_diagnostics(self):
        await self.client.aclose()
        await self.app.state.client.aclose()
        self.app = gateway.create_app(TOKEN, transport=self.transport, diagnostics_enabled=True)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def diagnostic_send(self, messages=None, **extra):
        return await self.client.post('/v1/chat/completions', headers={**user_auth(), 'X-Chat-Diagnostics': '1'}, json={
            'messages': messages or [{'role': 'user', 'content': 'Hi'}], 'stream': True, **extra,
        })

    async def test_diagnostics_require_server_opt_in_and_authentication(self):
        self.assertEqual((await self.diagnostic_send()).status_code, 403)
        self.assertEqual(self.forwarded, [])
        self.assertEqual((await self.client.get('/health', headers=user_auth())).json()['diagnosticsVersion'], 0)
        await self.enable_diagnostics()
        denied = await self.client.post('/v1/chat/completions', headers={'X-Chat-Diagnostics': '1'}, json={})
        self.assertEqual(denied.status_code, 401)
        self.assertNotIn('X-Chat-Diagnostics', denied.headers)
        self.assertEqual((await self.client.get('/health', headers=user_auth())).json()['diagnosticsVersion'], 1)

    async def test_diagnostics_reuse_exact_counts_and_normal_stream_is_unchanged(self):
        await self.enable_diagnostics()
        plain = await self.send()
        self.assertNotIn('X-Chat-Diagnostics', plain.headers)
        self.assertNotIn('stream_options', self.forwarded[-1])
        self.assertEqual(self.tokenizer_requests, 1)
        response = await self.diagnostic_send()
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual(record['inputTokensBefore'], 150)
        self.assertEqual(record['inputTokensAfter'], 150)
        self.assertEqual(record['gatewayTrimmedMessages'], 0)
        self.assertEqual(self.tokenizer_requests, 2)
        self.assertEqual(self.forwarded[-1]['stream_options'], {'include_usage': True})
        self.assertEqual(re.sub(rb'event: chat-timing\ndata: [^\n]+\n\n', b'', response.content), plain.content)
        self.assertNotIn(b'chat-timing', plain.content)
        self.assertNotIn('messages', record)
        self.assertNotIn('Hi', response.headers['X-Chat-Diagnostics'])
        self.assertNotIn(TOKEN, response.headers['X-Chat-Diagnostics'])

    async def test_timing_observes_content_events_across_utf8_and_network_boundaries(self):
        await self.enable_diagnostics()
        self.tokenize_delay = 0.025
        content = 'data: {"choices":[{"delta":{"content":"你好"}}]}\r\n\r\n'.encode()
        split = content.index('你'.encode()) + 1
        self.stream_parts = [b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
                             content[:split], content[split:],
                             b'data: {"choices":[],"usage":{"prompt_tokens":150,"completion_tokens":2}}\n\n',
                             b'data: [DO', b'NE]\n\n']
        response = await self.diagnostic_send([
            {'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}])
        event = json.loads(re.search(rb'event: chat-timing\ndata: ([^\n]+)', response.content)[1])
        self.assertEqual(event['requestId'], json.loads(response.headers['X-Chat-Diagnostics'])['requestId'])
        timing = event['timings']
        self.assertGreaterEqual(timing['contextFitMs'], 40)  # Both tokenizer calls, including trimming.
        self.assertGreaterEqual(timing['upstreamFirstContentMs'], 20)  # Ignore role/empty content.
        self.assertGreater(timing['gatewayFirstContentMs'], timing['upstreamFirstContentMs'])
        self.assertGreater(timing['gatewayTotalMs'], timing['gatewayFirstContentMs'])
        self.assertEqual(response.content.count(b'event: chat-timing'), 1)
        self.assertLess(response.content.index(b'event: chat-timing'), response.content.index(b'data: [DONE]'))
        self.assertEqual(re.sub(rb'event: chat-timing\ndata: [^\n]+\n\n', b'', response.content), b''.join(self.stream_parts))

    async def test_timing_keeps_missing_first_content_unavailable_on_timeout(self):
        await self.enable_diagnostics()
        self.reply_delay = 1.2
        response = await self.diagnostic_send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1})
        event = json.loads(re.search(rb'event: chat-timing\ndata: ([^\n]+)', response.content)[1])
        self.assertIsNone(event['timings']['upstreamFirstContentMs'])
        self.assertIsNone(event['timings']['gatewayFirstContentMs'])
        self.assertGreaterEqual(event['timings']['gatewayTotalMs'], 900)
        self.assertIn(b'interrupted', response.content)
        self.assertTrue(self.closed)
        self.reply_delay = 0
        self.assertEqual((await self.send()).status_code, 200)

    async def test_diagnostics_report_trimmed_and_rejected_prompts(self):
        await self.enable_diagnostics()
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}]
        response = await self.diagnostic_send(messages)
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual((record['inputTokensBefore'], record['inputTokensAfter']), (50000, 150))
        self.assertEqual(record['gatewayTrimmedMessages'], 2)
        self.assertEqual(self.tokenizer_requests, 2)
        response = await self.diagnostic_send(messages[:2] + messages)
        self.assertEqual(json.loads(response.headers['X-Chat-Diagnostics'])['gatewayTrimmedMessages'], 4)
        self.assertEqual(self.tokenizer_requests, 5)
        self.oversized = True
        response = await self.diagnostic_send()
        self.assertEqual(response.status_code, 400)
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual(record['inputTokensBefore'], 50000)
        self.assertEqual(record['lastCountedInputTokens'], 50000)
        self.assertIsNone(record['inputTokensAfter'])
        self.assertEqual(len(self.forwarded), 2)
        self.oversized = False; self.tokenizer_failure = True
        response = await self.diagnostic_send()
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(json.loads(response.headers['X-Chat-Diagnostics'])['inputTokensBefore'])

    def raw_chat(self, app, disconnect, user='11111111-1111-4111-8111-111111111111'):
        body = json.dumps({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True}).encode()
        received = False

        async def receive():
            nonlocal received
            if not received:
                received = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            await disconnect.wait()
            return {'type': 'http.disconnect'}

        async def send(_message):
            pass

        scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.3'},
                 'http_version': '1.1', 'method': 'POST', 'scheme': 'http',
                 'path': '/v1/chat/completions', 'raw_path': b'/v1/chat/completions',
                 'query_string': b'', 'root_path': '', 'server': ('test', 80), 'client': ('test', 1),
                 'headers': [(b'authorization', ('Bearer ' + TOKEN).encode()),
                             (b'content-type', b'application/json'), (b'x-chat-user-id', user.encode())]}
        return asyncio.create_task(app(scope, receive, send))

    async def test_disconnect_cancels_context_tokenization_and_prefill_before_headers(self):
        for blocked_path in ['/v1/models', '/tokenize', '/v1/chat/completions']:
            with self.subTest(stage=blocked_path):
                entered, cancelled, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()

                async def upstream(request):
                    if request.url.path == blocked_path:
                        entered.set()
                        try:
                            await asyncio.Event().wait()
                        finally:
                            cancelled.set()
                    if request.url.path == '/v1/models':
                        return httpx.Response(200, json={'data': [{'id': 'qwen-bf16', 'max_model_len': 40960}]})
                    return httpx.Response(200, json={'count': 10})

                app = gateway.create_app(TOKEN, transport=httpx.MockTransport(upstream))
                task = self.raw_chat(app, disconnect)
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    self.assertEqual(len(app.state.admission.active), 1)
                    disconnect.set()
                    await asyncio.wait_for(task, 2)
                    self.assertTrue(cancelled.is_set())
                    self.assertFalse(app.state.admission.active)
                    self.assertFalse(app.state.admission.owners)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await app.state.client.aclose()

    async def test_disconnect_holds_account_and_global_slot_until_delayed_close_completes(self):
        entered, close_started, close_release, closed, disconnect = [asyncio.Event() for _ in range(5)]

        class SlowClose(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield SSE

            async def aclose(self):
                close_started.set()
                await close_release.wait()
                closed.set()

        async def upstream(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'qwen-bf16', 'max_model_len': 40960}]})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'count': 10})
            return httpx.Response(200, stream=SlowClose())

        app = gateway.create_app(TOKEN, transport=httpx.MockTransport(upstream))
        user = '11111111-1111-4111-8111-111111111111'
        task = self.raw_chat(app, disconnect, user)
        try:
            await asyncio.wait_for(entered.wait(), 2)
            queued, _, _ = app.state.admission.reserve(user)
            disconnect.set()
            await asyncio.wait_for(close_started.wait(), 2)
            self.assertEqual(len(app.state.admission.active), 1)
            self.assertEqual(app.state.admission.accounts[user]['active'], 1)
            self.assertFalse(queued.done())
            close_release.set()
            await asyncio.wait_for(task, 2)
            self.assertTrue(closed.is_set())
            self.assertTrue(queued.done())
            self.assertEqual(app.state.admission.accounts[user]['active'], 1)
            app.state.admission.release(queued)
            self.assertFalse(app.state.admission.active)
        finally:
            close_release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.state.client.aclose()

    async def test_disconnect_during_error_response_close_keeps_one_owned_cleanup(self):
        close_started, close_release, closed, disconnect = [asyncio.Event() for _ in range(4)]
        close_calls = 0

        class SlowErrorClose(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'private failure detail'

            async def aclose(self):
                nonlocal close_calls
                close_calls += 1
                close_started.set()
                await close_release.wait()
                closed.set()

        async def upstream(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'qwen-bf16', 'max_model_len': 40960}]})
            if request.url.path == '/tokenize':
                return httpx.Response(200, json={'count': 10})
            return httpx.Response(500, stream=SlowErrorClose())

        app = gateway.create_app(TOKEN, transport=httpx.MockTransport(upstream))
        task = self.raw_chat(app, disconnect)
        try:
            await asyncio.wait_for(close_started.wait(), 2)
            disconnect.set()
            await asyncio.sleep(0.05)
            self.assertFalse(closed.is_set())
            self.assertEqual(len(app.state.admission.active), 1)
            close_release.set()
            await asyncio.wait_for(task, 2)
            self.assertTrue(closed.is_set())
            self.assertEqual(close_calls, 1)
            self.assertFalse(app.state.admission.active)
            self.assertFalse(app.state.admission.owners)
        finally:
            close_release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.state.client.aclose()

    async def test_identity_is_required_before_body_admission_and_health_needs_only_bearer(self):
        health = (await self.client.get('/health', headers=AUTH)).json()
        self.assertEqual((health['accountQuotaVersion'], health['accountQuotasEnforced']), (1, True))
        for value in [None, '', 'alice', '11111111111141118111111111111111', ' 11111111-1111-4111-8111-111111111111', 'urn:uuid:11111111-1111-4111-8111-111111111111']:
            headers = {**AUTH, **({'X-Chat-User-Id': value} if value is not None else {})}
            response = await self.client.post('/v1/chat/completions', headers=headers, content='invalid JSON')
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.tokenizer_requests, 0)
        self.assertFalse(self.app.state.admission.accounts)
        denied = await self.client.post('/v1/chat/completions', headers={'X-Chat-User-Id': 'bad'}, content='bad')
        self.assertEqual(denied.status_code, 401)

    async def test_account_rate_normalizes_uuid_and_ignores_body_identity(self):
        await self.enable_adapter_toggle()
        user = 'AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA'
        for index in range(6):
            response = await self.client.post('/v1/chat/completions', headers=user_auth(user if index % 2 else user.lower()), json={
                'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True, 'userId': str(uuid4()), 'useAdapter': bool(index % 2)})
            self.assertEqual(response.status_code, 200)
        response = await self.client.post('/v1/chat/completions', headers=user_auth(user), json={
            'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True})
        self.assertEqual((response.status_code, response.headers['X-Chat-Rejection']), (429, 'user_rate_limit'))
        self.assertTrue(1 <= int(response.headers['Retry-After']) <= 60)
        self.assertEqual((await self.send()).status_code, 200)
        self.assertEqual(len(self.app.state.admission.accounts[user.lower()]['recent']), 6)

    async def test_compatibility_missing_ids_share_bounded_account_but_invalid_ids_fail(self):
        await self.configure(require_user_id=False)
        health = (await self.client.get('/health', headers=AUTH)).json()
        self.assertFalse(health['accountQuotasEnforced'])
        for _ in range(6):
            response = await self.client.post('/v1/chat/completions', headers=AUTH, json={
                'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True})
            self.assertEqual(response.status_code, 200)
        response = await self.client.post('/v1/chat/completions', headers=AUTH, json={
            'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True})
        self.assertEqual(response.headers['X-Chat-Rejection'], 'user_rate_limit')
        self.assertEqual((await self.client.post('/v1/chat/completions', headers=user_auth('bad'), content='bad')).status_code, 400)
        self.assertEqual((await self.send()).status_code, 200)

    async def test_authentication_and_route_isolation(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            self.assertEqual((await self.client.get('/health', headers=headers)).status_code, 401)
        for path in ['/metrics', '/docs', '/openapi.json', '/v1/models']:
            self.assertEqual((await self.client.get(path, headers=user_auth())).status_code, 404)
        self.assertEqual((await self.client.get('/health', headers=user_auth())).status_code, 200)

    async def configure(self, **options):
        await self.client.aclose()
        await self.app.state.client.aclose()
        self.app = gateway.create_app(TOKEN, transport=self.transport, diagnostics_enabled=True, **options)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def enable_adapter_toggle(self, **options):
        self.model_cards = [{'id': 'qwen-bf16-base', 'max_model_len': 40960}, {'id': 'qwen-bf16', 'parent': 'qwen-bf16-base'}]
        await self.configure(base_model='qwen-bf16-base', **options)

    async def test_bf16_is_default_and_retired_profiles_fail_before_upstream(self):
        health = (await self.client.get('/health', headers=user_auth())).json()
        self.assertEqual(health['model'], 'qwen-bf16')
        before = (self.model_requests, self.tokenizer_requests, len(self.forwarded))
        for profile in ('4bit', 'unknown', '', None):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                gateway.create_app(TOKEN, model_profile=profile, transport=self.transport)
        self.assertEqual((self.model_requests, self.tokenizer_requests, len(self.forwarded)), before)

    async def test_image_input_requires_bf16_and_opt_in(self):
        image = image_uri()
        for profile, enabled in [('bf16', False), ('bf16', True)]:
            await self.configure(model_profile=profile, image_input_enabled=enabled)
            health = (await self.client.get('/health', headers=user_auth())).json()
            if profile == 'bf16' and enabled:
                self.assertEqual(health['imageInputVersion'], 1)
                self.assertEqual(health['imageInput'], {
                    'maxBytes': 262144, 'maxDimension': 2048,
                    'mimeTypes': ['image/jpeg', 'image/png', 'image/webp'],
                })
                response = await self.send(image=image)
                self.assertEqual(response.headers.get('X-Chat-Image-Accepted'), 'true')
                self.assertEqual(response.content, SSE)
            else:
                self.assertNotIn('imageInput', health)
                self.assertNotIn('imageInputVersion', health)
                response = await self.send(image=image)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.headers.get('X-Chat-Rejection'), 'image_unsupported')
        self.assertEqual(self.tokenizer_requests, 1)
        self.assertNotIn('X-Chat-Image-Accepted', (await self.send()).headers)

    async def test_image_validation_rejects_invalid_files_before_upstream(self):
        await self.configure(model_profile='bf16', image_input_enabled=True)
        valid = image_uri()
        mime, encoded = valid.split(',', 1)
        oversized = base64.b64encode(b'x' * (gateway.MAX_IMAGE_BYTES + 1)).decode('ascii')
        for image in (None, [valid], 'https://example.com/picture.png', valid.replace('image/png', 'image/jpeg'),
                      mime + ',Zm9v', mime + ',' + encoded[:40], mime + ', AB==', mime + ',' + oversized,
                      image_uri(size=(2049, 1)), image_uri(size=(1, 2049))):
            response = await self.send(image=image)
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.headers.get('X-Chat-Rejection'), 'image_invalid')
        response = await self.send(image=valid, settings={**gateway.SETTINGS_DEFAULT, 'requestBytes': 1024},
                                   messages=[{'role': 'user', 'content': 'x' * 1000}])
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.model_requests, 0)
        self.assertEqual(self.tokenizer_requests, 0)
        self.assertFalse(self.app.state.admission.active)

    async def test_image_uses_base_alias_and_survives_tokenizer_trimming(self):
        await self.enable_adapter_toggle(model_profile='bf16', image_input_enabled=True)
        for flags in ({}, {'useAdapter': True}):
            response = await self.send(image='invalid', **flags)
            self.assertEqual(response.headers.get('X-Chat-Rejection'), 'image_adapter')
        self.assertEqual(self.model_requests, 0)
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old answer'},
                    {'role': 'user', 'content': 'What is pictured?'}]
        for format in ('JPEG', 'PNG', 'WEBP'):
            image = image_uri(format)
            response = await self.diagnostic_send(messages, image=image, useAdapter=False,
                                                  settings={**gateway.SETTINGS_DEFAULT, 'reuseCachedPrefixes': False})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get('X-Chat-Image-Accepted'), 'true')
            self.assertEqual(response.headers.get('X-Chat-Adapter-Enabled'), 'false')
            self.assertEqual(self.tokenizer_models[-2:], ['qwen-bf16-base', 'qwen-bf16-base'])
            parts = [{'type': 'text', 'text': 'What is pictured?'}, {'type': 'image_url', 'image_url': {'url': image}}]
            self.assertEqual(self.tokenized[-2]['messages'][-1]['content'], parts)
            self.assertEqual(self.tokenized[-2]['messages'][1:-1], messages[:-1])
            self.assertEqual(self.forwarded[-1]['model'], 'qwen-bf16-base')
            self.assertEqual(self.forwarded[-1]['messages'], self.tokenized[-1]['messages'])
            self.assertEqual(self.forwarded[-1]['messages'][-1]['content'], parts)
            self.assertIn('cache_salt', self.forwarded[-1])
            diagnostic = json.loads(response.headers['X-Chat-Diagnostics'])
            self.assertEqual(diagnostic['gatewayTrimmedMessages'], 2)
            self.assertEqual((diagnostic['inputTokensBefore'], diagnostic['inputTokensAfter']), (50000, 150))
            self.assertNotIn(image, response.headers['X-Chat-Diagnostics'])
        self.assertEqual((await self.send()).headers.get('X-Chat-Adapter-Enabled'), 'true')

    async def test_image_upstream_errors_are_sanitized_and_release_admission(self):
        await self.configure(model_profile='bf16', image_input_enabled=True)
        image = image_uri()
        self.tokenizer_failure = True
        response = await self.send(image=image)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get('X-Chat-Rejection'), 'image_processing')
        self.assertNotIn('X-Chat-Image-Accepted', response.headers)
        self.tokenizer_failure = False
        self.stream_parts = [('data: ' + json.dumps({'error': {'message': image}}) + '\n\n').encode(), b'data: [DONE]\n\n']
        response = await self.send(image=image)
        self.assertNotIn(image, response.text)
        self.assertIn('interrupted', response.text)
        self.assertFalse(self.app.state.admission.active)

    async def test_adapter_toggle_routes_tokenizer_generation_and_diagnostics_per_request(self):
        await self.enable_adapter_toggle()
        health = (await self.client.get('/health', headers=user_auth())).json()
        self.assertTrue(health['adapterToggleAvailable'])
        self.assertEqual(health['adapterToggleVersion'], 1)
        for enabled, alias in [(True, 'qwen-bf16'), (False, 'qwen-bf16-base'), (True, 'qwen-bf16')]:
            response = await self.diagnostic_send(useAdapter=enabled, model='attacker-model')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['X-Chat-Adapter-Enabled'], str(enabled).lower())
            self.assertEqual(self.tokenizer_models[-1], alias)
            self.assertEqual(self.forwarded[-1]['model'], alias)
            self.assertEqual(json.loads(response.headers['X-Chat-Diagnostics'])['model'], alias)
        self.assertEqual((await self.send()).headers['X-Chat-Adapter-Enabled'], 'true')

    async def test_adapter_toggle_rejects_invalid_flags_unsupported_connections_and_missing_lineage(self):
        self.assertEqual((await self.send(useAdapter=False)).status_code, 400)
        await self.enable_adapter_toggle()
        for value in ['false', 0, None, {}, []]:
            self.assertEqual((await self.send(useAdapter=value)).status_code, 400)
        self.assertEqual(self.forwarded, [])
        self.model_cards[1]['parent'] = 'other'
        self.assertEqual((await self.client.get('/health', headers=user_auth())).status_code, 503)
        self.assertEqual((await self.send(useAdapter=False)).status_code, 503)
        self.assertEqual(self.forwarded, [])

    async def test_adapter_variants_share_admission_slots(self):
        await self.enable_adapter_toggle()
        self.block = True
        tasks = [asyncio.create_task(self.send(useAdapter=value)) for value in (True, False)]
        try:
            await self.until(lambda: len(self.forwarded) == 2)
            self.assertEqual(len(self.app.state.admission.active), 2)
            self.assertEqual({body['model'] for body in self.forwarded}, {'qwen-bf16', 'qwen-bf16-base'})
        finally:
            self.release.set()
            await asyncio.gather(*tasks)

    async def test_adapter_inherits_only_its_named_base_capacity_and_routes_to_adapter(self):
        self.model_cards = [{'id': 'unrelated', 'max_model_len': 40960},
                            {'id': 'qwen-bf16-base', 'max_model_len': 8192},
                            {'id': 'qwen-bf16', 'parent': 'qwen-bf16-base', 'max_model_len': None}]
        health = (await self.client.get('/health', headers=user_auth())).json()
        self.assertTrue(health['ready'])
        self.assertEqual(health['model'], 'qwen-bf16')
        self.assertEqual(health['limits']['contextTokens'], 8192)
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 8192, 'outputTokens': 128}
        self.assertEqual((await self.send(settings=settings, max_tokens=128)).status_code, 200)
        self.assertEqual(self.forwarded[-1]['model'], 'qwen-bf16')
        self.assertEqual(self.tokenizer_models[-1], 'qwen-bf16')
        self.assertEqual((await self.send()).status_code, 400)

    async def test_adapter_missing_or_invalid_base_capacity_fails_closed(self):
        for cards in ([{'id': 'other', 'max_model_len': 40960}],
                      [{'id': 'qwen-bf16', 'parent': 'missing'}, {'id': 'other', 'max_model_len': 40960}],
                      [{'id': 'qwen-bf16', 'parent': 'base'}, {'id': 'base', 'max_model_len': '40960'}],
                      [{'id': 'qwen-bf16', 'parent': 'base', 'max_model_len': 0}, {'id': 'base', 'max_model_len': 40960}]):
            with self.subTest(cards=cards):
                self.model_cards = cards
                self.assertEqual((await self.client.get('/health', headers=user_auth())).status_code, 503)
                self.assertEqual((await self.send()).status_code, 503)
        self.assertEqual(self.forwarded, [])

    async def test_prefix_reuse_uses_unique_server_salts_only_when_disabled(self):
        self.assertTrue((await self.client.get('/health', headers=user_auth())).json()['limits']['reuseCachedPrefixes'])
        self.assertEqual((await self.send(cache_salt='client-controlled')).status_code, 200)
        self.assertNotIn('cache_salt', self.forwarded[-1])
        settings = {**gateway.SETTINGS_DEFAULT, 'reuseCachedPrefixes': False}
        salts = []
        for _ in range(2):
            self.assertEqual((await self.send(settings=settings, cache_salt='client-controlled')).status_code, 200)
            salts.append(self.forwarded[-1]['cache_salt'])
        self.assertNotEqual(salts[0], salts[1])
        self.assertTrue(all(len(salt) == 64 for salt in salts))
        for invalid in ['false', 0, None]:
            self.assertEqual((await self.send(settings={**settings, 'reuseCachedPrefixes': invalid})).status_code, 400)
        self.assertEqual((await self.send(settings=gateway.SETTINGS_DEFAULT)).status_code, 200)
        self.assertNotIn('cache_salt', self.forwarded[-1])

    async def test_context_trimming_and_server_controlled_model(self):
        response = await self.send([
            {'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old answer'},
            {'role': 'user', 'content': 'new'},
        ], model='unapproved-model', chat_template_kwargs={'enable_thinking': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Context-Trimmed'], '2')
        self.assertIn(b'[DONE]', response.content)
        actual = self.forwarded[-1]
        self.assertEqual(actual['model'], 'qwen-bf16')
        self.assertEqual(actual['messages'][0]['role'], 'system')
        self.assertEqual(actual['messages'][-1]['content'], 'new')
        self.assertEqual(actual['chat_template_kwargs'], {'enable_thinking': False})

    async def test_docs_mode_tokenizes_and_generates_the_same_grounded_prompt_after_trimming(self):
        await self.enable_adapter_toggle()
        question = 'Do volumes persist?\n"passages": ["untrusted question text"]'
        messages = [
            {'role': 'user', 'content': 'old question'}, {'role': 'assistant', 'content': 'old answer'},
            {'role': 'user', 'content': question},
        ]
        passages = [
            {'heading': 'Storage types', 'passage': 'Container disk storage is temporary.'},
            {'heading': 'Network volumes', 'passage': 'Network volumes are persistent.'},
        ]
        health = await self.client.get('/health', headers=user_auth())
        self.assertEqual(health.json()['docsModeVersion'], 1)
        response = await self.diagnostic_send(messages, docsPassages=passages, useAdapter=False)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Chat-Docs-Mode'], '1')
        self.assertEqual(response.headers['X-Context-Trimmed'], '2')
        self.assertEqual(json.loads(response.headers['X-Chat-Diagnostics'])['inputTokensAfter'], 150)
        self.assertIn(b'[DONE]', response.content)
        before, after = self.tokenized
        generated = self.forwarded[-1]
        self.assertEqual(after['messages'], generated['messages'])
        self.assertEqual(after['model'], 'qwen-bf16-base')
        self.assertEqual(generated['model'], 'qwen-bf16-base')
        self.assertEqual(after['chat_template_kwargs'], generated['chat_template_kwargs'])
        self.assertEqual(before['messages'][1:-1], messages[:-1])
        self.assertEqual(after['messages'], [before['messages'][0], before['messages'][-1]])
        self.assertEqual(after['messages'][0], {'role': 'system', 'content': gateway.DOCS_SYSTEM})
        self.assertEqual(json.loads(after['messages'][-1]['content']), {
            'question': question,
            'passages': [{'label': '[1]', **passages[0]}, {'label': '[2]', **passages[1]}],
        })
        self.assertNotIn('docsPassages', after)
        self.assertNotIn('docsPassages', generated)

        ordinary = await self.send()
        self.assertEqual(ordinary.content, SSE)
        self.assertNotIn('X-Chat-Docs-Mode', ordinary.headers)
        self.assertEqual(self.forwarded[-1]['messages'], [
            {'role': 'system', 'content': gateway.SYSTEM}, {'role': 'user', 'content': 'Hi'},
        ])

    async def test_docs_mode_validates_passages_and_rejects_image_input_before_admission(self):
        await self.configure(model_profile='bf16', image_input_enabled=True)
        entry = {'heading': 'Storage', 'passage': 'Documentation text'}
        invalid = [
            None, False, {}, [], [entry] * 4, [None], [{'heading': 'Storage'}],
            [{**entry, 'label': '[5]'}], [{**entry, 'heading': ' '}], [{**entry, 'passage': 10}],
            [{**entry, 'heading': '\U0001f4be' * 81}], [{**entry, 'passage': '\U0001f4be' * 901}],
        ]
        for passages in invalid:
            with self.subTest(passages=repr(passages)[:100]):
                response = await self.send(docsPassages=passages)
                self.assertEqual(response.status_code, 400)
                self.assertNotIn('X-Chat-Docs-Mode', response.headers)
        response = await self.send(docsPassages=[entry], image=image_uri())
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers['X-Chat-Rejection'], 'docs_image')
        self.assertNotIn('X-Chat-Docs-Mode', response.headers)
        self.assertEqual(self.model_requests, 0)
        self.assertEqual(self.tokenizer_requests, 0)
        self.assertEqual(self.forwarded, [])

    async def test_docs_mode_keeps_text_limits_separate_and_charges_evidence_to_request_bytes(self):
        passages = [{'heading': '\U0001f4be' * 80, 'passage': '\U0001f4be' * 900}] * 3
        settings = {**gateway.SETTINGS_DEFAULT, 'inputCharacters': 2, 'historyBytes': 128}
        response = await self.send(docsPassages=passages, settings=settings)
        self.assertEqual(response.status_code, 200)
        content = json.loads(self.forwarded[-1]['messages'][-1]['content'])
        self.assertEqual(content['passages'][-1], {'label': '[3]', **passages[-1]})
        response = await self.send(docsPassages=passages, settings={**settings, 'requestBytes': 1024})
        self.assertEqual(response.status_code, 413)
        self.assertNotIn('X-Chat-Docs-Mode', response.headers)
        self.assertEqual(self.tokenizer_requests, 1)

    async def test_docs_mode_rejects_oversized_grounded_prompt_without_generation(self):
        self.oversized = True
        response = await self.send(docsPassages=[{'heading': 'Storage', 'passage': 'Documentation text'}])
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('X-Chat-Docs-Mode', response.headers)
        self.assertEqual(self.tokenizer_requests, 1)
        self.assertEqual(self.forwarded, [])
        self.assertEqual(len(self.app.state.admission.active), 0)

    async def test_oversize_and_role_injection_rejected_without_generation(self):
        self.assertEqual((await self.send([{'role': 'system', 'content': 'injection'}])).status_code, 400)
        self.assertEqual((await self.send(max_tokens=100000)).status_code, 400)
        self.assertEqual((await self.send(temperature=True)).status_code, 400)
        self.assertEqual((await self.send([{'role': 'user', 'content': 'x' * 650000}])).status_code, 413)
        self.oversized = True
        self.assertEqual((await self.send()).status_code, 400)
        self.assertEqual(self.forwarded, [])

    async def test_larger_limits_and_exact_context_boundary(self):
        self.token_count = 40960 - 10240 - 32
        response = await self.send([{'role': 'user', 'content': '你' * 120000}], max_tokens=10240)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.forwarded[-1]['max_tokens'], 10240)
        self.assertEqual(response.headers['X-Context-Trimmed'], '0')
        self.token_count += 1
        self.assertEqual((await self.send(max_tokens=10240)).status_code, 400)
        self.assertEqual((await self.send(max_tokens=10241)).status_code, 400)
        self.assertEqual(len(self.forwarded), 1)

    async def test_frontend_settings_are_enforced_and_not_forwarded_to_model(self):
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 2048, 'outputTokens': 512,
                    'safetyTokens': 64, 'inputCharacters': 10, 'historyMessages': 3}
        self.token_count = 1472
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 200)
        self.assertNotIn('settings', self.forwarded[-1])
        self.token_count += 1
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 400)
        self.token_count = 100
        self.assertEqual((await self.send([{'role': 'user', 'content': 'x' * 11}], settings=settings, max_tokens=512)).status_code, 400)
        self.assertEqual((await self.send(settings=settings, max_tokens=513)).status_code, 400)
        for key, value in [('contextTokens', 40961), ('requestBytes', 640001), ('timeoutSeconds', True), ('safetyTokens', -1)]:
            invalid = {**settings, key: value}
            self.assertEqual((await self.send(settings=invalid, max_tokens=512)).status_code, 400)

    async def test_request_and_history_limits_cannot_be_bypassed(self):
        for key, value, status in [('historyBytes', 128, 400), ('requestBytes', 1024, 413)]:
            settings = {**gateway.SETTINGS_DEFAULT, key: value}
            response = await self.send([{'role': 'user', 'content': 'x' * 1500}], settings=settings)
            self.assertEqual(response.status_code, status)
        settings = {**gateway.SETTINGS_DEFAULT, 'historyMessages': 1}
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}]
        self.assertEqual((await self.send(messages, settings=settings)).status_code, 400)
        self.assertEqual(self.forwarded, [])

    async def test_context_ceiling_comes_from_running_model(self):
        self.model_context = 4096
        health = await self.client.get('/health', headers=user_auth())
        self.assertEqual(health.json()['limits']['contextTokens'], 4096)
        self.assertEqual((await self.send(settings=gateway.SETTINGS_DEFAULT)).status_code, 400)
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 4096, 'outputTokens': 512}
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 200)

    async def test_configured_timeout_covers_prefill_and_streaming_and_releases_capacity(self):
        settings = {**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1}
        self.tokenize_delay = 1.2
        self.assertEqual((await self.send(settings=settings)).status_code, 503)
        self.tokenize_delay = 0
        self.reply_delay = 1.2
        response = await self.send(settings=settings)
        self.assertIn(b'interrupted', response.content)
        self.assertTrue(self.closed)
        self.reply_delay = 0
        self.assertEqual((await self.send(settings=settings)).status_code, 200)

    async def test_failures_release_capacity_and_hide_upstream_details(self):
        self.tokenizer_failure = True
        self.assertEqual((await self.send()).status_code, 503)
        self.tokenizer_failure = False; self.tokenizer_invalid = True
        self.assertEqual((await self.send()).status_code, 503)
        self.tokenizer_invalid = False; self.model_failure = True
        response = await self.send()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('internal details', response.text)
        self.model_failure = False
        self.assertEqual((await self.send()).status_code, 200)

    async def until(self, predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(poll(), 3)

    async def controlled(self, prompt, **extra):
        self.blocked_prompts.add(prompt)
        return asyncio.create_task(self.send([{'role': 'user', 'content': prompt}], **extra))

    async def stop_tasks(self, tasks):
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_two_streams_run_without_waiting_for_a_pair(self):
        tasks = []
        try:
            tasks.append(await self.controlled('first'))
            await self.until(lambda: self.started_prompts == ['first'])
            # The first stream is already running before the second is submitted.
            tasks.append(await self.controlled('second'))
            await self.until(lambda: self.started_prompts == ['first', 'second'])
            self.assertEqual(len(self.app.state.admission.active), 2)
            self.assertFalse(any(task.done() for task in tasks))
            for event in self.prompt_releases.values():
                event.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertTrue(all(result.status_code == 200 and b'[DONE]' in result.content for result in results))
        finally:
            await self.stop_tasks(tasks)

    async def test_six_waiters_are_fifo_and_ninth_request_is_rejected(self):
        tasks = []
        try:
            for index in range(8):
                tasks.append(await self.controlled(str(index)))
                await self.until(lambda: len(self.app.state.admission.active) + len(self.app.state.admission.waiting) == len(tasks))
            await self.until(lambda: len(self.started_prompts) == 2)
            self.assertEqual(self.started_prompts, ['0', '1'])
            self.assertEqual(len(self.app.state.admission.waiting), 6)
            response = await self.send()
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers['X-Chat-Rejection'], 'queue_full')
            self.assertEqual(len(self.forwarded), 2)
            for index in range(8):
                await self.until(lambda: str(index) in self.prompt_releases)
                self.prompt_releases[str(index)].set()
                self.assertEqual((await asyncio.wait_for(tasks[index], 3)).status_code, 200)
            self.assertEqual(self.started_prompts, [str(index) for index in range(8)])
            self.assertEqual(len(self.app.state.admission.active), 0)
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            # Eight reservations plus twelve more reach the rate limit; overflow did not count.
            for _ in range(12):
                self.assertEqual((await self.send()).status_code, 200)
            self.assertEqual((await self.send()).headers['X-Chat-Rejection'], 'rate_limit')
        finally:
            await self.stop_tasks(tasks)

    async def test_cancelled_waiter_is_removed_and_never_reaches_model(self):
        tasks = []
        try:
            for prompt in ['first', 'second', 'cancelled']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            tasks[2].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[2]
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            self.assertEqual(len(self.forwarded), 2)
            tasks.append(await self.controlled('replacement'))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            self.prompt_releases['first'].set()
            await tasks[0]
            await self.until(lambda: 'replacement' in self.started_prompts)
            self.assertNotIn('cancelled', self.started_prompts)
        finally:
            await self.stop_tasks(tasks)

    async def test_queue_wait_counts_toward_timeout_and_frees_waiting_slot(self):
        tasks = []
        try:
            for prompt in ['first', 'second']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2)
            response = await asyncio.wait_for(self.send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1}), 3)
            self.assertEqual(response.status_code, 504)
            self.assertEqual(response.headers['X-Chat-Rejection'], 'queue_timeout')
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            self.assertEqual(len(self.forwarded), 2)
        finally:
            await self.stop_tasks(tasks)

    async def test_waiting_consumes_the_same_deadline_as_tokenization(self):
        tasks = []
        try:
            for prompt in ['first', 'second']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2)
            tasks.append(asyncio.create_task(self.send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1})))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            await asyncio.sleep(0.6)
            self.tokenize_delay = 0.6
            self.prompt_releases['first'].set()
            response = await asyncio.wait_for(tasks[2], 2)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(len(self.forwarded), 2)  # Queue wait did not reset the generation deadline.
        finally:
            await self.stop_tasks(tasks)

    async def test_rate_limit_is_global_even_when_slots_are_free(self):
        for _ in range(20):
            self.assertEqual((await self.send()).status_code, 200)
        response = await self.send()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['X-Chat-Rejection'], 'rate_limit')
        self.assertEqual(len(self.app.state.admission.active), 0)

    async def test_cancelling_one_active_stream_leaves_the_other_running(self):
        tasks = []
        try:
            for prompt in ['first', 'second', 'third']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2 and len(self.app.state.admission.waiting) == 1)
            tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[0]
            await self.until(lambda: 'third' in self.started_prompts)
            self.assertIn('first', self.finished_prompts)
            self.assertNotIn('second', self.finished_prompts)
            self.assertFalse(tasks[1].done())
            self.assertEqual(len(self.app.state.admission.active), 2)
        finally:
            await self.stop_tasks(tasks)

    async def test_active_stream_cancellation_releases_upstream(self):
        self.block = True
        first = asyncio.create_task(self.send())
        await asyncio.wait_for(self.entered.wait(), 3)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.closed)
        self.block = False
        self.assertEqual((await self.send()).status_code, 200)


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_eligible_fifo_and_newcomers_cannot_bypass(self):
        q = gateway.AdmissionQueue()
        a1, _, _ = q.reserve('a')
        a2, _, _ = q.reserve('a')
        b1, _, _ = q.reserve('b')
        b2, _, _ = q.reserve('b')
        c1, _, _ = q.reserve('c')
        self.assertEqual(q.reserve('a')[1:], ('user_capacity', None))
        q.release(b1)
        # A's older waiter is ineligible. B's waiter precedes C and starts.
        self.assertTrue(b2.done())
        self.assertFalse(a2.done())
        self.assertFalse(c1.done())
        d1, _, _ = q.reserve('d')
        q.release(a1)
        self.assertTrue(a2.done())
        q.release(b2)
        self.assertTrue(c1.done())
        self.assertFalse(d1.done())
        for ticket in [a1, a2, b1, b2, c1, d1]:
            q.release(ticket)
            q.release(ticket)
        self.assertFalse(q.active)
        self.assertFalse(q.waiting)
        self.assertFalse(q.owners)
        self.assertTrue(all(state['active'] == state['waiting'] == 0 for state in q.accounts.values()))

    async def test_rates_consumed_only_on_acceptance_and_expire_at_boundary(self):
        now = [100.0]
        q = gateway.AdmissionQueue(clock=lambda: now[0])
        for _ in range(6):
            ticket, rejection, retry = q.reserve('a')
            self.assertIsNone(rejection)
            q.release(ticket)
        self.assertEqual(q.reserve('a')[1:], ('user_rate_limit', 60))
        self.assertEqual(len(q.recent), 6)
        now[0] = 159.9
        self.assertEqual(q.reserve('a')[1:], ('user_rate_limit', 1))
        now[0] = 160
        q.prune()
        self.assertFalse(q.accounts)
        self.assertFalse(q.recent)
        ticket, _, _ = q.reserve('a')
        q.release(ticket)
        # No automatic refund for cancelled/failed accepted requests.
        self.assertEqual(len(q.accounts['a']['recent']), 1)

    async def test_cancelled_waiter_promotion_race_and_account_cleanup(self):
        now = [0]
        q = gateway.AdmissionQueue(clock=lambda: now[0])
        a1, _, _ = q.reserve('a')
        a2, _, _ = q.reserve('a')
        a2.cancel()
        q.release(a1)
        q.release(a2)
        self.assertFalse(q.active)
        self.assertFalse(q.waiting)
        self.assertEqual(q.accounts['a']['active'], 0)
        self.assertEqual(q.accounts['a']['waiting'], 0)
        now[0] = 60
        q.prune()
        self.assertFalse(q.accounts)

    async def test_global_capacity_rate_and_config_bounds(self):
        q = gateway.AdmissionQueue()
        tickets = [q.reserve(str(index))[0] for index in range(8)]
        self.assertEqual(q.reserve('overflow')[1:], ('queue_full', None))
        self.assertEqual(len(q.recent), 8)
        for ticket in tickets:
            q.release(ticket)
        for index in range(12):
            ticket, _, _ = q.reserve('more-' + str(index))
            q.release(ticket)
        self.assertEqual(q.reserve('overflow')[1:], ('rate_limit', 60))
        for options in [{'user_concurrency': 3}, {'user_concurrency': 0}, {'user_queue_limit': -1},
                        {'user_queue_limit': 7}, {'user_rate_limit': 21}, {'user_rate_limit': 0}]:
            with self.assertRaises(ValueError):
                gateway.AdmissionQueue(**options)


class CapacityEnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def test_source_fallback_and_explicit_bounded_capacity(self):
        cases = [({}, (2,6,20)), ({'MAX_NUM_SEQS':'1'}, (1,6,20)),
                 ({'MAX_NUM_SEQS':'8'}, (2,6,20)),
                 ({'MAX_NUM_SEQS':'8','CHAT_MAX_CONCURRENT':'8','CHAT_MAX_WAITING':'24','CHAT_RATE_LIMIT_PER_MINUTE':'80'}, (8,24,80))]
        for values, expected in cases:
            with patch.dict(os.environ, values, clear=True):
                self.assertEqual(gateway.capacity_from_env(), expected)
                q=gateway.AdmissionQueue(*expected)
                self.assertEqual((q.user_concurrency,q.user_queue_limit,q.user_rate_limit),(1,1,6))
        for key in ['MAX_NUM_SEQS','CHAT_MAX_CONCURRENT','CHAT_MAX_WAITING','CHAT_RATE_LIMIT_PER_MINUTE',
                    'CHAT_USER_MAX_CONCURRENT','CHAT_USER_MAX_WAITING','CHAT_USER_RATE_LIMIT_PER_MINUTE']:
            for value in ['', '-1', '1.5', '+2', ' 2', '1_0', '81']:
                with self.subTest(key=key,value=value),patch.dict(os.environ,{key:value},clear=True),self.assertRaises(ValueError):
                    gateway.create_app(TOKEN)
        for values in [{'MAX_NUM_SEQS':'0'},{'MAX_NUM_SEQS':'9'},{'CHAT_MAX_CONCURRENT':'0'},
                       {'CHAT_MAX_CONCURRENT':'8'}, {'CHAT_MAX_WAITING':'25'}, {'CHAT_RATE_LIMIT_PER_MINUTE':'0'}]:
            with patch.dict(os.environ,values,clear=True),self.assertRaises(ValueError): gateway.create_app(TOKEN)

    async def test_eighty_global_and_six_per_account_limits(self):
        q=gateway.AdmissionQueue(8,24,80)
        for _ in range(6):
            ticket,why,_=q.reserve('same'); self.assertIsNone(why); q.release(ticket)
        self.assertEqual(q.reserve('same')[1], 'user_rate_limit')
        for i in range(6,80):
            ticket,why,_=q.reserve(str(i)); self.assertIsNone(why); q.release(ticket)
        self.assertEqual(q.reserve('overflow')[1], 'rate_limit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
