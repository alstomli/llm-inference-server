"""Local integration fixture: real gateway, deliberately fake model, no GPU."""
import base64
from collections import deque
from hashlib import sha256
import importlib.util
import json
import os
import signal
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uvicorn

TOKEN = 'integration-test-token-' + 'x' * 40
IMAGE_INPUT = os.environ.get('CHAT_FIXTURE_IMAGE_INPUT') == 'true'
ADAPTER = IMAGE_INPUT or os.environ.get('CHAT_FIXTURE_ADAPTER') == 'true'
IMAGE_PROMPT_TOKENS = 900
tokenized_images = deque(maxlen=32)
tokenized_lock = threading.Lock()


def image_prompt(body):
    messages = body['messages']
    latest = messages[-1]
    text, image = latest['content']
    assert latest['role'] == 'user' and body['model'] == 'qwen-bf16-base'
    assert all(isinstance(message['content'], str) for message in messages[:-1])
    assert text['type'] == 'text' and isinstance(text['text'], str) and text['text'].strip()
    assert image['type'] == 'image_url'
    media_type, encoded = image['image_url']['url'].split(',', 1)
    assert media_type in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64')
    assert base64.b64decode(encoded, validate=True)
    return text['text']


def image_signature(body):
    fields = [body['model'], body['messages'], body.get('chat_template_kwargs')]
    return sha256(json.dumps(fields, sort_keys=True).encode()).digest()


class Model(BaseHTTPRequestHandler):
    def do_GET(self):
        cards = ([{'id': 'qwen-bf16-base', 'max_model_len': 40960}, {'id': 'qwen-bf16', 'parent': 'qwen-bf16-base'}]
                 if ADAPTER else [{'id': 'qwen-bf16', 'max_model_len': 40960}])
        data = json.dumps({'data': cards}).encode() if self.path == '/v1/models' else b''
        self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        prompt = body['messages'][-1]['content']
        is_image = isinstance(prompt, list)
        if is_image:
            try:
                assert IMAGE_INPUT
                prompt = image_prompt(body)
            except (AssertionError, KeyError, ValueError, AttributeError, TypeError):
                self.send_error(400, 'Invalid fixture image prompt')
                return
        prompt_tokens = IMAGE_PROMPT_TOKENS if is_image else 100
        if self.path == '/tokenize':
            count = 50000 if len(body['messages']) > 2 or prompt == 'oversized' else prompt_tokens
            if is_image and count == prompt_tokens:
                with tokenized_lock:
                    tokenized_images.append(image_signature(body))
            data = json.dumps({'count': count}).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        if is_image:
            signature = image_signature(body)
            with tokenized_lock:
                if signature not in tokenized_images:
                    self.send_error(400, 'Image generation prompt did not match tokenization')
                    return
                tokenized_images.remove(signature)
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        slow = prompt == 'slow'
        try:
            if is_image:
                words = ['Fixture received ', f"the image using {body['model']}."]
            elif ADAPTER:
                words = [f"Fixture response using {body['model']}. "] * 4
            else:
                words = ['Hello ', 'from ', 'the integration fixture.']
            for word in (['sample '] * 100 if slow else words):
                data = {'choices': [{'delta': {'content': word}}]}
                self.wfile.write(('data: ' + json.dumps(data) + '\n\n').encode()); self.wfile.flush(); time.sleep(0.1)
            reason = 'length' if prompt == 'length-test' else 'stop'
            final = {'choices': [{'delta': {}, 'finish_reason': reason}]}
            self.wfile.write(('data: ' + json.dumps(final) + '\n\n').encode())
            if body.get('stream_options', {}).get('include_usage'):
                usage = {'choices': [], 'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 4, 'total_tokens': prompt_tokens + 4}}
                self.wfile.write(('data: ' + json.dumps(usage, separators=(',', ':')) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except ConnectionError:
            pass

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    model_port = int(os.environ.get('CHAT_FIXTURE_MODEL_PORT', '8000'))
    server = ThreadingHTTPServer(('127.0.0.1', model_port), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    if binary := os.environ.get('CHAT_GATEWAY_BINARY'):
        # Optional real Go gateway for the existing frontend integration runner.
        # This fixture always supplies disposable credentials and a fake model.
        with tempfile.TemporaryDirectory() as temporary:
            token_file = Path(temporary) / 'token'
            token_file.write_text(TOKEN)
            env = {**os.environ, 'CHAT_TOKEN_FILE': str(token_file),
                   'CHAT_HOST': '127.0.0.1' if os.environ.get('CHAT_TEST_FIXTURE_PYTHON') else '0.0.0.0',
                   'CHAT_PORT': os.environ.get('CHAT_FIXTURE_PORT', '8001'),
                   'VLLM_PORT': str(model_port), 'VLLM_API_KEY': '',
                   'SERVED_MODEL_NAME': 'qwen-bf16',
                   'MODEL_PROFILE': 'bf16',
                   'CHAT_IMAGE_INPUT_ENABLED': str(IMAGE_INPUT).lower(),
                   'LORA_ADAPTER_PATH': 'fixture-adapter' if ADAPTER else '',
                   'VLLM_BASE_MODEL_NAME': 'qwen-bf16-base' if ADAPTER else 'qwen-bf16',
                   'CHAT_REQUIRE_USER_ID': 'true', 'MAX_MODEL_LEN': '40960',
                   'MAX_NUM_SEQS': '2', 'CHAT_USER_MAX_CONCURRENT': '1',
                   'CHAT_USER_MAX_WAITING': '1', 'CHAT_USER_RATE_LIMIT_PER_MINUTE': '6'}
            child = subprocess.Popen([binary], env=env)
            def stop_fixture(signum, frame):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, stop_fixture)
            signal.signal(signal.SIGINT, stop_fixture)
            try:
                raise SystemExit(child.wait())
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                server.shutdown()
                server.server_close()
    spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('chat-gateway.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    uvicorn.run(module.create_app(TOKEN, require_user_id=True, upstream=f'http://127.0.0.1:{model_port}',
        model='qwen-bf16', base_model='qwen-bf16-base' if ADAPTER else None,
        model_profile='bf16', image_input_enabled=IMAGE_INPUT),
        host='127.0.0.1' if os.environ.get('CHAT_TEST_FIXTURE_PYTHON') else '0.0.0.0', port=int(os.environ.get('CHAT_FIXTURE_PORT', '8001')), access_log=False, log_level='warning')
