"""Local integration fixture: real gateway, deliberately fake model, no GPU."""
import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
import uvicorn

TOKEN = 'integration-test-token-' + 'x' * 40


class Model(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({'data': [{'id': 'qwen', 'max_model_len': 40960}]}).encode() if self.path == '/v1/models' else b''
        self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            data = json.dumps({'count': 50000 if len(body['messages']) > 2 or body['messages'][-1]['content'] == 'oversized' else 100}).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
        slow = body['messages'][-1]['content'] == 'slow'
        try:
            for word in (['sample '] * 100 if slow else ['Hello ', 'from ', 'the integration fixture.']):
                data = {'choices': [{'delta': {'content': word}}]}
                self.wfile.write(('data: ' + json.dumps(data) + '\n\n').encode()); self.wfile.flush(); time.sleep(0.1)
            reason = 'length' if body['messages'][-1]['content'] == 'length-test' else 'stop'
            final = {'choices': [{'delta': {}, 'finish_reason': reason}]}
            self.wfile.write(('data: ' + json.dumps(final) + '\n\n').encode())
            if body.get('stream_options', {}).get('include_usage'):
                self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":4,"total_tokens":104}}\n\n')
            self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    model_port = int(os.environ.get('CHAT_FIXTURE_MODEL_PORT', '8000'))
    server = ThreadingHTTPServer(('127.0.0.1', model_port), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('chat-gateway.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    uvicorn.run(module.create_app(TOKEN, upstream=f'http://127.0.0.1:{model_port}'), host='127.0.0.1' if os.environ.get('CHAT_TEST_FIXTURE_PYTHON') else '0.0.0.0', port=int(os.environ.get('CHAT_FIXTURE_PORT', '8001')), access_log=False, log_level='warning')
