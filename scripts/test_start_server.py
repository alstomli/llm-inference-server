"""Run inside the inference image: exercise startup without loading a GPU model."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fake = self.root / 'vllm'
        fake.write_text('''#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
if os.environ.get('FAKE_ARG_FILE'):
    with open(os.environ['FAKE_ARG_FILE'], 'w') as f:
        json.dump(sys.argv[1:], f)
if os.environ.get('FAKE_VLLM_EXIT'):
    sys.exit(int(os.environ['FAKE_VLLM_EXIT']))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        if self.path == '/v1/models':
            self.wfile.write(json.dumps({'data': [{'id': 'qwen', 'max_model_len': 40960}]}).encode())
        else:
            self.wfile.write(b'vllm:num_requests_running 0\\n')
    def log_message(self, *args):
        pass
ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
''')
        fake.chmod(0o755)
        self.token = 'test-only-token-' + 'x' * 40
        self.token_file = self.root / 'token'
        self.token_file.write_text(self.token)
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.chat_port = s.getsockname()[1]
        self.env = {
            **os.environ,
            'PATH': str(self.root) + ':' + os.environ['PATH'],
            'METRICS_ENABLED': 'true',
            'METRICS_HOST': '127.0.0.1',
            'METRICS_PORT': str(self.port),
            'METRICS_TOKEN_FILE': str(self.token_file),
            'CHAT_ENABLED': 'false',
            'CHAT_HOST': '127.0.0.1',
            'CHAT_PORT': str(self.chat_port),
            'CHAT_TOKEN_FILE': str(self.token_file),
        }
        self.process = None

    def tearDown(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.process:
            self.process.communicate(timeout=10)
        self.temp.cleanup()

    def start(self, **overrides):
        self.process = subprocess.Popen(
            ['/usr/local/bin/start-server'], env={**self.env, **overrides},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def ready(self, port, authorized=False, path='/metrics'):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.fail('Startup exited: ' + self.process.communicate()[0])
            headers = {'Authorization': 'Bearer ' + self.token} if authorized else {}
            try:
                with urlopen(Request(f'http://127.0.0.1:{port}{path}', headers=headers), timeout=1) as response:
                    return response.read()
            except (HTTPError, URLError, OSError):
                time.sleep(0.1)
        self.fail('Service did not become ready')

    def closed(self, port):
        with socket.socket() as s:
            self.assertNotEqual(s.connect_ex(('127.0.0.1', port)), 0)

    def test_two_cold_starts_launch_authenticated_relay_and_stop_cleanly(self):
        for _ in range(2):
            self.start()
            self.assertIn(b'vllm:num_requests_running', self.ready(self.port, True))
            with self.assertRaises(HTTPError) as error:
                urlopen(f'http://127.0.0.1:{self.port}/metrics', timeout=1)
            self.assertEqual(error.exception.code, 401)
            self.process.terminate()
            self.process.communicate(timeout=10)
            self.assertEqual(self.process.returncode, 143)
            self.closed(8000)
            self.closed(self.port)

    def test_missing_token_refuses_startup(self):
        self.start(METRICS_TOKEN_FILE=str(self.root / 'missing'))
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('token file is missing', output)
        self.closed(8000)

    def test_model_failure_stops_relay(self):
        self.start(FAKE_VLLM_EXIT='7')
        self.process.communicate(timeout=10)
        self.assertEqual(self.process.returncode, 7)
        self.closed(self.port)

    def test_relay_failure_stops_model(self):
        self.start()
        self.ready(self.port, True)
        children = Path(f'/proc/{self.process.pid}/task/{self.process.pid}/children').read_text().split()
        relay_pid = next(int(pid) for pid in children if b'/usr/local/lib/metrics-relay.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
        os.kill(relay_pid, signal.SIGKILL)
        self.process.communicate(timeout=10)
        self.assertEqual(self.process.returncode, 137)
        self.closed(8000)

    def test_disabled_preserves_vllm_only_startup(self):
        self.start(METRICS_ENABLED='false')
        self.ready(8000)
        self.closed(self.port)

    def test_ttft_buckets_cover_tenths_through_ten_seconds_and_keep_other_bounds(self):
        args_file = self.root / 'vllm-args.json'
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file))
        self.ready(8000)
        args = json.loads(args_file.read_text())
        buckets = json.loads(args[args.index('--custom-histogram-buckets') + 1])
        self.assertEqual(set(buckets), {'time_to_first_token'})
        expected = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.25, 0.75]
        expected += [i / 10 for i in range(1, 101)]
        expected += [20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
        self.assertEqual(buckets['time_to_first_token'], sorted(expected))

    def test_chat_only_starts_its_gateway_without_metrics(self):
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true')
        self.assertIn(b'"ready":true', self.ready(self.chat_port, True, '/health'))
        self.closed(self.port)

    def test_both_gateways_start_and_chat_failure_stops_every_service(self):
        self.start(CHAT_ENABLED='true')
        self.ready(self.port, True)
        self.ready(self.chat_port, True, '/health')
        children = Path(f'/proc/{self.process.pid}/task/{self.process.pid}/children').read_text().split()
        chat_pid = next(int(pid) for pid in children if b'/usr/local/lib/chat-gateway.py' in Path(f'/proc/{pid}/cmdline').read_bytes())
        os.kill(chat_pid, signal.SIGKILL)
        self.process.communicate(timeout=10)
        self.assertEqual(self.process.returncode, 137)
        self.closed(8000)
        self.closed(self.port)

    def test_missing_chat_token_refuses_startup(self):
        self.start(CHAT_ENABLED='true', CHAT_TOKEN_FILE=str(self.root / 'missing-chat-token'))
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('Chat token file is missing', output)
        self.closed(8000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
