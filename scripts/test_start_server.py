"""Run inside the inference image: exercise startup without loading a GPU model."""
import json
import struct
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
            self.wfile.write(json.dumps({'data': [{'id': os.environ.get('SERVED_MODEL_NAME', 'qwen'), 'max_model_len': int(os.environ.get('MAX_MODEL_LEN', '40960'))}]}).encode())
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
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '2')
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

    def test_single_sequence_override_also_lowers_gateway_concurrency(self):
        args_file = self.root / 'vllm-args.json'
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true', MAX_NUM_SEQS='1', FAKE_ARG_FILE=str(args_file))
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '1')
        self.assertEqual(health['admission']['maxConcurrentRequests'], 1)
        self.assertEqual(health['admission']['maxWaitingRequests'], 6)

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


    def bf16_environment(self):
        root = self.root / 'bf16'
        root.mkdir()
        (root / 'config.json').write_text('{"model_id":"Qwen/Qwen3.8-27B"}')
        source = self.root / 'checkpoint'
        source.mkdir()
        (source / 'config.json').write_text('{"dtype":"bfloat16"}')
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
        (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0\0')
        (self.root / 'huggingface_hub.py').write_text('''
import hashlib, os, shutil, types
from pathlib import Path
class HfApi:
    def model_info(self, *args, **kwargs):
        files = [types.SimpleNamespace(rfilename=p.name, size=p.stat().st_size,
                 lfs=types.SimpleNamespace(sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                 for p in Path(os.environ['FAKE_CHECKPOINT']).iterdir()]
        return types.SimpleNamespace(sha='a' * 40, siblings=files)
def hf_hub_download(model_id, filename, revision, local_dir):
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    target = Path(local_dir) / filename
    shutil.copyfile(Path(os.environ['FAKE_CHECKPOINT']) / filename, target)
    return str(target)
''')
        return {'MODEL_PROFILE': 'bf16', 'BF16_ROOT': str(root),
                'FAKE_CHECKPOINT': str(source), 'PYTHONPATH': str(self.root)}

    def test_bf16_profile_stages_original_weights_and_advertises_its_actual_limits(self):
        args_file = self.root / 'vllm-args.json'
        env = self.bf16_environment()
        self.start(CHAT_ENABLED='true', FAKE_ARG_FILE=str(args_file), **env)
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertEqual(health['model'], 'qwen-bf16')
        self.assertEqual(health['limits']['contextTokens'], 40960)
        self.assertEqual(health['admission']['maxConcurrentRequests'], 2)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[0:2], ['serve', str(Path(env['BF16_ROOT']) / 'model')])
        for key, value in [('--dtype', 'bfloat16'), ('--cpu-offload-gb', '30'),
                           ('--max-model-len', '40960'), ('--max-num-seqs', '2')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--enable-prefix-caching', args)
        self.assertIn('--enforce-eager', args)
        self.assertNotIn('--language-model-only', args)
        state = json.loads((Path(env['BF16_ROOT']) / 'model-manifest.json').read_text())
        self.assertTrue(state['complete'])
        self.assertEqual(state['tensor_dtypes'], {'BF16': 1})
        self.assertIn(b'vllm:num_requests_running', self.ready(self.port, True))

    def test_bf16_accepts_the_same_context_and_concurrency_overrides(self):
        args_file = self.root / 'vllm-args.json'
        self.start(CHAT_ENABLED='true', FAKE_ARG_FILE=str(args_file),
                   MAX_MODEL_LEN='8192', MAX_NUM_SEQS='1', CPU_OFFLOAD_GB='40',
                   **self.bf16_environment())
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertEqual(health['limits']['contextTokens'], 8192)
        self.assertEqual(health['admission']['maxConcurrentRequests'], 1)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--cpu-offload-gb') + 1], '40')
        self.assertEqual(args[args.index('--max-model-len') + 1], '8192')
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '1')

    def test_invalid_profile_fails_before_starting_services(self):
        self.start(MODEL_PROFILE='unknown')
        output = self.process.communicate(timeout=10)[0]
        self.assertNotEqual(self.process.returncode, 0)
        self.assertIn('MODEL_PROFILE must be', output)
        self.closed(8000)
        self.closed(self.port)

    def test_bf16_refuses_an_unmounted_workspace(self):
        mountpoint = self.root / 'mountpoint'
        mountpoint.write_text('#!/bin/sh\nexit 1\n')
        mountpoint.chmod(0o755)
        self.start(MODEL_PROFILE='bf16')
        output = self.process.communicate(timeout=10)[0]
        self.assertNotEqual(self.process.returncode, 0)
        self.assertIn('attached persistent volume', output)
        self.closed(8000)

    def test_jupyter_without_token_fails_before_starting_services(self):
        self.start(JUPYTER_ENABLED='true', JUPYTER_PASSWORD='', JUPYTER_TOKEN='')
        output = self.process.communicate(timeout=10)[0]
        self.assertNotEqual(self.process.returncode, 0)
        self.assertIn('Jupyter requires', output)
        self.closed(8000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
