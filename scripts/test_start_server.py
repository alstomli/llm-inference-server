"""Run inside the inference image: exercise startup without loading a GPU model."""
import json
import struct
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PINNED_4BIT = '71dface38d4cbfa34121f1d446d4a0fa1025f13c'


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
            base = sys.argv[sys.argv.index('--served-model-name') + 1]
            cards = [{'id': base, 'max_model_len': int(os.environ.get('MAX_MODEL_LEN', '40960'))}]
            if '--lora-modules' in sys.argv:
                module = json.loads(sys.argv[sys.argv.index('--lora-modules') + 1])
                cards.append({'id': module['name'], 'root': module['path'], 'parent': module['base_model_name'], 'max_model_len': None})
            self.wfile.write(json.dumps({'data': cards}).encode())
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
            'JUPYTER_ENABLED': 'false',
        }
        for key in ('MODEL_PROFILE', 'MODEL_ID', 'SERVED_MODEL_NAME', 'BF16_ROOT', 'QWEN_4BIT_ROOT',
                    'FAKE_HUB_SHA', 'CHAT_IMAGE_INPUT_ENABLED',
                    'VLLM_BASE_MODEL_NAME', 'LORA_ADAPTER_PATH', 'MAX_LORA_RANK',
                    'CPU_OFFLOAD_GB', 'MAX_NUM_SEQS', 'MAX_MODEL_LEN', 'CHAT_GATEWAY_IMPLEMENTATION',
                    'CHAT_WEB_SEARCH_ENABLED', 'CHAT_WEB_SEARCH_MODEL_VERIFIED',
                    'CHAT_WEB_SEARCH_TOOL_PARSER', 'CHAT_WEB_SEARCH_VERIFIED_MODEL',
                    'KV_OFFLOAD_CPU_GIB', 'CUDA_DECODE_GRAPHS', 'KV_RUNTIME_OBSERVE', 'KV_RUNTIME_PROOF'):
            self.env.pop(key, None)
        # Disposable BF16 checkpoint; ordinary tests leave the profile unset to exercise its default.
        self.env.update(self.bf16_environment())
        self.env.pop('MODEL_PROFILE', None)
        self.process = None
        self.launcher = '/usr/local/bin/start-server'

    def tearDown(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.process:
            self.process.communicate(timeout=10)
        self.temp.cleanup()

    def start(self, **overrides):
        self.process = subprocess.Popen(
            [self.launcher], env={**self.env, **overrides},
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
        self.assertNotIn('--enable-lora', args)
        self.assertNotIn('--enable-auto-tool-choice', args)
        self.assertNotIn('--tool-call-parser', args)
        self.assertIn('--enforce-eager', args)
        self.assertNotIn('--kv-transfer-config', args)
        self.assertNotIn('--compilation-config', args)
        buckets = json.loads(args[args.index('--custom-histogram-buckets') + 1])
        self.assertEqual(set(buckets), {'time_to_first_token'})
        expected = [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.25, 0.75]
        expected += [i / 10 for i in range(1, 101)]
        expected += [20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]
        self.assertEqual(buckets['time_to_first_token'], sorted(expected))

    def test_explicit_native_offload_and_decode_graphs_preserve_serving_limits(self):
        # Resource guard has its own boundary tests. This fixture isolates argv
        # construction from the CI runner's intentionally small shared memory.
        guard = self.root / 'guard.py'
        guard.write_text('pass\n')
        launcher = self.root / 'start-server'
        launcher.write_text(Path(self.launcher).read_text().replace('/usr/local/lib/kv-runtime-config.py', str(guard)))
        launcher.chmod(0o755)
        self.launcher = str(launcher)
        args_file = self.root / 'feature-args.json'
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file), CPU_OFFLOAD_GB='0',
                   KV_OFFLOAD_CPU_GIB='64', CUDA_DECODE_GRAPHS='true')
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertNotIn('--enforce-eager', args)
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '2')
        self.assertEqual(args[args.index('--max-model-len') + 1], '40960')
        self.assertEqual(args[args.index('--gpu-memory-utilization') + 1], '0.90')
        self.assertEqual(args[args.index('--cpu-offload-gb') + 1], '0')
        self.assertNotIn('--max-num-batched-tokens', args)
        self.assertNotIn('--kv-cache-memory-bytes', args)
        self.assertEqual(args[args.index('--mamba-cache-mode') + 1], 'align')
        self.assertEqual(json.loads(args[args.index('--compilation-config') + 1]),
                         {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1, 2]})
        self.assertEqual(json.loads(args[args.index('--kv-transfer-config') + 1]),
                         {'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
                          'kv_connector_extra_config': {'cpu_bytes_to_use': 68719476736,
                          'blocks_per_chunk': 1, 'offload_prompt_only': False}})
        self.assertNotIn('--worker-cls', args)
        self.assertNotIn('--middleware', args)
        self.assertIn('--cudagraph-metrics', args)
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]), {'image': 1, 'video': 0})

    def test_eight_sequence_graph_profile_uses_padded_capture_buckets(self):
        args_file = self.root / 'eight-args.json'
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file), CPU_OFFLOAD_GB='0',
                   MAX_NUM_SEQS='8', CUDA_DECODE_GRAPHS='true')
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '8')
        self.assertEqual(args[args.index('--max-model-len') + 1], '40960')
        self.assertEqual(args[args.index('--cpu-offload-gb') + 1], '0')
        self.assertEqual(json.loads(args[args.index('--compilation-config') + 1]),
                         {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1, 2, 4, 8]})
        self.assertNotIn('--enforce-eager', args)
        self.assertNotIn('--worker-cls', args)
        self.assertNotIn('--middleware', args)

    def test_graph_profile_still_rejects_cpu_weight_offload(self):
        self.start(CPU_OFFLOAD_GB='1', MAX_NUM_SEQS='8', CUDA_DECODE_GRAPHS='true')
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('Decode graphs require zero weight offload', output)
        self.closed(8000)

    def test_proof_flags_preserve_lora_registration_and_base_alias(self):
        guard = self.root / 'guard.py'
        guard.write_text('pass\n')
        launcher = self.root / 'start-server'
        launcher.write_text(Path(self.launcher).read_text().replace('/usr/local/lib/kv-runtime-config.py', str(guard)))
        launcher.chmod(0o755)
        self.launcher = str(launcher)
        args_file = self.root / 'proof-args.json'
        adapter = self.lora_adapter(self.env)
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file), CPU_OFFLOAD_GB='0',
                   KV_OFFLOAD_CPU_GIB='64', CUDA_DECODE_GRAPHS='true',
                   KV_RUNTIME_OBSERVE='true', KV_RUNTIME_PROOF='true', LORA_ADAPTER_PATH=adapter)
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--worker-cls') + 1], 'kv_runtime.Worker')
        self.assertEqual(args[args.index('--middleware') + 1], 'kv_runtime.ProofMiddleware')
        self.assertEqual(args[args.index('--served-model-name') + 1], 'qwen-bf16-base')
        self.assertIn('--enable-lora', args)
        self.assertEqual(args[args.index('--max-lora-rank') + 1], '8')
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]), {'image': 1, 'video': 0})

    def test_every_concurrency_from_two_to_eight_derives_its_decode_capture_sizes(self):
        expected = {2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 4], 5: [1, 2, 4, 5],
                    6: [1, 2, 4, 6], 7: [1, 2, 4, 7], 8: [1, 2, 4, 8]}
        for sequences, captures in expected.items():
            with self.subTest(sequences=sequences):
                args_file = self.root / f'graph-{sequences}-args.json'
                self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file), CPU_OFFLOAD_GB='0',
                           MAX_NUM_SEQS=str(sequences), CUDA_DECODE_GRAPHS='true')
                self.ready(8000)
                args = json.loads(args_file.read_text())
                self.process.terminate()
                self.process.communicate(timeout=10)
                self.assertEqual(args[args.index('--max-num-seqs') + 1], str(sequences))
                self.assertEqual(json.loads(args[args.index('--compilation-config') + 1]),
                                 {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': captures})

    def test_concurrency_outside_two_to_eight_refuses_start(self):
        for graphs in ('false', 'true'):
            for sequences in ('0', '1', '9', '08', ' 4', '4x', ''):
                with self.subTest(graphs=graphs, sequences=sequences):
                    self.start(CPU_OFFLOAD_GB='0', MAX_NUM_SEQS=sequences, CUDA_DECODE_GRAPHS=graphs)
                    output = self.process.communicate(timeout=10)[0]
                    self.assertEqual(self.process.returncode, 1)
                    self.assertIn('MAX_NUM_SEQS must be an integer from 2 through 8', output)
                    self.closed(8000)

    def test_invalid_native_capacity_fails_closed(self):
        self.start(KV_OFFLOAD_CPU_GIB='32')
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('KV_OFFLOAD_CPU_GIB must be 0 or 64', output)
        self.closed(8000)

    def test_chat_only_starts_its_gateway_without_metrics(self):
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true')
        self.assertIn(b'"ready":true', self.ready(self.chat_port, True, '/health'))
        self.closed(self.port)

    def test_web_search_requires_explicit_model_parser_verification(self):
        self.start(CHAT_WEB_SEARCH_ENABLED='true')
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('explicit model/parser compatibility verification', output)
        self.closed(8000)

    def test_web_search_uses_only_explicitly_selected_parser(self):
        args_file = self.root / 'web-vllm-args.json'
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file),
                   CHAT_WEB_SEARCH_ENABLED='true', CHAT_WEB_SEARCH_MODEL_VERIFIED='true',
                   CHAT_WEB_SEARCH_VERIFIED_MODEL='qwen-bf16', CHAT_WEB_SEARCH_TOOL_PARSER='fixture-parser')
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertIn('--enable-auto-tool-choice', args)
        self.assertEqual(args[args.index('--tool-call-parser') + 1], 'fixture-parser')

    def test_parser_can_be_staged_without_advertising_gateway_search(self):
        args_file = self.root / 'staged-parser-args.json'
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true', FAKE_ARG_FILE=str(args_file),
                   CHAT_WEB_SEARCH_ENABLED='false', CHAT_WEB_SEARCH_MODEL_VERIFIED='false',
                   CHAT_WEB_SEARCH_TOOL_PARSER='fixture-parser')
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        args = json.loads(args_file.read_text())
        self.assertIn('--enable-auto-tool-choice', args)
        self.assertEqual(args[args.index('--tool-call-parser') + 1], 'fixture-parser')
        self.assertFalse(health['webSearchAvailable'])
        self.assertEqual(health['webSearch']['reason'], 'disabled')

    def test_concurrency_override_sets_gateway_active_and_three_times_waiting_slots(self):
        args_file = self.root / 'vllm-args.json'
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true', MAX_NUM_SEQS='4', FAKE_ARG_FILE=str(args_file))
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '4')
        self.assertEqual(health['admission']['maxConcurrentRequests'], 4)
        self.assertEqual(health['admission']['maxWaitingRequests'], 12)

    def test_both_gateways_start_and_chat_failure_stops_every_service(self):
        self.start(CHAT_ENABLED='true')
        self.ready(self.port, True)
        self.ready(self.chat_port, True, '/health')
        children = Path(f'/proc/{self.process.pid}/task/{self.process.pid}/children').read_text().split()
        chat_pid = next(int(pid) for pid in children if b'/usr/local/bin/chat-gateway' in Path(f'/proc/{pid}/cmdline').read_bytes())
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

    def test_explicit_python_rollback_keeps_authenticated_health(self):
        self.start(METRICS_ENABLED='false', CHAT_ENABLED='true', CHAT_GATEWAY_IMPLEMENTATION='python')
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertTrue(health['accountQuotasEnforced'])
        self.assertEqual(health['accountQuotaVersion'], 1)

    def test_unknown_gateway_implementation_refuses_startup(self):
        self.start(CHAT_ENABLED='true', CHAT_GATEWAY_IMPLEMENTATION='unknown')
        output = self.process.communicate(timeout=10)[0]
        self.assertEqual(self.process.returncode, 1)
        self.assertIn('CHAT_GATEWAY_IMPLEMENTATION must be', output)
        self.closed(8000)


    def bf16_environment(self):
        root = self.root / 'bf16'
        root.mkdir(exist_ok=True)
        (root / 'config.json').write_text('{"model_id":"Qwen/Qwen3.8-27B"}')
        source = self.root / 'checkpoint'
        source.mkdir(exist_ok=True)
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
        return types.SimpleNamespace(sha=os.environ.get('FAKE_HUB_SHA', 'a' * 40), siblings=files)
def hf_hub_download(model_id, filename, revision, local_dir):
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    target = Path(local_dir) / filename
    shutil.copyfile(Path(os.environ['FAKE_CHECKPOINT']) / filename, target)
    return str(target)
''')
        return {'MODEL_PROFILE': 'bf16', 'BF16_ROOT': str(root),
                'FAKE_CHECKPOINT': str(source), 'PYTHONPATH': str(self.root) + os.pathsep + os.environ.get('PYTHONPATH', '')}

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
        for key, value in [('--dtype', 'bfloat16'), ('--cpu-offload-gb', '40'),
                           ('--max-model-len', '40960'), ('--max-num-seqs', '2')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertIn('--enable-prefix-caching', args)
        self.assertIn('--enforce-eager', args)
        self.assertNotIn('--language-model-only', args)
        self.assertNotIn('AWQ', json.dumps(args))
        self.assertNotIn('--enable-lora', args)
        state = json.loads((Path(env['BF16_ROOT']) / 'model-manifest.json').read_text())
        self.assertTrue(state['complete'])
        self.assertEqual(state['tensor_dtypes'], {'BF16': 1})
        self.assertIn(b'vllm:num_requests_running', self.ready(self.port, True))

    def test_bf16_accepts_the_same_context_and_concurrency_overrides(self):
        args_file = self.root / 'vllm-args.json'
        self.start(CHAT_ENABLED='true', FAKE_ARG_FILE=str(args_file),
                   MAX_MODEL_LEN='8192', MAX_NUM_SEQS='6', CPU_OFFLOAD_GB='45',
                   **self.bf16_environment())
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertEqual(health['limits']['contextTokens'], 8192)
        self.assertEqual(health['admission']['maxConcurrentRequests'], 6)
        self.assertEqual(health['admission']['maxWaitingRequests'], 18)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--cpu-offload-gb') + 1], '45')
        self.assertEqual(args[args.index('--max-model-len') + 1], '8192')
        self.assertEqual(args[args.index('--max-num-seqs') + 1], '6')
        self.assertIn('--enforce-eager', args)

    def lora_adapter(self, env):
        adapter = self.root / 'adapter with spaces'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8,
            'base_model_name_or_path': str(Path(env['BF16_ROOT']) / 'model'), 'bias': 'none'}))
        header = json.dumps({
            'base_model.model.layer.lora_A.weight': {'dtype': 'F32', 'shape': [8, 1], 'data_offsets': [0, 32]},
            'base_model.model.layer.lora_B.weight': {'dtype': 'F32', 'shape': [1, 8], 'data_offsets': [32, 64]},
        }).encode()
        (adapter / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 64)
        return str(adapter)

    def test_bf16_lora_registers_adapter_and_preserves_gateway_alias_and_limits(self):
        env = self.bf16_environment()
        adapter = self.lora_adapter(env)
        args_file = self.root / 'vllm-args.json'
        self.start(CHAT_ENABLED='true', LORA_ADAPTER_PATH=adapter, FAKE_ARG_FILE=str(args_file), **env)
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertEqual(health['model'], 'qwen-bf16')
        self.assertEqual(health['limits']['contextTokens'], 40960)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--served-model-name') + 1], 'qwen-bf16-base')
        self.assertIn('--enable-lora', args)
        self.assertEqual(args[args.index('--max-lora-rank') + 1], '8')
        module = json.loads(args[args.index('--lora-modules') + 1])
        self.assertEqual(module, {'name': 'qwen-bf16', 'path': adapter, 'base_model_name': 'qwen-bf16-base'})
        self.assertIn(b'vllm:num_requests_running', self.ready(self.port, True))

    def test_lora_supports_explicit_base_alias_and_rank_capacity(self):
        env = self.bf16_environment()
        args_file = self.root / 'vllm-args.json'
        self.start(METRICS_ENABLED='false', LORA_ADAPTER_PATH=self.lora_adapter(env),
                   VLLM_BASE_MODEL_NAME='bf16-original', MAX_LORA_RANK='16', FAKE_ARG_FILE=str(args_file), **env)
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertEqual(args[args.index('--served-model-name') + 1], 'bf16-original')
        self.assertEqual(args[args.index('--max-lora-rank') + 1], '16')

    def test_invalid_lora_configuration_refuses_startup(self):
        env = self.bf16_environment()
        adapter = self.lora_adapter(env)
        for overrides in ({'LORA_ADAPTER_PATH': str(self.root / 'missing')},
                          {'VLLM_BASE_MODEL_NAME': 'qwen-bf16'}, {'MAX_LORA_RANK': '1'}, {'MAX_LORA_RANK': '4'}):
            with self.subTest(overrides=overrides):
                settings = {**env, 'LORA_ADAPTER_PATH': adapter, **overrides}
                self.start(**settings)
                output = self.process.communicate(timeout=10)[0]
                self.assertNotEqual(self.process.returncode, 0)
                self.assertTrue('LoRA' in output or 'LORA' in output)
                self.closed(8000)
                self.closed(self.port)
                self.closed(self.chat_port)

    def test_separate_base_alias_without_adapter_refuses_startup(self):
        self.start(VLLM_BASE_MODEL_NAME='unrouted-base')
        output = self.process.communicate(timeout=10)[0]
        self.assertNotEqual(self.process.returncode, 0)
        self.assertIn('must match SERVED_MODEL_NAME', output)
        self.closed(8000)

    def test_invalid_profile_fails_before_starting_services(self):
        for profile in ('4BIT', 'unknown', ''):
            with self.subTest(profile=profile):
                args_file = self.root / 'forbidden-model-args.json'
                self.start(MODEL_PROFILE=profile, FAKE_ARG_FILE=str(args_file))
                output = self.process.communicate(timeout=10)[0]
                self.assertNotEqual(self.process.returncode, 0)
                self.assertIn('MODEL_PROFILE', output)
                self.assertFalse(args_file.exists(), 'Rejected profile must not invoke a GPU model')
                self.closed(8000)
                self.closed(self.port)
                self.closed(self.chat_port)

    def four_bit_environment(self, quantization=None, tensors=None):
        # Disposable stand-in for RedHatAI/Qwen3.8-27B-INT4: compressed-tensors W4A16 with a BF16 vision tower.
        env = self.bf16_environment()  # Writes the fake huggingface_hub module.
        root = self.root / 'qwen-4bit'
        root.mkdir(exist_ok=True)
        (root / 'config.json').write_text('{"model_id":"RedHatAI/Qwen3.8-27B-INT4"}')
        source = self.root / 'checkpoint-4bit'
        source.mkdir(exist_ok=True)
        group = {'targets': ['Linear'], 'weights': {'num_bits': 4, 'type': 'int', 'group_size': 128},
                 'input_activations': None}
        (source / 'config.json').write_text(json.dumps({'quantization_config': quantization or {
            'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
            'config_groups': {'group_0': group}}}))
        header = json.dumps({name: {'dtype': dtype, 'shape': [1], 'data_offsets': [0, 4]}
                             for name, dtype in (tensors or {
                                 'model.language_model.layers.0.mlp.down_proj.weight_packed': 'I32',
                                 'model.language_model.layers.0.mlp.down_proj.weight_scale': 'BF16',
                                 'model.visual.blocks.0.attn.qkv.weight': 'BF16'}).items()}).encode()
        (source / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
        env.pop('BF16_ROOT')
        return {**env, 'MODEL_PROFILE': '4bit', 'QWEN_4BIT_ROOT': str(root), 'FAKE_CHECKPOINT': str(source),
                'FAKE_HUB_SHA': PINNED_4BIT}

    def test_4bit_profile_stages_pinned_int4_base_and_serves_qwen_4bit(self):
        args_file = self.root / 'vllm-args.json'
        env = self.four_bit_environment()
        self.start(CHAT_ENABLED='true', CHAT_IMAGE_INPUT_ENABLED='true', MAX_NUM_SEQS='5',
                   FAKE_ARG_FILE=str(args_file), **env)
        health = json.loads(self.ready(self.chat_port, True, '/health'))
        self.assertEqual(health['model'], 'qwen-4bit')
        self.assertFalse(health['adapterToggleAvailable'])
        self.assertEqual(health['imageInputVersion'], 1)
        self.assertEqual([m['model'] for m in health['webSearch']['models']], ['qwen-4bit'])
        self.assertEqual(health['limits']['contextTokens'], 40960)
        self.assertEqual((health['admission']['maxConcurrentRequests'],
                          health['admission']['maxWaitingRequests']), (5, 15))
        args = json.loads(args_file.read_text())
        self.assertEqual(args[0:2], ['serve', str(Path(env['QWEN_4BIT_ROOT']) / 'model')])
        for key, value in [('--served-model-name', 'qwen-4bit'), ('--kv-cache-dtype', 'fp8'),
                           ('--max-model-len', '40960'), ('--max-num-seqs', '5'),
                           ('--reasoning-parser', 'qwen3')]:
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertEqual(json.loads(args[args.index('--limit-mm-per-prompt') + 1]), {'image': 1, 'video': 0})
        self.assertIn('--enable-prefix-caching', args)
        for flag in ('--cpu-offload-gb', '--enable-lora', '--lora-modules', '--max-lora-rank', '--dtype'):
            self.assertNotIn(flag, args)
        state = json.loads((Path(env['QWEN_4BIT_ROOT']) / 'model-manifest.json').read_text())
        self.assertTrue(state['complete'])
        self.assertEqual(state['revision'], PINNED_4BIT)
        self.assertEqual(state['quantization'], 'compressed-tensors W4A16')
        request = Request(f'http://127.0.0.1:{self.chat_port}/v1/chat/completions', method='POST',
                          data=json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True,
                                           'useAdapter': True}).encode(),
                          headers={'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json',
                                   'X-Chat-User-ID': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'})
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request, timeout=5)
        self.assertEqual(rejected.exception.code, 400)
        self.assertIn(b'vllm:num_requests_running', self.ready(self.port, True))

    def test_4bit_keeps_cpu_kv_offload_and_the_tool_parser(self):
        guard = self.root / 'guard.py'
        guard.write_text('pass\n')
        launcher = self.root / 'start-server'
        launcher.write_text(Path(self.launcher).read_text().replace('/usr/local/lib/kv-runtime-config.py', str(guard)))
        launcher.chmod(0o755)
        self.launcher = str(launcher)
        args_file = self.root / 'feature-args.json'
        self.start(METRICS_ENABLED='false', FAKE_ARG_FILE=str(args_file), KV_OFFLOAD_CPU_GIB='64',
                   CUDA_DECODE_GRAPHS='true', CHAT_WEB_SEARCH_TOOL_PARSER='fixture-parser',
                   **self.four_bit_environment())
        self.ready(8000)
        args = json.loads(args_file.read_text())
        self.assertEqual(json.loads(args[args.index('--kv-transfer-config') + 1])['kv_connector'],
                         'OffloadingConnector')
        self.assertEqual(args[args.index('--kv-cache-dtype') + 1], 'fp8')
        self.assertEqual(args[args.index('--tool-call-parser') + 1], 'fixture-parser')
        self.assertIn('--compilation-config', args)
        self.assertNotIn('--cpu-offload-gb', args)

    def assert_refused(self, message, **settings):
        args_file = self.root / 'forbidden-model-args.json'
        self.start(FAKE_ARG_FILE=str(args_file), **settings)
        output = self.process.communicate(timeout=30)[0]
        self.assertNotEqual(self.process.returncode, 0, output)
        self.assertIn(message, output)
        self.assertFalse(args_file.exists(), 'A refused start must not invoke a GPU model')
        self.closed(8000)
        self.closed(self.port)
        self.closed(self.chat_port)

    def test_4bit_refuses_adapter_settings_and_cpu_weight_offload(self):
        for key, value in (('LORA_ADAPTER_PATH', str(self.root)), ('VLLM_BASE_MODEL_NAME', 'qwen-4bit-base'),
                           ('MAX_LORA_RANK', '8')):
            with self.subTest(key):
                self.assert_refused(key, CHAT_ENABLED='true', **{key: value}, **self.four_bit_environment())
        with self.subTest('CPU_OFFLOAD_GB'):
            self.assert_refused('CPU_OFFLOAD_GB', CPU_OFFLOAD_GB='10', **self.four_bit_environment())

    def test_4bit_staging_refuses_unpinned_non_int4_and_vision_less_checkpoints(self):
        cases = {'pinned revision': ({}, {'FAKE_HUB_SHA': 'd' * 40}),
                 'not compressed-tensors W4A16': ({'quantization': {'quant_method': 'awq', 'bits': 4}}, {}),
                 'no vision tower': ({'tensors': {'model.language_model.layers.0.weight_packed': 'I32'}}, {})}
        for message, (checkpoint, overrides) in cases.items():
            with self.subTest(message):
                shutil.rmtree(self.root / 'qwen-4bit', ignore_errors=True)
                shutil.rmtree(self.root / 'checkpoint-4bit', ignore_errors=True)
                self.assert_refused(message, METRICS_ENABLED='false',
                                    **{**self.four_bit_environment(**checkpoint), **overrides})

    def test_4bit_refuses_to_start_without_the_flashinfer_jit_toolchain(self):
        self.assert_refused('nvcc', CUDA_HOME=str(self.root / 'no-cuda'), **self.four_bit_environment())

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
