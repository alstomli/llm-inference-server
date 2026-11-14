"""Optional, local-only enablement observations; absent from ordinary serving.

No tensor values, prompts, tokens, credentials or memory addresses are recorded.
Ordinary serving uses the vendor worker without this subclass.
"""
import asyncio
import hmac
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path('/tmp/qwen-kv-proof')


def secure_root():
    ROOT.mkdir(mode=0o700, exist_ok=True)
    info = ROOT.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Proof directory must be an owned private directory')
    return ROOT


def write(name, value):
    secure_root()
    temporary = ROOT / (name + '.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(ROOT / name)


def take_control(path, now=None):
    """Consume an owner-only, regular, short-lived one-use capability."""
    now = time.time() if now is None else now
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Invalid proof control permissions')
        with os.fdopen(descriptor, closefd=False) as stream:
            value = json.load(stream)
        if not now < value['expires'] <= now + 600 or len(value['token']) < 32:
            raise ValueError('Expired or invalid proof capability')
        os.rename(path, str(path) + '.consumed')
        return value
    finally:
        os.close(descriptor)


class ProofMiddleware:
    """Maintenance gate plus one-shot Unix socket; no HTTP administration route."""
    socket_lifetime = 600
    def __init__(self, app):
        self.app = app
        self.started = False
        self.server = None

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and not self.started:
            self.started = True
            self.engine = scope['app'].state.engine_client
            secure_root()
            socket = ROOT / 'reset.sock'
            socket.unlink(missing_ok=True)
            self.server = await asyncio.start_unix_server(self.reset, path=str(socket), limit=2048)
            socket.chmod(0o600)
            asyncio.create_task(self.expire_socket(socket))
        if scope['type'] == 'http' and (ROOT / 'maintenance').exists():
            headers = dict(scope.get('headers', []))
            # Only the private verifier can submit inference while ingress waits.
            control_path = ROOT / 'inference-token'
            token = control_path.read_text().strip() if control_path.exists() else ''
            allowed = token and hmac.compare_digest(headers.get(b'x-kv-proof', b''), token.encode())
            if not allowed and scope['path'] not in ('/metrics', '/v1/models'):
                await send({'type': 'http.response.start', 'status': 503, 'headers': [(b'content-type', b'application/json')]})
                await send({'type': 'http.response.body', 'body': b'{"error":"Model enablement verification in progress"}'})
                return
        await self.app(scope, receive, send)

    async def expire_socket(self, socket):
        await asyncio.sleep(self.socket_lifetime)
        self.server.close()
        socket.unlink(missing_ok=True)
        await self.server.wait_closed()

    async def require_idle(self):
        if self.engine.get_num_unfinished_requests() != 0:
            raise RuntimeError('API requests are unfinished')
        import httpx
        from prometheus_client.parser import text_string_to_metric_families
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            response = await client.get('http://127.0.0.1:' + os.environ.get('VLLM_PORT', '8000') + '/metrics')
            response.raise_for_status()
        needed = {'vllm:num_requests_running', 'vllm:num_requests_waiting',
                  'vllm:kv_offload_cpu_cache_usage_perc',
                  'vllm:kv_offload_cpu_cache_write_usage_perc',
                  'vllm:kv_offload_cpu_cache_read_usage_perc'}
        found = set()
        for family in text_string_to_metric_families(response.text):
            for sample in family.samples:
                if sample.name in needed:
                    found.add(sample.name)
                    if sample.value != 0:
                        raise RuntimeError('Engine or native transfers are busy')
        if found != needed:
            raise RuntimeError('Required idle metrics are missing')

    async def reset(self, reader, writer):
        try:
            message = json.loads(await asyncio.wait_for(reader.readline(), 3))
            control = take_control(ROOT / 'reset-control.json')
            if message.keys() != {'token'} or not hmac.compare_digest(message['token'], control['token']):
                raise ValueError('Invalid capability')
            if not (ROOT / 'maintenance').exists():
                raise ValueError('Maintenance gate is required')
            # Native reset refuses busy blocks; never abort work or reset the CPU connector.
            await self.require_idle()
            result = await asyncio.wait_for(self.engine.reset_prefix_cache(reset_running_requests=False, reset_connector=False), 30)
            if result is not True:
                raise RuntimeError('Native idle GPU reset was refused')
            writer.write(b'{"reset":true,"reset_running_requests":false,"reset_connector":false}\n')
            self.server.close()
            (ROOT / 'reset.sock').unlink(missing_ok=True)
            await writer.drain()
        except Exception as error:
            writer.write(json.dumps({'reset': False, 'error_type': type(error).__name__}).encode() + b'\n')
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def worker_class():
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    class ObservedWorker(GPUWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import torch
            scheduler = self.vllm_config.scheduler_config
            write('scheduler.json', {**{name: getattr(scheduler, name) for name in
                  ('max_num_seqs', 'max_num_batched_tokens', 'enable_chunked_prefill')},
                  'max_model_len': self.vllm_config.model_config.max_model_len})
            native_replay = torch.cuda.CUDAGraph.replay
            counts = {'completed_native_replays': 0}

            def replay(graph):
                result = native_replay(graph)
                counts['completed_native_replays'] += 1
                # Every completed native call is evidence, not dispatch/capture.
                # Only temporary proof runs install this callback.
                write('replays.json', {**counts, 'time': time.time()})
                return result

            torch.cuda.CUDAGraph.replay = replay
        def initialize_from_config(self, config):
            super().initialize_from_config(config)
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            connector = get_kv_transfer_group()
            worker = connector.connector_worker
            groups = worker.spec.config.groups
            selected = [g.group_id for g in groups]
            actual = self.model_runner.kv_cache_config
            expected = set(actual.prefix_cacheable_group_ids)
            if set(selected) != expected:
                raise RuntimeError('Native offloader did not bind all cache groups')
            descriptions = []
            for group in actual.kv_cache_groups:
                leafs = getattr(group.kv_cache_spec, 'kv_cache_specs', {'self': group.kv_cache_spec})
                descriptions.append(sorted({type(v).__name__ for v in leafs.values()}))
            storage = {}

            def visit(value):
                if isinstance(value, dict):
                    for item in value.values(): visit(item)
                elif isinstance(value, (list, tuple)):
                    for item in value: visit(item)
                elif hasattr(value, 'untyped_storage'):
                    block = value.untyped_storage()
                    storage[(str(value.device), block.data_ptr())] = block.nbytes()

            visit(self.model_runner.kv_caches)
            write('allocation.json', {'num_gpu_blocks': actual.num_blocks,
                  'actual_gpu_cache_bytes': sum(n for (device, _), n in storage.items() if device.startswith('cuda')),
                  'group_types': descriptions, 'offload_group_ids': selected,
                  'cpu_chunks': worker.spec.num_chunks, 'cpu_bytes_per_chunk': worker.spec.kv_bytes_per_chunk,
                  'allocated_cpu_bytes': worker.spec.num_chunks * worker.spec.kv_bytes_per_chunk,
                  'hash_block_size': actual.hash_block_size,
                  'cache_hit_alignment_tokens': actual.cache_hit_alignment_tokens,
                  'kv_cache_layout': str(actual.kv_cache_layout),
                  'store_group_reference_counts': [len(group) for group in worker.worker._store_handler.layer_refs_per_group],
                  'load_group_reference_counts': [len(group) for group in worker.worker._load_handler.layer_refs_per_group]})

    return ObservedWorker


# vLLM resolves this only in proof worker processes. API middleware can be
# imported without initializing CUDA or importing the GPU worker.
def __getattr__(name):
    if name == 'Worker':
        return worker_class()
    raise AttributeError(name)
