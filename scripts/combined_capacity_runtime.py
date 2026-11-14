"""Strict temporary argv transformation and local-only runtime contracts."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

IMAGE = 'zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d'
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CPU_BYTES = 68719476736
BASE_ITL_BUCKETS = [.01,.025,.05,.075,.1,.15,.2,.3,.4,.5,.75,1,2.5,5,7.5,10,20,40,80]
FINE_ITL_BUCKETS = [.001,.002,.005] + [index / 1000 for index in range(10, 201, 5)]


def secure_root(path):
    path = Path(path)
    if path.parent != Path('/tmp') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', path.name):
        raise ValueError('Capacity controls require an owned direct child of /tmp')
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Capacity directory must be owned and mode0700')
    return path


def canonical_argv(argv, env=None):
    env = os.environ if env is None else env
    args = list(argv)
    if 'serve' not in args or args.count('serve') != 1:
        raise ValueError('Expected exactly one native serve command')
    args = args[args.index('serve'):]
    flags = [arg for arg in args if arg.startswith('--')]
    if len(flags) != len(set(flags)):
        raise ValueError('Duplicate native options are forbidden')
    if '--api-key' in args:
        index = args.index('--api-key') + 1
        if not env.get('VLLM_API_KEY') or args[index] != env['VLLM_API_KEY']:
            raise ValueError('API key must come from the unchanged container environment')
        args[index] = '@ENV:VLLM_API_KEY@'
    return args


def value(args, flag):
    if args.count(flag) != 1:
        raise ValueError('Missing or duplicate required native option')
    return args[args.index(flag) + 1]


def contract(argv, env=None):
    args = canonical_argv(argv, env)
    assert value(args, '--served-model-name') == 'qwen-bf16-base'
    assert value(args, '--max-num-seqs') == '2'
    assert value(args, '--max-model-len') == '40960'
    assert float(value(args, '--gpu-memory-utilization')) == .9
    assert value(args, '--dtype') == 'bfloat16' and value(args, '--cpu-offload-gb') == '0'
    assert value(args, '--host') == '127.0.0.1'
    assert '--enable-prefix-caching' in args and value(args, '--mamba-cache-mode') == 'align'
    assert '--cudagraph-metrics' in args
    assert '--enable-lora' in args and value(args, '--max-lora-rank') == '8'
    assert json.loads(value(args, '--limit-mm-per-prompt')) == {'image': 1, 'video': 0}
    assert '--enforce-eager' not in args and '--disable-hybrid-kv-cache-manager' not in args
    assert '--worker-cls' not in args and '--middleware' not in args
    assert '--kv-cache-memory-bytes' not in args and '--num-gpu-blocks-override' not in args
    compilation = json.loads(value(args, '--compilation-config'))
    assert compilation == {'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY', 'cudagraph_capture_sizes': [1, 2]}
    transfer = json.loads(value(args, '--kv-transfer-config'))
    assert transfer == {'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
        'kv_connector_extra_config': {'cpu_bytes_to_use': CPU_BYTES, 'blocks_per_chunk': 1, 'offload_prompt_only': False}}
    return {'schema_version': 1, 'image': IMAGE, 'native_argv': args,
            'native_argv_sha256': hashlib.sha256(json.dumps(args, separators=(',', ':')).encode()).hexdigest(),
            'native_itl_buckets': BASE_ITL_BUCKETS}


def test_argv(argv, frozen, env=None):
    args = list(argv)
    actual = canonical_argv(args, env)
    if contract(args, env) != frozen:
        raise ValueError('Native startup does not match the fresh production contract')
    # Keep every nonallowlisted value and its order byte-for-byte intact.
    args = args[args.index('serve'):]
    args[args.index('--max-num-seqs') + 1] = '256'
    if '--max-num-batched-tokens' in args:
        args[args.index('--max-num-batched-tokens') + 1] = '8192'
    else:
        args += ['--max-num-batched-tokens', '8192']
    compilation = json.loads(value(args, '--compilation-config'))
    compilation['cudagraph_capture_sizes'] = CAPTURE_SIZES
    args[args.index('--compilation-config') + 1] = json.dumps(compilation, separators=(',', ':'))
    buckets = json.loads(value(args, '--custom-histogram-buckets'))
    buckets['inter_token_latency'] = sorted(set(frozen['native_itl_buckets'] + FINE_ITL_BUCKETS + buckets.get('inter_token_latency', [])))
    args[args.index('--custom-histogram-buckets') + 1] = json.dumps(buckets, separators=(',', ':'))
    args += ['--worker-cls', 'capacity_graph_observer.Worker']
    return args


def validate_lease(lease, now):
    assert lease['created'] <= now < lease['testing_stop_at']
    assert lease['testing_stop_at'] == lease['created'] + 110 * 60
    assert lease['restore_at'] == lease['created'] + 120 * 60
    assert lease['exclusive_end'] == lease['created'] + 150 * 60
    assert 0 < lease['startup_cap_seconds'] <= 2100
    assert lease['run_id'] and lease['lease_id']


def resources():
    root = Path('/sys/fs/cgroup')
    memory = {name: int((root / name).read_text()) for name in ['memory.current', 'memory.max', 'memory.swap.current']}
    memory['memory.events'] = {key: int(value) for key, value in (line.split() for line in (root / 'memory.events').read_text().splitlines())}
    shm = os.statvfs('/dev/shm')
    memory['shm_available_bytes'] = shm.f_bavail * shm.f_frsize
    return memory


def safe(memory, baseline, projected_cpu_bytes=0):
    assert memory['memory.max'] - memory['memory.current'] - projected_cpu_bytes >= 32 * 1024**3, 'RAM reserve'
    assert memory['shm_available_bytes'] - projected_cpu_bytes >= 8 * 1024**3, 'Shared memory reserve'
    assert memory['memory.swap.current'] == baseline['memory.swap.current'], 'Swap growth'
    assert all(memory['memory.events'].get(name, 0) == baseline['memory.events'].get(name, 0) for name in ['oom', 'oom_kill']), 'OOM growth'


def atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
