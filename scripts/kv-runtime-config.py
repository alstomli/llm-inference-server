"""Fail closed before allocating the optional native 64 GiB host tier."""
import json
import os
import secrets
from pathlib import Path
import sys
import stat


def check_memory(root=Path('/sys/fs/cgroup'), shm='/dev/shm'):
    current = int((root / 'memory.current').read_text())
    maximum = int((root / 'memory.max').read_text())
    stats = os.statvfs(shm)
    if maximum - current - 64 * 1024**3 < 32 * 1024**3:
        raise RuntimeError('Native KV allocation would leave less than 32 GiB RAM reserve')
    if stats.f_bavail * stats.f_frsize - 64 * 1024**3 < 8 * 1024**3:
        raise RuntimeError('Native KV allocation would leave less than 8 GiB shared memory reserve')
    print(json.dumps({'native_kv_preflight': True, 'cpu_tier_bytes': 64 * 1024**3}))


def prepare_proof(root=Path('/tmp/qwen-kv-proof')):
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('Proof directory must be an owned private directory')
    for name, content in [('maintenance', ''), ('inference-token', secrets.token_hex(32))]:
        descriptor = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(content)


if __name__ == '__main__':
    if sys.argv[1:] == ['prepare-proof']:
        prepare_proof()
    elif sys.argv[1:] == ['check-memory']:
        check_memory()
    else:
        raise SystemExit('Unknown runtime preparation operation')
