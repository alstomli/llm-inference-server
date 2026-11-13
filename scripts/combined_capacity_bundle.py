"""Pure offline builder: freeze source first, materialize the fresh lease last."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import zlib

SOURCES = ['benchmark-capacity.py', 'combined_capacity_runtime.py', 'combined_capacity_measure.py', 'combined-capacity-supervisor.py',
           'combined-capacity-run.py', 'capacity_graph_observer.py']


def files(repo):
    repo = Path(repo)
    result = {name: (repo / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n') for name in SOURCES}
    result['bin/vllm'] = (repo / 'scripts/capacity-vllm-shim.py').read_bytes().replace(b'\r\n', b'\n')
    result['canonical-plan.json'] = (repo / 'docs/COMBINED-CAPACITY-PLAN.json').read_bytes().replace(b'\r\n', b'\n')
    return result


def build_args(run_root, source, lease, args_contract):
    run_root = str(run_root)
    if not re.fullmatch(r'/tmp/[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', run_root):
        raise ValueError('Invalid temporary run root')
    if Path(run_root).name != lease['run_id']:
        raise ValueError('Run root and lease identity differ')
    for name in source:
        if name.startswith('/') or '..' in Path(name).parts:
            raise ValueError('Invalid bundle member')
    bundle = dict(source)
    bundle['args-contract.json'] = json.dumps(args_contract, indent=2, allow_nan=False).encode()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in bundle.items()}
    bundle['source-files.json'] = json.dumps(hashes, sort_keys=True, indent=2).encode()
    bundle['lease.json'] = json.dumps(lease, indent=2, allow_nan=False).encode()
    payload = {'root': run_root, 'files': {name: base64.b64encode(data).decode() for name, data in bundle.items()}}
    encoded = base64.b64encode(zlib.compress(json.dumps(payload, separators=(',', ':')).encode(), 9)).decode()
    # The payload contains source, numerical plan, fresh lease and scrubbed argv;
    # container secrets remain in the unchanged native environment.
    program = '''import base64,json,os,pathlib,sys,zlib
p=json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
root=pathlib.Path(p['root']);root.mkdir(mode=0o700)
for name,value in p['files'].items():
 path=root/name
 if path.parent!=root:path.parent.mkdir(mode=0o700,exist_ok=True)
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as stream:stream.write(base64.b64decode(value))
(root/'bin/vllm').chmod(0o700)
os.environ['CAPACITY_SESSION_ROOT']=str(root)
os.execv(sys.executable,[sys.executable,str(root/'combined-capacity-supervisor.py')])
'''
    program = 'PAYLOAD=' + repr(encoded) + '\n' + program
    return 'bash -lc ' + shlex.quote('exec python3 -c ' + shlex.quote(program))
