"""Build a frozen, inline, local-only pod entrypoint from credential-free files."""
import base64,gzip,hashlib,json
def build_override(contents,root):
 assert root.startswith('/tmp/kv-residency-')
 assert all('/' not in n and '\\' not in n and n not in ['.','..'] for n in contents)
 payload={name:{'data':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()} for name,data in contents.items()}
 encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
 code="""import base64,gzip,hashlib,json,os,sys
from pathlib import Path
try:
 root=Path(LOCAL_PATH);assert not root.exists();root.mkdir(mode=0o700)
 bundle=json.loads(gzip.decompress(base64.b64decode(ENCODED)))
 for name,item in bundle.items():
  assert '/' not in name and '\\\\' not in name and name not in ['.','..']
  data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256']
  (root/name).write_bytes(data)
 public=json.loads((root/'original-launch.private.json').read_text())['public_exports'];os.environ.update(public)
 os.environ['KV_EXPERIMENT_ROOT']=str(root);os.environ['JUPYTER_SERVER_ROOT']=str(root)
 os.execv(sys.executable,[sys.executable,str(root/'supervisor.py')])
except BaseException:
 for key in ['KV_EXPERIMENT_ROOT','JUPYTER_SERVER_ROOT']:os.environ.pop(key,None)
 os.execv('/usr/local/bin/start-server',['/usr/local/bin/start-server'])
""".replace('LOCAL_PATH',repr(root)).replace('ENCODED',repr(encoded))
 assert len(code.encode())<120000
 return json.dumps({'entrypoint':['python3','-c',code],'cmd':[]},separators=(',',':'))
