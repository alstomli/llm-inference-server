"""Authenticated, bounded observation of the owned local experiment directory."""
import hmac,http.server,json,os,re,threading,time
from pathlib import Path
ROOT=Path(os.environ['KV_EXPERIMENT_ROOT'])
TOKEN=os.environ.get('JUPYTER_PASSWORD') or os.environ['JUPYTER_TOKEN']
PUBLIC={'runtime.json','lease.json','engine.status.json','session.events.jsonl','layout-B.json'}
STATE={'phase':'initializing'}
LOCK=threading.Lock()
def update(**values):
 with LOCK:STATE.update(values)
def allowed(name):
 return name in PUBLIC or bool(re.fullmatch(r'results/[A-Za-z0-9_.-]+\.(json|jsonl)',name)) and '.private.' not in name
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  if not self.path.startswith('/__kv_'):self.reply(503,{'error':'Scheduled inference experiment'});return
  if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+TOKEN):self.reply(401,{'error':'Unauthorized'});return
  if self.path=='/__kv_status':
   with LOCK:result={**STATE,'time':time.time()}
   self.reply(200,result);return
  if self.path=='/__kv_files':
   names=[name for name in PUBLIC if (ROOT/name).is_file()]
   names += ['results/'+p.name for p in (ROOT/'results').glob('*') if allowed('results/'+p.name)]
   self.reply(200,{'files':names});return
  prefix='/__kv_file/'
  if self.path.startswith(prefix) and allowed(self.path[len(prefix):]):
   p=ROOT/self.path[len(prefix):]
   try:
    if p.stat().st_size>8*1024**2:self.reply(413,{'error':'Artifact too large'});return
    data=p.read_bytes();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
   except FileNotFoundError:pass
  self.reply(404,{'error':'Not found'})
 def reply(self,status,data):
  raw=json.dumps(data,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 do_POST=lambda self:self.reply(503,{'error':'Scheduled inference experiment'})
 def log_message(self,*args):pass
def start():
 server=http.server.ThreadingHTTPServer(('0.0.0.0',8001),Handler)
 threading.Thread(target=server.serve_forever,daemon=True).start();return server
