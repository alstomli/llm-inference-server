"""Temporary lease-limited process tree; existing image/config/model stay intact."""
import http.server,json,os,signal,subprocess,threading,time
from pathlib import Path
import requests

ROOT=Path('/workspace/capacity-session')
LEASE=json.loads((ROOT/'lease.json').read_text())
DEADLINE=LEASE['restore_at']
children=[]
engine=None
loadgen=None
current=None
def event(kind,**values):
    with (ROOT/'session.events.jsonl').open('a') as f:f.write(json.dumps({'t':time.time(),'event':kind,**values})+'\n')

def terminate(process):
    if process is None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)

def spawn(args,env=None,name='service'):
    log=(ROOT/(name+'.private.log')).open('ab')
    p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close();return p

def stock():
    terminate(loadgen)
    for p in children:terminate(p)
    terminate(engine)
    event('stock_launcher_exec')
    os.execv('/usr/local/bin/start-server',['/usr/local/bin/start-server'])

class Maintenance(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503);self.send_header('Content-Type','application/json');self.send_header('Retry-After','120');self.end_headers();self.wfile.write(b'{"error":"Scheduled capacity test"}')
    do_POST=do_GET
    def log_message(self,*args):pass

def start_engine(config):
    global engine,current
    terminate(engine)
    # No second model process is launched until the previous tree releases VRAM.
    time.sleep(3)
    state=json.loads((ROOT/'original-process.private.json').read_text())
    args=list(state['args'])
    args[args.index('--max-num-seqs')+1]=str(config['sequences'])
    if '--max-num-batched-tokens' in args:args[args.index('--max-num-batched-tokens')+1]=str(config['batched_tokens'])
    else:args += ['--max-num-batched-tokens',str(config['batched_tokens'])]
    current=dict(config)
    engine=spawn(args,state['env'],'engine-seq'+str(config['sequences'])+'-batch'+str(config['batched_tokens'])+'-attempt'+str(int(time.time())))
    event('engine_start',pid=engine.pid,**config)
    (ROOT/'engine.status.json').write_text(json.dumps({'state':'starting','config':config,'pid':engine.pid}))
    return time.time()

def main():
    global engine,loadgen
    if time.time()>=DEADLINE or (ROOT/'restore.remote').exists():stock()
    event('lease_started',deadline=DEADLINE)
    server=http.server.ThreadingHTTPServer(('0.0.0.0',8001),Maintenance)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    children.append(spawn(['python3','/usr/local/lib/start-jupyter.py'],name='jupyter'))
    children.append(spawn(['python3','/usr/local/lib/metrics-relay.py'],name='metrics'))
    config={'sequences':256,'batched_tokens':8192}
    (ROOT/'desired.json').write_text(json.dumps(config))
    started=start_engine(config);healthy=False;attempts=0
    h={}
    if os.getenv('VLLM_API_KEY'):h['Authorization']='Bearer '+os.environ['VLLM_API_KEY']
    while time.time()<DEADLINE:
        if (ROOT/'restore.remote').exists():break
        try:desired=json.loads((ROOT/'desired.json').read_text())
        except (ValueError,FileNotFoundError):desired=config
        assert type(desired['sequences']) is int and 1<=desired['sequences']<=256
        assert desired['batched_tokens'] in (4096,8192,16384)
        if desired!=current:
            config=desired;started=start_engine(config);healthy=False
        if engine.poll() is not None:
            event('engine_exit',returncode=engine.returncode,healthy=healthy,**current)
            if healthy or time.time()>LEASE['created']+780 or attempts>=7:
                event('safety_restore',reason='engine_failure')
                break
            attempts+=1
            config={**current,'sequences':max(1,current['sequences']//2)}
            (ROOT/'desired.json').write_text(json.dumps(config))
            started=start_engine(config);healthy=False
        if not healthy:
            try:
                r=requests.get('http://127.0.0.1:8000/health',headers=h,timeout=2)
                if r.status_code==200:
                    healthy=True
                    event('engine_ready',startup_seconds=time.time()-started,**current)
                    (ROOT/'engine.status.json').write_text(json.dumps({'state':'ready','config':current,'pid':engine.pid,'ready_at':time.time()}))
                    if loadgen is None:
                        loadgen=spawn(['python3',str(ROOT/'experiment.py')],name='experiment')
                        event('experiment_start',pid=loadgen.pid)
            except requests.RequestException:pass
            if time.time()-started>900:
                event('startup_timeout');break
        time.sleep(1)
    server.shutdown();stock()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        event('supervisor_exception',error_type=type(e).__name__)
        stock()
