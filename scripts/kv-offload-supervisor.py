"""Bounded A/B maintenance supervisor; owns only children it launches.

The independent local watchdog restores original pod args at minute150. This
remote supervisor independently stops load and execs the stock launcher then.
"""
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import requests

ROOT=Path(__file__).resolve().parent
LEASE=json.loads((ROOT/'lease.json').read_text())
children=[]
engine=controller=None
current=None
B_SHM_BEFORE=None


def event(kind,**fields):
    with (ROOT/'session.events.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),'event':kind,**fields})+'\n')


def atomic(name,value):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value));t.replace(p)


def spawn(argv,env=None,name='service'):
    with (ROOT/(name+'.private.log')).open('ab') as log:
        return subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def terminate(p):
    if p is None:return
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
    # Leader exit does not prove its worker children have exited.
    until=time.time()+45
    while time.time()<until:
        try:os.killpg(p.pid,0)
        except ProcessLookupError:return
        time.sleep(.5)
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def memory():
    base=Path('/sys/fs/cgroup')
    r={k:int((base/k).read_text()) for k in ['memory.current','memory.max','memory.swap.current']}
    r['memory.events']={k:int(v) for k,v in (line.split() for line in (base/'memory.events').read_text().splitlines())}
    s=os.statvfs('/dev/shm');r['shm_available_bytes']=s.f_bavail*s.f_frsize
    return r


def safety():
    r=memory();event('resources',**r)
    if r['memory.max']-r['memory.current']<32*1024**3:raise RuntimeError('RAM headroom')
    if r['shm_available_bytes']<8*1024**3:raise RuntimeError('Shared memory headroom')
    if r['memory.swap.current']>INITIAL['memory.swap.current']:raise RuntimeError('Swap growth')
    if any(r['memory.events'].get(k,0)>INITIAL['memory.events'].get(k,0) for k in ['oom','oom_kill']):raise RuntimeError('OOM')


def stock(reason):
    event('restoring_stock',reason=reason)
    atomic('engine.status.json',{'state':'restoring_stock','reason':reason})
    terminate(controller);terminate(engine)
    event('benchmark_workers_released',resources=memory())
    for p in children:terminate(p)
    os.execv('/usr/local/bin/start-server',['/usr/local/bin/start-server'])


def set_flag(args,key,value):
    if key in args:
        i=args.index(key);args[i+1]=str(value)
    else:args.extend([key,str(value)])


def start(config):
    global engine,current,B_SHM_BEFORE
    terminate(engine)
    # Confirm all old GPU processes disappeared before replacing the engine.
    until=time.time()+60
    while time.time()<until:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
        if not output:break
        time.sleep(1)
    else:raise RuntimeError('Old GPU workers remain')
    if current and current['arm']=='B':
        released=memory();event('B_memory_release',before_shm=B_SHM_BEFORE,after=released)
        assert released['shm_available_bytes']>=B_SHM_BEFORE-1024**3,'CPU cache shared memory not released'
    if time.time()>=LEASE['restore_at']:raise RuntimeError('Lease expired before spawn')
    if config['arm']=='B' and time.time()>=LEASE['created']+125*60:raise RuntimeError('B cutoff')
    safety()
    state=json.loads((ROOT/'original-process.private.json').read_text())
    args=list(state['args'])
    assert '--kv-transfer-config' not in args and '--kv-offloading-size' not in args
    assert '--enable-prefix-caching' in args and '--enforce-eager' in args
    assert '--disable-hybrid-kv-cache-manager' not in args and '--no-enable-prefix-caching' not in args
    assert args[args.index('--cpu-offload-gb')+1]=='0'
    for key,value in [('--max-num-seqs',256),('--max-num-batched-tokens',8192),('--mamba-cache-mode','align'),('--worker-cls','kv_offload_observer.Worker')]:set_flag(args,key,value)
    if config.get('gpu_cache_bytes'):set_flag(args,'--kv-cache-memory-bytes',config['gpu_cache_bytes'])
    if config['arm']=='B':
        B_SHM_BEFORE=memory()['shm_available_bytes']
        set_flag(args,'--kv-transfer-config',json.dumps({'kv_connector':'OffloadingConnector','kv_role':'kv_both','kv_connector_extra_config':{'cpu_bytes_to_use':68719476736,'blocks_per_chunk':1,'offload_prompt_only':False}},separators=(',',':')))
    env={**state['env'],'PYTHONPATH':str(ROOT)+os.pathsep+state['env'].get('PYTHONPATH',''),'KV_EXPERIMENT_ROOT':str(ROOT),'KV_EXPERIMENT_ARM':config['arm']}
    current=dict(config)
    engine=spawn(args,env,name='engine-'+config['arm'])
    atomic('engine.status.json',{'state':'starting','config':current,'pid':engine.pid})
    event('engine_start',config=current,pid=engine.pid)
    return time.time()


class Maintenance(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(503);self.send_header('Content-Type','application/json');self.send_header('Retry-After','120');self.end_headers();self.wfile.write(b'{"error":"Scheduled inference experiment"}')
    do_POST=do_GET
    def log_message(self,*args):pass


def main():
    global controller,INITIAL
    INITIAL=memory()
    if time.time()>=LEASE['restore_at']:stock('lease_expired')
    safety()
    server=http.server.ThreadingHTTPServer(('0.0.0.0',8001),Maintenance)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    children.append(spawn(['python3','/usr/local/lib/start-jupyter.py'],name='jupyter'))
    children.append(spawn(['python3','/usr/local/lib/metrics-relay.py'],name='metrics'))
    atomic('desired.json',{'arm':'A','gpu_cache_bytes':None})
    started=start({'arm':'A','gpu_cache_bytes':None});healthy=False
    h={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
    client=requests.Session();client.trust_env=False;last_sample=0
    while time.time()<LEASE['restore_at']:
        if (ROOT/'restore.remote').exists():break
        if time.time()-last_sample>2:safety();last_sample=time.time()
        desired=json.loads((ROOT/'desired.json').read_text())
        assert desired['arm'] in ['A','B','A2']
        if desired['arm']=='B' and time.time()>LEASE['created']+125*60:raise RuntimeError('B cutoff')
        if desired!=current:started=start(desired);healthy=False
        if engine.poll() is not None:raise RuntimeError('Engine exited')
        if not healthy:
            try:
                if client.get('http://127.0.0.1:8000/health',headers=h,timeout=2).status_code==200:
                    assert (ROOT/('layout-'+current['arm']+'.json')).exists()
                    healthy=True;atomic('engine.status.json',{'state':'ready','config':current,'pid':engine.pid,'ready_at':time.time()})
                    event('engine_ready',config=current,startup_seconds=time.time()-started)
                    if controller is None:controller=spawn(['python3',str(ROOT/'experiment.py')],name='experiment')
            except requests.RequestException:pass
            if time.time()-started>18*60:raise RuntimeError('Startup deadline')
        elif controller is not None and controller.poll() is not None:
            stock('controller_finished')
        time.sleep(.5)
    stock('remote_absolute_deadline')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        event('supervisor_failure',error_type=type(e).__name__,reason=str(e) if str(e) in ['RAM headroom','Shared memory headroom','Swap growth','OOM','Engine exited','Startup deadline','B cutoff','Old GPU workers remain'] else 'suppressed')
        stock('supervisor_safety_stop')
