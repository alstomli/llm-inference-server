"""One fixed engine; local-only control and independent bounded recovery."""
import importlib.util,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
LEASE=json.loads((ROOT/'lease.json').read_text())
spec=importlib.util.spec_from_file_location('status',ROOT/'kv_status.py');status=importlib.util.module_from_spec(spec);spec.loader.exec_module(status)
engine=controller=None;children=[]
def event(kind,**values):
 try:
  with (ROOT/'session.events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':kind,**values},allow_nan=False)+'\n')
 except Exception:pass
def atomic(name,value):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,allow_nan=False));t.replace(p)
def memory():
 p=Path('/sys/fs/cgroup');s=os.statvfs('/dev/shm')
 return {'current':int((p/'memory.current').read_text()),'limit':int((p/'memory.max').read_text()),'swap':int((p/'memory.swap.current').read_text()),
  'events':{k:int(v) for k,v in (line.split() for line in (p/'memory.events').read_text().splitlines())},'shm_available':s.f_bavail*s.f_frsize}
def spawn(argv,env=None,name='service'):
 with (ROOT/(name+'.private.log')).open('ab') as log:
  return subprocess.Popen(argv,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def terminate(p):
 if p is None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=20)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 # A leader exit may leave its workers alive.
 try:os.killpg(p.pid,signal.SIGKILL)
 except ProcessLookupError:pass
def stock(reason):
 status.update(phase='restoring',reason=reason);event('restore',reason=reason)
 terminate(controller);terminate(engine)
 for p in children:terminate(p)
 for key in ['JUPYTER_SERVER_ROOT','KV_EXPERIMENT_ROOT','KV_EXPERIMENT_ARM']:os.environ.pop(key,None)
 os.execv('/usr/local/bin/start-server',['/usr/local/bin/start-server'])
def safety(initial):
 m=memory();status.update(resources=m)
 assert m['limit']-m['current']>=32*1024**3,'RAM headroom'
 assert m['shm_available']>=8*1024**3,'Shared memory headroom'
 assert m['swap']<=initial['swap'],'Swap growth'
 assert all(m['events'].get(k,0)<=initial['events'].get(k,0) for k in ['oom','oom_kill']),'OOM'
 return m
def capacity(m):
 assert m['current']+8589934592+32*1024**3<=m['limit'],'Preallocation RAM headroom'
 assert m['shm_available']>=8589934592+8*1024**3,'Preallocation shared memory headroom'
def idle(client,headers,timeout=30):
 until=time.time()+timeout
 while time.time()<until:
  try:
   if client.get('http://127.0.0.1:8000/health',headers=headers,timeout=2).status_code!=200:return False
   r=client.get('http://127.0.0.1:8000/metrics',headers=headers,timeout=2);values={}
   for line in r.text.splitlines():
    if not line.startswith('vllm:'):continue
    k,v=line.rsplit(' ',1);name=k.split('{')[0].removeprefix('vllm:');values[name]=values.get(name,0)+float(v)
   required=['num_requests_running','num_requests_waiting']
   names=required+['kv_offload_cpu_cache_usage_perc','kv_offload_cpu_cache_write_usage_perc','kv_offload_cpu_cache_read_usage_perc']
   if r.status_code==200 and all(k in values for k in required) and all(values.get(k,0)==0 for k in names):return True
  except (requests.RequestException,ValueError):pass
  time.sleep(.5)
 return False
def released(baseline,timeout=45):
 until=time.time()+timeout
 while time.time()<until:
  m=memory()
  try:empty=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
  except (OSError,subprocess.SubprocessError):empty=False
  if empty and m['shm_available']>=baseline-1024**2:return m
  time.sleep(.5)
 raise RuntimeError('Worker or CPU buffer release timeout')
def command(original):
 args=list(original)
 for i,v in enumerate(args):
  if v=='__VLLM_API_KEY__':args[i]=os.environ['VLLM_API_KEY']
 assert '--kv-transfer-config' not in args and '--worker-cls' not in args
 assert '--enable-prefix-caching' in args and '--enforce-eager' in args and '--disable-hybrid-kv-cache-manager' not in args
 assert args[args.index('--max-num-seqs')+1]=='2' and args[args.index('--cpu-offload-gb')+1]=='0'
 def flag(key,value):
  if key in args:args[args.index(key)+1]=str(value)
  else:args.extend([key,str(value)])
 flag('--kv-cache-memory-bytes',4294967296);flag('--mamba-cache-mode','align');flag('--stream-interval',1)
 flag('--worker-cls','kv_residency_observer.Worker')
 flag('--kv-transfer-config',json.dumps({'kv_connector':'OffloadingConnector','kv_role':'kv_both','kv_connector_extra_config':{'cpu_bytes_to_use':8589934592,'blocks_per_chunk':1,'offload_prompt_only':False}}))
 hist=json.loads(args[args.index('--custom-histogram-buckets')+1]) if '--custom-histogram-buckets' in args else {}
 hist['inter_token_latency']=[i/1000 for i in range(1,201)]+[.25,.5,1,2,5]
 flag('--custom-histogram-buckets',json.dumps(hist,separators=(',',':')))
 return args
def main():
 global engine,controller
 status.start();initial=memory();safety(initial);capacity(initial)
 def absolute_guard():
  while time.time()<LEASE['restore_at']:time.sleep(min(1,max(.01,LEASE['restore_at']-time.time())))
  stock('independent_local_absolute_deadline')
 threading.Thread(target=absolute_guard,daemon=True).start()
 env={**os.environ,'JUPYTER_SERVER_ROOT':str(ROOT)}
 children.append(spawn(['python3','/usr/local/lib/start-jupyter.py'],env,name='jupyter'))
 children.append(spawn(['python3','/usr/local/lib/metrics-relay.py'],name='metrics'))
 launch=json.loads((ROOT/'original-launch.private.json').read_text());args=command(launch['args'])
 model_env={**os.environ,**launch['public_exports'],'PYTHONPATH':str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),'KV_EXPERIMENT_ROOT':str(ROOT),'KV_EXPERIMENT_ARM':'B'}
 engine=spawn(args,model_env,name='engine');started=time.time();status.update(phase='starting',engine_pid=engine.pid,started=started)
 event('engine_start',pid=engine.pid,gpu_cache_bytes=4294967296,cpu_cache_bytes=8589934592)
 client=requests.Session();client.trust_env=False
 h={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
 healthy=False;last_sample=0;failed_at=None;restore_wait=False
 while time.time()<LEASE['restore_at']:
  if (ROOT/'restore.remote').exists() and not restore_wait:
   reason=(ROOT/'restore.remote').read_text()
   if reason=='fatal correctness failure':stock('fatal_correctness_failure')
   terminate(controller);terminate(engine);controller=engine=None;restore_wait=True
   resources=released(initial['shm_available']);event('offload_workers_released',**resources)
   status.update(phase='ready_for_original_args_restore',resources=resources,shm_baseline=initial['shm_available'])
  if time.time()-last_sample>2:
   m=safety(initial);event('resources',**m);last_sample=time.time()
  if restore_wait:time.sleep(.5);continue
  if engine.poll() is not None:stock('engine_exit')
  if not healthy:
   try:
    if client.get('http://127.0.0.1:8000/health',headers=h,timeout=2).status_code==200:
     layout=json.loads((ROOT/'layout-B.json').read_text());assert 0<layout['actual_gpu_cache_bytes']<=4294967296
     healthy=True;status.update(phase='ready',engine_pid=engine.pid,layout={'bytes':layout['actual_gpu_cache_bytes'],'blocks':layout['num_blocks'],'fingerprint':layout['layout_fingerprint']})
     atomic('engine.status.json',{'state':'ready','pid':engine.pid,'started':started,'ready':time.time()})
     event('engine_ready',startup_s=time.time()-started)
     controller=spawn(['python3',str(ROOT/'experiment.py')],name='experiment')
   except requests.RequestException:pass
   if time.time()-started>35*60:stock('bounded_startup_timeout')
  elif controller and controller.poll() is not None:
   if failed_at is None:
    if not idle(client,h):stock('failed_controller_not_idle')
    failed_at=time.time()
   status.update(phase='experiment_failed_idle' if (ROOT/'experiment.failed').exists() else 'experiment_finished',controller_exit=controller.returncode)
   if time.time()-failed_at>=600 or time.time()+600>=LEASE['restore_at']:stock('diagnostic_hold_expired')
   retry=ROOT/'retry.controller'
   if retry.exists():
    if not idle(client,h):stock('retry_not_idle')
    retry.unlink();failed_at=None;controller=spawn(['python3',str(ROOT/'experiment.py')],name='experiment-retry')
  time.sleep(.5)
 stock('absolute_deadline')
if __name__=='__main__':
 try:main()
 except BaseException as error:
  event('supervisor_failure',error_type=type(error).__name__)
  stock('supervisor_safety_stop')
