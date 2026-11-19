"""Runs ON THE POD through RunPod exec. Read-only snapshot of engine, gateway and resources."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import requests
base=Path('/sys/fs/cgroup')
memory={k:int((base/k).read_text()) for k in ['memory.current','memory.max','memory.swap.current']}
memory['memory.events']={k:int(v) for k,v in (line.split() for line in (base/'memory.events').read_text().splitlines())}
v=os.statvfs('/dev/shm');memory['shm_available_bytes']=v.f_bavail*v.f_frsize
processes=[];regions={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode().split('\0')[:-1]
  if 'serve' in args and any('vllm' in a for a in args):
   if '--api-key' in args:args[args.index('--api-key')+1]='[REDACTED]'
   processes.append({'pid':int(p.name),'args':args})
  for fd in (p/'fd').iterdir():
   try:
    if '/dev/shm/vllm_offload_' in os.readlink(fd):
     info=fd.stat();regions[(info.st_dev,info.st_ino)]=info.st_size
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
s=requests.Session();s.trust_env=False
h={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
def get(path,port=8000,headers=h):
 try:
  r=s.get('http://127.0.0.1:'+str(port)+path,headers=headers,timeout=3)
  return r.status_code,r.text
 except requests.RequestException:return None,''
health,_=get('/health');mc,metrics=get('/metrics');models_code,models=get('/v1/models')
chat_token=Path(os.getenv('CHAT_TOKEN_FILE','/workspace/chat/chat-token')).read_text().strip()
gc,gh=get('/health',8001,{'Authorization':'Bearer '+chat_token})
try:gh=json.loads(gh)
except ValueError:gh={}
idle={}
for line in metrics.splitlines():
 if line.startswith('#'):continue
 name=line.split('{')[0].split(' ')[0]
 if name in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_offload_cpu_cache_usage_perc','vllm:kv_offload_cpu_cache_write_usage_perc','vllm:kv_offload_cpu_cache_read_usage_perc']:
  idle[name]=idle.get(name,0)+float(line.rsplit(' ',1)[1])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps({'time':time.time(),'health_status':health,'metrics_http':mc,'gateway_http':gc,'gateway_health':gh,'memory':memory,'processes':processes,'native_cpu_offload_region_bytes':list(regions.values()),'idle_counts':idle,'metrics':metrics,'models':json.loads(models) if models_code==200 else None,'gpu_csv':gpu,'start_sha256':hashlib.sha256(Path('/usr/local/bin/start-server').read_bytes()).hexdigest() if Path('/usr/local/bin/start-server').exists() else None}))
