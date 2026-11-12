"""Focused, single-engine GPU-resident versus CPU-restored prefix experiment."""
import asyncio,hashlib,importlib.util,json,math,os,random,re,sys,time
from pathlib import Path
import aiohttp

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
HEADERS={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
PREFIX=3136;INPUT=3200;OUTPUT=128
STATS_INTERVAL=10.0
FILE_PREFIX=''
class CorrectnessFailure(RuntimeError):pass
spec=importlib.util.spec_from_file_location('capacity',ROOT/'benchmark-capacity.py')
b=importlib.util.module_from_spec(spec);sys.modules['capacity']=b;spec.loader.exec_module(b)

def write(name,value):
 p=RESULTS/(FILE_PREFIX+name);assert not p.exists(),'Evidence already finalized';t=p.with_suffix('.tmp');t.write_text(json.dumps(value,allow_nan=False));t.replace(p)
def event(kind,**values):
 with (RESULTS/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':kind,**values},allow_nan=False)+'\n')
def total(m,name,source=None):
 return sum(v for k,v in m.items() if k.split('{')[0]=='vllm:'+name and (source is None or f'source="{source}"' in k))
def delta(before,after):
 assert all(k in after for k in before if k.split('{')[0].endswith(('_total','_count','_sum','_bucket'))),'Cumulative metric series disappeared'
 out={k:v-before.get(k,0) for k,v in after.items() if not k.split('{')[0].endswith('_created')}
 for k,v in out.items():
  assert math.isfinite(v),'Nonfinite metric delta'
  if k.split('{')[0].endswith(('_total','_count','_sum','_bucket')):assert v>=0,'Counter reset'
 return out
def transfer(m,kind):
 return sum(v for k,v in m.items() if k.split('{')[0] in [f'vllm:kv_offload_{kind}_bytes_total',f'vllm:kv_offload_{kind}_bytes_sum'])
def verify(d,mode,count=1):
 checks={'finished_requests':total(d,'request_success_total')==count,
  'prompt_count':total(d,'request_prompt_tokens_count')==count,
  'prompt_tokens':total(d,'request_prompt_tokens_sum')==INPUT*count,
  'generation_tokens':total(d,'generation_tokens_total')==(OUTPUT if mode in ['gpu','cpu','prime'] else count),
  'prefill_count':total(d,'request_prefill_kv_computed_tokens_count')==count,
  'computed_tokens':total(d,'request_prefill_kv_computed_tokens_sum')==((64 if mode in ['gpu','cpu'] else INPUT)*count),
  'device_hits':total(d,'prompt_tokens_cached_by_source_total','device')==(PREFIX if mode=='gpu' else 0),
  'host_hits':total(d,'prompt_tokens_cached_by_source_total','host')==(PREFIX if mode=='cpu' else 0),
  'zero_preemptions':total(d,'num_preemptions_total')==0}
 if mode in ['gpu','cpu']:
  checks['no_cpu_store']=transfer(d,'store')==0
  checks['server_ttft_count']=total(d,'time_to_first_token_seconds_count')==1
 if mode in ['gpu','cpu','prime']:checks['itl_count']=total(d,'inter_token_latency_seconds_count')==OUTPUT-1
 if mode=='cpu':checks['cpu_load']=transfer(d,'load')>0
 else:checks['no_cpu_load']=transfer(d,'load')==0
 if mode=='prime':checks['cpu_store']=transfer(d,'store')>0
 if mode=='fill':checks['no_cpu_store']=transfer(d,'store')==0
 return checks

async def metrics():
 async with HTTP.get('http://127.0.0.1:8000/metrics',headers=HEADERS,timeout=aiohttp.ClientTimeout(total=5)) as r:
  assert r.status==200;raw=await r.text()
 out={}
 for line in raw.splitlines():
  if not line.startswith('vllm:'):continue
  k,v=line.rsplit(' ',1);v=float(v);assert math.isfinite(v);out[k]=v
 return out
def quiescent(m):
 return all(total(m,k)==0 for k in ['num_requests_running','num_requests_waiting','kv_offload_cpu_cache_usage_perc','kv_offload_cpu_cache_write_usage_perc','kv_offload_cpu_cache_read_usage_perc'])
def signature(m):
 return (transfer(m,'load'),transfer(m,'store'),total(m,'generation_tokens_total'),total(m,'request_success_total'),total(m,'inter_token_latency_seconds_count'))
async def stable(expected=None):
 until=min(time.time()+90,LEASE['restore_at']-5);last=None;since=time.monotonic()
 while time.time()<until:
  m=await metrics();sig=signature(m)
  if not quiescent(m) or sig!=last or (expected is not None and not expected(m)):
   since=time.monotonic();last=sig
  elif time.monotonic()-since>=STATS_INTERVAL:
   await asyncio.sleep(1)
   final=await metrics()
   if quiescent(final) and signature(final)==sig:return final
   since=time.monotonic();last=signature(final)
  await asyncio.sleep(.5)
 raise RuntimeError('Finalized quiet metric interval unavailable')

async def request(ids,salt,limit,output=OUTPUT,stream=True):
 assert time.time()+310<LEASE['restore_at'],'Restoration reserve'
 payload={'model':'qwen-bf16-base','prompt':ids,'max_tokens':output,'min_tokens':output,'ignore_eos':True,'temperature':0,'seed':12345,
  'cache_salt':salt,'kv_transfer_params':{'max_offload_tokens':limit},'stream':stream}
 if stream:payload['stream_options']={'include_usage':True}
 began=time.monotonic();first=last=None;gaps=[];text='';usage=None;done=False;finish=None
 async with HTTP.post('http://127.0.0.1:8000/v1/completions',headers=HEADERS,json=payload,timeout=aiohttp.ClientTimeout(total=300)) as r:
  if r.status!=200:
   error=RuntimeError('Target HTTP failure');error.http_status=r.status;raise error
  if not stream:
   data=await r.json();usage=data.get('usage');text=data['choices'][0].get('text','');done=True;finish=data['choices'][0].get('finish_reason')
  else:
   parser=b.SSE()
   async for block in r.content.iter_any():
    for frame in parser.feed(block):
     now=time.monotonic()
     if frame=='[DONE]':done=True;continue
     data=json.loads(frame);assert not data.get('error'),'Stream error'
     if data.get('usage'):usage=data['usage']
     for choice in data.get('choices',[]):
      if choice.get('finish_reason'):finish=choice['finish_reason']
      part=choice.get('text','')
      if part:
       text+=part
       if first is None:first=now
       if last is not None:gaps.append(now-last)
       last=now
   assert not parser.buffer.strip(),'Truncated stream'
 assert done and finish=='length' and usage and usage['prompt_tokens']==INPUT and usage['completion_tokens']==output,'Token/stream accounting failure'
 return {'ttft_s':first-began if first else None,'e2e_s':time.monotonic()-began,'approx_tpot_s':(last-first)/(output-1) if first and output>1 else None,
  'chunk_gap_p95_s':b.percentile(gaps,95),'chunk_count':len(gaps)+bool(first),'prompt_tokens':usage['prompt_tokens'],'completion_tokens':output,
  'response_cached_tokens':(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),'answer_pass':bool(re.match(r'^\s*7329(?:\s|[.!<]|$)',text)) if output==OUTPUT else None}

class Prompts:
 def __init__(self):
  from transformers import AutoTokenizer
  self.t=AutoTokenizer.from_pretrained('/workspace/qwen-bf16/model',local_files_only=True,trust_remote_code=False)
  template=self.t.apply_chat_template([{'role':'user','content':'SENTINEL'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  p,s=template.split('SENTINEL');self.p=self.t.encode(p,add_special_tokens=False);self.s=self.t.encode(s,add_special_tokens=False)
  q=self.t.encode('\nWhat is the archive code for Amber Harbor given at the beginning? Reply with only the integer.',add_special_tokens=False)
  pad=self.t.encode(' quiet river garden.'*10000,add_special_tokens=False)
  self.suffix=pad[:64-len(q)-len(self.s)]+q+self.s;assert len(self.suffix)==64
  self.pad=pad
 def get(self,identity,filler=False):
  fact=('The archive code for Amber Harbor is 7329. Remember it exactly. ' if not filler else 'Read the following neutral notes. ')
  body=self.t.encode(hashlib.sha256(identity.encode()).hexdigest()+' '+fact,add_special_tokens=False)
  prefix=self.p+body+self.pad[:PREFIX-len(self.p)-len(body)]
  ids=prefix+self.suffix;assert len(prefix)==PREFIX and len(ids)==INPUT
  return ids
def accepted_trials():
 out=set()
 for p in RESULTS.glob('*-pair.json'):
  row=json.loads(p.read_text())
  if row.get('accepted') and not row.get('warmup'):out.add(row['trial'])
 return out
def expected_fillers(blocks):
 assert blocks==83,'Unexpected cache allocation requires review'
 return 22
def pair_summary(trial,order,pair):
 return {'trial':trial,'order':order,'accepted':True,'warmup':trial=='warmup','gpu':pair['gpu'],'cpu':pair['cpu'],
  'server_ttft_cpu_minus_gpu_s':pair['cpu']['server_ttft_s']-pair['gpu']['server_ttft_s'],
  'client_ttft_cpu_minus_gpu_s':pair['cpu']['ttft_s']-pair['gpu']['ttft_s']}

async def _checked(ids,salt,mode,label):
 before=await stable();row=await request(ids,salt,PREFIX)
 def expected(m):
  d=delta(before,m)
  return total(d,'request_success_total')>=1 and total(d,'request_prefill_kv_computed_tokens_count')>=1 and total(d,'inter_token_latency_seconds_count')>=OUTPUT-1
 after=await stable(expected);d=delta(before,after);checks=verify(d,mode);checks['known_answer']=row['answer_pass']
 row['server_ttft_s']=total(d,'time_to_first_token_seconds_sum')
 row.update(label=label,condition=mode,checks=checks,passed=all(checks.values()),metrics_delta=d,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),time=time.time())
 write(label+'.json',row);event('request_gate',label=label,condition=mode,passed=row['passed'],failed=[k for k,v in checks.items() if not v])
 if not row['answer_pass']:raise CorrectnessFailure('Known-answer gate failed')
 assert row['passed'],'Residency/correctness proof failed'
 return row
async def checked(ids,salt,mode,label):
 began=time.monotonic();event('attempt_start',label=label,condition=mode,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
 try:return await _checked(ids,salt,mode,label)
 except BaseException as error:
  write(label+'-failure.json',{'label':label,'condition':mode,'time':time.time(),'elapsed_s':time.monotonic()-began,'error_type':type(error).__name__,'http_status':getattr(error,'http_status',None),'accepted':False})
  raise
async def evict(trial):
 before=await stable()
 for i in range(FILLERS):
  identity=f"{LEASE['lease_id']}/filler/{FILE_PREFIX}/{trial}/{i}"
  await request(PROMPTS.get(identity,True),hashlib.sha256(identity.encode()).hexdigest(),0,output=1,stream=False)
 after=await stable(lambda m:total(delta(before,m),'request_success_total')>=FILLERS)
 d=delta(before,after);checks=verify(d,'fill',FILLERS)
 write(f'{trial}-fill.json',{'count':FILLERS,'checks':checks,'passed':all(checks.values()),'metrics_delta':d})
 assert all(checks.values()),'Filler contamination or transfer failure'

async def main():
 global HTTP,LEASE,PROMPTS,FILLERS,FILE_PREFIX
 LEASE=json.loads((ROOT/'lease.json').read_text());RESULTS.mkdir(exist_ok=True)
 FILE_PREFIX=f'attempt{time.time_ns()}-'
 layout=json.loads((ROOT/'layout-B.json').read_text());assert layout['actual_gpu_cache_bytes']<=4294967296 and layout['actual_gpu_cache_bytes']>3*1024**3
 assert {'MambaSpec','FullAttentionSpec'}.issubset({t for g in layout['groups'] for t in g['leaf_types']})
 FILLERS=expected_fillers(layout['num_blocks'])
 PROMPTS=Prompts();orders=['gpu-cpu']*10+['cpu-gpu']*10;random.Random(20261112).shuffle(orders)
 write('plan.json',{'orders':orders,'fillers_per_eviction':FILLERS,'stats_stability_seconds':STATS_INTERVAL,'input':INPUT,'prefix':PREFIX,'suffix':64,'output':OUTPUT,'layout_fingerprint':layout['layout_fingerprint']})
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),trust_env=False) as HTTP:
  # Complete one scored, proof-gated warm pair before accepting measurements.
  for trial,order in [('warmup','gpu-cpu'),*[(str(i),o) for i,o in enumerate(orders)]]:
   if trial in accepted_trials():
    event('accepted_pair_preserved',trial=trial);continue
   if time.time()+900>LEASE['restore_at']:event('budget_stop',next_trial=trial);break
   salt=hashlib.sha256(f"{LEASE['lease_id']}/target/{FILE_PREFIX}/{trial}".encode()).hexdigest();ids=PROMPTS.get(salt)
   await checked(ids,salt,'prime',trial+'-prime')
   pair={}
   if order=='gpu-cpu':
    pair['gpu']=await checked(ids,salt,'gpu',trial+'-gpu');await evict(trial);pair['cpu']=await checked(ids,salt,'cpu',trial+'-cpu')
   else:
    await evict(trial);pair['cpu']=await checked(ids,salt,'cpu',trial+'-cpu');pair['gpu']=await checked(ids,salt,'gpu',trial+'-gpu')
   write(trial+'-pair.json',pair_summary(trial,order,pair))
   event('pair_complete',trial=trial,order=order,ttft_gpu_s=pair['gpu']['ttft_s'],ttft_cpu_s=pair['cpu']['ttft_s'])
  event('experiment_complete')
if __name__=='__main__':
 succeeded=False;fatal=False
 try:asyncio.run(main());succeeded=True
 except BaseException as error:
  fatal=isinstance(error,CorrectnessFailure)
  try:event('experiment_stopped',error_type=type(error).__name__,reason=str(error) if str(error) in ['Residency/correctness proof failed','Filler contamination or transfer failure','Finalized quiet metric interval unavailable','Unexpected cache allocation'] else 'suppressed')
  except Exception:pass
 finally:
  if succeeded or fatal:(ROOT/'restore.remote').write_text('experiment finished' if succeeded else 'fatal correctness failure')
  else:(ROOT/'experiment.failed').write_text('paused for bounded diagnosis; absolute restoration lease remains active')
