"""Matched, resumable, budget-bounded native CPU KV-cache A/B experiment.

Raw numerical evidence is written eagerly. Prompt IDs and generated text never
leave memory; only input hashes and bounded correctness booleans are retained.
"""
import asyncio
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time

import aiohttp

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
LEASE=json.loads((ROOT/'lease.json').read_text())
RUNTIME=json.loads((ROOT/'runtime.json').read_text())
spec=importlib.util.spec_from_file_location('capacity',ROOT/'benchmark-capacity.py')
b=importlib.util.module_from_spec(spec);sys.modules['capacity']=b;spec.loader.exec_module(b)
with (ROOT/'prompts.private.pickle').open('rb') as f:POOL=pickle.load(f)
HEADERS={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
START=LEASE['created']
ARM='A'


def event(kind,**fields):
    row={'time':time.time(),'elapsed_s':time.time()-START,'arm':ARM,'event':kind,**fields}
    with (RESULTS/'experiment.events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(row),flush=True)


def save(name,value):
    p=RESULTS/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)


class Prompts:
    def __init__(self,offset=0,output=None):self.offset,self.output=offset,output
    def get(self,workload,i):
        ids,out=POOL['items'][workload][self.offset+i]
        return ids,out if self.output is None else self.output


async def metrics():
    async with HTTP.get('http://127.0.0.1:8000/metrics',headers=HEADERS,timeout=aiohttp.ClientTimeout(total=5)) as r:
        assert r.status==200
        raw=await r.text()
    values={}
    for line in raw.splitlines():
        if not line.startswith('vllm:'):continue
        key,val=line.rsplit(' ',1)
        try:
            number=float(val)
            assert math.isfinite(number),'Nonfinite metric'
            values[key]=number
        except ValueError:pass
    return values


def total(values,fragment):
    return sum(v for k,v in values.items() if k.split('{')[0]==('vllm:'+fragment))


def differences(before,after):
    result={k:v-before.get(k,0) for k,v in after.items() if not k.endswith('_created')}
    assert all(v>=0 for k,v in result.items() if k.split('{')[0].endswith(('_total','_count','_sum','_bucket'))),'Cumulative metric reset'
    # Histogram bucket keys have labels; ignore created gauges by base name.
    return {k:v for k,v in result.items() if not k.split('{')[0].endswith('_created')}


async def drain():
    until=min(time.time()+40,LEASE['restore_at'])
    while time.time()<until:
        m=await metrics()
        if total(m,'num_requests_running')==0 and total(m,'num_requests_waiting')==0:
            await asyncio.sleep(1.2)
            return await metrics()
        await asyncio.sleep(.25)
    raise RuntimeError('Engine failed to drain')


async def run(name,workload,c,count,cutoff,offset=0,warm=False,output=None):
    full=ARM+'-'+name
    existing=RESULTS/(full+'.summary.json')
    if existing.exists():
        data=json.loads(existing.read_text())
        if data.get('all_outcomes_accounted') and data.get('outcomes')==count and not data.get('stopped'):
            event('resumed',name=full);return data
        event('skipped',name=full,reason='partial_evidence_preserved');return None
    if time.time()+65>=cutoff:
        event('skipped',name=full,reason='phase_budget');return None
    before=await drain()
    wall=min(cutoff-time.time()-45,300+math.ceil(count/c)*70)
    cfg=b.Config(name=full,workload=workload,concurrency=c,max_requests=count,duration=max(1,wall-1),wall_seconds=wall,
        request_deadline=300,seed=12345,force_length=True,max_num_seqs=256,max_num_batched_tokens=8192,
        runtime_fingerprint=RUNTIME['fingerprint'],cache_namespace=LEASE['lease_id']+'/'+ARM,
        scenario_kind='warmup' if warm else 'capacity')
    prompts=Prompts(offset,output)
    input_hash=hashlib.sha256(json.dumps([prompts.get(workload,i) for i in range(count)],separators=(',',':')).encode()).hexdigest()
    event('scenario_start',name=full,workload=workload,concurrency=c,count=count,prompt_ids_sha256=input_hash)
    summary=await b.Runner(cfg,prompts,b.Writer(RESULTS,cfg),api_key=os.getenv('VLLM_API_KEY')).run()
    after=await drain()
    expected=sum(max(0,r.get('completion_tokens',0)-1) for r in [json.loads(x) for x in (RESULTS/(full+'.requests.jsonl')).read_text().splitlines()] if r['outcome']=='success')
    # Await asynchronous engine statistics publication, bounded independently.
    until=time.time()+10
    while time.time()<until:
        if (total(after,'generation_tokens_total')-total(before,'generation_tokens_total')>=summary['total_output_tokens']
            and total(after,'inter_token_latency_seconds_count')-total(before,'inter_token_latency_seconds_count')>=expected):break
        await asyncio.sleep(.5);after=await metrics()
    delta=differences(before,after)
    generated=total(after,'generation_tokens_total')-total(before,'generation_tokens_total')
    itl=total(after,'inter_token_latency_seconds_count')-total(before,'inter_token_latency_seconds_count')
    token_verified=summary['successful']==count and not summary.get('stopped') and itl==expected and generated==summary['total_output_tokens']
    control={'prompt_ids_sha256':input_hash,'before':before,'after':after,'delta':delta,
        'server_itl_count':itl,'expected_decode_token_intervals':expected,'generated_tokens_delta':generated,
        'server_itl_token_semantics_verified':token_verified,
        'preemptions_delta':total(after,'num_preemptions_total')-total(before,'num_preemptions_total'),
        'drained':True,'histogram_quantiles':'bucket estimates from drained before/after deltas'}
    host_hits=sum(v for k,v in delta.items() if 'prompt_tokens_cached_by_source_total' in k and 'source="host"' in k)
    load_bytes=sum(v for k,v in delta.items() if 'kv_offload_load_bytes' in k and k.split('{')[0].endswith(('_total','_sum')))
    control['unique_request_preemption_resume_proven']=(ARM=='B' and 'long-c' in name and 'wave' in name and control['preemptions_delta']>0 and host_hits>0 and load_bytes>0)
    control['resume_host_hit_tokens']=host_hits;control['resume_load_bytes']=load_bytes
    save(full+'.control.json',control)
    assert summary['all_outcomes_accounted']
    event('scenario_end',name=full,outcomes=summary['outcomes'],success=summary['success_rate'],ttft_p95_s=summary['ttft_s']['p95'],tpot_p95_s=summary['avg_tpot_s']['p95'],e2e_p95_s=summary['e2e_s']['p95'],output_tokens_s=summary['output_tokens_s'],itl_token_semantics_verified=token_verified,preemptions=control['preemptions_delta'])
    return summary


async def direct(ids,out,salt,model='qwen-bf16-base'):
    payload={'model':model,'prompt':ids,'max_tokens':out,'temperature':0,'seed':12345,'cache_salt':salt,'stream':False,'ignore_eos':True,'min_tokens':out}
    async with HTTP.post('http://127.0.0.1:8000/v1/completions',json=payload,headers=HEADERS,timeout=aiohttp.ClientTimeout(total=300)) as r:
        assert r.status==200,'Correctness HTTP failed'
        data=await r.json()
    assert data.get('usage',{}).get('prompt_tokens')==len(ids)
    assert data.get('usage',{}).get('completion_tokens')==out
    return data


def known_ids():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/workspace/qwen-bf16/model',local_files_only=True,trust_remote_code=False)
    template=tokenizer.apply_chat_template([{'role':'user','content':'KV_SENTINEL'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix,suffix=template.split('KV_SENTINEL')
    p=tokenizer.encode(prefix,add_special_tokens=False);s=tokenizer.encode(suffix,add_special_tokens=False)
    fact=tokenizer.encode('The archive code for Amber Harbor is 7329. Remember this exact code.\n',add_special_tokens=False)
    question=tokenizer.encode('\nWhat is the archive code for Amber Harbor given at the beginning? Reply with only the integer.',add_special_tokens=False)
    filler=tokenizer.encode('quiet river garden. '*8192,add_special_tokens=False)
    ids=p+fact+filler[:8192-len(p)-len(s)-len(fact)-len(question)]+question+s
    assert len(ids)==8192
    return ids


def answer_pass(data):
    return bool(re.match(r'^\s*7329(?:\s|[.!<]|$)',data['choices'][0].get('text','')))


async def correctness(cutoff):
    ids=known_ids()
    checks=[]
    for model in ['qwen-bf16-base','qwen-bf16']:
        if time.time()+320>=cutoff:raise RuntimeError('Correctness budget')
        before=await drain()
        data=await direct(ids,16,LEASE['lease_id']+'/known/'+ARM,model)
        after=await drain();delta=differences(before,after)
        hits=sum(v for k,v in delta.items() if 'prompt_tokens_cached_by_source_total' in k)
        checks.append({'model':model,'expected_answer_pass':answer_pass(data),'token_accounting_pass':True,'cache_hit_tokens':hits})
        assert checks[-1]['expected_answer_pass'],'Known answer failed'
        if model=='qwen-bf16':assert hits==0,'Adapter isolation failed'
    save(ARM+'-correctness.json',{'checks':checks,'passed':True,'output_persisted':False})
    event('correctness_pass',models=len(checks))


async def _reuse(trial,cutoff):
    name=f'reuse-{trial}'
    if time.time()+360>=cutoff:event('skipped',name=name,reason='reuse_budget');return
    ids=known_ids()
    salt=LEASE['lease_id']+f'/reuse/{ARM}/{trial}'
    await drain();fresh=await direct(ids,16,salt);assert answer_pass(fresh),'Known answer failed';prime=await drain()
    # Distinct early prompts and salts fill >GPU token capacity but <64GiB host
    # capacity. This is a controlled eviction proof, never a dev cache reset.
    fill=await run(name+'-eviction','long',32,80,cutoff,offset=700+trial*20,output=1)
    if fill is None:return
    before=await drain();replayed=await direct(ids,16,salt);after=await drain()
    assert answer_pass(replayed),'Restored known answer failed'
    delta=differences(before,after)
    host_hits=sum(v for k,v in delta.items() if 'prompt_tokens_cached_by_source_total' in k and 'source="host"' in k)
    load_bytes=sum(v for k,v in delta.items() if 'kv_offload_load_bytes' in k and k.split('{')[0].endswith(('_total','_sum')))
    load_observations=sum(v for k,v in delta.items() if 'kv_offload_load_bytes_count' in k)
    # Distinct salts must not reuse the preceding replay's cached identity.
    isolation_before=await drain();await direct(ids,16,salt+'-different');isolation_after=await drain()
    isolation_delta=differences(isolation_before,isolation_after)
    isolation_host_hits=sum(v for k,v in isolation_delta.items() if 'prompt_tokens_cached_by_source_total' in k and 'source="host"' in k)
    isolation_device_hits=sum(v for k,v in isolation_delta.items() if 'prompt_tokens_cached_by_source_total' in k and 'source="device"' in k)
    proof={'prime_metrics':prime,'before':before,'after':after,'delta':delta,'host_hit_tokens':host_hits,
        'load_bytes':load_bytes,'load_observations':load_observations,'cpu_hit_proven':host_hits>0 and (load_bytes>0 or load_observations>0),
        'different_salt_host_hits':isolation_host_hits,'different_salt_device_hits':isolation_device_hits,
        'salt_isolation_pass':isolation_host_hits==0 and isolation_device_hits==0,'isolation_delta':isolation_delta,
        'fresh_known_answer_pass':True,'restored_known_answer_pass':True,'prompt_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    save(ARM+'-'+name+'.json',proof)
    assert proof['salt_isolation_pass'],'Salt isolation failed'
    if ARM=='B':assert proof['cpu_hit_proven'],'CPU retrieval not demonstrated'
    event('reuse_proof',trial=trial,cpu_hit_proven=proof['cpu_hit_proven'],host_hit_tokens=host_hits,load_bytes=load_bytes,salt_isolation_pass=True)


async def reuse(trial,cutoff):
    mandatory=ARM=='B' and trial==0
    try:
        async with asyncio.timeout(max(.01,cutoff-time.time()-45)):
            await _reuse(trial,cutoff-45)
        if mandatory:
            proof=RESULTS/(ARM+'-reuse-0.json')
            assert proof.exists() and json.loads(proof.read_text()).get('cpu_hit_proven'),'CPU retrieval not demonstrated'
    except TimeoutError:
        event('reuse_budget_censored',trial=trial,mandatory=mandatory)
        await drain()
        if mandatory:raise RuntimeError('Mandatory CPU retrieval proof budget exhausted')


async def change(arm,cutoff):
    global ARM
    ARM=arm
    a=json.loads((ROOT/'layout-A.json').read_text())
    desired={'arm':arm,'gpu_cache_bytes':a['actual_gpu_cache_bytes']}
    p=ROOT/'desired.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(desired));t.replace(p)
    until=min(cutoff,time.time()+19*60)
    while time.time()<until:
        await asyncio.sleep(2)
        status=json.loads((ROOT/'engine.status.json').read_text())
        if status.get('state')=='ready' and status.get('config')==desired:
            layout=json.loads((ROOT/('layout-'+arm+'.json')).read_text())
            assert layout['layout_fingerprint']==a['layout_fingerprint'],'GPU cache layout mismatch'
            event('matched_layout_verified',bytes=layout['actual_gpu_cache_bytes'],num_blocks=layout['num_blocks'],fingerprint=layout['layout_fingerprint'])
            return
    raise RuntimeError('Arm startup budget')


async def measurements(cutoff,drift=False,correctness_done=False):
    if not correctness_done:await correctness(cutoff)
    # Startup and shape-specific warmups are excluded from measured artifacts.
    await run('warm-short-c8','short',8,8,cutoff,offset=600,warm=True)
    if drift:
        await run('drift-short-c8','short',8,34,cutoff,offset=0)
        return
    await run('warm-medium-c8','medium',8,8,cutoff,offset=600,warm=True)
    for workload in ['short','medium']:
        offset=0
        for trial,count in enumerate([34,33,33]):
            await run(f'{workload}-c8-t{trial}',workload,8,count,cutoff,offset=offset)
            offset+=count
    for c in [32,64]:
        await run(f'warm-long-c{c}','long',c,c,cutoff,offset=500,warm=True)
        for trial in range(3):await run(f'long-c{c}-wave{trial}','long',c,c,cutoff,offset=trial*c)
    for workload in ['short','medium']:
        await run(f'warm-{workload}-c2',workload,2,2,cutoff,offset=620,warm=True)
        await run(f'{workload}-c2-overhead',workload,2,20,cutoff,offset=200)
    # Store/retrieve proof is required; extra repeats are budget-optional.
    for trial in (range(1,3) if ARM=='B' else range(3)):await reuse(trial,cutoff)


async def main():
    global HTTP,ARM
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),trust_env=False) as HTTP:
        event('experiment_ready',runtime_fingerprint=RUNTIME['fingerprint'])
        a=json.loads((ROOT/'layout-A.json').read_text())
        assert {'MambaSpec','FullAttentionSpec'}.issubset({t for x in a['groups'] for t in x['leaf_types']})
        assert a['actual_gpu_cache_bytes']<68719476736,'CPU cache must exceed GPU cache'
        await measurements(START+60*60)
        await change('B',START+80*60)
        # Mandatory retrieval proof before performance measurements.
        await correctness(START+123*60)
        await reuse(0,START+123*60)
        await measurements(START+123*60,correctness_done=True)
        await change('A2',START+146*60)
        await measurements(START+149*60,drift=True)
        event('experiment_complete')


if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException as e:
        event('experiment_stopped',error_type=type(e).__name__,reason=str(e) if str(e) in ['GPU cache layout mismatch','CPU cache must exceed GPU cache','Salt isolation failed','Known answer failed','Engine failed to drain','Correctness budget','Arm startup budget'] else 'suppressed')
    finally:
        (ROOT/'restore.remote').write_text('controller finished')
