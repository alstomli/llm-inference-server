"""Deadline-budgeted experiment controller, executed inside the existing pod."""
import asyncio,hashlib,importlib.util,json,os,pickle,signal,subprocess,sys,time
from pathlib import Path
import aiohttp

ROOT=Path('/workspace/capacity-session');RESULTS=ROOT/'results';RESULTS.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('capacity',ROOT/'benchmark-capacity.py');b=importlib.util.module_from_spec(spec);sys.modules['capacity']=b;spec.loader.exec_module(b)
LEASE=json.loads((ROOT/'lease.json').read_text());RUNTIME=json.loads((ROOT/'runtime.json').read_text())
STOP=LEASE['created']+5700
STARTED=LEASE['created']
history=[]

class Prompts:
    def __init__(self):
        with (ROOT/'prompts.private.pickle').open('rb') as f:data=pickle.load(f)
        self.items,self.size=data['items'],data['size']
    def get(self,workload,i):
        if i>=self.size:raise ValueError('Pool exhausted')
        return self.items[workload][i]

PROMPTS=Prompts()

def event(kind,**values):
    row={'t':time.time(),'elapsed_s':time.time()-STARTED,'event':kind,**values}
    with (RESULTS/'experiment.events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def config():
    status=json.loads((ROOT/'engine.status.json').read_text())
    assert status['state']=='ready'
    return status['config']

async def healthy(session):
    try:
        async with session.get('http://127.0.0.1:8000/health',headers=HEADERS,timeout=aiohttp.ClientTimeout(total=5)) as r:return r.status==200
    except (aiohttp.ClientError,TimeoutError):return False

HEADERS={'Authorization':'Bearer '+os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}

async def reset_cache(session):
    assert await healthy(session),'Engine unhealthy before scenario'
    await drain()
    async with session.post('http://127.0.0.1:8000/reset_prefix_cache',headers=HEADERS,timeout=aiohttp.ClientTimeout(total=10)) as r:
        assert r.status==200,'Prefix reset unavailable'

async def run(name,workload,c,duration=120,max_requests=1024,rate=0,force=True,wall=None):
    remaining=STOP-time.time()
    if remaining<45:
        event('skipped',name=name,reason='global_budget');return None
    conf=config();await reset_cache(SESSION)
    wall=min(wall or duration+180,remaining)
    cfg=b.Config(name=name,workload=workload,concurrency=c,duration=min(duration,max(1,wall-15)),wall_seconds=wall,
                 max_requests=max_requests,arrival_rate=rate,max_backlog=min(1024,max(16,c*2)),seed=12345,
                 force_length=force,max_num_seqs=conf['sequences'],max_num_batched_tokens=conf['batched_tokens'],
                 runtime_fingerprint=RUNTIME['fingerprint'],scenario_kind=('warmup' if name.startswith('warm-') else 'natural_stop' if not force else 'sustained' if rate else 'tuning' if name.startswith('batch-') else 'capacity'))
    event('scenario_start',name=name,workload=workload,concurrency=c,rate=rate,config=conf,duration=cfg.duration)
    runner=b.Runner(cfg,PROMPTS,b.Writer(RESULTS,cfg),api_key=os.getenv('VLLM_API_KEY'))
    async def saturation_probe():
        await asyncio.sleep(min(30,cfg.duration/2))
        await gpu(name+'-under-load')
    probe=asyncio.create_task(saturation_probe())
    try:summary=await runner.run();history.append(summary)
    finally:
        probe.cancel();await asyncio.gather(probe,return_exceptions=True)
    event('scenario_end',name=name,outcomes=summary['outcomes'],success=summary['success_rate'],
          output_tokens_s=summary['output_tokens_s'],ttft_p95_s=summary['ttft_s']['p95'],tpot_p95_s=summary['avg_tpot_s']['p95'],stopped=summary['stopped'])
    assert summary['all_outcomes_accounted'],'Missing outcomes'
    if not await healthy(SESSION):raise RuntimeError('Engine lost health')
    if summary['stopped']=='failure_rate_exceeded_5_percent':raise RuntimeError('Failure threshold')
    # Ensure no accepted requests remain active before resetting cache/config.
    await drain()
    return summary

async def gpu(label):
    fields=['power.draw','power.limit','clocks.current.sm','clocks_event_reasons.active','clocks_event_reasons.hw_power_brake_slowdown']
    values={}
    for field in fields:
        p=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu='+field,'--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
        raw,_=await p.communicate()
        if p.returncode==0:values[field]=raw.decode().strip()
    event('gpu_diagnostics',label=label,values=values)

async def counters():
    values={}
    try:
        async with SESSION.get('http://127.0.0.1:8000/metrics',headers=HEADERS,timeout=aiohttp.ClientTimeout(total=2)) as r:
            text=await r.text()
        for line in text.splitlines():
            if line.startswith('#'):continue
            pieces=line.rsplit(' ',1)
            if len(pieces)!=2:continue
            key=pieces[0].split('{')[0].removeprefix('vllm:')
            if key in ['num_requests_running','num_requests_waiting','prompt_tokens_total','generation_tokens_total']:
                values[key]=values.get(key,0)+float(pieces[1])
    except (aiohttp.ClientError,ValueError,TimeoutError):pass
    return values

async def drain():
    until=min(time.time()+30,STOP)
    while time.time()<until:
        sample=await counters()
        if sample.get('num_requests_running')==0 and sample.get('num_requests_waiting')==0:return
        await asyncio.sleep(.25)
    raise RuntimeError('Engine did not drain to zero running and waiting')

async def pair(phase,repetition):
    if time.time()>min(STOP,STARTED+4500):return
    name=f'pair-{phase}-{repetition}';await reset_cache(SESSION);conf=config()
    cfg=b.Config(name=name,workload='long',concurrency=2,duration=180,wall_seconds=200,
                 max_num_seqs=conf['sequences'],max_num_batched_tokens=conf['batched_tokens'],runtime_fingerprint=RUNTIME['fingerprint'],scenario_kind='stagger')
    runner=b.Runner(cfg,PROMPTS,b.Writer(RESULTS,cfg),api_key=os.getenv('VLLM_API_KEY'))
    runner.started=time.monotonic();runner.arrivals_end=runner.started+180
    baseline=await counters();evidence=[];verified=phase=='simultaneous'
    a=runner.new_row(time.monotonic());a['workload']='long'
    first=asyncio.create_task(runner.request(SESSION,a))
    try:
        if phase!='simultaneous':
            end=time.monotonic()+30
            while time.monotonic()<end and not first.done():
                current=await counters()
                delta_prompt=current.get('prompt_tokens_total',0)-baseline.get('prompt_tokens_total',0)
                delta_output=current.get('generation_tokens_total',0)-baseline.get('generation_tokens_total',0)
                active=current.get('num_requests_running',0)
                evidence.append({'t_s':time.monotonic()-runner.started,'prompt_delta':delta_prompt,'output_delta':delta_output,'running':active})
                if phase=='prefill' and active==1 and delta_prompt>0 and delta_output==0:verified=True;break
                if phase=='decode' and active==1 and delta_output>0:verified=True;break
                await asyncio.sleep(.02)
        second=runner.new_row(time.monotonic());second['workload']='short'
        event('pair_B_submission',name=name,phase=phase,phase_verified=verified,a_still_active=not first.done(),server_evidence=evidence[-3:])
        await asyncio.wait_for(asyncio.gather(first,runner.request(SESSION,second)),180)
    finally:
        if not first.done():first.cancel();await asyncio.gather(first,return_exceptions=True)
        summary=b.aggregate(cfg,runner.records,time.monotonic()-runner.started,runner.peak_outstanding)
        summary.update(phase=phase,phase_verified=verified,phase_evidence=evidence,all_outcomes_accounted=len(runner.records)==runner.next_id)
        runner.writer.close(summary);history.append(summary)

async def change_batch(tokens):
    desired={**config(),'batched_tokens':tokens}
    event('scheduler_change_start',config=desired)
    temp=ROOT/'desired.tmp';temp.write_text(json.dumps(desired));temp.replace(ROOT/'desired.json')
    until=min(time.time()+900,STOP-240)
    while time.time()<until:
        await asyncio.sleep(3)
        try:
            s=json.loads((ROOT/'engine.status.json').read_text())
            if s['state']=='ready' and s['config']==desired and await healthy(SESSION):
                event('scheduler_change_ready',config=desired);return True
        except (ValueError,FileNotFoundError):pass
    event('scheduler_change_failed',config=desired);return False

async def main():
    global SESSION
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None),trust_env=False) as SESSION:
        event('experiment_ready',runtime=RUNTIME,config=config())
        await gpu('idle')
        # Preparation/warmup capped at15min from exclusive activation.
        await run('warm-short','short',1,duration=60,max_requests=2,wall=90)
        await run('warm-medium','medium',1,duration=60,max_requests=2,wall=90)
        await run('warm-long','long',1,duration=60,max_requests=2,wall=90)
        await run('natural-stop-validation','short',2,duration=120,max_requests=12,force=False,wall=180)
        ceiling=config()['sequences'];sweep=[];low_gains=0;previous=None
        sweep_end=min(STARTED+3000,STOP-2100)
        for c in [1,2,4,8,16,32,64,128,256]:
            if c>ceiling or time.time()+200>sweep_end:
                event('skipped',name='medium-sweep-'+str(c),reason='ceiling_or_sweep_budget');break
            sample=await run('medium-sweep-'+str(c),'medium',c,duration=120,wall=min(300,sweep_end-time.time()))
            if sample is None:break
            sweep.append(sample)
            if previous and sample['output_tokens_s']<previous*1.05:low_gains+=1
            else:low_gains=0
            previous=sample['output_tokens_s']
            if low_gains>=2:
                event('sweep_stop',reason='two_consecutive_gains_below_5_percent',concurrency=c);break
        assert sweep,'No primary sweep evidence'
        peak=max(s['output_tokens_s'] for s in sweep)
        knee=min(s['config']['concurrency'] for s in sweep if s['output_tokens_s']>=peak*.95)
        provisional=[s['config']['concurrency'] for s in sweep if s['success_rate']>=.99 and (s['ttft_s']['p95'] or 1e9)<=5 and (s['avg_tpot_s']['p95'] or 1e9)<=.1]
        candidate=max(provisional) if provisional else 1
        event('primary_boundary',knee=knee,provisional_chat_candidate=candidate,ceiling=ceiling)
        # Verify both short and medium at the same exact settings, >=100 outcomes.
        await run('validate-medium-'+str(candidate),'medium',candidate,duration=420,max_requests=100,wall=540)
        await run('validate-short-'+str(candidate),'short',candidate,duration=420,max_requests=100,wall=540)
        for c in sorted({max(1,knee//2),knee,min(ceiling,knee*2)}):
            if time.time()>STARTED+3900:break
            await run('short-boundary-'+str(c),'short',c,duration=75,wall=180)
            await run('long-boundary-'+str(c),'long',c,duration=75,wall=240)
        mixed=await run('mixed-boundary','mixed',min(ceiling,max(candidate,knee)),duration=150,wall=330)
        for rep in range(3):
            for phase in ['simultaneous','prefill','decode']:
                await pair(phase,rep)
        if mixed:
            rate=mixed['completed_requests_s']
            for ratio in [.5,.8,1,1.2]:
                if time.time()>STOP-720:
                    event('skipped',name='open-loop-'+str(ratio),reason='sustain_budget');continue
                await run('open-loop-'+str(ratio),'mixed',min(ceiling,max(candidate,knee)),duration=100,
                          rate=rate*ratio,wall=min(280,STOP-600-time.time()))
        for tokens in [4096,16384]:
            if time.time()>STOP-1200:
                event('skipped',name='batch-'+str(tokens),reason='restart_budget');continue
            if not await change_batch(tokens):break
            await run('warm-batch-'+str(tokens),'mixed',2,duration=60,max_requests=4,wall=120)
            await run('batch-'+str(tokens),'mixed',min(ceiling,max(candidate,knee)),duration=120,max_requests=200,wall=240)
        event('experiment_complete',scenarios=len(history),knee=knee,candidate=candidate)

if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException as e:event('experiment_stopped',error_type=type(e).__name__)
    finally:
        try:b.analyze(RESULTS)
        except Exception as e:event('analysis_partial',error_type=type(e).__name__)
        (ROOT/'restore.remote').write_text('deadline or completed')
        event('restore_requested')
