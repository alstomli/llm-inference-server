import asyncio,importlib.util,json,os,threading,time,tempfile,unittest
from pathlib import Path
from unittest import mock
import requests
BASE=Path(__file__).parent
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,BASE/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class ProofTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=load('residency','kv-residency-experiment.py')
 def values(self,mode):
  d={f'vllm:{k}':v for k,v in {'request_success_total':1,'request_prompt_tokens_count':1,'request_prompt_tokens_sum':3200,'generation_tokens_total':128,
   'request_prefill_kv_computed_tokens_count':1,'request_prefill_kv_computed_tokens_sum':64,'num_preemptions_total':0,'inter_token_latency_seconds_count':127,'time_to_first_token_seconds_count':1}.items()}
  d['vllm:prompt_tokens_cached_by_source_total{source="device"}']=3136 if mode=='gpu' else 0
  d['vllm:prompt_tokens_cached_by_source_total{source="host"}']=3136 if mode=='cpu' else 0
  d['vllm:kv_offload_load_bytes_total']=100 if mode=='cpu' else 0
  return d
 def test_exact_gpu_and_cpu_proofs(self):
  for mode in ['gpu','cpu']:self.assertTrue(all(self.m.verify(self.values(mode),mode).values()))
 def test_partial_mixed_hits_rejected(self):
  d=self.values('cpu');d['vllm:prompt_tokens_cached_by_source_total{source="host"}']=3000;d['vllm:prompt_tokens_cached_by_source_total{source="device"}']=136
  self.assertFalse(all(self.m.verify(d,'cpu').values()))
 def test_cpu_requires_bytes_and_targets_forbid_stores(self):
  for mode in ['cpu','gpu']:
   d=self.values(mode);d['vllm:kv_offload_store_bytes_total']=1;self.assertFalse(self.m.verify(d,mode)['no_cpu_store'])
  d=self.values('cpu');d['vllm:kv_offload_load_bytes_total']=0;self.assertFalse(self.m.verify(d,'cpu')['cpu_load'])
 def test_exact_observations_and_no_unrelated_request(self):
  for key,bad in [('inter_token_latency_seconds_count',126),('generation_tokens_total',129),('request_success_total',2),('request_prefill_kv_computed_tokens_sum',65),('time_to_first_token_seconds_count',2)]:
   d=self.values('gpu');d['vllm:'+key]=bad;self.assertFalse(all(self.m.verify(d,'gpu').values()))
 def test_counter_reset_and_nonfinite(self):
  for before,after in [({'vllm:x_total':2},{'vllm:x_total':1}),({'vllm:x_count':1},{}),({}, {'vllm:gauge':float('nan')})]:
   with self.assertRaises(AssertionError):self.m.delta(before,after)
 def test_exact_reviewed_geometry(self):
  self.assertEqual(self.m.expected_fillers(83),22)
  for blocks in [82,84,667]:
   with self.assertRaises(AssertionError):self.m.expected_fillers(blocks)
 def test_known_answer_failure_is_fatal_and_retained(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(self.m,'RESULTS',Path(tmp)),mock.patch.object(self.m,'stable',new=mock.AsyncMock(side_effect=[{},self.values('gpu')])),mock.patch.object(self.m,'request',new=mock.AsyncMock(return_value={'answer_pass':False})):
   with self.assertRaises(self.m.CorrectnessFailure):asyncio.run(self.m.checked([0]*3200,'test','gpu','known-answer'))
   row=json.loads((Path(tmp)/'known-answer.json').read_text());self.assertFalse(row['checks']['known_answer'])
   self.assertTrue((Path(tmp)/'known-answer-failure.json').exists())
 def test_server_ttft_primary_client_secondary(self):
  result=self.m.pair_summary('0','gpu-cpu',{'gpu':{'server_ttft_s':.2,'ttft_s':.3},'cpu':{'server_ttft_s':.4,'ttft_s':.8}})
  self.assertAlmostEqual(result['server_ttft_cpu_minus_gpu_s'],.2)
  self.assertAlmostEqual(result['client_ttft_cpu_minus_gpu_s'],.5)
 def test_immutable_accepted_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   old=self.m.RESULTS;self.m.RESULTS=Path(tmp)
   try:
    self.m.write('0-pair.json',{'trial':'0','accepted':True,'warmup':False})
    self.m.write('warmup-pair.json',{'trial':'warmup','accepted':True,'warmup':True})
    self.assertEqual(self.m.accepted_trials(),{'0'})
    with self.assertRaises(AssertionError):self.m.write('0-pair.json',{'accepted':False})
   finally:self.m.RESULTS=old
 def test_fill_batch_zero_cpu_transfer(self):
  d={f'vllm:{k}':v for k,v in {'request_success_total':22,'request_prompt_tokens_count':22,'request_prompt_tokens_sum':70400,'generation_tokens_total':22,'request_prefill_kv_computed_tokens_count':22,'request_prefill_kv_computed_tokens_sum':70400}.items()}
  self.assertTrue(all(self.m.verify(d,'fill',22).values()));d['vllm:kv_offload_store_bytes_total']=1;self.assertFalse(all(self.m.verify(d,'fill',22).values()))
class StatusTests(unittest.TestCase):
 def test_actual_auth_private_allowlist_and_maintenance(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.dict(os.environ,{'KV_EXPERIMENT_ROOT':tmp,'JUPYTER_PASSWORD':'test-only-token'}):
   m=load('status_test','kv-residency-status.py');server=m.http.server.ThreadingHTTPServer(('127.0.0.1',0),m.Handler)
   thread=threading.Thread(target=server.serve_forever);thread.start();url='http://127.0.0.1:'+str(server.server_port)
   try:
    self.assertEqual(requests.get(url+'/__kv_status',timeout=2).status_code,401)
    h={'Authorization':'Bearer test-only-token'}
    self.assertEqual(requests.get(url+'/__kv_status',headers=h,timeout=2).status_code,200)
    for name in ['original-launch.private.json','../lease.json','results/x.private.json']:
     self.assertIn(requests.get(url+'/__kv_file/'+name,headers=h,timeout=2).status_code,[404,503])
    self.assertEqual(requests.post(url+'/v1/chat/completions',timeout=2).status_code,503)
   finally:server.shutdown();thread.join(2);server.server_close()
class LaunchTests(unittest.TestCase):
 def supervisor(self,tmp):
  root=Path(tmp);(root/'lease.json').write_text(json.dumps({'restore_at':time.time()+100}))
  __import__('shutil').copy(BASE/'kv-residency-status.py',root/'kv_status.py')
  script=(BASE/'kv-residency-supervisor.py').read_text().replace('ROOT=Path(__file__).resolve().parent','ROOT=Path('+repr(tmp)+')')
  m={'__name__':'supervisor_test'};exec(compile(script,'supervisor_test','exec'),m);return m
 def test_stock_cleanup_survives_diagnostic_write_failure(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.dict(os.environ,{'KV_EXPERIMENT_ROOT':tmp,'JUPYTER_PASSWORD':'test'}):
   m=self.supervisor(tmp);owned=[object(),object(),object()];m.update(controller=owned[0],engine=owned[1],children=[owned[2]])
   class ExecStock(Exception):pass
   with mock.patch.object(m['status'],'update'),mock.patch.dict(m,{'terminate':mock.Mock()}),mock.patch.object(m['Path'],'open',side_effect=OSError()),mock.patch.object(m['os'],'execv',side_effect=ExecStock):
    with self.assertRaises(ExecStock):m['stock']('test')
    self.assertEqual([c.args[0] for c in m['terminate'].call_args_list],owned)
 def test_safety_rejects_swap_oom_and_low_headroom(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.dict(os.environ,{'KV_EXPERIMENT_ROOT':tmp,'JUPYTER_PASSWORD':'test'}):
   m=self.supervisor(tmp);good={'current':1,'limit':100*1024**3,'swap':0,'shm_available':20*1024**3,'events':{'oom':0,'oom_kill':0}}
   for bad in [{**good,'swap':1},{**good,'current':90*1024**3},{**good,'shm_available':0},{**good,'events':{'oom':1}}]:
    with mock.patch.dict(m,{'memory':lambda:bad}):
     with self.assertRaises(AssertionError):m['safety'](good)
 def test_fixed_pool_flags_and_histogram_preserved(self):
  with tempfile.TemporaryDirectory() as tmp,mock.patch.dict(os.environ,{'KV_EXPERIMENT_ROOT':tmp,'JUPYTER_PASSWORD':'test'}):
   root=Path(tmp);(root/'lease.json').write_text(json.dumps({'restore_at':time.time()+100}))
   __import__('shutil').copy(BASE/'kv-residency-status.py',root/'kv_status.py')
   script=(BASE/'kv-residency-supervisor.py').read_text().replace('ROOT=Path(__file__).resolve().parent','ROOT=Path('+repr(tmp)+')')
   m={'__name__':'supervisor_test'};exec(compile(script,'supervisor_test','exec'),m)
   args=m['command'](['vllm','serve','fixture','--max-num-seqs','2','--cpu-offload-gb','0','--enable-prefix-caching','--enforce-eager','--custom-histogram-buckets','{"time_to_first_token":[1,2]}'])
   self.assertEqual(args[args.index('--kv-cache-memory-bytes')+1],'4294967296')
   hist=json.loads(args[args.index('--custom-histogram-buckets')+1]);self.assertEqual(hist['time_to_first_token'],[1,2]);self.assertEqual(len(hist['inter_token_latency']),205)
   config=json.loads(args[args.index('--kv-transfer-config')+1]);self.assertEqual(config['kv_connector_extra_config']['cpu_bytes_to_use'],8589934592)
class BootstrapTests(unittest.TestCase):
 def test_inline_materialization_and_fallback(self):
  m=load('bootstrap_test','kv-residency-bootstrap.py')
  with tempfile.TemporaryDirectory(prefix='kv-residency-') as tmp,mock.patch.dict(os.environ,{}):
   root=str(Path(tmp)/'owned')
   # The runtime path restriction is deliberate; use an owned /tmp fixture string
   # while substituting only the filesystem path in the executable test body.
   args=json.loads(m.build_override({'supervisor.py':b'# frozen','original-launch.private.json':b'{"public_exports":{}}'},'/tmp/kv-residency-fixture'))
   code=args['entrypoint'][2].replace("'/tmp/kv-residency-fixture'",repr(root))
   with mock.patch.object(os,'execv') as execute:exec(compile(code,'inline-bootstrap','exec'),{})
   self.assertEqual((Path(root)/'supervisor.py').read_bytes(),b'# frozen')
   self.assertEqual(execute.call_args.args[1][-1],str(Path(root)/'supervisor.py'))
   with mock.patch.object(os,'execv') as execute:exec(compile(code,'inline-bootstrap','exec'),{})
   self.assertEqual(execute.call_args.args[0],'/usr/local/bin/start-server')
 def test_rejects_traversal(self):
  m=load('bootstrap_reject','kv-residency-bootstrap.py')
  with self.assertRaises(AssertionError):m.build_override({'../escape':b'x'},'/tmp/kv-residency-fixture')
if __name__=='__main__':unittest.main()
