"""Regression coverage for the experiment's consequential safety/evidence gates."""
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        root=Path(cls.tmp.name)
        for name in ['kv-offload-experiment.py','benchmark-capacity.py']:
            shutil.copy(Path(__file__).with_name(name),root/name)
        (root/'lease.json').write_text(json.dumps({'created':0,'lease_id':'test','restore_at':9999999999}))
        (root/'runtime.json').write_text(json.dumps({'fingerprint':'test'}))
        with (root/'prompts.private.pickle').open('wb') as f:pickle.dump({'items':{'short':[([1,2],128),([3,4],128)]}},f)
        spec=importlib.util.spec_from_file_location('kv_experiment_test',root/'kv-offload-experiment.py')
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_histogram_delta_rejects_reset(self):
        with self.assertRaises(AssertionError):
            self.m.differences({'vllm:inter_token_latency_seconds_count':100},{'vllm:inter_token_latency_seconds_count':99})

    def test_gauge_can_decrease(self):
        self.assertEqual(self.m.differences({'vllm:kv_cache_usage_perc':.8},{'vllm:kv_cache_usage_perc':.3})['vllm:kv_cache_usage_perc'],-.5)

    def test_histogram_label_and_new_series(self):
        key='vllm:inter_token_latency_seconds_bucket{le="+Inf"}'
        self.assertEqual(self.m.differences({key:5},{key:8,'vllm:kv_offload_load_bytes_count':2})[key],3)

    def test_known_answer_strict_prefix(self):
        for text in ['7329',' 7329\n','7329.']:
            self.assertTrue(self.m.answer_pass({'choices':[{'text':text}]}))
        for text in ['17329','The code might be 7329','73299','wrong']:
            self.assertFalse(self.m.answer_pass({'choices':[{'text':text}]}))

    def test_prompt_offset_and_override(self):
        self.assertEqual(self.m.Prompts(1,1).get('short',0),([3,4],1))

    def test_common_300_second_deadline_and_fraction(self):
        self.m.b.Config(request_deadline=300).validate()
        with self.assertRaises(ValueError):self.m.b.Config(request_deadline=301).validate()
        rows=[{'outcome':'success','e2e_s':179,'completion_tokens':128},
              {'outcome':'success','e2e_s':181,'completion_tokens':128},
              {'outcome':'deadline','e2e_s':300}]
        r=self.m.b.aggregate(self.m.b.Config(),rows,300)
        self.assertEqual(r['successful_within_180s'],1)
        self.assertEqual(r['fraction_successful_within_180s'],1/3)

    def test_nonfinite_numeric_evidence_rejected(self):
        with self.assertRaises(ValueError):self.m.save('nonfinite.json',{'bad':float('nan')})

    def test_supervisor_reaches_engine_start_without_starting_services(self):
        root=Path(self.tmp.name)
        shutil.copy(Path(__file__).with_name('kv-offload-supervisor.py'),root/'supervisor.py')
        spec=importlib.util.spec_from_file_location('kv_supervisor_test',root/'supervisor.py')
        supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
        class ReachedEngineStart(Exception):pass
        with mock.patch.object(supervisor,'memory',return_value={}), \
             mock.patch.object(supervisor,'safety'), \
             mock.patch.object(supervisor.http.server,'ThreadingHTTPServer'), \
             mock.patch.object(supervisor.threading,'Thread'), \
             mock.patch.object(supervisor,'spawn'), \
             mock.patch.object(supervisor,'atomic'), \
             mock.patch.object(supervisor,'start',side_effect=ReachedEngineStart):
            with self.assertRaises(ReachedEngineStart):supervisor.main()

    def supervisor(self):
        root=Path(self.tmp.name)
        shutil.copy(Path(__file__).with_name('kv-offload-supervisor.py'),root/'supervisor.py')
        spec=importlib.util.spec_from_file_location('kv_supervisor_fixture',root/'supervisor.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.LEASE={'created':0,'restore_at':9000}
        return module

    def test_supervisor_a_b_a2_flags_and_reclamation(self):
        s=self.supervisor()
        argv=['vllm','serve','fixture','--max-num-seqs','2','--cpu-offload-gb','0','--enable-prefix-caching','--enforce-eager','--enable-lora']
        (s.ROOT/'original-process.private.json').write_text(json.dumps({'args':argv,'env':{}}))
        resource={'shm_available_bytes':100*1024**3}
        with mock.patch.object(s,'terminate') as terminate, mock.patch.object(s,'safety'), \
             mock.patch.object(s,'memory',return_value=resource), mock.patch.object(s,'event'), \
             mock.patch.object(s,'atomic'), mock.patch.object(s.time,'time',return_value=100), \
             mock.patch.object(s.subprocess,'check_output',return_value=''), \
             mock.patch.object(s,'spawn',return_value=mock.Mock(pid=123)) as spawn:
            s.start({'arm':'A','gpu_cache_bytes':None})
            a=spawn.call_args.args[0]
            s.start({'arm':'B','gpu_cache_bytes':1024})
            b=spawn.call_args.args[0]
            s.start({'arm':'A2','gpu_cache_bytes':1024})
            a2=spawn.call_args.args[0]
            for command in [a,b,a2]:
                self.assertEqual(command[command.index('--max-num-seqs')+1],'256')
                self.assertEqual(command[command.index('--max-num-batched-tokens')+1],'8192')
                self.assertEqual(command[command.index('--mamba-cache-mode')+1],'align')
            self.assertNotIn('--kv-transfer-config',a)
            self.assertNotIn('--kv-transfer-config',a2)
            config=json.loads(b[b.index('--kv-transfer-config')+1])
            self.assertEqual(config['kv_connector_extra_config']['cpu_bytes_to_use'],68719476736)
            self.assertFalse(config['kv_connector_extra_config']['offload_prompt_only'])
            self.assertEqual(terminate.call_count,3)

    def test_supervisor_refuses_unreleased_cpu_cache_and_expired_lease(self):
        s=self.supervisor();s.current={'arm':'B'};s.B_SHM_BEFORE=100*1024**3
        with mock.patch.object(s,'terminate'), mock.patch.object(s,'event'), \
             mock.patch.object(s.subprocess,'check_output',return_value=''), \
             mock.patch.object(s,'memory',return_value={'shm_available_bytes':40*1024**3}):
            with self.assertRaisesRegex(AssertionError,'not released'):s.start({'arm':'A2','gpu_cache_bytes':1024})
        s.current=None;s.LEASE['restore_at']=0
        with mock.patch.object(s,'terminate'), mock.patch.object(s.subprocess,'check_output',return_value=''):
            with self.assertRaisesRegex(RuntimeError,'Lease expired'):s.start({'arm':'A','gpu_cache_bytes':None})

    def test_stock_restoration_cleans_every_owned_group(self):
        s=self.supervisor();s.controller=object();s.engine=object();s.children=[object(),object()]
        class ExecStock(Exception):pass
        with mock.patch.object(s,'event'),mock.patch.object(s,'atomic'),mock.patch.object(s,'memory',return_value={}), \
             mock.patch.object(s,'terminate') as terminate,mock.patch.object(s.os,'execv',side_effect=ExecStock) as execute:
            with self.assertRaises(ExecStock):s.stock('test')
            self.assertEqual([call.args[0] for call in terminate.call_args_list],[s.controller,s.engine,*s.children])
            self.assertEqual(execute.call_args.args[0],'/usr/local/bin/start-server')

    def test_stock_recovery_survives_logging_and_state_write_failure(self):
        s=self.supervisor();s.controller=object();s.engine=object();s.children=[object()]
        class ExecStock(Exception):pass
        with mock.patch.object(s,'event',side_effect=OSError('disk full')), \
             mock.patch.object(s,'atomic',side_effect=OSError('disk full')), \
             mock.patch.object(s,'terminate') as terminate, \
             mock.patch.object(s.os,'execv',side_effect=ExecStock) as execute:
            with self.assertRaises(ExecStock):s.stock('write_failure')
            self.assertEqual(terminate.call_count,3)
            self.assertEqual(execute.call_args.args[0],'/usr/local/bin/start-server')


if __name__=='__main__':unittest.main()
