"""Offline contracts: exact history, safe transformation, accounting and feed."""
import asyncio
import ast
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import zlib

import combined_capacity_runtime as runtime
import combined_capacity_bundle as bundle
import combined_capacity_measure as measure

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('combined_core', ROOT / 'scripts/benchmark-capacity.py')
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def native():
    return ['python3', '/usr/local/bin/vllm', 'serve', '/workspace/qwen-bf16/model',
            '--served-model-name', 'qwen-bf16-base', '--max-num-seqs', '2',
            '--max-model-len', '40960', '--gpu-memory-utilization', '0.90',
            '--dtype', 'bfloat16', '--cpu-offload-gb', '0', '--host', '127.0.0.1',
            '--enable-prefix-caching', '--mamba-cache-mode', 'align', '--cudagraph-metrics',
            '--enable-lora', '--max-lora-rank', '8', '--lora-modules', '{"name":"qwen-bf16","path":"/adapter"}',
            '--limit-mm-per-prompt', '{"image":1,"video":0}',
            '--compilation-config', '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}',
            '--kv-transfer-config', json.dumps({'kv_connector': 'OffloadingConnector', 'kv_role': 'kv_both',
                'kv_connector_extra_config': {'cpu_bytes_to_use': 68719476736, 'blocks_per_chunk': 1, 'offload_prompt_only': False}}),
            '--custom-histogram-buckets', '{"time_to_first_token":[0.001,0.1,1,5,10,2560]}',
            '--api-key', 'test-only-secret']


class RuntimeTests(unittest.TestCase):
    def test_only_approved_delta_and_no_secret_in_contract(self):
        original = native()
        env = {'VLLM_API_KEY': 'test-only-secret'}
        frozen = runtime.contract(original, env)
        self.assertNotIn('test-only-secret', json.dumps(frozen))
        args = runtime.test_argv(original, frozen, env)
        self.assertEqual(runtime.value(args, '--max-num-seqs'), '256')
        self.assertEqual(runtime.value(args, '--max-num-batched-tokens'), '8192')
        self.assertEqual(json.loads(runtime.value(args, '--compilation-config'))['cudagraph_capture_sizes'], runtime.CAPTURE_SIZES)
        self.assertEqual(runtime.value(args, '--lora-modules'), runtime.value(original, '--lora-modules'))
        self.assertEqual(runtime.value(args, '--api-key'), 'test-only-secret')
        buckets = json.loads(runtime.value(args, '--custom-histogram-buckets'))
        self.assertEqual(buckets['time_to_first_token'], [.001,.1,1,5,10,2560])
        self.assertTrue(set(runtime.BASE_ITL_BUCKETS).issubset(buckets['inter_token_latency']))
        self.assertIn(.095, buckets['inter_token_latency'])

    def test_drift_or_duplicates_fail_closed(self):
        env = {'VLLM_API_KEY': 'test-only-secret'}
        baseline = native()
        frozen = runtime.contract(baseline, env)
        for flag, replacement in [('--max-model-len', '8192'), ('--served-model-name', 'qwen-bf16'),
                                  ('--max-lora-rank', '16'), ('--limit-mm-per-prompt', '{"image":0}')]:
            changed = list(baseline)
            changed[changed.index(flag) + 1] = replacement
            with self.assertRaises((ValueError, AssertionError)):
                runtime.test_argv(changed, frozen, env)
        with self.assertRaises(ValueError):
            runtime.test_argv(baseline + ['--max-num-seqs', '2'], frozen, env)
        with self.assertRaises(ValueError):
            runtime.contract(baseline, {'VLLM_API_KEY': 'wrong'})

    def test_lease_has_fixed_absolute_budget(self):
        lease = {'created': 100, 'testing_stop_at': 6700, 'restore_at': 7300,
                 'exclusive_end': 9100, 'startup_cap_seconds': 2100, 'run_id': 'run', 'lease_id': 'fresh'}
        runtime.validate_lease(lease, 101)
        with self.assertRaises(AssertionError):
            runtime.validate_lease({**lease, 'restore_at': 7400}, 101)
        with self.assertRaises(AssertionError):
            runtime.validate_lease(lease, 6700)

    def test_bundle_is_transitive_lf_and_lease_materialized_last(self):
        source = bundle.files(ROOT)
        self.assertIn('combined_capacity_measure.py', source)
        self.assertNotIn(b'\r', source['bin/vllm'])
        frozen = runtime.contract(native(), {'VLLM_API_KEY': 'test-only-secret'})
        command = bundle.build_args('/tmp/offline-run', source, {'run_id': 'offline-run', 'lease_id': 'fresh'}, frozen)
        self.assertNotIn('test-only-secret', command)
        outer = json.loads(command)
        self.assertEqual(outer['entrypoint'][:2], ['python3', '-c'])
        self.assertEqual(outer['cmd'], [])
        self.assertLess(len(command.encode()), 65536)
        program = outer['entrypoint'][2]
        encoded = ast.literal_eval(program.splitlines()[0].split('=', 1)[1])
        payload = json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(payload['root'], '/tmp/offline-run')
        self.assertEqual(json.loads(payload['files']['lease.json'])['lease_id'], 'fresh')
        hashes = json.loads(payload['files']['source-files.json'])
        for name, digest in hashes.items():
            self.assertEqual(hashlib.sha256(payload['files'][name].encode('utf-8')).hexdigest(), digest)
        self.assertNotIn('lease.json', hashes)
        self.assertNotIn('test-only-secret', json.dumps(payload))
        # Execute the exact structured-ENTRYPOINT program under mocks, then
        # compare every materialized byte. No real supervisor is launched.
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / 'owned-root'
            fake_pathlib = SimpleNamespace(Path=lambda value: staged)
            with patch.dict(sys.modules, {'pathlib': fake_pathlib}), patch.object(os, 'execv') as execute, \
                 patch.object(os, 'O_NOFOLLOW', getattr(os, 'O_NOFOLLOW', 0), create=True), patch.dict(os.environ):
                exec(compile(program, 'exact-bootstrap', 'exec'), {'__name__': 'bootstrap_offline'})
                execute.assert_called_once_with(sys.executable, [sys.executable, str(staged / 'combined-capacity-supervisor.py')])
            for name, text in payload['files'].items():
                self.assertEqual((staged / name).read_bytes(), text.encode('utf-8'))
            for name, expected in source.items():
                self.assertEqual((staged / name).read_bytes(), expected)
        with self.assertRaises(ValueError):
            bundle.build_args('/workspace/not-allowed', source, {'run_id': 'not-allowed'}, frozen)

    def test_compact_transport_preserves_nonascii_utf8_and_rejects_binary(self):
        source = bundle.files(ROOT)
        source['unicode-test.txt'] = 'archive 港湾\n'.encode('utf-8')
        frozen = runtime.contract(native(), {'VLLM_API_KEY': 'test-only-secret'})
        command = bundle.build_args('/tmp/offline-run', source, {'run_id': 'offline-run'}, frozen)
        program = json.loads(command)['entrypoint'][2]
        encoded = ast.literal_eval(program.splitlines()[0].split('=', 1)[1])
        payload = json.loads(zlib.decompress(base64.b64decode(encoded)))
        self.assertEqual(payload['files']['unicode-test.txt'].encode('utf-8'), source['unicode-test.txt'])
        with self.assertRaises(UnicodeDecodeError):
            bundle.build_args('/tmp/offline-run', {**source, 'bad.bin': b'\xff'}, {'run_id': 'offline-run'}, frozen)


class PlanTests(unittest.TestCase):
    def test_canonical_cases_arrivals_and_sample_contract(self):
        plan = json.loads((ROOT / 'docs/COMBINED-CAPACITY-PLAN.json').read_text())
        self.assertEqual(len(plan['cases']), 29)
        self.assertEqual(plan['prompt_pool_size_per_workload'], 1024)
        self.assertAlmostEqual(plan['historical_case_elapsed_sum_s'] / 60, 48.56056934713852)
        self.assertEqual([case['case_id'] for case in plan['cases'][:3]], ['warm-short', 'warm-medium', 'warm-long'])
        open_cases = [case for case in plan['cases'] if case['config']['arrival_rate']]
        self.assertEqual([len(case['arrival_schedule']) for case in open_cases], [80,121,147,181])
        for case in open_cases:
            self.assertEqual([row['id'] for row in case['arrival_schedule']], list(range(len(case['arrival_schedule']))))
            self.assertEqual(case['arrival_schedule'], sorted(case['arrival_schedule'], key=lambda row: row['scheduled_s']))
        for case in plan['cases']:
            self.assertEqual(case['config']['request_deadline'], 180)
            if case['case_id'].startswith('long-boundary'):
                self.assertEqual(case['config']['max_requests'], case['config']['concurrency'])
            if case['case_id'].startswith('short-boundary'):
                self.assertEqual(case['config']['duration'], 75)


class MeasureTests(unittest.TestCase):
    def test_counts_and_censored_intervals_do_not_claim_token_itl(self):
        summary = {'stopped': None, 'all_outcomes_accounted': True, 'successful': 1, 'outcomes': 1}
        records = [{'outcome': 'success', 'completion_tokens': 4}]
        counters = {'vllm:inter_token_latency_seconds_count': 3}
        self.assertEqual(measure.qualify(summary, records, counters)['itl_interpretation'], 'per_token_intervals')
        self.assertEqual(measure.qualify({**summary, 'stopped': 'stage_wall_deadline'}, records, counters)['itl_interpretation'], 'event_intervals_or_partial')
        self.assertEqual(measure.qualify(summary, records, {'vllm:inter_token_latency_seconds_count': 2})['itl_interpretation'], 'event_intervals_or_partial')

    def test_counter_resets_and_missing_series_are_rejected(self):
        with self.assertRaises(ValueError): measure.delta({'vllm:x_total': 2}, {'vllm:x_total': 1})
        with self.assertRaises(ValueError): measure.delta({'vllm:x_total': 2}, {})
        self.assertEqual(measure.delta({}, {'vllm:x_total': 3})['vllm:x_total'], 3)

    def test_drained_cpu_metric_is_active_pins_not_retained_occupancy(self):
        names = ['num_requests_running','num_requests_waiting','kv_offload_cpu_cache_usage_perc',
                 'kv_offload_cpu_cache_write_usage_perc','kv_offload_cpu_cache_read_usage_perc']
        values = {'vllm:' + name: 0 for name in names}
        self.assertTrue(measure.drained(values))
        self.assertFalse(measure.drained({**values, 'vllm:kv_offload_cpu_cache_usage_perc': .1}))
        self.assertFalse(measure.drained({key: value for key, value in values.items() if 'offload' not in key}))

    def test_immutable_completion_has_actual_window_and_finalized_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results').mkdir()
            (root / 'results/case.summary.json').write_text('{}')
            feed = measure.CompletionFeed(root, {'run_id': 'run', 'lease_id': 'fresh'}, 'manifest', 'source')
            case = {'case_id': 'case', 'sequence': 1, 'category': 'capacity', 'cold_warm_status': 'warm'}
            event = feed.emit(case, 1000, 1002.5, 'complete', {'outcomes': 2, 'successful': 2}, ['results/case.summary.json'])
            self.assertEqual(event['capture_from_ms'], 970000)
            self.assertEqual(event['capture_to_ms'], 1032500)
            self.assertEqual(event['end_epoch_ms'], 1002500)
            index = json.loads((root / 'completion-index.json').read_text())
            item = index['entries'][0]
            self.assertEqual(item['sha256'], hashlib.sha256((root / item['path']).read_bytes()).hexdigest())
            with self.assertRaises(ValueError): feed.emit(case, 1003, 1004, 'complete', {}, [])
            with self.assertRaises(ValueError): measure.CompletionFeed(root, {'run_id': 'run', 'lease_id': 'fresh'}, 'm', 's')


class FrozenArrivalTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_offsets_workloads_and_all_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = core.Config(name='replay', duration=.08, wall_seconds=1, concurrency=2,
                              arrival_rate=999, collect_gpu=False, max_requests=10)
            schedule = [{'id': 0, 'scheduled_s': 0, 'workload': 'long'},
                        {'id': 1, 'scheduled_s': .02, 'workload': 'short'},
                        {'id': 2, 'scheduled_s': .04, 'workload': 'medium'}]
            runner = core.Runner(cfg, None, core.Writer(directory, cfg), arrival_schedule=schedule)
            async def request(session, row):
                row.update(outcome='success', completion_tokens=4, prompt_tokens=3, ended_s=time.monotonic()-runner.started)
                runner.record(row)
            async def metrics(session): await runner.stop.wait()
            runner.request, runner.metrics = request, metrics
            result = await runner.run()
            self.assertEqual([row['scheduled_s'] for row in runner.records], [0,.02,.04])
            self.assertEqual([row['workload'] for row in runner.records], ['long','short','medium'])
            self.assertEqual(result['outcomes'], 3)
            self.assertTrue(result['arrival_replay_complete'])

    async def test_negative_or_unordered_replay_is_rejected(self):
        cfg = core.Config(arrival_rate=1, duration=10)
        for schedule in [[{'id':0,'scheduled_s':-.1,'workload':'short'}],
                         [{'id':1,'scheduled_s':0,'workload':'short'}]]:
            with self.assertRaises(ValueError): core.Runner(cfg, None, None, arrival_schedule=schedule)

    async def test_controller_propagates_swallowed_cancel_once_and_cannot_start_next_case(self):
        source = (ROOT / 'scripts/combined-capacity-run.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        functions = [node for node in tree.body if getattr(node, 'name', None) in ('run_case', 'finalized_artifacts')]
        with tempfile.TemporaryDirectory() as directory:
            runroot = Path(directory)
            results = runroot / 'results'
            results.mkdir()
            summary = {'stopped': 'controller_cancelled', 'all_outcomes_accounted': True, 'outcomes': 1, 'successful': 0}
            (results / 'cancel.summary.json').write_text(json.dumps(summary))
            class FakeRunner:
                records = [{'outcome': 'stage_cancel'}]
                async def run(self): return summary
            feed = measure.CompletionFeed(runroot, {'run_id':'run','lease_id':'lease'}, 'manifest', 'source')
            config = SimpleNamespace(validate=lambda:None, request_deadline=180)
            scope = {'asyncio':asyncio,'json':json,'os':os,'time':time,'ROOT':runroot,'RESULTS':results,
                     'MANIFEST_SHA':'manifest','LEASE':{'lease_id':'lease'}, 'HEADERS':{},
                     'b':SimpleNamespace(Config=lambda **kw:config, Runner=lambda *a,**kw:FakeRunner(), Writer=lambda *a:None),
                     'settled':AsyncMock(return_value=('',{})), 'snapshot':AsyncMock(return_value=('',{})),
                     'atomic':runtime.atomic,'progress':lambda *a,**kw:None}
            exec(compile(ast.Module(body=functions,type_ignores=[]),'controller-functions','exec'),scope)
            case = {'case_id':'cancel','sequence':4,'category':'capacity','config':{},'arrival_schedule':[], 'historical_name':'cancel-old'}
            next_case_started = False
            with self.assertRaises(asyncio.CancelledError):
                await scope['run_case'](None,feed,None,{'fingerprint':'runtime'},case)
                next_case_started = True
            self.assertFalse(next_case_started)
            self.assertEqual(len(feed.entries),1)
            event = json.loads((runroot / feed.entries[0]['path']).read_text())
            self.assertEqual(event['status'],'cancelled')
            self.assertFalse((results/'cancel.failure.json').exists())
            self.assertEqual(set(event['artifacts']),{str(path.relative_to(runroot)) for path in results.iterdir()})


class ObserverTests(unittest.TestCase):
    def test_stop_flush_positively_restores_original_method(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            class Graph:
                def replay(self): calls.append('native')
            original = Graph.replay
            class Worker:
                def __init__(self,*a,**kw): pass
            native = SimpleNamespace(Worker=Worker)
            torch = SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph))
            with patch.dict(sys.modules, {'vllm.v1.worker.gpu_worker':native,'torch':torch}), patch.object(runtime,'secure_root',return_value=root), patch.dict(os.environ,{'CAPACITY_SESSION_ROOT':str(root)}):
                spec = importlib.util.spec_from_file_location('observer_offline',ROOT/'scripts/capacity_graph_observer.py')
                observer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(observer)
                observed = observer.worker_class()()
                Graph().replay()
                self.assertEqual(json.loads((root/'observer-replays.json').read_text())['completed_native_replays'],1)
                (root/'observer-stop').touch()
                Graph().replay()
                self.assertIs(Graph.replay,original)
                self.assertTrue(json.loads((root/'observer-inactive.json').read_text())['native_replay_restored'])
                Graph().replay()
                self.assertEqual(len(calls),3)
                self.assertEqual(json.loads((root/'observer-replays.json').read_text())['completed_native_replays'],1)


if __name__ == '__main__':
    unittest.main()
