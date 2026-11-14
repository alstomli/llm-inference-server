"""B-only fixed historical suite. No adaptive rates, scenarios or scheduler changes."""
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

import aiohttp
from combined_capacity_measure import CompletionFeed, delta, drained, parse_metrics, qualify, signature
from combined_capacity_runtime import atomic, secure_root

ROOT = secure_root(os.environ['CAPACITY_SESSION_ROOT'])
RESULTS = ROOT / 'results'
RESULTS.mkdir(mode=0o700, exist_ok=True)
LEASE = json.loads((ROOT / 'lease.json').read_text())
PLAN = json.loads((ROOT / 'canonical-plan.json').read_text())
MANIFEST_SHA = hashlib.sha256((ROOT / 'canonical-plan.json').read_bytes()).hexdigest()
SOURCE_SHA = hashlib.sha256((ROOT / 'source-files.json').read_bytes()).hexdigest()
HEADERS = {'Authorization': 'Bearer ' + os.environ['VLLM_API_KEY']} if os.getenv('VLLM_API_KEY') else {}
spec = importlib.util.spec_from_file_location('capacity', ROOT / 'benchmark-capacity.py')
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def progress(phase, case=None, **extra):
    atomic(ROOT / 'progress.json', {'run_id': LEASE['run_id'], 'lease_id': LEASE['lease_id'],
           'time': time.time(), 'phase': phase, 'case_id': case['case_id'] if case else None,
           'canonical_sequence': case.get('sequence') if case else None, **extra})


async def snapshot(session):
    async with session.get('http://127.0.0.1:8000/metrics', headers=HEADERS,
                           timeout=aiohttp.ClientTimeout(total=5)) as response:
        if response.status != 200:
            raise RuntimeError('Native metrics unavailable')
        text = await response.text()
    return text, parse_metrics(text)


async def settled(session, allow_missing_cpu=False):
    deadline = min(time.time() + 60, LEASE['testing_stop_at'])
    previous, since = None, time.monotonic()
    while time.time() < deadline:
        raw, values = await snapshot(session)
        current = signature(values)
        if not drained(values, allow_missing_cpu) or current != previous:
            previous, since = current, time.monotonic()
        elif time.monotonic() - since >= 11:
            return raw, values
        await asyncio.sleep(.5)
    raise RuntimeError('Drained published metric interval unavailable')


def finalized_artifacts(case_id):
    return [str(path.relative_to(ROOT)) for path in sorted(RESULTS.glob(case_id + '.*')) if path.is_file()]


async def run_case(session, feed, prompts, runtime, case):
    case = dict(case)
    case_id = case['case_id']
    case['cold_warm_status'] = ('cold_start_warmup_with_observer' if case_id == 'warm-short' else
                              'workload_or_shape_warmup' if case['category'] == 'warmup' else 'warm_after_canonical_warmups')
    allow_missing = case['category'] == 'warmup'
    before_raw, before = await settled(session, allow_missing)
    observe = case['sequence'] <= 3
    replay_file = ROOT / 'observer-replays.json'
    replay_before = json.loads(replay_file.read_text())['completed_native_replays'] if replay_file.exists() else 0
    (RESULTS / (case_id + '.before.prometheus.txt')).write_text(before_raw, encoding='utf-8')
    config = b.Config(name=case_id, model='qwen-bf16-base', **case['config'],
                      max_num_seqs=256, max_num_batched_tokens=8192,
                      runtime_fingerprint=runtime['fingerprint'], cache_namespace=LEASE['lease_id'])
    config.validate()
    assert config.request_deadline == 180
    runner = b.Runner(config, prompts, b.Writer(RESULTS, config), api_key=os.getenv('VLLM_API_KEY'),
                      arrival_schedule=case['arrival_schedule'] or None)
    started = time.time()
    progress('case_running', case)
    emitted = False
    try:
        summary = await runner.run()
        ended = time.time()
        assert summary['all_outcomes_accounted'], 'Missing request outcomes'
        if summary.get('stopped') == 'controller_cancelled':
            # The core deliberately finalizes partial rows instead of raising.
            # Preserve that evidence, then propagate cancellation to the suite.
            try:
                raw, _ = await snapshot(session)
                (RESULTS / (case_id + '.cancelled.prometheus.txt')).write_text(raw, encoding='utf-8')
            except Exception:
                pass
            atomic(RESULTS / (case_id + '.cancelled.json'), {'time': ended,
                   'all_outcomes_accounted': True, 'metric_publication_unsettled': True})
            feed.emit(case, started, ended, 'cancelled', summary, finalized_artifacts(case_id))
            emitted = True
            raise asyncio.CancelledError
        after_raw, after = await settled(session, allow_missing)
        replay_after = json.loads(replay_file.read_text())['completed_native_replays'] if replay_file.exists() else 0
        if observe:
            assert replay_after > replay_before, 'No request-time native replay in canonical warmup'
        (RESULTS / (case_id + '.after.prometheus.txt')).write_text(after_raw, encoding='utf-8')
        counters = delta(before, after)
        measurement = qualify(summary, runner.records, counters)
        atomic(RESULTS / (case_id + '.counter-delta.json'), {'counters': counters, 'qualification': measurement})
        summary['native_metric_qualification'] = measurement
        summary['canonical_manifest_sha256'] = MANIFEST_SHA
        summary['historical_name'] = case['historical_name']
        if observe:
            summary['warmup_native_replay_observation'] = {'before': replay_before, 'after': replay_after,
                 'request_time_delta': replay_after - replay_before}
        summary['measurement_state'] = 'warmup_instrumented' if case['category'] == 'warmup' and case['sequence'] <= 3 else 'warmup' if case['category'] == 'warmup' else 'steady'
        if case_id in ('validate-short-8', 'validate-medium-8', 'short-boundary-32', 'short-boundary-64'):
            summary['required_minimum_outcomes'] = 100
            summary['minimum_sample_met'] = summary['outcomes'] >= 100
        atomic(RESULTS / (case_id + '.summary.json'), summary)
        status = 'complete' if summary['stopped'] is None else 'capacity_boundary_or_censored'
        feed.emit(case, started, ended, status, summary, finalized_artifacts(case_id))
        emitted = True
        progress('case_complete', case, outcomes=summary['outcomes'], status=status)
        return summary
    except BaseException as error:
        ended = time.time()
        if not emitted:
            atomic(RESULTS / (case_id + '.failure.json'), {'error_type': type(error).__name__, 'time': ended})
            feed.emit(case, started, ended, 'cancelled' if isinstance(error, asyncio.CancelledError) else 'failed',
                      {'outcomes': len(runner.records), 'successful': sum(row['outcome'] == 'success' for row in runner.records)},
                      finalized_artifacts(case_id))
        raise


async def disable_observer(session, feed, tokenizer):
    case = {'case_id': 'graph-observer-disable-flush', 'category': 'qualification',
            'cold_warm_status': 'excluded_warm_correctness_flush'}
    before_raw, before = await settled(session)
    started = time.time()
    progress('case_running', case)
    (ROOT / 'observer-stop').touch(mode=0o600, exist_ok=False)
    messages = [{'role': 'user', 'content': 'What is 2 + 2? Reply with only the integer.'}]
    payload = {'model': 'qwen-bf16-base', 'messages': messages, 'temperature': 0,
               'max_tokens': 32, 'min_tokens': 32, 'ignore_eos': True,
               'chat_template_kwargs': {'enable_thinking': False}}
    try:
        async with session.post('http://127.0.0.1:8000/v1/chat/completions', headers=HEADERS,
                                json=payload, timeout=aiohttp.ClientTimeout(total=180)) as response:
            assert response.status == 200
            data = await response.json()
        assert re.match(r'^\s*4(?:\s|[.!<]|$)', data['choices'][0]['message']['content']), 'Base correctness failed'
        ended = time.time()
        inactive = json.loads((ROOT / 'observer-inactive.json').read_text())
        assert inactive['native_replay_restored'] and inactive['completed_warmup_replays'] > 0
        observations = [json.loads((RESULTS / (name + '.summary.json')).read_text())['warmup_native_replay_observation']
                        for name in ['warm-short', 'warm-medium', 'warm-long']]
        assert all(row['request_time_delta'] > 0 for row in observations)
        after_raw, after = await settled(session)
        name = case['case_id']
        (RESULTS / (name + '.before.prometheus.txt')).write_text(before_raw, encoding='utf-8')
        (RESULTS / (name + '.after.prometheus.txt')).write_text(after_raw, encoding='utf-8')
        atomic(RESULTS / (name + '.control.json'), {'scored_base_correctness': True,
               'observer_inactive': inactive, 'warmup_request_time_replay_observations': observations,
               'warmup_request_time_replay_delta': sum(row['request_time_delta'] for row in observations),
               'counters': delta(before, after), 'excluded_from_capacity': True})
        feed.emit(case, started, ended, 'complete', {'outcomes': 1, 'successful': 1}, finalized_artifacts(name))
    except BaseException as error:
        ended = time.time()
        atomic(RESULTS / (case['case_id'] + '.failure.json'), {'error_type': type(error).__name__, 'time': ended})
        feed.emit(case, started, ended, 'failed', {'outcomes': 1, 'successful': 0}, finalized_artifacts(case['case_id']))
        raise


async def main():
    from transformers import AutoTokenizer
    assert PLAN['model'] == 'qwen-bf16-base' and len(PLAN['cases']) == 29
    contract = json.loads((ROOT / 'args-contract.json').read_text())
    model_path = contract['native_argv'][1]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    prompts = b.Prompts(tokenizer, 12345, size=1024)
    assert prompts.template_suffix_sha256 == PLAN['template_suffix_sha256']
    runtime = {'image': contract['image'], 'manifest_sha256': MANIFEST_SHA,
               'source_sha256': SOURCE_SHA, 'prompt_pool_size_per_workload': 1024,
               'template_suffix_sha256': prompts.template_suffix_sha256,
               'generator_source_sha256': hashlib.sha256(inspect.getsource(b.Prompts).encode()).hexdigest(),
               'model_config_sha256': hashlib.sha256((Path(model_path) / 'config.json').read_bytes()).hexdigest(),
               'launch': json.loads((ROOT / 'launch.json').read_text()),
               'allocation': json.loads((ROOT / 'allocation.json').read_text())}
    assert runtime['model_config_sha256'] == PLAN['model_config_sha256']
    corpus = {}
    for workload, items in prompts.items.items():
        digest = hashlib.sha256()
        for ids, output_tokens in items:
            digest.update(json.dumps([ids, output_tokens], separators=(',', ':')).encode())
        corpus[workload] = digest.hexdigest()
    runtime['corpus_sha256_by_workload'] = corpus
    runtime['fingerprint'] = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()[:24]
    atomic(ROOT / 'runtime.json', runtime)
    feed = CompletionFeed(ROOT, LEASE, MANIFEST_SHA, SOURCE_SHA)
    summaries = []
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    def request_stop():
        if not main_task.cancelling():
            main_task.cancel()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(request_stop))
    async def absolute_stop():
        await asyncio.sleep(max(0, LEASE['testing_stop_at'] - time.time()))
        progress('testing_deadline_reached')
        request_stop()
    deadline_task = asyncio.create_task(absolute_stop())
    progress('prepared', completed_canonical_cases=0)
    failed = False
    try:
      async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), trust_env=False) as session:
        for case in PLAN['cases']:
            # Do not shorten a case or its historical request deadline to fit.
            if time.time() + case['config']['wall_seconds'] + 75 >= LEASE['testing_stop_at']:
                progress('budget_stop', case, completed_canonical_cases=len(summaries))
                break
            if case['sequence'] == 4:
                await disable_observer(session, feed, tokenizer)
                assert json.loads((ROOT / 'observer-inactive.json').read_text())['native_replay_restored']
            summary = await run_case(session, feed, prompts, runtime, case)
            summaries.append(summary)
    except BaseException:
        failed = True
        raise
    finally:
        deadline_task.cancel()
        await asyncio.gather(deadline_task, return_exceptions=True)
        result = {'run_id': LEASE['run_id'], 'lease_id': LEASE['lease_id'],
                  'completed_canonical_cases': len(summaries), 'expected_canonical_cases': 29,
                  'completion_events': len(feed.entries), 'all_cases_complete': len(summaries) == 29,
                  'finished_at': time.time(), 'manifest_sha256': MANIFEST_SHA, 'source_sha256': SOURCE_SHA,
                  'failed_or_cancelled': failed}
        atomic(ROOT / 'run-result.json', result)
        progress('failed' if failed else 'complete', completed_canonical_cases=len(summaries), all_cases_complete=len(summaries) == 29)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except BaseException as error:
        progress('failed', error_type=type(error).__name__)
        raise SystemExit(1)
