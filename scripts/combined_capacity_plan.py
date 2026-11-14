"""Freeze the 29 historical cases and exact open-loop arrivals, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

FOUNDATION = 'ae82a1c'
EXCLUDED = ['medium-sweep-2-retry1', 'medium-sweep-4-retry1']
ORDER = ['warm-short', 'warm-medium', 'warm-long', 'natural-stop-validation', 'medium-sweep-1']
for concurrency in [2, 4, 8, 16, 32, 64]:
    ORDER += [f'warm-shape-{concurrency}-retry2', f'medium-sweep-{concurrency}-retry2']
ORDER += ['warm-shape-128-retry2', 'mixed-boundary-retry2']
ORDER += [f'open-loop-{ratio}-retry2' for ratio in ['0.5', '0.8', '1', '1.2']]
ORDER += ['validate-medium-8-retry2', 'validate-short-8-retry2',
          'short-boundary-32-retry2', 'long-boundary-32-retry2',
          'short-boundary-64-retry2', 'long-boundary-64-retry2']
CONFIG_KEYS = ['workload', 'concurrency', 'duration', 'wall_seconds', 'request_deadline',
               'max_requests', 'arrival_rate', 'max_backlog', 'seed', 'force_length',
               'metrics_interval', 'collect_gpu', 'scenario_kind']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    root = Path(root)
    runtime = json.loads((root / 'runtime.json').read_text(encoding='utf-8'))
    assert runtime['prompt_pool_size_per_workload'] == 1024
    cases = []
    for index, name in enumerate(ORDER, 1):
        summary_path = root / (name + '.summary.json')
        request_path = root / (name + '.requests.jsonl')
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        config = {key: previous['config'][key] for key in CONFIG_KEYS}
        case_id = name.removesuffix('-retry2')
        if case_id in ('long-boundary-32', 'long-boundary-64'):
            config['max_requests'] = config['concurrency']
        case = {'sequence': index, 'case_id': case_id, 'historical_name': name,
                'category': config['scenario_kind'], 'config': config,
                'historical_elapsed_s': previous['elapsed_s'],
                'historical_outcomes': previous['outcomes'],
                'historical_summary_sha256': digest(summary_path),
                'historical_requests_sha256': digest(request_path),
                'arrival_schedule': []}
        case['historical_metrics'] = {key: previous.get(key) for key in
            ['outcomes', 'successful', 'success_rate', 'output_tokens_s', 'completed_requests_s',
             'total_output_tokens', 'outcome_counts', 'ttft_s', 'avg_tpot_s', 'e2e_s', 'stopped',
             'chat_slo_pass', 'chat_slo_provisional']}
        if config['arrival_rate']:
            rows = [json.loads(line) for line in request_path.read_text(encoding='utf-8').splitlines()]
            rows.sort(key=lambda row: row['id'])
            assert [row['id'] for row in rows] == list(range(len(rows)))
            case['arrival_schedule'] = [{key: row[key] for key in ('id', 'scheduled_s', 'workload')} for row in rows]
            assert all(0 <= row['scheduled_s'] < config['duration'] for row in case['arrival_schedule'])
        cases.append(case)
    assert len(cases) == 29
    result = {'schema_version': 1, 'foundation_commit': FOUNDATION,
              'historical_run': '2026-11-11-capacity', 'excluded_cold_retry_cases': EXCLUDED,
              'model': 'qwen-bf16-base', 'prompt_pool_size_per_workload': 1024,
              'template_suffix_sha256': runtime['template_suffix_sha256'],
              'model_revision': runtime['model_revision'],
              'model_config_sha256': runtime['model_config_sha256'],
              'historical_case_elapsed_sum_s': sum(case['historical_elapsed_s'] for case in cases),
              'test_profile': {'max_num_seqs': 256, 'max_num_batched_tokens': 8192,
                   'context_tokens': 40960, 'cpu_weight_offload_gib': 0,
                   'gpu_memory_utilization': .9, 'cpu_kv_budget_bytes': 68719476736,
                   'compilation_mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                   'capture_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256]},
              'cases': cases}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('historical_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plan = build(args.historical_root)
    args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(plan['cases']), 'historical_minutes': plan['historical_case_elapsed_sum_s'] / 60,
                      'manifest_sha256': digest(args.output)}))
