"""Compare matched B cases to frozen historical client metrics, never infer ITL."""
import argparse
import json
from pathlib import Path


def compare(plan, results):
    rows = []
    for case in plan['cases']:
        path = Path(results) / (case['case_id'] + '.summary.json')
        baseline = case['historical_metrics']
        row = {'case_id': case['case_id'], 'category': case['category'],
               'historical': baseline, 'current': None,
               'historical_server_itl_comparison': 'unavailable_no_comparable_histograms'}
        if path.exists():
            current = json.loads(path.read_text(encoding='utf-8'))
            row['current'] = {key: current.get(key) for key in baseline}
            row['current']['native_metric_qualification'] = current.get('native_metric_qualification')
            row['relative_changes'] = {}
            for field in ['output_tokens_s', 'completed_requests_s']:
                old, new = baseline.get(field), current.get(field)
                row['relative_changes'][field] = new / old - 1 if old and new is not None else None
            for field in ['ttft_s', 'avg_tpot_s', 'e2e_s']:
                old, new = (baseline.get(field) or {}).get('p95'), (current.get(field) or {}).get('p95')
                row['relative_changes'][field + '_p95'] = new / old - 1 if old and new is not None else None
            row['comparison_scope'] = ('warmup_operational_only' if case['category'] == 'warmup' else
                  'censored_or_capacity_boundary' if current.get('stopped') or not current.get('all_outcomes_accounted') else
                  'same_fixed_arrivals' if case['arrival_schedule'] else
                  'same_observed_single_wave' if case['case_id'].startswith('long-boundary') else 'same_closed_loop_controls')
            row['capacity_slo_qualified'] = bool(current.get('chat_slo_pass'))
        rows.append(row)
    capacities = {}
    for workload in ['short', 'medium']:
        capacities[workload] = [case['config']['concurrency'] for case, row in zip(plan['cases'], rows)
                 if case['config']['workload'] == workload and row.get('capacity_slo_qualified')]
    intersection = set(capacities['short']) & set(capacities['medium'])
    return {'schema_version': 1, 'cases': rows, 'historical_arm_rerun': False,
            'base_only_engine_scope': True, 'public_gateway_capacity_measured': False,
            'client_tpot_definition': 'Legacy approximate TPOT from first/last content chunk and validated usage tokens; compare only to the same historical client metric',
            'current_server_itl_definition': 'Native histogram deltas qualified by successful uncensored decode-interval count; otherwise event/partial labels',
            'chat_capacity_slo': {'minimum_outcomes': 100, 'minimum_success_rate': .99,
                 'ttft_p95_seconds': 5, 'legacy_approx_tpot_p95_seconds': .1},
            'validated_chat_concurrency': max(intersection) if intersection else None,
            'gpu_utilization_interpretation': 'Resource context only; neither CPU KV reload nor throughput causality is inferred from utilization'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.plan.read_text(encoding='utf-8')), args.results)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'cases_present': sum(row['current'] is not None for row in report['cases']),
                      'validated_chat_concurrency': report['validated_chat_concurrency']}))
