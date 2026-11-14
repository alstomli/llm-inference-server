"""Numerical metric qualification and immutable case completion feed."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from combined_capacity_runtime import atomic


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        if not line.startswith('vllm:'):
            continue
        key, value = line.rsplit(' ', 1)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite native metric')
        result[key] = value
    return result


def total(values, name, source=None):
    return sum(value for key, value in values.items() if key.split('{', 1)[0] == 'vllm:' + name
               and (source is None or f'source="{source}"' in key))


def cumulative(key):
    return key.split('{', 1)[0].endswith(('_total', '_count', '_sum', '_bucket'))


def delta(before, after):
    if any(key not in after for key in before if cumulative(key)):
        raise ValueError('Cumulative metric series disappeared')
    result = {key: value - before.get(key, 0) for key, value in after.items() if not key.split('{', 1)[0].endswith('_created')}
    if any(value < 0 for key, value in result.items() if cumulative(key)):
        raise ValueError('Native metric counter reset')
    return result


def transfer(values, direction):
    return total(values, f'kv_offload_{direction}_bytes_total') + total(values, f'kv_offload_{direction}_bytes_sum')


def signature(values):
    return tuple(sorted((key, value) for key, value in values.items() if cumulative(key)))


def drained(values, allow_missing_cpu=False):
    for name in ['num_requests_running', 'num_requests_waiting', 'kv_offload_cpu_cache_usage_perc',
                 'kv_offload_cpu_cache_write_usage_perc', 'kv_offload_cpu_cache_read_usage_perc']:
        present = any(key.split('{', 1)[0] == 'vllm:' + name for key in values)
        if not present and (not allow_missing_cpu or not name.startswith('kv_offload_')):
            return False
        if total(values, name) != 0:
            return False
    return True


def qualify(summary, records, counters):
    expected = sum(max(0, row.get('completion_tokens', 0) - 1) for row in records if row['outcome'] == 'success')
    observed = total(counters, 'inter_token_latency_seconds_count')
    clean = (summary['stopped'] is None and summary['all_outcomes_accounted'] and
             summary['successful'] == summary['outcomes'] and summary.get('arrival_replay_complete', True))
    itl = 'per_token_intervals' if clean and observed == expected else 'event_intervals_or_partial'
    preemptions = total(counters, 'num_preemptions_total')
    host = total(counters, 'prompt_tokens_cached_by_source_total', 'host')
    loads = transfer(counters, 'load')
    return {'itl_interpretation': itl, 'itl_count': observed, 'expected_successful_decode_intervals': expected,
            'all_successful_uncensored': clean, 'preemptions': preemptions, 'host_hit_tokens': host,
            'cpu_load_bytes': loads, 'cpu_store_bytes': transfer(counters, 'store'),
            'unique_salt_preemption_reload_counters': preemptions > 0 and host > 0 and loads > 0,
            'resume_attribution': 'counter_evidence_only_no_per_request_trace',
            'historical_itl_comparison': 'unavailable_no_comparable_historical_histograms'}


class CompletionFeed:
    def __init__(self, root, lease, manifest_sha256, source_sha256):
        self.root = Path(root)
        self.lease, self.manifest, self.source = lease, manifest_sha256, source_sha256
        self.entries = []
        (self.root / 'completions').mkdir(mode=0o700, exist_ok=True)
        if (self.root / 'completion-index.json').exists():
            raise ValueError('Completion feed already exists; refusing an implicit rerun')
        self.write_index()

    def write_index(self):
        atomic(self.root / 'completion-index.json', {'schema_version': 1, 'run_id': self.lease['run_id'],
               'lease_id': self.lease['lease_id'], 'manifest_sha256': self.manifest, 'entries': self.entries})

    def emit(self, case, started, ended, status, summary, artifacts):
        case_id = case['case_id']
        if any(entry['case_id'] == case_id for entry in self.entries):
            raise ValueError('Scenario completion is immutable and single-use')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', case_id):
            raise ValueError('Unsafe case identity')
        if not math.isfinite(started) or not math.isfinite(ended) or ended < started:
            raise ValueError('Invalid scenario interval')
        hashes = {}
        for name in artifacts:
            path = self.root / name
            if not path.is_file() or self.root.resolve() not in path.resolve().parents:
                raise ValueError('Finalized artifact is missing or outside the run')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        sequence = len(self.entries) + 1
        to_utc = lambda stamp: datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).isoformat(timespec='milliseconds')
        start_ms, end_ms = int(started * 1000), int(ended * 1000)
        event = {'schema_version': 1, 'run_id': self.lease['run_id'], 'lease_id': self.lease['lease_id'],
                 'sequence': sequence, 'canonical_sequence': case.get('sequence'), 'case_id': case_id,
                 'category': case['category'], 'status': status, 'cold_warm_status': case['cold_warm_status'],
                 'start_epoch_ms': start_ms, 'end_epoch_ms': end_ms, 'start_utc': to_utc(started), 'end_utc': to_utc(ended),
                 'capture_from_ms': start_ms - 30000, 'capture_to_ms': end_ms + 30000,
                 'capture_not_before_ms': end_ms + 40000, 'manifest_sha256': self.manifest, 'source_sha256': self.source,
                 'outcomes': summary.get('outcomes', 0), 'successful': summary.get('successful', 0),
                 'artifacts': hashes}
        name = f'completions/{sequence:04d}-{case_id}.json'
        with (self.root / name).open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(event, indent=2, allow_nan=False))
        self.entries.append({'sequence': sequence, 'case_id': case_id, 'path': name,
                            'sha256': hashlib.sha256((self.root / name).read_bytes()).hexdigest()})
        self.write_index()
        return event
