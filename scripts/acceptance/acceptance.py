#!/usr/bin/env python3
"""Profile acceptance for a serving Pod, run from a workstation through the public gateway.

Derived from the 2026-11-14 batch-eight acceptance, generalized to any profile.
It assumes a NEW or idle Pod: it never changes Pod configuration and has no
rollback machinery. Rolling back means switching traffic to another profile.

Waves (each sends exactly `concurrency` requests at once with known answers):
  short        512 tokens in / 128 out, all must reach `concurrency` active
  medium       2048 in / 256 out, plus queued and active cancellation probes
  mixed        base text, adapter text (only if the profile has one), vision
  nearcontext  context - 128 - 32 tokens in
  tool         one web-search request in `required` mode

Usage:
  python acceptance.py --profile 4bit --pod <POD_ID> --concurrency 4 prepare
  python acceptance.py --profile 4bit --pod <POD_ID> --concurrency 4 all
Results go to acceptance-results/<profile>-<UTC> (git-ignored) or --run-dir.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import re
import socket
import ssl
import struct
import sys
import threading
import time
from urllib.request import Request, urlopen
import uuid
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SAFETY_TOKENS = 32
NEAR_OUTPUT = 128
MAX_WALL = 240
WAVES = ('short', 'medium', 'mixed', 'nearcontext')
IDLE_GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting')

PROFILES = {
    'bf16': {'base_alias': 'qwen-bf16-base', 'adapter_alias': 'qwen-bf16', 'kv_cache_dtype': None,
             'model_path': '/workspace/qwen-bf16/model', 'min_ram_headroom_gib': 32, 'min_shm_gib': 8},
    '4bit': {'base_alias': 'qwen-4bit', 'adapter_alias': None, 'kv_cache_dtype': 'fp8',
             'model_path': '/workspace/qwen-4bit/model', 'min_ram_headroom_gib': 8, 'min_shm_gib': 4},
}


def load_bench():
    spec = importlib.util.spec_from_file_location('benchmark_capacity', HERE.parent / 'benchmark-capacity.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Config:
    profile: str
    pod: str
    concurrency: int
    waiting: int
    context: int
    token_file: Path
    run_dir: Path
    model_path: str

    @property
    def spec(self):
        return PROFILES[self.profile]

    @property
    def host(self):
        return f'{self.pod}-8001.proxy.runpod.net'


# ---------- helpers ----------

def save(cfg, name, value, exclusive=False):
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.run_dir / name
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    return path


def runpod_key():
    for line in (REPO / '.env').read_text(encoding='utf-8-sig').splitlines():
        m = re.match(r'^\s*RUNPOD_API_KEY\s*=\s*(.*?)\s*$', line)
        if m:
            return m[1].strip().strip('"').strip("'")
    raise RuntimeError('RUNPOD_API_KEY missing from .env')


def execute(cfg, code, timeout=25):
    """Read-only python3 -c on the Pod through the RunPod exec API."""
    body = json.dumps({'command': ['python3', '-c', code], 'timeoutSeconds': timeout}).encode()
    req = Request(f'https://api.runpod.io/v2/pods/{cfg.pod}/exec', data=body, method='POST',
                  headers={'Authorization': 'Bearer ' + runpod_key(), 'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout + 10) as r:
        row = json.loads(r.read())
    if row.get('exitCode') or row.get('timedOut') or row.get('truncated'):
        raise RuntimeError('Remote exec incomplete')
    return row.get('stdout', '')


def build_prompt_code(cfg, seed=12345):
    params = {'model_path': cfg.model_path, 'concurrency': cfg.concurrency, 'context': cfg.context,
              'safety': SAFETY_TOKENS, 'adapter': bool(cfg.spec['adapter_alias']), 'seed': seed}
    return f'PARAMS = {params!r}\n' + (HERE / 'remote_prompts.py').read_text(encoding='utf-8')


def png_uri():
    """2x2 solid-red PNG, the proven gateway vision fixture."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    raw = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0))
    raw += chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 2) * 2)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(raw).decode()


# ---------- pure checks (unit tested) ----------

def expected_model(cfg, use_adapter):
    return cfg.spec['adapter_alias'] if use_adapter else cfg.spec['base_alias']


def arithmetic_correct(answer):
    numbers = re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?', answer)
    return bool(numbers) and all(x == '4' for x in numbers)


def vision_correct(answer):
    colors = re.findall(r'\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b', answer.lower())
    return bool(colors) and all(x == 'red' for x in colors)


def check_headers(cfg, row, headers):
    h = {k.lower(): v for k, v in headers}
    adapter_header = h.get('x-chat-adapter-enabled')
    if cfg.spec['adapter_alias']:
        assert adapter_header == str(row['useAdapter']).lower(), 'Adapter header'
    else:
        assert adapter_header in (None, 'false'), 'Adapter header on a profile without adapter'
    assert h.get('x-context-trimmed') == '0' and h.get('cache-control') == 'no-store', 'Trim/cache contract'
    if row['vision']:
        assert h.get('x-chat-image-accepted') == 'true', 'Vision header'
    diagnostic = json.loads(h['x-chat-diagnostics'])
    assert diagnostic['model'] == expected_model(cfg, row['useAdapter']), 'Model diagnostics'
    assert diagnostic['gatewayTrimmedMessages'] == 0, 'Trim diagnostics'
    if not row['vision']:
        assert diagnostic['inputTokensAfter'] == row['target_input_tokens'], 'Exact input token count'
    assert diagnostic['inputTokensAfter'] + row['max_tokens'] + SAFETY_TOKENS <= cfg.context, 'Context bound'
    return diagnostic


def arg_value(args, key):
    assert args.count(key) == 1, 'Engine setting ' + key
    return args[args.index(key) + 1]


def validate_status(cfg, row, baseline=None, idle=True):
    """Engine, gateway and resource contract for the profile. `row` comes from remote_status.py."""
    spec = cfg.spec
    assert len(row['processes']) == 1, 'Engine count'
    args = row['processes'][0]['args']
    assert arg_value(args, '--max-num-seqs') == str(cfg.concurrency), 'Engine setting --max-num-seqs'
    assert arg_value(args, '--max-model-len') == str(cfg.context), 'Engine setting --max-model-len'
    assert spec['base_alias'] in args[args.index('--served-model-name') + 1:], 'Served base alias'
    if spec['kv_cache_dtype']:
        assert arg_value(args, '--kv-cache-dtype') == spec['kv_cache_dtype'], 'KV cache dtype'
    assert ('--enable-lora' in args) == bool(spec['adapter_alias']), 'LoRA flag matches profile'
    gh = row['gateway_health']
    a = gh.get('admission', {})
    assert row['health_status'] == 200 and row['gateway_http'] == 200 and gh.get('ready'), 'Readiness'
    assert a.get('maxConcurrentRequests') == cfg.concurrency, 'Gateway concurrency limit'
    assert a.get('maxWaitingRequests') == cfg.waiting, 'Gateway waiting limit'
    assert gh.get('imageInputVersion') == 1, 'Vision available'
    assert gh.get('limits', {}).get('contextTokens') == cfg.context, 'Gateway context'
    if idle:
        assert a.get('activeRequests') == 0 and a.get('waitingRequests') == 0, 'Gateway not idle'
        assert all(row['idle_counts'].get(k, 0) == 0 for k in IDLE_GAUGES), 'Engine not idle'
    m = row['memory']
    assert m['memory.max'] - m['memory.current'] >= spec['min_ram_headroom_gib'] * 1024**3, 'RAM headroom'
    assert m['shm_available_bytes'] >= spec['min_shm_gib'] * 1024**3, 'Shared memory headroom'
    if baseline:
        b = baseline['memory']
        assert m['memory.swap.current'] <= b['memory.swap.current'], 'Swap growth'
        assert all(m['memory.events'][k] <= b['memory.events'][k] for k in ('oom', 'oom_kill')), 'OOM growth'


def check_corpus(cfg, rows):
    for wave in WAVES:
        part = [r for r in rows if r['wave'] == wave]
        assert len(part) == cfg.concurrency, f'{wave}: expected {cfg.concurrency} rows'
    mixed = [r for r in rows if r['wave'] == 'mixed']
    assert any(r['vision'] for r in mixed), 'Mixed wave needs vision'
    assert any(r['useAdapter'] for r in mixed) == bool(cfg.spec['adapter_alias']), 'Adapter rows match profile'
    near = [r for r in rows if r['wave'] == 'nearcontext']
    assert all(r['target_input_tokens'] + r['max_tokens'] + SAFETY_TOKENS == cfg.context for r in near), 'Near-context size'


# ---------- gateway I/O ----------

class Gateway:
    def __init__(self, cfg):
        self.cfg = cfg
        self.token = cfg.token_file.read_text().strip()

    def connect(self, payload=None, principal=None, path='/v1/chat/completions', deadline=None):
        conn = http.client.HTTPSConnection(self.cfg.host, timeout=8, context=ssl.create_default_context())
        h = {'Authorization': 'Bearer ' + self.token, 'User-Agent': f'{self.cfg.profile}-acceptance/1'}
        if payload is not None:
            h.update({'Content-Type': 'application/json', 'X-Chat-User-ID': principal, 'X-Chat-Diagnostics': '1'})
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        conn.request('POST' if payload is not None else 'GET', path, body=body, headers=h)
        if deadline:
            conn.sock.settimeout(max(.1, deadline - time.monotonic()))
        return conn

    @staticmethod
    def abort(conn):
        if conn:
            try:
                if conn.sock:
                    conn.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def health(self):
        conn = self.connect(path='/health')
        try:
            r = conn.getresponse()
            assert r.status == 200, 'Health status'
            value = json.loads(r.read())
            assert value['ready'], 'Gateway not ready'
            return value
        finally:
            self.abort(conn)

    def admission(self):
        return self.health()['admission']

    def wait_counts(self, active, waiting, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            a = self.admission()
            if a['activeRequests'] == active and a['waitingRequests'] == waiting:
                return a
            time.sleep(.15)
        raise AssertionError('Bounded admission recovery')


def adapter_choice(cfg, use_adapter):
    """The useAdapter field, sent only to a profile with an adapter; others answer 400 to it."""
    return {'useAdapter': use_adapter} if cfg.spec['adapter_alias'] else {}


def payload(cfg, row):
    content = row['content']
    if row['vision']:
        content += ('\nFor this vision check, ignore the earlier arithmetic output instruction. '
                    'What solid color is the attached image? Reply with only the color name.')
    value = {'messages': [{'role': 'user', 'content': content}], 'max_tokens': row['max_tokens'],
             'temperature': 0, 'stream': True, **adapter_choice(cfg, row['useAdapter']),
             'settings': {'outputTokens': row['max_tokens'], 'contextTokens': cfg.context,
                          'safetyTokens': SAFETY_TOKENS, 'timeoutSeconds': MAX_WALL, 'reuseCachedPrefixes': False}}
    if row['vision']:
        value['image'] = png_uri()
    return value


def request(cfg, gw, bench, row, principal, launch_at):
    delay = launch_at - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    started = time.monotonic()
    result = {'wave': row['wave'], 'id': row['id'], 'outcome': 'error'}
    conn = timer = usage = None
    try:
        conn = gw.connect(payload(cfg, row), principal, deadline=started + MAX_WALL)
        timer = threading.Timer(max(.01, MAX_WALL - (time.monotonic() - started)), gw.abort, args=(conn,))
        timer.daemon = True
        timer.start()
        response = conn.getresponse()
        result['http'] = response.status
        assert response.status == 200, 'HTTP status'
        diagnostic = check_headers(cfg, row, response.getheaders())
        parser = bench.SSE()
        text, done, finish, first, last = [], False, None, None, None
        while time.monotonic() - started < MAX_WALL and not done:
            data = response.read1(4096)
            if not data:
                break
            for frame in parser.feed(data):
                if frame == '[DONE]':
                    done = True
                    break
                value = json.loads(frame)
                assert 'error' not in value, 'Stream error'
                usage = value.get('usage') or usage
                for choice in value.get('choices', []):
                    chunk = choice.get('delta', {}).get('content')
                    if chunk:
                        last = time.monotonic()
                        first = first or last
                        text.append(chunk)
                    finish = choice.get('finish_reason') or finish
        answer = ''.join(text).strip()
        correct = vision_correct(answer) if row['vision'] else arithmetic_correct(answer)
        assert done and finish in ('stop', 'length') and correct, 'Completion/correctness'
        assert isinstance(usage, dict) and usage.get('completion_tokens', 0) > 0, 'Usage'
        result.update(outcome='success', input_tokens=diagnostic['inputTokensAfter'],
                      completion_tokens=usage['completion_tokens'], ttft_s=first - started,
                      e2e_s=time.monotonic() - started, finish=finish, model=diagnostic['model'],
                      vision=row['vision'], answer_sha256=hashlib.sha256(answer.encode()).hexdigest())
    except Exception as e:
        result.update(error_type=type(e).__name__, error=str(e)[:200], e2e_s=time.monotonic() - started)
    finally:
        if timer:
            timer.cancel()
        gw.abort(conn)
    return result


def queued_cancel(cfg, gw, principal):
    """Send one extra request for an already-active principal, see it queue, cancel it, see it drain."""
    conn = None
    try:
        conn = gw.connect({'messages': [{'role': 'user', 'content': 'What is 2+2? Reply only 4.'}],
                           'max_tokens': 128, 'temperature': 0, 'stream': True, **adapter_choice(cfg, False),
                           'settings': {'outputTokens': 128, 'timeoutSeconds': 20}}, principal)
        gw.wait_counts(cfg.concurrency, 1, timeout=4)
        gw.abort(conn)
        conn = None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if gw.admission()['waitingRequests'] == 0:
                return {'probe': 'queued_cancellation', 'passed': True}
            time.sleep(.1)
        raise AssertionError('Queued cancellation did not drain')
    finally:
        gw.abort(conn)


def active_cancel(cfg, gw):
    conn = None
    try:
        conn = gw.connect({'messages': [{'role': 'user', 'content': 'Count from 1 to 1000, one number per line.'}],
                           'max_tokens': 2048, 'temperature': 0, 'stream': True, **adapter_choice(cfg, False),
                           'settings': {'outputTokens': 2048, 'timeoutSeconds': 30}}, str(uuid.uuid4()))
        r = conn.getresponse()
        assert r.status == 200 and r.readline(4096), 'Active stream'
        assert gw.admission()['activeRequests'] == 1, 'Active cancellation was not active'
        gw.abort(conn)
        conn = None
        gw.wait_counts(0, 0, timeout=15)
        return {'probe': 'active_cancellation', 'passed': True}
    finally:
        gw.abort(conn)


# ---------- commands ----------

def get_status(cfg):
    return json.loads(execute(cfg, (HERE / 'remote_status.py').read_text(encoding='utf-8')))


def cmd_prepare(cfg):
    raw = execute(cfg, build_prompt_code(cfg), timeout=25)
    rows = json.loads(zlib.decompress(base64.b64decode(raw.strip())))
    check_corpus(cfg, rows)
    save(cfg, 'corpus.private.json', rows)
    save(cfg, 'corpus-manifest.json', [{k: v for k, v in r.items() if k != 'content'} for r in rows])
    print(json.dumps({'requests': len(rows), 'concurrency': cfg.concurrency, 'inference': False}))


def cmd_status(cfg, baseline=None, name='status.private.json'):
    row = get_status(cfg)
    validate_status(cfg, row, baseline)
    save(cfg, name, row)
    return row


def run_wave(cfg, gw, bench, name, baseline):
    gw.wait_counts(0, 0)
    rows = [r for r in json.loads((cfg.run_dir / 'corpus.private.json').read_text(encoding='utf-8')) if r['wave'] == name]
    n = cfg.concurrency
    assert len(rows) == n, 'Corpus size'
    principals = [str(uuid.uuid4()) for _ in rows]
    start = time.monotonic() + .2
    samples, queued = [], None
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(request, cfg, gw, bench, row, principals[i], start + i * .05)
                   for i, row in enumerate(rows)]
        deadline = start + MAX_WALL
        while not all(f.done() for f in futures) and time.monotonic() < deadline:
            a = gw.admission()
            samples.append({'time': time.time(), **a})
            if name == 'medium' and queued is None and a['activeRequests'] == n:
                queued = queued_cancel(cfg, gw, principals[0])
            time.sleep(.1)
        results = [f.result() for f in futures]
    save(cfg, name + '-requests.json', results)
    save(cfg, name + '-admission.json', samples)
    peak = max((x['activeRequests'] for x in samples), default=0)
    failed = [r for r in results if r['outcome'] != 'success']
    summary = {'wave': name, 'passed': not failed, 'requests': n, 'failed': len(failed), 'peak_gateway_active': peak,
               'e2e_s': {'p50': bench.percentile([r['e2e_s'] for r in results], 50),
                         'max': max(r['e2e_s'] for r in results)},
               'ttft_s_p50': bench.percentile([r['ttft_s'] for r in results if 'ttft_s' in r], 50),
               'completion_tokens': sum(r.get('completion_tokens', 0) for r in results)}
    if name in ('short', 'medium') and peak != n:
        summary.update(passed=False, reason=f'peak active {peak} != {n}')
    if name == 'medium':
        summary['queued_cancellation'] = queued
        if not (queued and queued['passed']):
            summary.update(passed=False, reason='queued cancellation probe missing')
    gw.wait_counts(0, 0)
    time.sleep(12)  # let published metrics settle before the idle check
    cmd_status(cfg, baseline, name + '-idle.private.json')
    if name == 'medium':
        summary['active_cancellation'] = active_cancel(cfg, gw)
    save(cfg, name + '-summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def run_tool(cfg, gw, bench):
    """One web-search request in required mode: the model must call the search tool and finish with sources."""
    started = time.monotonic()
    body = {'messages': [{'role': 'user', 'content': 'Search the web: what is the latest stable Python release? '
                                                      'Answer in one sentence and cite the source.'}],
            'max_tokens': 512, 'temperature': 0, 'stream': True, **adapter_choice(cfg, False),
            'webSearch': {'version': 1, 'mode': 'required'},
            'settings': {'outputTokens': 512, 'timeoutSeconds': MAX_WALL}}
    conn = gw.connect(body, str(uuid.uuid4()), deadline=started + MAX_WALL)
    result = {'wave': 'tool', 'passed': False}
    try:
        r = conn.getresponse()
        result['http'] = r.status
        assert r.status == 200, 'HTTP status'
        assert dict((k.lower(), v) for k, v in r.getheaders()).get('x-chat-web-search') == '1', 'Web search header'
        parser, done, sources, outcome = bench.SSE(), False, 0, None
        while not done and time.monotonic() - started < MAX_WALL:
            data = r.read1(4096)
            if not data:
                break
            for frame in parser.feed(data):
                if frame == '[DONE]':
                    done = True
                    break
                value = json.loads(frame)
                sources = max(sources, len(value.get('sources', [])))
                outcome = value.get('outcome', outcome)
        assert done and outcome == 'completed' and sources > 0, 'Search completed with sources'
        result.update(passed=True, sources=sources, e2e_s=time.monotonic() - started)
    except Exception as e:
        result.update(error_type=type(e).__name__, error=str(e)[:200])
    finally:
        gw.abort(conn)
    save(cfg, 'tool-summary.json', result)
    print(json.dumps(result), flush=True)
    return result


def cmd_all(cfg, waves):
    bench = load_bench()
    gw = Gateway(cfg)
    if not (cfg.run_dir / 'corpus.private.json').exists():
        cmd_prepare(cfg)
    baseline = cmd_status(cfg, name='baseline-status.private.json')
    summaries = []
    for name in waves:
        summaries.append(run_tool(cfg, gw, bench) if name == 'tool' else run_wave(cfg, gw, bench, name, baseline))
    passed = all(s['passed'] for s in summaries)
    save(cfg, 'acceptance.json', {'profile': cfg.profile, 'pod': cfg.pod, 'concurrency': cfg.concurrency,
                                  'context': cfg.context, 'passed': passed, 'time': time.time(), 'waves': summaries})
    print(json.dumps({'passed': passed, 'run_dir': str(cfg.run_dir)}))
    return passed


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--profile', choices=sorted(PROFILES), required=True)
    p.add_argument('--pod', required=True, help='RunPod Pod ID')
    p.add_argument('--concurrency', type=int, required=True, help='Engine MAX_NUM_SEQS and gateway active limit')
    p.add_argument('--waiting', type=int, help='Gateway waiting limit (default 3 x concurrency)')
    p.add_argument('--context', type=int, default=40960)
    p.add_argument('--token-file', type=Path, help='Default .secrets/<profile>-chat-token')
    p.add_argument('--model-path', help='Tokenizer path on the Pod (default per profile)')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('command', choices=['prepare', 'status', 'all', *WAVES, 'tool'])
    a = p.parse_args(argv)
    if a.concurrency < 1:
        p.error('--concurrency must be at least 1')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return Config(profile=a.profile, pod=a.pod, concurrency=a.concurrency, waiting=a.waiting or 3 * a.concurrency,
                  context=a.context, token_file=a.token_file or REPO / '.secrets' / f'{a.profile}-chat-token',
                  run_dir=a.run_dir or REPO / 'acceptance-results' / f'{a.profile}-{stamp}',
                  model_path=a.model_path or PROFILES[a.profile]['model_path']), a.command


def main(argv=None):
    cfg, command = parse(argv)
    if command == 'prepare':
        cmd_prepare(cfg)
    elif command == 'status':
        print(json.dumps({k: v for k, v in cmd_status(cfg).items() if k in ('time', 'health_status', 'gpu_csv')}))
    elif command == 'all':
        return 0 if cmd_all(cfg, [*WAVES, 'tool']) else 1
    else:
        baseline_path = cfg.run_dir / 'baseline-status.private.json'
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        bench, gw = load_bench(), Gateway(cfg)
        summary = run_tool(cfg, gw, bench) if command == 'tool' else run_wave(cfg, gw, bench, command, baseline)
        return 0 if summary['passed'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
