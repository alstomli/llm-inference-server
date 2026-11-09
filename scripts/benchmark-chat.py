"""Compare single and concurrent gateway streams; no GPU/server settings are changed."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPException
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID


PROMPT = ('If an image is attached, briefly describe it first. '
          'Write a detailed explanation of how rain forms, using numbered steps. '
          'Continue with examples until you reach the output limit.')
TIMING_KEYS = ('contextFitMs', 'upstreamFirstContentMs', 'gatewayFirstContentMs', 'gatewayTotalMs')
DIAGNOSTIC_KEYS = ('model', 'inputTokensAfter', 'outputBudget', 'reuseCachedPrefixes')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a private gateway credential to another origin.


class BenchmarkError(Exception):
    pass


def events(response):
    """Read complete SSE frames, including CRLF, comments and multiline data."""
    name, data, size = 'message', [], 0
    while True:
        raw = response.readline(1_048_577)
        if not raw:
            if data:
                raise BenchmarkError('Truncated SSE frame')
            return
        size += len(raw)
        if size > 1_048_576:
            raise BenchmarkError('SSE frame too large')
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                yield name, '\n'.join(data)
            name, data, size = 'message', [], 0
        elif line.startswith('event:'):
            name = line[6:].lstrip(' ')
        elif line.startswith('data:'):
            data.append(line[5:].removeprefix(' '))


def stream_metrics(response, started, clock=time.perf_counter):
    first = None
    done = False
    completion_tokens = None
    finish_reason = None
    timings = {}
    for name, data in events(response):
        observed = clock()
        if data == '[DONE]':
            done = True
            ended = observed
            break
        item = json.loads(data)
        if not isinstance(item, dict) or 'error' in item:
            raise BenchmarkError('Gateway returned a stream error')
        if name == 'chat-timing':
            timings = {key: item.get('timings', {}).get(key) for key in TIMING_KEYS}
            continue
        usage = item.get('usage')
        if isinstance(usage, dict) and type(usage.get('completion_tokens')) is int:
            completion_tokens = usage['completion_tokens']
        for choice in item.get('choices', []):
            content = choice.get('delta', {}).get('content')
            if isinstance(content, str) and content and first is None:
                first = observed
            if choice.get('finish_reason') is not None:
                finish_reason = choice['finish_reason']
    if not done or first is None or finish_reason not in ('stop', 'length'):
        raise BenchmarkError('Missing complete text response or successful finish reason')
    if completion_tokens is None or completion_tokens <= 0:
        raise BenchmarkError('Missing completion token usage; throughput cannot be measured')
    total = ended - started
    after_first = ended - first
    return {
        'ttft_seconds': first - started,
        'total_seconds': total,
        'completion_tokens': completion_tokens,
        'output_tokens_per_total_second': completion_tokens / total if total > 0 else None,
        'post_ttft_tokens_per_second': completion_tokens / after_first if after_first > 0 else None,
        'finish_reason': finish_reason,
        'gateway_timings_ms': timings,
    }


def request(opener, base, token, path, timeout, payload=None, user=None):
    headers = {'Authorization': 'Bearer ' + token, 'User-Agent': 'qwen-chat-benchmark/1.0'}
    if payload is not None:
        headers.update({'Content-Type': 'application/json', 'X-Chat-User-Id': user,
                        'X-Chat-Diagnostics': '1'})
    return opener.open(Request(base + path, headers=headers,
                               data=json.dumps(payload).encode() if payload is not None else None),
                       timeout=timeout)


def run_request(base, token, timeout, payload, user, label, barrier):
    opener = build_opener(NoRedirect)
    barrier.wait(timeout=10)
    started = time.perf_counter()
    result = {'request': label}
    try:
        with request(opener, base, token, '/v1/chat/completions', timeout, payload, user) as response:
            if not response.headers.get('Content-Type', '').startswith('text/event-stream'):
                raise BenchmarkError('Expected an SSE response')
            if 'image' in payload and response.headers.get('X-Chat-Image-Accepted') != 'true':
                raise BenchmarkError('Gateway did not confirm image receipt')
            if payload.get('useAdapter') is False and response.headers.get('X-Chat-Adapter-Enabled') != 'false':
                raise BenchmarkError('Gateway did not confirm the base model')
            diagnostic = json.loads(response.headers.get('X-Chat-Diagnostics', '{}'))
            result.update(stream_metrics(response, started))
            result['diagnostics'] = {key: diagnostic.get(key) for key in DIAGNOSTIC_KEYS}
            result['ok'] = True
    except HTTPError as error:
        # Do not log response bodies, URLs, headers, tokens, prompts or image data.
        result.update(ok=False, error=f'HTTP {error.code}')
    except (URLError, TimeoutError, OSError, HTTPException):
        result.update(ok=False, error='Connection failed or timed out')
    except (BenchmarkError, ValueError, TypeError, AttributeError, KeyError) as error:
        result.update(ok=False, error=str(error) if isinstance(error, BenchmarkError) else 'Invalid stream metadata')
    result['started_offset_seconds'] = started  # Converted relative to scenario start below.
    result['ended_offset_seconds'] = time.perf_counter()
    return result


def run_scenario(base, token, timeout, jobs):
    barrier = threading.Barrier(len(jobs))
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run_request, base, token, timeout, payload, user, label, barrier)
                   for label, user, payload in jobs]
        results = [future.result() for future in futures]
    for result in results:
        result['started_offset_seconds'] -= started
        result['ended_offset_seconds'] -= started
    overlap = (max(0, min(r['ended_offset_seconds'] for r in results)
                   - max(r['started_offset_seconds'] for r in results)) if len(results) > 1 else None)
    return {'requests': results, 'client_overlap_seconds': overlap}


def image_data(path, health):
    mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(path.suffix.lower())
    limits = health['imageInput']
    if mime not in limits['mimeTypes']:
        raise BenchmarkError('Unsupported image extension or MIME type')
    if not 0 < path.stat().st_size <= limits['maxBytes']:
        raise BenchmarkError('Image exceeds advertised byte limit or is empty')
    # The gateway remains responsible for decoding, dimensions and animation checks.
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('RUNPOD_BF16_CHAT_URL'))
    parser.add_argument('--image', type=Path, required=True, help='Existing representative image within gateway limits')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path (will not overwrite)')
    parser.add_argument('--expected-model', default='qwen-bf16')
    parser.add_argument('--user-a', default='00000000-0000-4000-8000-000000000101')
    parser.add_argument('--user-b', default='00000000-0000-4000-8000-000000000102')
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--cooldown', type=float, default=65, help='Seconds between measured repetitions')
    parser.add_argument('--timeout', type=float, default=180, help='Socket operation timeout, seconds')
    parser.add_argument('--cache-mode', choices=('isolated', 'reuse'), default='isolated')
    parser.add_argument('--skip-warmup', action='store_true')
    args = parser.parse_args()
    token = os.environ.get('RUNPOD_BF16_CHAT_TOKEN')
    if not token or not args.url:
        parser.error('Set RUNPOD_BF16_CHAT_TOKEN and RUNPOD_BF16_CHAT_URL (or --url)')
    parts = urlsplit(args.url)
    if (parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.path not in ('', '/')
            or (parts.scheme == 'http' and parts.hostname not in ('localhost', '127.0.0.1', '::1'))):
        parser.error('URL must be an HTTPS gateway origin (HTTP allowed only on loopback)')
    try:
        users = [str(UUID(args.user_a)), str(UUID(args.user_b))]
    except ValueError:
        parser.error('Both user IDs must be UUIDs')
    if users[0] == users[1]:
        parser.error('Two distinct stable user IDs are required')
    if not (128 <= args.max_tokens <= 10240 and 1 <= args.repeats <= 20
            and 1 <= args.timeout <= 900 and 60 <= args.cooldown <= 3600):
        parser.error('Use 128..10240 tokens, 1..20 repeats, 1..900 timeout, 60..3600 cooldown')
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    base = args.url.rstrip('/')
    try:
        with request(build_opener(NoRedirect), base, token, '/health', args.timeout) as response:
            health = json.load(response)
        if health.get('ready') is not True or health.get('model') != args.expected_model:
            raise BenchmarkError('Expected model is not ready')
        if health.get('diagnosticsVersion') != 1:
            raise BenchmarkError('Gateway diagnostics must already be enabled for token usage and timing')
        if health.get('imageInputVersion') != 1 or not health.get('imageInput'):
            raise BenchmarkError('Gateway does not advertise image input')
        if health.get('admission', {}).get('maxConcurrentRequests', 0) < 2:
            raise BenchmarkError('Gateway must already permit two concurrent requests')
        if args.max_tokens > health['limits']['outputTokens']:
            raise BenchmarkError('Output budget exceeds gateway limit')
        image = image_data(args.image, health)
    except (HTTPError, URLError, OSError, HTTPException, ValueError, KeyError, TypeError, BenchmarkError) as error:
        parser.error(str(error) if isinstance(error, BenchmarkError) else 'Health/image preflight failed')
    payload = {'messages': [{'role': 'user', 'content': PROMPT}], 'max_tokens': args.max_tokens,
               'temperature': 0, 'stream': True,
               'settings': {'outputTokens': args.max_tokens, 'reuseCachedPrefixes': args.cache_mode == 'reuse',
                            'contextTokens': health['limits']['contextTokens']}}
    if health.get('adapterToggleAvailable'):
        payload['useAdapter'] = False  # Same base weights for every scenario, including text.
    image_payload = {**payload, 'image': image}
    report = {'version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'model': health['model'], 'max_tokens': args.max_tokens, 'cache_mode': args.cache_mode,
              'warmup': not args.skip_warmup, 'cooldown_seconds': args.cooldown,
              'admission_at_start': health.get('admission'), 'runs': [],
              'metric_definitions': {
                  'ttft_seconds': 'Request start to first nonempty content delta; includes network and gateway preparation.',
                  'output_tokens_per_total_second': 'Reported completion tokens / request-start-to-DONE seconds.',
                  'post_ttft_tokens_per_second': 'Reported completion tokens / first-content-to-DONE seconds; approximate decode rate, not per-token latency.',
                  'client_overlap_seconds': 'Overlap of client request lifetimes; does not prove GPU scheduling overlap.',
              }}
    failed = False
    try:
        if not args.skip_warmup:
            print('Warming text and image paths (two sequential requests)...', flush=True)
            for label, body in (('warmup_text', payload), ('warmup_image', image_payload)):
                run = run_scenario(base, token, args.timeout, [(label, users[0], body)])
                report['runs'].append({'scenario': label, 'repeat': 0, **run})
                if not run['requests'][0]['ok']:
                    raise BenchmarkError('Warmup failed; see report')
        for repeat in range(1, args.repeats + 1):
            if repeat > 1:
                print(f'Cooling down for {args.cooldown:g} seconds...', flush=True)
                # Short sleeps keep interactive cancellation responsive.
                deadline = time.monotonic() + args.cooldown
                while (remaining := deadline - time.monotonic()) > 0:
                    time.sleep(min(1, remaining))
            for scenario, jobs in (
                ('single_text', [('text_a', users[0], payload)]),
                ('two_text', [('text_a', users[0], payload), ('text_b', users[1], payload)]),
                ('text_and_image', [('text_a', users[0], payload), ('image_b', users[1], image_payload)]),
            ):
                print(f'{scenario}, repetition {repeat}...', flush=True)
                run = run_scenario(base, token, args.timeout, jobs)
                report['runs'].append({'scenario': scenario, 'repeat': repeat, **run})
                for result in run['requests']:
                    if result['ok']:
                        rate = result['post_ttft_tokens_per_second']
                        rate_text = f'{rate:.2f}' if rate is not None else 'unavailable'
                        print(f"  {result['request']}: TTFT {result['ttft_seconds']:.3f}s; post-TTFT tokens/s {rate_text}", flush=True)
                    else:
                        raise BenchmarkError(f"{scenario} failed: {result['error']}; see report")
    except (BenchmarkError, KeyboardInterrupt) as error:
        failed = True
        report['error'] = str(error) if isinstance(error, BenchmarkError) else 'Interrupted'
    report['ok'] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    print('Report saved.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
