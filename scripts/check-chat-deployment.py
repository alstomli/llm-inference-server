"""Verify the authenticated Runpod chat deployment without printing its token."""
import argparse
import json
import os
import re
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', type=int, default=0, help='Maximum readiness wait in seconds')
    parser.add_argument('--expected-model', default='qwen-bf16')
    parser.add_argument('--read-timeout', type=int, default=30)
    parser.add_argument('--user-id', default=os.environ.get('CHAT_CHECK_USER_ID', '00000000-0000-4000-8000-000000000001'),
                        help='Stable trusted smoke-check principal UUID (shares its normal account budget)')
    args = parser.parse_args()
    if re.fullmatch(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', args.user_id) is None:
        parser.error('user-id must be a canonical UUID')
    args.user_id = args.user_id.lower()
    if not 1 <= args.read_timeout <= 900:
        parser.error('read-timeout must be between 1 and 900 seconds')
    local_env = ROOT.parent / 'qwen-chat/.env.local'
    values = dict(line.split('=', 1) for line in local_env.read_text().splitlines() if '=' in line and not line.startswith('#')) if local_env.exists() else {}
    token = os.environ.get('RUNPOD_BF16_CHAT_TOKEN') or values.get('RUNPOD_BF16_CHAT_TOKEN')
    base = os.environ.get('RUNPOD_BF16_CHAT_URL') or values.get('RUNPOD_BF16_CHAT_URL')
    if not token or not base:
        parser.error('Set RUNPOD_BF16_CHAT_URL and RUNPOD_BF16_CHAT_TOKEN or configure qwen-chat/.env.local')
    base = base.rstrip('/')
    deadline = time.monotonic() + args.wait

    def request(path, authorization=None, payload=None):
        headers = {'User-Agent': 'qwen-chat-deployment-check/1.0'}
        if authorization is not None:
            headers['Authorization'] = 'Bearer ' + authorization
        if payload is not None:
            headers['Content-Type'] = 'application/json'
            headers['X-Chat-User-Id'] = args.user_id
        data = json.dumps(payload).encode() if payload is not None else None
        return urlopen(Request(base + path, headers=headers, data=data), timeout=args.read_timeout)

    def expect(path, status, authorization=None, payload=None, wait=False):
        while True:
            try:
                with request(path, authorization, payload) as response:
                    actual, body = response.status, response.read()
            except HTTPError as error:
                actual, body = error.code, b''
            except (URLError, TimeoutError):
                actual, body = 0, b''
            if actual == status:
                return body
            if wait and time.monotonic() < deadline and actual in (0, 404, 502, 503, 504):
                time.sleep(5)
                continue
            raise AssertionError(f'{path}: expected {status}, got {actual}')

    health = json.loads(expect('/health', 200, token, wait=True))
    assert health['ready'] is True and health['model'] == args.expected_model
    assert health.get('accountQuotaVersion') == 1 and health.get('accountQuotasEnforced') is True, 'Gateway account quota enforcement is unavailable'
    print('PASS: The deployed model is ready with account quotas enforced.', flush=True)

    payload = {
        'messages': [{'role': 'user', 'content': 'Reply with exactly: ready'}],
        'max_tokens': 128, 'temperature': 0.1, 'stream': True,
    }
    start = time.monotonic()
    answer, done, first = '', False, None
    with request('/v1/chat/completions', token, payload) as response:
        assert response.headers['Content-Type'].startswith('text/event-stream')
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            assert not event.get('error'), 'Streaming returned an error'
            choices = event.get('choices', [])
            if choices:
                text = choices[0].get('delta', {}).get('content')
                if text:
                    if first is None:
                        first = time.monotonic() - start
                    answer += text
    assert done and answer.strip(), 'No complete model reply was received'
    print('PASS: A nonempty response to hello streamed to completion.')
    print('Reply:', answer.strip())
    print(f'Time to first text: {first:.2f}s; total: {time.monotonic() - start:.2f}s')


if __name__ == '__main__':
    main()
