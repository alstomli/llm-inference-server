"""Verify the authenticated Runpod chat deployment without printing its token."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', type=int, default=0, help='Maximum readiness wait in seconds')
    args = parser.parse_args()
    local_env = ROOT.parent / 'qwen-chat/.env.local'
    values = dict(line.split('=', 1) for line in local_env.read_text().splitlines() if '=' in line and not line.startswith('#')) if local_env.exists() else {}
    token = os.environ.get('RUNPOD_CHAT_TOKEN') or values.get('RUNPOD_CHAT_TOKEN')
    base = os.environ.get('RUNPOD_CHAT_URL') or values.get('RUNPOD_CHAT_URL') or 'https://fj04zaugwd1u83-8001.proxy.runpod.net'
    if not token:
        parser.error('Set RUNPOD_CHAT_TOKEN or configure qwen-chat/.env.local')
    base = base.rstrip('/')
    deadline = time.monotonic() + args.wait

    def request(path, authorization=None, payload=None):
        headers = {'User-Agent': 'qwen-chat-deployment-check/1.0'}
        if authorization is not None:
            headers['Authorization'] = 'Bearer ' + authorization
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode() if payload is not None else None
        return urlopen(Request(base + path, headers=headers, data=data), timeout=30)

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

    expect('/health', 401, wait=True)
    expect('/health', 401, 'invalid')
    expect('/v1/chat/completions', 401, payload={'messages': []})
    expect('/v1/models', 404, token)
    expect('/metrics', 404, token)
    expect('/docs', 404, token)
    print('PASS: Chat requires its token; model-management, metrics, and docs routes are blocked.', flush=True)
    health = json.loads(expect('/health', 200, token, wait=True))
    assert health['ready'] is True and health['model'] == 'qwen'
    print('PASS: The deployed model is ready.', flush=True)
    expect('/v1/chat/completions', 400, token, {'messages': [{'role': 'system', 'content': 'invalid'}], 'stream': True})

    # One real generation also verifies exact-token counting and old-turn trimming.
    payload = {
        'messages': [
            {'role': 'user', 'content': 'one two three four ' * 1400},
            {'role': 'assistant', 'content': 'I understand.'},
            {'role': 'user', 'content': 'Reply with one short sentence confirming you can answer chat messages.'},
        ],
        'max_tokens': 128, 'temperature': 0.1, 'stream': True,
    }
    start = time.monotonic()
    answer, done, first = '', False, None
    with request('/v1/chat/completions', token, payload) as response:
        assert response.headers['Content-Type'].startswith('text/event-stream')
        assert response.headers['X-Context-Trimmed'] == '2'
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
    print('PASS: Real Qwen response streamed to completion; older messages were trimmed to fit context.')
    print('Reply:', answer.strip())
    print(f'Time to first text: {first:.2f}s; total: {time.monotonic() - start:.2f}s')


if __name__ == '__main__':
    main()
