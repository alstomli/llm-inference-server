"""Check the deployed metrics relay without printing its credential."""
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
TOKEN = (ROOT / '.secrets/metrics-token').read_text().strip()
BASE = 'https://fj04zaugwd1u83-8888.proxy.runpod.net'


def expect_status(path, headers, status):
    try:
        with urlopen(Request(BASE + path, headers={'User-Agent': 'monitoring-check/1.0', **headers}), timeout=30) as response:
            actual = response.status
            body = response.read().decode()
    except HTTPError as error:
        actual, body = error.code, ''
    assert actual == status, f'{path}: expected {status}, got {actual}'
    return body


if __name__ == '__main__':
    expect_status('/metrics', {}, 401)
    expect_status('/metrics', {'Authorization': 'Bearer invalid'}, 401)
    headers = {'Authorization': 'Bearer ' + TOKEN}
    expect_status('/v1/chat/completions', headers, 404)
    text = expect_status('/metrics', headers, 200)
    for metric in ('vllm:num_requests_running', 'gpu_memory_used_bytes', 'gpu_exporter_up 1'):
        assert metric in text, f'Missing metric: {metric}'
    print('HTTPS endpoint: correct token accepted; missing/wrong tokens rejected; model route blocked.')
    print('Live vLLM and GPU metrics: present.')
