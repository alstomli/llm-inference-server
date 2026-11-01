"""Check the deployed cloud dashboard without printing credentials."""
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://grafana-production-5c5e.up.railway.app'
root = Path(__file__).resolve().parents[1]
settings = dict(line.split('=', 1) for line in (root / 'monitoring/.secrets/railway-grafana.env').read_text().splitlines() if '=' in line)
authorization = 'Basic ' + base64.b64encode(('admin:' + settings['GRAFANA_PASSWORD']).encode()).decode()


def get(path, authenticated=False):
    headers = {'User-Agent': 'monitoring-check/1.0'}
    if authenticated:
        headers['Authorization'] = authorization
    with urlopen(Request(BASE + path, headers=headers), timeout=30) as response:
        return json.load(response)


assert get('/api/health')['database'] == 'ok'
try:
    get('/api/search')
except HTTPError as error:
    assert error.code == 401
else:
    raise AssertionError('Grafana must require login')
print('PASS: HTTPS health and login protection')

dashboard = get('/api/dashboards/uid/inference-baseline', True)['dashboard']
assert len(dashboard['panels']) == 20
datasource = get('/api/datasources/uid/prometheus', True)
assert datasource['url'] == 'http://prometheus.railway.internal:9090'
assert get('/api/datasources/uid/prometheus/health', True)['status'] == 'OK'
print('PASS: 20 dashboard panels and private Prometheus connection')

for query in ['up{job="vllm"}', 'gpu_exporter_up']:
    path = '/api/datasources/proxy/uid/prometheus/api/v1/query?' + urlencode({'query': query})
    result = get(path, True)['data']['result']
    assert result and all(item['value'][1] == '1' for item in result), f'{query} is not healthy'
    print(f'PASS: {query} = 1')

print('PASS: Cloud monitoring is collecting live Runpod metrics')
