"""Validate the Railway images on the isolated local smoke-test ports."""
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]
settings = dict(line.split('=', 1) for line in (root / 'monitoring/.env').read_text().splitlines() if '=' in line)
authorization = 'Basic ' + base64.b64encode(('admin:' + settings['GRAFANA_PASSWORD']).encode()).decode()


def get(path, authenticated=False):
    headers = {'Authorization': authorization} if authenticated else {}
    with urlopen(Request('http://127.0.0.1:13000' + path, headers=headers), timeout=15) as response:
        return json.load(response)


assert get('/api/health')['database'] == 'ok'
try:
    get('/api/search')
except HTTPError as error:
    assert error.code == 401
else:
    raise AssertionError('Grafana must require login')
dashboard = get('/api/dashboards/uid/inference-baseline', True)['dashboard']
assert len(dashboard['panels']) == 20
datasource = get('/api/datasources/uid/prometheus', True)
assert datasource['url'] == 'http://prometheus:9090'
assert get('/api/datasources/uid/prometheus/health', True)['status'] == 'OK'
query = 'http://127.0.0.1:19090/api/v1/query?' + urlencode({'query': 'up{job="vllm"}'})
with urlopen(query, timeout=15) as response:
    result = json.load(response)['data']['result']
assert result and result[0]['value'][1] == '1', 'Live metrics connection is not healthy'
print('PASS: Grafana login, database, 20 dashboard panels, private data source, and live Runpod scrape.')
