"""Record actual readiness, BF16 checkpoint evidence, and one real generation."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path('/workspace/qwen-bf16')
BASE = 'http://127.0.0.1:8000'


def request(path, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=timeout) as response:
        body = response.read()
        return response.status, json.loads(body) if body else None


def main():
    deadline = time.monotonic() + 1800
    while True:
        try:
            status, _ = request('/health')
            if status == 200:
                break
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Model did not become ready within 30 minutes')
        time.sleep(10)
    _, models = request('/v1/models')
    model = next(item for item in models['data'] if item['id'] == 'qwen-bf16')
    assert model['max_model_len'] == 4096, model
    manifest = json.loads((ROOT / 'model-manifest.json').read_text())
    assert manifest['complete'] and manifest['tensor_dtypes'].get('BF16')
    config = json.loads((ROOT / 'config.json').read_text())
    prompt = 'Reply with a short greeting and confirm you can respond.'
    started = time.monotonic()
    _, response = request('/v1/chat/completions', {
        'model': 'qwen-bf16',
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 48, 'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
    }, timeout=600)
    elapsed = time.monotonic() - started
    choice = response['choices'][0]
    assert choice['message'].get('content', '').strip(), response
    assert choice['finish_reason'] == 'stop', choice
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
        '--format=csv,noheader'], text=True).strip()
    result = {
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'pod_id': 'ftntkktzid9f4p', 'global_volume': 'spicy_coffee_meerkat',
        'volume_mount': '/workspace', 'api_scope': 'Pod loopback only',
        'health_status': status, 'served_model': model,
        'configuration': config, 'checkpoint_revision': manifest['revision'],
        'checkpoint_file_count': len(manifest['files']),
        'weight_shards': sum(name.endswith('.safetensors') for name in manifest['files']),
        'checkpoint_bytes': sum(item['size'] for item in manifest['files'].values()),
        'tensor_dtypes': manifest['tensor_dtypes'], 'gpu': gpu,
        'container_memory_limit_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
        'container_memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['vllm', 'transformers', 'torch', 'huggingface-hub', 'safetensors']},
        'generation': {'prompt': prompt, 'answer': choice['message']['content'],
                       'finish_reason': choice['finish_reason'],
                       'usage': response.get('usage'), 'elapsed_seconds': round(elapsed, 3)},
        'limitations': ['One short text generation tested; no load or maximum-context benchmark.',
                        'The API is private; no public endpoint or frontend migration was configured.'],
    }
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    freeze = subprocess.check_output([str(Path(__import__('sys').executable)), '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'requirements-resolved.txt').write_text(freeze)
    print(json.dumps(result, indent=2), flush=True)
    print('LIVE_BF16_DEPLOYMENT_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
