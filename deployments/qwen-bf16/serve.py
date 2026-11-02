"""Launch the pinned local BF16 checkpoint with adjustable partial offloading."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/workspace/qwen-bf16')


def command(config):
    if config['dtype'] != 'bfloat16':
        raise ValueError('The approved deployment uses BF16')
    if config['host'] != '127.0.0.1':
        raise ValueError('Public exposure requires a separate ingress configuration')
    return [str(Path(sys.executable).with_name('vllm')), 'serve', str(ROOT / 'model'),
            '--served-model-name', config['served_model_name'],
            '--dtype', config['dtype'],
            '--cpu-offload-gb', str(config['cpu_offload_gib']),
            '--max-model-len', str(config['max_model_len']),
            '--max-num-seqs', str(config['max_num_seqs']),
            '--gpu-memory-utilization', str(config['gpu_memory_utilization']),
            '--host', config['host'], '--port', str(config['port']),
            '--reasoning-parser', 'qwen3', '--enforce-eager',
            '--safetensors-load-strategy', 'eager',
            '--limit-mm-per-prompt', '{"image":1,"video":0}']


if __name__ == '__main__':
    state = json.loads((ROOT / 'model-manifest.json').read_text())
    if not state.get('complete'):
        raise RuntimeError('The original checkpoint download has not completed')
    args = command(json.loads((ROOT / 'config.json').read_text()))
    print('Launching:', ' '.join(args), flush=True)
    os.execv(args[0], args)
