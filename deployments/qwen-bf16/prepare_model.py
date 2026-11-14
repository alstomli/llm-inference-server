"""Pin and stage the original checkpoint without locks on the global volume."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(os.environ.get('BF16_ROOT', '/workspace/qwen-bf16'))
STAGING = Path('/opt/qwen-bf16/download')
MODEL = ROOT / 'model'
MANIFEST = ROOT / 'model-manifest.json'


def write_manifest(state):
    # Only one writer, protected by a local flock in scripts/start-server.sh. Global volumes
    # do not provide atomic rename; an interrupted JSON write fails closed.
    MANIFEST.write_text(json.dumps(state, indent=2) + '\n')


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    if config['model_id'] != 'Qwen/Qwen3.8-27B':
        raise ValueError('This deployment requires the original Qwen checkpoint')
    state = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    info = HfApi().model_info(config['model_id'],
                            revision=state.get('revision', 'main'),
                            files_metadata=True)
    if state and state.get('model_id') != config['model_id']:
        raise ValueError('Existing manifest belongs to another model')
    state = state or {'model_id': config['model_id'], 'revision': info.sha,
                      'files': {}, 'complete': False}
    state['complete'] = False
    MODEL.mkdir(parents=True, exist_ok=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    write_manifest(state)
    files = [f for f in info.siblings if
             f.rfilename.endswith(('.safetensors', '.json', '.txt', '.jinja', '.model', '.tiktoken'))
             or f.rfilename == 'LICENSE']
    if not any(f.rfilename.endswith('.safetensors') for f in files):
        raise ValueError('No original safetensors weights found')
    print(f'Pinned {config["model_id"]} at {info.sha}', flush=True)
    for item in sorted(files, key=lambda f: (f.rfilename.endswith('.safetensors'), f.rfilename)):
        relative = PurePosixPath(item.rfilename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint filename')
        destination = MODEL.joinpath(*relative.parts)
        recorded = state['files'].get(item.rfilename)
        if recorded and destination.is_file() and destination.stat().st_size == recorded['size']:
            print(f'Already staged: {item.rfilename}', flush=True)
            continue
        print(f'Downloading: {item.rfilename} ({item.size} bytes)', flush=True)
        staged = Path(hf_hub_download(config['model_id'], item.rfilename,
                                    revision=info.sha, local_dir=STAGING))
        if item.size is not None and staged.stat().st_size != item.size:
            raise ValueError(f'Download size mismatch: {item.rfilename}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with staged.open('rb') as source, destination.open('wb') as target:
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
                digest.update(block)
        expected = item.lfs.sha256 if item.lfs else None
        if expected and digest.hexdigest() != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {item.rfilename}')
        state['files'][item.rfilename] = {'size': staged.stat().st_size,
                                         'sha256': digest.hexdigest()}
        write_manifest(state)
        staged.unlink()  # Only the one temporary file on the container disk.
        print(f'Staged and verified: {item.rfilename}', flush=True)
    model_config = json.loads((MODEL / 'config.json').read_text())
    if model_config.get('quantization_config'):
        raise ValueError('The checkpoint declares quantization')
    dtype_counts = {}
    for shard in MODEL.glob('*.safetensors'):
        with shard.open('rb') as source:
            header_size = struct.unpack('<Q', source.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            header = json.loads(source.read(header_size))
        for name, tensor in header.items():
            if name != '__metadata__':
                dtype = tensor['dtype']
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if not dtype_counts.get('BF16'):
        raise ValueError('BF16 tensors were not found')
    state['tensor_dtypes'] = dtype_counts
    state['complete'] = True
    write_manifest(state)
    print(f'Checkpoint ready; tensor dtypes: {dtype_counts}', flush=True)


if __name__ == '__main__':
    main()
