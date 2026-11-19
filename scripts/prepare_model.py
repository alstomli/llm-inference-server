"""Pin and stage a profile's checkpoint without locks on the global volume.

The shared logic (pinned revision, extension filter, scratch download, checksum copy to
the volume, resumable manifest, traversal guard) is the same for every profile; each
Profile supplies its model, folders and checkpoint validation.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
from typing import Callable

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_EXTENSIONS = ('.safetensors', '.json', '.txt', '.jinja', '.model', '.tiktoken')


@dataclass(frozen=True)
class Profile:
    name: str
    model_id: str
    revision: str | None  # None pins the current main commit on the first run.
    root_env: str
    default_root: str
    staging: str
    label: str
    validate: Callable[[Path], dict]  # Returns extra manifest fields; raises to refuse.
    extensions: tuple = DEFAULT_EXTENSIONS


def tensor_dtypes(model_dir):
    dtype_counts = {}
    for shard in model_dir.glob('*.safetensors'):
        with shard.open('rb') as source:
            header_size = struct.unpack('<Q', source.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            header = json.loads(source.read(header_size))
        for name, tensor in header.items():
            if name != '__metadata__':
                dtype = tensor['dtype']
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return dtype_counts


def validate_bf16(model_dir):
    model_config = json.loads((model_dir / 'config.json').read_text())
    if model_config.get('quantization_config'):
        raise ValueError('The checkpoint declares quantization')
    dtype_counts = tensor_dtypes(model_dir)
    if not dtype_counts.get('BF16'):
        raise ValueError('BF16 tensors were not found')
    return {'tensor_dtypes': dtype_counts}


PROFILES = {
    'bf16': Profile(name='bf16', model_id='Qwen/Qwen3.8-27B', revision=None,
                    root_env='BF16_ROOT', default_root='/workspace/qwen-bf16',
                    staging='/opt/qwen-bf16/download', label='original Qwen',
                    validate=validate_bf16),
}


def select_profile(environ):
    name = environ.get('MODEL_PROFILE', 'bf16')
    if name not in PROFILES:
        raise ValueError(f'Unknown model profile: {name}')
    return PROFILES[name]


PROFILE = select_profile(os.environ)
ROOT = Path(os.environ.get(PROFILE.root_env, PROFILE.default_root))
STAGING = Path(PROFILE.staging)
MODEL = ROOT / 'model'
MANIFEST = ROOT / 'model-manifest.json'


def write_manifest(manifest, state):
    # Only one writer, protected by a local flock in scripts/start-server.sh. Global volumes
    # do not provide atomic rename; an interrupted JSON write fails closed.
    manifest.write_text(json.dumps(state, indent=2) + '\n')


def stage(profile, root, staging):
    model, manifest = root / 'model', root / 'model-manifest.json'
    config = json.loads((root / 'config.json').read_text())
    if config['model_id'] != profile.model_id:
        raise ValueError(f'This deployment requires the {profile.label} checkpoint')
    state = json.loads(manifest.read_text()) if manifest.exists() else {}
    if state and state.get('model_id') != profile.model_id:
        raise ValueError('Existing manifest belongs to another model')
    if state and profile.revision and state.get('revision') != profile.revision:
        raise ValueError('Existing manifest pins another revision')
    info = HfApi().model_info(profile.model_id,
                            revision=profile.revision or state.get('revision', 'main'),
                            files_metadata=True)
    state = state or {'model_id': profile.model_id, 'revision': info.sha,
                      'files': {}, 'complete': False}
    state['complete'] = False
    model.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, state)
    files = [f for f in info.siblings if
             f.rfilename.endswith(profile.extensions) or f.rfilename == 'LICENSE']
    if not any(f.rfilename.endswith('.safetensors') for f in files):
        raise ValueError(f'No {profile.label} safetensors weights found')
    print(f'Pinned {profile.model_id} at {info.sha}', flush=True)
    for item in sorted(files, key=lambda f: (f.rfilename.endswith('.safetensors'), f.rfilename)):
        relative = PurePosixPath(item.rfilename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe checkpoint filename')
        destination = model.joinpath(*relative.parts)
        recorded = state['files'].get(item.rfilename)
        if recorded and destination.is_file() and destination.stat().st_size == recorded['size']:
            print(f'Already staged: {item.rfilename}', flush=True)
            continue
        print(f'Downloading: {item.rfilename} ({item.size} bytes)', flush=True)
        staged = Path(hf_hub_download(profile.model_id, item.rfilename,
                                    revision=info.sha, local_dir=staging))
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
        write_manifest(manifest, state)
        staged.unlink()  # Only the one temporary file on the container disk.
        print(f'Staged and verified: {item.rfilename}', flush=True)
    checks = profile.validate(model)
    state.update(checks)
    state['complete'] = True
    write_manifest(manifest, state)
    summary = '; '.join(f'{key.replace("_", " ")}: {value}' for key, value in checks.items())
    print(f'Checkpoint ready; {summary}', flush=True)


def main():
    stage(PROFILE, ROOT, STAGING)


if __name__ == '__main__':
    main()
