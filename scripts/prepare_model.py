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
    model_id: str
    revision: str | None  # None pins the current main commit on the first run.
    root_env: str  # Environment variable that overrides default_root.
    default_root: Path
    staging: Path  # Scratch download folder on the container disk.
    label: str  # Glossary name of the checkpoint, used in errors.
    validate: Callable[[Path], dict]  # Returns extra manifest fields; raises to refuse.
    extensions: tuple = DEFAULT_EXTENSIONS
    require_revision: bool = False  # Refuse to stage without a reviewed pin.


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


def tensor_names(model_dir):
    names = set()
    for shard in model_dir.glob('*.safetensors'):
        with shard.open('rb') as source:
            header_size = struct.unpack('<Q', source.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            names.update(json.loads(source.read(header_size)))
    names.discard('__metadata__')
    return names


def validate_w4a16(model_dir):
    """Accept only compressed-tensors 4-bit integer weights with 16-bit activations."""
    quantization = json.loads((model_dir / 'config.json').read_text()).get('quantization_config') or {}
    groups = list((quantization.get('config_groups') or {}).values())
    if quantization.get('quant_method') != 'compressed-tensors' or not groups or not all(
            (group.get('weights') or {}).get('num_bits') == 4
            and (group.get('weights') or {}).get('type') == 'int'
            and not group.get('input_activations') for group in groups):
        raise ValueError('The checkpoint is not compressed-tensors W4A16')
    names = tensor_names(model_dir)
    if not any(name.endswith('.weight_packed') for name in names):
        raise ValueError('Packed W4A16 weights were not found')
    if not any(name.startswith('model.visual.') for name in names):
        raise ValueError('The checkpoint has no vision tower')
    return {'quantization': 'compressed-tensors W4A16', 'tensor_dtypes': tensor_dtypes(model_dir)}


PROFILES = {
    'bf16': Profile(model_id='Qwen/Qwen3.8-27B', revision=None,
                    root_env='BF16_ROOT', default_root=Path('/workspace/qwen-bf16'),
                    staging=Path('/opt/qwen-bf16/download'), label='BF16 base',
                    validate=validate_bf16),
    # Reviewed on 2026-11-19 and loaded by the fit test (alstomli/llm-inference-server#50).
    '4bit': Profile(model_id='RedHatAI/Qwen3.8-27B-INT4',
                    revision='71dface38d4cbfa34121f1d446d4a0fa1025f13c',
                    root_env='QWEN_4BIT_ROOT', default_root=Path('/workspace/qwen-4bit'),
                    staging=Path('/opt/qwen-4bit/download'), label='4-bit base',
                    validate=validate_w4a16, require_revision=True),
}


def select_profile(environ):
    name = environ.get('MODEL_PROFILE', 'bf16')
    if name not in PROFILES:
        raise ValueError(f'Unknown model profile: {name}')
    return PROFILES[name]


# Optional overrides of the selected profile's folders, used by offline tests.
ROOT = None
STAGING = None


def write_manifest(manifest, state):
    # Only one writer, protected by a local flock in scripts/start-server.sh. Global volumes
    # do not provide atomic rename; an interrupted JSON write fails closed.
    manifest.write_text(json.dumps(state, indent=2) + '\n')


def stage(profile, root, staging):
    model, manifest = root / 'model', root / 'model-manifest.json'
    config = json.loads((root / 'config.json').read_text())
    if config['model_id'] != profile.model_id:
        raise ValueError(f'This deployment requires the {profile.label} checkpoint')
    if profile.require_revision and not profile.revision:
        raise ValueError(f'The {profile.label} must be staged at a pinned revision')
    state = json.loads(manifest.read_text()) if manifest.exists() else {}
    if state and state.get('model_id') != profile.model_id:
        raise ValueError('Existing manifest belongs to another model')
    if state and profile.revision and state.get('revision') != profile.revision:
        raise ValueError('Existing manifest pins another revision')
    info = HfApi().model_info(profile.model_id,
                            revision=profile.revision or state.get('revision', 'main'),
                            files_metadata=True)
    if profile.revision and info.sha != profile.revision:
        raise ValueError(f'The Hub returned {info.sha}, not the pinned revision {profile.revision}')
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
    profile = select_profile(os.environ)
    root = ROOT or Path(os.environ.get(profile.root_env, profile.default_root))
    stage(profile, root, STAGING or profile.staging)


if __name__ == '__main__':
    main()
