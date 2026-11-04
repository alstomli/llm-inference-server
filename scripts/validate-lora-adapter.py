#!/usr/bin/env python3
"""Validate a local language LoRA before starting vLLM; never import/load the base model."""
import argparse
import json
import math
from pathlib import Path
import struct

SUPPORTED_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def validate_adapter(path, base_model, base_name, name, max_rank):
    if not name or not base_name or name == base_name:
        raise ValueError('LoRA and base model names must be nonempty and distinct')
    if max_rank not in SUPPORTED_RANKS:
        raise ValueError(f'MAX_LORA_RANK must be one of {SUPPORTED_RANKS}')
    path = Path(path).resolve(strict=True)
    config = json.loads((path / 'adapter_config.json').read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Adapter configuration must be a JSON object')
    rank = config.get('r')
    if type(rank) is not int or not 1 <= rank <= max_rank:
        raise ValueError('Adapter rank must be positive and no larger than MAX_LORA_RANK')
    if config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Expected a causal language-model LoRA adapter')
    if config.get('bias', 'none') != 'none' or config.get('modules_to_save') or config.get('use_dora'):
        raise ValueError('Only standard language LoRA weights without saved full modules/bias/DoRA are supported')
    source = config.get('base_model_name_or_path')
    if not isinstance(source, str) or not Path(source).is_absolute() or Path(source).resolve() != Path(base_model).resolve():
        raise ValueError('Adapter base_model_name_or_path must match the cached BF16 model path')
    weights = path / 'adapter_model.safetensors'
    size = weights.stat().st_size
    with weights.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Adapter weight header is incomplete')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(16 * 2**20, size - 8):
            raise ValueError('Invalid adapter weight header length')
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        raise ValueError('Adapter weight header must be a JSON object')
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    if not tensors:
        raise ValueError('Adapter contains no tensors')
    pairs, ranges = {}, []
    for key, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise ValueError('Invalid adapter tensor metadata')
        kind = next((kind for kind in ('A', 'B') if key.endswith(f'.lora_{kind}.weight')), None)
        if kind is None:
            raise ValueError('Adapter may contain only linear LoRA A/B tensors')
        shape, offsets, dtype = tensor.get('shape'), tensor.get('data_offsets'), tensor.get('dtype')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(value) is not int or value < 1 for value in shape)
                or shape[0 if kind == 'A' else 1] != rank):
            raise ValueError('Adapter tensor shape does not match its configured rank')
        width = {'F32': 4, 'F16': 2, 'BF16': 2}.get(dtype)
        if (width is None or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != math.prod(shape) * width):
            raise ValueError('Invalid adapter tensor storage')
        ranges.append(tuple(offsets))
        pairs.setdefault(key.rsplit('.lora_', 1)[0], set()).add(kind)
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError('Adapter tensor storage has gaps or overlaps')
        end = stop
    if end != size - 8 - length or any(kinds != {'A', 'B'} for kinds in pairs.values()):
        raise ValueError('Adapter is incomplete or has unmatched LoRA A/B tensors')
    return {'name': name, 'path': str(path), 'base_model_name': base_name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--base-name', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--max-rank', required=True, type=int)
    args = parser.parse_args()
    try:
        result = validate_adapter(args.path, args.base_model, args.base_name, args.name, args.max_rank)
    except (ValueError, OSError, TypeError, AttributeError) as error:
        parser.exit(1, f'Invalid LoRA adapter: {error}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
