"""Warmup-only native replay counter, positively removed before measurement."""
import json
import os
from pathlib import Path
import time
from combined_capacity_runtime import atomic, secure_root

ROOT = secure_root(os.environ['CAPACITY_SESSION_ROOT'])


def worker_class():
    from vllm.v1.worker.gpu_worker import Worker as NativeWorker

    class ObservedWorker(NativeWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import torch
            original = torch.cuda.CUDAGraph.replay
            count = 0

            def replay(graph):
                nonlocal count
                if (ROOT / 'observer-stop').exists():
                    torch.cuda.CUDAGraph.replay = original
                    atomic(ROOT / 'observer-inactive.json', {'native_replay_restored': True,
                           'completed_warmup_replays': count, 'time': time.time()})
                    return original(graph)
                result = original(graph)
                count += 1
                atomic(ROOT / 'observer-replays.json', {'completed_native_replays': count, 'time': time.time()})
                return result

            torch.cuda.CUDAGraph.replay = replay

        def initialize_from_config(self, config):
            super().initialize_from_config(config)
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            worker = get_kv_transfer_group().connector_worker
            actual = self.model_runner.kv_cache_config
            groups = []
            for group in actual.kv_cache_groups:
                leafs = getattr(group.kv_cache_spec, 'kv_cache_specs', {'self': group.kv_cache_spec})
                groups.append(sorted({type(value).__name__ for value in leafs.values()}))
            selected = [group.group_id for group in worker.spec.config.groups]
            assert set(selected) == set(actual.prefix_cacheable_group_ids)
            storage = {}
            def visit(value):
                if isinstance(value, dict):
                    for child in value.values(): visit(child)
                elif isinstance(value, (list, tuple)):
                    for child in value: visit(child)
                elif hasattr(value, 'untyped_storage'):
                    item = value.untyped_storage()
                    storage[(str(value.device), item.data_ptr())] = item.nbytes()
            visit(self.model_runner.kv_caches)
            atomic(ROOT / 'allocation.json', {'gpu_blocks': actual.num_blocks,
                   'actual_gpu_cache_bytes': sum(size for (device, _), size in storage.items() if device.startswith('cuda')),
                   'cpu_chunks': worker.spec.num_chunks, 'cpu_bytes_per_chunk': worker.spec.kv_bytes_per_chunk,
                   'cpu_mapped_bytes': worker.spec.num_chunks * worker.spec.kv_bytes_per_chunk,
                   'group_types': groups, 'offload_groups': selected,
                   'cache_hit_alignment_tokens': actual.cache_hit_alignment_tokens,
                   'hash_block_size': actual.hash_block_size, 'layout': str(actual.kv_cache_layout),
                   'store_group_refs': [len(group) for group in worker.worker._store_handler.layer_refs_per_group],
                   'load_group_refs': [len(group) for group in worker.worker._load_handler.layer_refs_per_group]})

    return ObservedWorker


def __getattr__(name):
    if name == 'Worker':
        return worker_class()
    raise AttributeError(name)
