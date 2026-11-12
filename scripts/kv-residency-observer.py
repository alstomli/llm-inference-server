"""Observation-only vLLM worker subclass used identically in both A/B arms.

Imported as kv_offload_observer from the owned temporary session directory.
No tensor contents or storage addresses are written.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker as GPUWorker


def describe(value):
    if dataclasses.is_dataclass(value):
        return {f.name: describe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Worker(GPUWorker):
    def initialize_from_config(self, kv_cache_config):
        super().initialize_from_config(kv_cache_config)
        runner = self.model_runner
        config = runner.kv_cache_config
        storages = {}
        tensors = []

        def visit(value, name):
            if isinstance(value, dict):
                for k, v in value.items():
                    visit(v, name + '/' + str(k))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    visit(v, name + '/' + str(i))
            elif hasattr(value, 'untyped_storage'):
                storage = value.untyped_storage()
                identity = (str(value.device), storage.data_ptr())
                if identity not in storages:
                    storages[identity] = {'device': str(value.device), 'nbytes': storage.nbytes(), 'index': len(storages)}
                tensors.append({'name': name, 'shape': list(value.shape), 'stride': list(value.stride()),
                                'dtype': str(value.dtype), 'storage_nbytes': storage.nbytes(),
                                'storage_index': storages[identity]['index'], 'storage_offset': value.storage_offset()})

        visit(runner.kv_caches, 'kv_caches')
        groups = [{'type': type(g.kv_cache_spec).__name__, 'spec': describe(g.kv_cache_spec),
                   'leaf_types': sorted({type(v).__name__ for v in getattr(g.kv_cache_spec, 'kv_cache_specs', {'self': g.kv_cache_spec}).values()}),
                   'role': describe(getattr(g,'role',None)), 'host_resident': getattr(g,'host_resident',None),
                   'page_size_bytes': g.kv_cache_spec.page_size_bytes, 'layers': list(g.layer_names)}
                  for g in config.kv_cache_groups]
        evidence = {'num_blocks': config.num_blocks, 'groups': groups,
                    'tensor_descriptors': describe(config.kv_cache_tensors),
                    'kernel_block_sizes': describe(getattr(runner, '_kernel_block_sizes', None)),
                    'layout': describe(config.kv_cache_layout), 'hash_block_size': config.hash_block_size,
                    'cache_hit_alignment_tokens': config.cache_hit_alignment_tokens,
                    'storages': list(storages.values()), 'tensors': tensors,
                    'actual_gpu_cache_bytes': sum(v['nbytes'] for v in storages.values() if v['device'].startswith('cuda'))}
        assert evidence['actual_gpu_cache_bytes'] > 0 and evidence['num_blocks'] > 0
        evidence['layout_fingerprint'] = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        # Connector-specific observations do not enter the common GPU layout hash.
        if os.environ['KV_EXPERIMENT_ARM'] == 'B':
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            connector = get_kv_transfer_group()
            worker = connector.connector_worker
            groups = worker.spec.config.groups
            selected = [g.group_id for g in groups]
            evidence['connector'] = {'type': type(connector).__name__, 'groups': describe(groups),
                'selected_group_ids': selected, 'num_cpu_chunks': getattr(worker.spec,'num_chunks',None)}
            expected = set(config.prefix_cacheable_group_ids)
            assert set(selected) == expected, 'Not all prefix-cacheable groups bound to offloader'
            eligible_types = {type(v).__name__ for i in expected for v in getattr(config.kv_cache_groups[i].kv_cache_spec,'kv_cache_specs',{'self':config.kv_cache_groups[i].kv_cache_spec}).values()}
            assert {'MambaSpec','FullAttentionSpec'}.issubset(eligible_types)
            evidence['connector']['expected_group_ids'] = sorted(expected)
            offloader = worker.worker
            for key in ['_store_handler', '_load_handler']:
                handler = getattr(offloader, key)
                refs = getattr(handler, 'layer_refs_per_group', None)
                evidence['connector'][key] = {'type': type(handler).__name__,
                    'group_reference_counts': [len(v) for v in refs] if refs is not None else None,
                    'num_cpu_chunks': getattr(handler, 'num_cpu_chunks', None)}
        destination = Path(os.environ['KV_EXPERIMENT_ROOT']) / ('layout-' + os.environ['KV_EXPERIMENT_ARM'] + '.json')
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence, indent=2))
        temporary.replace(destination)
