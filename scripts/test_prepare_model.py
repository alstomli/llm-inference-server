"""Exercise pinned restart reuse and fail-closed checkpoint staging offline."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class StagingTests(unittest.TestCase):
    def run_case(self, scenario):
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [1],
                                       'data_offsets': [0, 2]}}).encode()
        weight = struct.pack('<Q', len(header)) + header + b'\0\0'
        parent = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(prefix='qwen-stage-check-', dir=parent) as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(parent))
            data = {'config.json': b'{"text_config":{"dtype":"bfloat16"}}',
                    'model-00001.safetensors': weight}
            if scenario == 'unsafe_path':
                data['../escape.safetensors'] = weight
            downloads, revisions = [], []

            def info(*args, **kwargs):
                revisions.append(kwargs.get('revision'))
                siblings = [types.SimpleNamespace(
                    rfilename=name, size=len(value), lfs=types.SimpleNamespace(
                        sha256=hashlib.sha256(value).hexdigest()))
                    for name, value in data.items()]
                return types.SimpleNamespace(sha='a' * 40, siblings=siblings)

            def download(model_id, filename, revision, local_dir):
                downloads.append(filename)
                target = Path(local_dir) / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                value = data[filename]
                if scenario == 'checksum_failure' and filename.endswith('.safetensors'):
                    value = value[:-1] + b'x'
                target.write_bytes(value)
                return str(target)

            fake = types.ModuleType('huggingface_hub')
            fake.HfApi = lambda: types.SimpleNamespace(model_info=info)
            fake.hf_hub_download = download
            with patch.dict(sys.modules, {'huggingface_hub': fake}):
                spec = importlib.util.spec_from_file_location(
                    'prepare_model_check', Path(__file__).with_name('prepare_model.py'))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            module.ROOT = root / 'volume'
            module.STAGING = root / 'local'
            module.MODEL = module.ROOT / 'model'
            module.MANIFEST = module.ROOT / 'model-manifest.json'
            module.ROOT.mkdir()
            (module.ROOT / 'config.json').write_text('{"model_id":"Qwen/Qwen3.8-27B"}')
            with redirect_stdout(io.StringIO()):
                if scenario == 'resume':
                    module.main()
                    state = json.loads(module.MANIFEST.read_text())
                    self.assertTrue(state['complete'])
                    self.assertEqual(state['tensor_dtypes'], {'BF16': 1})
                    module.main()
                    self.assertEqual(len(downloads), 2)
                    self.assertEqual(revisions, ['main', 'a' * 40])
                else:
                    with self.assertRaisesRegex(ValueError, 'checksum mismatch|Unsafe checkpoint'):
                        module.main()
                    self.assertFalse(json.loads(module.MANIFEST.read_text())['complete'])
                    self.assertFalse((root / 'escape.safetensors').exists())

    def test_reuses_completed_files_and_pins_revision(self):
        self.run_case('resume')

    def test_rejects_corrupt_checkpoint(self):
        self.run_case('checksum_failure')

    def test_rejects_filename_traversal(self):
        self.run_case('unsafe_path')


class ProfileStagingTests(unittest.TestCase):
    """The shared staging logic takes every model-specific rule from the profile."""

    def setUp(self):
        self.downloads, self.revisions = [], []
        self.data = {'config.json': b'{}', 'weights.safetensors': b'weights',
                     'extra.bin': b'bin', 'notes.md': b'skip'}
        fake = types.ModuleType('huggingface_hub')

        def info(*args, **kwargs):
            self.revisions.append(kwargs.get('revision'))
            siblings = [types.SimpleNamespace(
                rfilename=name, size=len(value), lfs=types.SimpleNamespace(
                    sha256=hashlib.sha256(value).hexdigest()))
                for name, value in self.data.items()]
            return types.SimpleNamespace(sha=kwargs.get('revision'), siblings=siblings)

        def download(model_id, filename, revision, local_dir):
            self.downloads.append((model_id, filename, revision))
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.data[filename])
            return str(target)

        fake.HfApi = lambda: types.SimpleNamespace(model_info=info)
        fake.hf_hub_download = download
        with patch.dict(sys.modules, {'huggingface_hub': fake}):
            spec = importlib.util.spec_from_file_location(
                'prepare_model_profile_check', Path(__file__).with_name('prepare_model.py'))
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory(prefix='qwen-profile-check-', dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'volume'
        self.root.mkdir()
        (self.root / 'config.json').write_text('{"model_id":"example/model"}')
        self.validated = []

    def profile(self, **overrides):
        def validate(model_dir):
            self.validated.append(model_dir)
            return {'checked_by': 'fake-profile'}
        fields = dict(model_id='example/model', revision='b' * 40,
                      root_env='FAKE_ROOT', default_root=self.root,
                      staging=self.root.parent / 'local', label='fake',
                      validate=validate, extensions=('.safetensors', '.json', '.bin'))
        fields.update(overrides)
        return self.module.Profile(**fields)

    def stage(self, profile):
        with redirect_stdout(io.StringIO()):
            self.module.stage(profile, self.root, profile.staging)
        return json.loads((self.root / 'model-manifest.json').read_text())

    def test_profile_supplies_revision_extensions_and_validation(self):
        state = self.stage(self.profile())
        self.assertEqual(self.revisions, ['b' * 40])
        self.assertEqual(sorted(name for _, name, _ in self.downloads),
                         ['config.json', 'extra.bin', 'weights.safetensors'])
        self.assertEqual({revision for _, _, revision in self.downloads}, {'b' * 40})
        self.assertEqual(self.validated, [self.root / 'model'])
        self.assertEqual(state['revision'], 'b' * 40)
        self.assertEqual(state['checked_by'], 'fake-profile')
        self.assertTrue(state['complete'])

    def test_pinned_profile_refuses_manifest_at_another_revision(self):
        (self.root / 'model-manifest.json').write_text(json.dumps(
            {'model_id': 'example/model', 'revision': 'c' * 40, 'files': {},
             'complete': True}))
        with redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'another revision'):
                self.module.stage(self.profile(), self.root, self.root.parent / 'local')
        self.assertEqual(self.downloads, [])

    def test_config_must_name_the_profile_model(self):
        with self.assertRaisesRegex(ValueError, 'requires the fake checkpoint'):
            self.stage(self.profile(model_id='example/other'))

    def test_selects_profile_from_environment(self):
        self.assertIs(self.module.select_profile({}), self.module.PROFILES['bf16'])
        self.assertIs(self.module.select_profile({'MODEL_PROFILE': 'bf16'}),
                      self.module.PROFILES['bf16'])
        with self.assertRaisesRegex(ValueError, 'Unknown model profile'):
            self.module.select_profile({'MODEL_PROFILE': 'nope'})

    def test_bf16_profile_keeps_current_settings(self):
        bf16 = self.module.PROFILES['bf16']
        self.assertEqual((bf16.model_id, bf16.revision, bf16.root_env, bf16.default_root,
                          bf16.staging),
                         ('Qwen/Qwen3.8-27B', None, 'BF16_ROOT', Path('/workspace/qwen-bf16'),
                          Path('/opt/qwen-bf16/download')))


PINNED_4BIT = '71dface38d4cbfa34121f1d446d4a0fa1025f13c'


def safetensors(tensors):
    """A tiny safetensors file holding the given {name: dtype} tensors."""
    header = {name: {'dtype': dtype, 'shape': [1], 'data_offsets': [0, 4]}
              for name, dtype in tensors.items()}
    raw = json.dumps(header).encode()
    return struct.pack('<Q', len(raw)) + raw + b'\0' * 4


def int4_config(**weights):
    """config.json of the reviewed RedHat checkpoint: compressed-tensors W4A16."""
    group = {'targets': ['Linear'],
             'weights': {'num_bits': 4, 'type': 'int', 'symmetric': True,
                         'group_size': 128, 'strategy': 'group', 'actorder': 'static',
                         **weights},
             'input_activations': None, 'output_activations': None}
    return {'architectures': ['Qwen3_5ForConditionalGeneration'],
            'quantization_config': {
                'quant_method': 'compressed-tensors', 'format': 'pack-quantized',
                'quantization_status': 'compressed', 'config_groups': {'group_0': group},
                'ignore': ['lm_head', 'model.visual.blocks.0.attn.qkv'],
                'kv_cache_scheme': {'num_bits': 8, 'type': 'float', 'strategy': 'tensor'}}}


INT4_TENSORS = {'model.language_model.layers.0.mlp.down_proj.weight_packed': 'I32',
                'model.language_model.layers.0.mlp.down_proj.weight_scale': 'BF16',
                'model.language_model.embed_tokens.weight': 'BF16',
                'model.visual.blocks.0.attn.qkv.weight': 'BF16'}


class FourBitStagingTests(unittest.TestCase):
    """The 4bit profile stages only the reviewed compressed-tensors W4A16 checkpoint."""

    def setUp(self):
        self.hub_sha = PINNED_4BIT
        self.revisions = []
        self.data = {'config.json': json.dumps(int4_config()).encode(),
                     'model.safetensors': safetensors(INT4_TENSORS),
                     'tokenizer.json': b'{}'}
        fake = types.ModuleType('huggingface_hub')

        def info(*args, **kwargs):
            self.revisions.append(kwargs.get('revision'))
            siblings = [types.SimpleNamespace(
                rfilename=name, size=len(value), lfs=types.SimpleNamespace(
                    sha256=hashlib.sha256(value).hexdigest()))
                for name, value in self.data.items()]
            return types.SimpleNamespace(sha=self.hub_sha, siblings=siblings)

        def download(model_id, filename, revision, local_dir):
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.data[filename])
            return str(target)

        fake.HfApi = lambda: types.SimpleNamespace(model_info=info)
        fake.hf_hub_download = download
        with patch.dict(sys.modules, {'huggingface_hub': fake}):
            spec = importlib.util.spec_from_file_location(
                'prepare_model_4bit_check', Path(__file__).with_name('prepare_model.py'))
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory(prefix='qwen-4bit-check-', dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'volume'
        self.root.mkdir()
        (self.root / 'config.json').write_text('{"model_id":"RedHatAI/Qwen3.8-27B-INT4"}')
        self.profile = self.module.PROFILES['4bit']

    def stage(self, profile=None):
        with redirect_stdout(io.StringIO()):
            self.module.stage(profile or self.profile, self.root, self.root.parent / 'local')
        return json.loads((self.root / 'model-manifest.json').read_text())

    def manifest(self):
        return json.loads((self.root / 'model-manifest.json').read_text())

    def test_4bit_profile_pins_the_reviewed_revision_on_its_own_volume(self):
        self.assertEqual((self.profile.model_id, self.profile.revision, self.profile.root_env,
                          self.profile.default_root, self.profile.staging),
                         ('RedHatAI/Qwen3.8-27B-INT4', PINNED_4BIT, 'QWEN_4BIT_ROOT',
                          Path('/workspace/qwen-4bit'), Path('/opt/qwen-4bit/download')))
        self.assertIs(self.module.select_profile({'MODEL_PROFILE': '4bit'}), self.profile)

    def test_stages_w4a16_checkpoint_with_vision_tower_at_the_pin(self):
        state = self.stage()
        self.assertTrue(state['complete'])
        self.assertEqual(state['revision'], PINNED_4BIT)
        self.assertEqual(self.revisions, [PINNED_4BIT])
        self.assertEqual(state['quantization'], 'compressed-tensors W4A16')
        self.assertEqual(state['tensor_dtypes'], {'I32': 1, 'BF16': 3})

    def test_refuses_a_hub_revision_other_than_the_pin(self):
        self.hub_sha = 'd' * 40
        with self.assertRaisesRegex(ValueError, 'pinned revision'):
            self.stage()
        self.assertFalse((self.root / 'model').exists())

    def test_refuses_to_stage_without_a_pinned_revision(self):
        with self.assertRaisesRegex(ValueError, 'pinned revision'):
            self.stage(self.module.Profile(**{**self.profile.__dict__, 'revision': None}))
        self.assertEqual(self.revisions, [])

    def test_refuses_checkpoints_that_are_not_compressed_tensors_w4a16(self):
        bf16 = {'architectures': ['Qwen3_5ForConditionalGeneration']}
        awq = {**int4_config(), 'quantization_config': {'quant_method': 'awq', 'bits': 4}}
        w8 = int4_config(num_bits=8)
        w4a8 = int4_config()
        w4a8['quantization_config']['config_groups']['group_0']['input_activations'] = {
            'num_bits': 8, 'type': 'int'}
        cases = {'unquantized': (bf16, {'model.visual.blocks.0.weight': 'BF16',
                                        'model.language_model.layers.0.weight': 'BF16'}),
                 'awq': (awq, INT4_TENSORS), 'eight-bit weights': (w8, INT4_TENSORS),
                 'quantized activations': (w4a8, INT4_TENSORS),
                 'no packed weights': (int4_config(), {'model.visual.blocks.0.weight': 'BF16',
                                                       'model.language_model.layers.0.weight': 'BF16'})}
        for name, (config, tensors) in cases.items():
            with self.subTest(name):
                (self.root / 'model-manifest.json').unlink(missing_ok=True)
                self.data['config.json'] = json.dumps(config).encode()
                self.data['model.safetensors'] = safetensors(tensors)
                with self.assertRaisesRegex(ValueError, 'W4A16'):
                    self.stage()
                self.assertFalse(self.manifest()['complete'])

    def test_refuses_a_checkpoint_without_the_vision_tower(self):
        self.data['model.safetensors'] = safetensors(
            {name: dtype for name, dtype in INT4_TENSORS.items() if 'visual' not in name})
        with self.assertRaisesRegex(ValueError, 'vision tower'):
            self.stage()
        self.assertFalse(self.manifest()['complete'])


if __name__ == '__main__':
    unittest.main()
