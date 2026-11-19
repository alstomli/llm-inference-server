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
        fields = dict(name='fake', model_id='example/model', revision='b' * 40,
                      root_env='FAKE_ROOT', default_root=str(self.root),
                      staging=str(self.root.parent / 'local'), label='fake',
                      validate=validate, extensions=('.safetensors', '.json', '.bin'))
        fields.update(overrides)
        return self.module.Profile(**fields)

    def stage(self, profile):
        with redirect_stdout(io.StringIO()):
            self.module.stage(profile, self.root, Path(profile.staging))
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
                         ('Qwen/Qwen3.8-27B', None, 'BF16_ROOT', '/workspace/qwen-bf16',
                          '/opt/qwen-bf16/download'))


if __name__ == '__main__':
    unittest.main()
