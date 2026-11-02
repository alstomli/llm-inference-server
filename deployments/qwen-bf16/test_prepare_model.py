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


if __name__ == '__main__':
    unittest.main()
