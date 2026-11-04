"""Validate local adapter metadata and complete tensor storage without Torch or a GPU."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('validate_lora', Path(__file__).with_name('validate-lora-adapter.py'))
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / 'model'
        self.path = self.root / 'adapter'
        self.path.mkdir()
        self.config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 8,
                       'base_model_name_or_path': str(self.base), 'bias': 'none'}
        self.header = {
            'layer.lora_A.weight': {'dtype': 'F32', 'shape': [8, 1], 'data_offsets': [0, 32]},
            'layer.lora_B.weight': {'dtype': 'F32', 'shape': [1, 8], 'data_offsets': [32, 64]},
        }
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.path / 'adapter_config.json').write_text(json.dumps(self.config))
        header = json.dumps(self.header).encode()
        (self.path / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 64)

    def validate(self, **kwargs):
        return lora.validate_adapter(self.path, self.base, kwargs.get('base_name', 'base'),
                                     kwargs.get('name', 'tuned'), kwargs.get('max_rank', 8))

    def test_valid_local_adapter_and_base_lineage(self):
        self.assertEqual(self.validate(), {'name': 'tuned', 'path': str(self.path.resolve()), 'base_model_name': 'base'})

    def test_alias_collision_and_invalid_rank_capacity(self):
        for kwargs in ({'name': 'base'}, {'name': ''}, {'max_rank': 1}, {'max_rank': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.validate(**kwargs)

    def test_wrong_base_and_unsupported_adapter_configuration(self):
        original = dict(self.config)
        for changes in ({'base_model_name_or_path': str(self.root / 'other')}, {'r': 16},
                        {'r': True}, {'peft_type': 'IA3'}, {'bias': 'all'}, {'use_dora': True},
                        {'modules_to_save': ['lm_head']}, {'task_type': 'SEQ_CLS'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config = {**original, **changes}
                self.write()
                self.validate()

    def test_truncated_or_missing_weights(self):
        path = self.path / 'adapter_model.safetensors'
        for data in (b'', b'123', struct.pack('<Q', 99999999), path.read_bytes()[:-1]):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                path.write_bytes(data)
                self.validate()
        path.unlink()
        with self.assertRaises(OSError):
            self.validate()

    def test_wrong_tensor_rank_missing_pair_and_overlapping_storage(self):
        original = json.loads(json.dumps(self.header))
        for change in ('rank', 'pair', 'overlap', 'full-weight'):
            self.header = json.loads(json.dumps(original))
            if change == 'rank':
                self.header['layer.lora_A.weight']['shape'] = [4, 2]
            elif change == 'pair':
                del self.header['layer.lora_B.weight']
            elif change == 'overlap':
                self.header['layer.lora_B.weight']['data_offsets'] = [0, 32]
            else:
                self.header['layer.weight'] = self.header.pop('layer.lora_B.weight')
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.write()
                self.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
