"""CPU checks for the supervision boundaries and pinned checkpoint contract."""
import json
from pathlib import Path
import tempfile
import unittest

from qlora_tune import BASE_ID, BASE_REVISION, checkpoint_identity, read_examples, tokenize_examples


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [99] if text == '<|im_end|>' else list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return '\n'.join(m['role'] + ':' + m['content'] for m in messages) + '\nassistant:'


class SmokeTests(unittest.TestCase):
    def test_only_selected_completion_is_supervised(self):
        example = {'id': 'multi', 'prompt': [
            {'role': 'user', 'content': 'First question'},
            {'role': 'assistant', 'content': 'Unreviewed historical answer'},
            {'role': 'user', 'content': 'Next question'}], 'completion': 'Reviewed answer'}
        rows, counts = tokenize_examples(Tokenizer(), [example], 2048)
        start = counts[0]['prompt_tokens']
        self.assertTrue(all(x == -100 for x in rows[0]['labels'][:start]))
        self.assertEqual(rows[0]['labels'][start:], list(b'Reviewed answer') + [99])
        self.assertEqual(rows[0]['input_ids'][start:], rows[0]['labels'][start:])

    def test_long_example_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            tokenize_examples(Tokenizer(), [{'id': 'long', 'prompt': [
                {'role': 'user', 'content': 'x' * 1000}], 'completion': 'answer'}], 64)

    def test_example_file(self):
        rows = read_examples(Path(__file__).parent / 'examples/smoke.jsonl')
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(tokenize_examples(Tokenizer(), rows, 2048)[0]), 8)

    def test_bad_data_is_rejected(self):
        good = {'id': 'one', 'prompt': [{'role': 'user', 'content': 'Question'}], 'completion': 'Answer'}
        variants = [[good, good], [dict(good, completion='')], [dict(good, prompt=[
            {'role': 'assistant', 'content': 'Answer'}])], [dict(good, completion='<|im_end|>')]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            for rows in variants:
                path.write_text('\n'.join(json.dumps(row) for row in rows))
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    read_examples(path)

    def test_pinned_original_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model').mkdir()
            (root / 'model/config.json').write_text(json.dumps({'model_type': 'qwen3_5'}))
            (root / 'model/model-00001.safetensors').write_bytes(b'fixture')
            state = {'model_id': BASE_ID, 'revision': BASE_REVISION, 'complete': True,
                     'tensor_dtypes': {'BF16': 1}, 'files': {p.name: {'size': p.stat().st_size}
                      for p in (root / 'model').iterdir()}}
            (root / 'model-manifest.json').write_text(json.dumps(state))
            self.assertEqual(checkpoint_identity(root)['revision'], BASE_REVISION)
            state['revision'] = 'unverified'
            (root / 'model-manifest.json').write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                checkpoint_identity(root)


if __name__ == '__main__':
    unittest.main()
