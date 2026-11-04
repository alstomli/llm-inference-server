"""CPU checks for dataset isolation, run provenance and interruption recovery."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from qlora_smoke import (check_run_directory, parse_args, prompt_key, read_examples,
                         run_contract, split_examples, trainer_settings, write_json)


def example(identifier, question=None):
    return {'id': identifier, 'prompt': [{'role': 'user', 'content': question or identifier}],
            'completion': 'A reviewed answer'}


class TrainingTests(unittest.TestCase):
    def test_json_array_and_jsonl_are_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            array, lines = Path(tmp) / 'train.json', Path(tmp) / 'smoke.jsonl'
            rows = [example('a'), example('b')]
            array.write_text(json.dumps(rows), encoding='utf-8-sig')
            lines.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            self.assertEqual(read_examples(array), read_examples(lines))
            array.write_text(json.dumps({'examples': rows}))
            with self.assertRaisesRegex(ValueError, 'JSON array'):
                read_examples(array)
            array.write_text('[]')
            with self.assertRaisesRegex(ValueError, 'empty'):
                read_examples(array)

    def test_missing_data_does_not_fall_back_to_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Dataset not found'):
                read_examples(Path(tmp) / 'train.json')
        args = parse_args([])
        self.assertEqual(args.data.name, 'train.json')
        self.assertIsNone(args.steps)

    def test_split_is_seeded_and_keeps_prompt_variants_together(self):
        rows = [example(str(i), 'same prompt' if i < 3 else str(i)) for i in range(20)]
        training, validation = split_examples(rows, 0.2, 42)
        self.assertEqual((training, validation), split_examples(rows, 0.2, 42))
        other_train, other_eval = split_examples(list(reversed(rows)), 0.2, 42)
        self.assertEqual({e['id'] for e in validation}, {e['id'] for e in other_eval})
        self.assertFalse({prompt_key(e) for e in training} & {prompt_key(e) for e in validation})
        self.assertEqual(len(training) + len(validation), len(rows))
        self.assertTrue(training and validation and other_train)

    def test_explicit_validation_prevents_shared_id_and_prompt(self):
        with self.assertRaisesRegex(ValueError, 'IDs'):
            split_examples([example('a')], 0.1, 42, [example('a', 'different')])
        with self.assertRaisesRegex(ValueError, 'prompts overlap'):
            split_examples([example('a', 'same')], 0.1, 42, [example('b', 'same')])
        self.assertEqual(split_examples([example('a')], 0.1, 42, [example('b')]),
                         ([example('a')], [example('b')]))
        with self.assertRaisesRegex(ValueError, 'distinct prompts'):
            split_examples([example('a', 'same'), example('b', 'same')], 0.1, 42)

    def test_invalid_training_settings_fail_before_loading_model(self):
        for flags in (['--learning-rate', 'nan'], ['--epochs', '0'], ['--steps', '-1'],
                      ['--validation-fraction', '1'], ['--rank', '0'], ['--alpha', '0'],
                      ['--save-steps', '0'], ['--save-total-limit', '1'], ['--max-length', '99999'],
                      ['--seed', '-1'],
                      ['--smoke-test', '--eval-data', 'eval.json']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_epoch_training_and_step_override_compute_warmup(self):
        args = parse_args(['--epochs', '3', '--warmup-ratio', '0.1'])
        settings = trainer_settings(args, 17)  # ceil(17 / 4) * 3 = 15 updates
        self.assertEqual(settings['warmup_steps'], 2)
        self.assertEqual(settings['max_steps'], -1)
        self.assertTrue(settings['load_best_model_at_end'])
        self.assertFalse(settings['save_only_model'])
        args = parse_args(['--steps', '100', '--warmup-ratio', '0.1', '--save-steps', '10'])
        settings = trainer_settings(args, 17)
        self.assertEqual(settings['warmup_steps'], 10)
        self.assertEqual(settings['eval_steps'], settings['save_steps'])
        self.assertEqual(trainer_settings(parse_args(['--steps', '1']), 1)['warmup_steps'], 0)
        smoke = trainer_settings(parse_args(['--smoke-test', '--gradient-accumulation-steps', '1']), 8)
        self.assertEqual(smoke['max_steps'], 8)
        self.assertEqual(smoke['save_strategy'], 'no')

    def test_resume_preserves_configuration_and_requires_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'train.json'
            write_json(data, [example('a'), example('b')])
            output = root / 'run'
            args = parse_args(['--data', str(data), '--output', str(output)])
            training, evaluation = split_examples(read_examples(data), 0.1, 42)
            contract = run_contract(args, {'revision': 'fixture'}, training, evaluation)
            self.assertIsNone(check_run_directory(args, contract))
            checkpoint = output / 'checkpoints/checkpoint-1'
            checkpoint.mkdir(parents=True)
            for name in ('trainer_state.json', 'adapter_config.json', 'adapter_model.safetensors',
                         'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                (checkpoint / name).write_text('{}')
            write_json(output / 'run.json', {'status': 'interrupted', 'contract': contract})
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                check_run_directory(args, contract)
            args.resume_from_checkpoint = checkpoint
            self.assertEqual(check_run_directory(args, contract)['status'], 'interrupted')
            changed = dict(contract, data_sha256='changed')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                check_run_directory(args, changed)
            (checkpoint / 'optimizer.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'optimizer.pt'):
                check_run_directory(args, contract)
            inferred = parse_args(['--resume-from-checkpoint', str(checkpoint)])
            self.assertEqual(inferred.output, output.resolve())


if __name__ == '__main__':
    unittest.main()
