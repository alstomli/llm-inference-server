"""Optional installed-stack test: tiny random CPU model, no downloads or Qwen weights."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from qlora_tune import (CompletionCollator, check_run_directory, parse_args,
                         run_contract, trainer_settings, write_json)

HAS_STACK = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'peft'))


@unittest.skipUnless(HAS_STACK, 'Run in the QLoRA venv to test the installed Trainer/PEFT stack')
class TrainerIntegrationTests(unittest.TestCase):
    def test_checkpoint_resume_and_best_adapter_export(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments, set_seed
        from peft import LoraConfig, get_peft_model

        torch.set_num_threads(1)

        def model():
            set_seed(42)
            base = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=32, n_embd=8,
                                            n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2))
            return get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['c_attn'],
                                                  fan_in_fan_out=True, task_type='CAUSAL_LM'))

        class InterruptAfterCheckpoint(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                raise RuntimeError('test interruption after checkpoint')

        for schedule in ('epoch', 'steps'):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / 'train.json'
                data.write_text('[]')
                flags = ['--output', str(root / 'run'), '--data', str(data), '--epochs', '2',
                         '--gradient-accumulation-steps', '1']
                if schedule == 'steps':
                    flags += ['--steps', '3', '--save-steps', '1']
                args = parse_args(flags)
                contract = run_contract(args, {'revision': 'synthetic-cpu-test'}, [], [])
                settings = trainer_settings(args, 4)
                settings.update(use_cpu=True, bf16=False, optim='adamw_torch', gradient_checkpointing=False,
                                dataloader_pin_memory=False, disable_tqdm=True)
                rows = [{'input_ids': [1, 2, 3, 4, 5, 6], 'attention_mask': [1] * 6,
                         'labels': [-100, -100, -100, 4, 5, 6]} for _ in range(4)]

                def trainer(callbacks=None):
                    return Trainer(model=model(), args=TrainingArguments(**settings), train_dataset=rows,
                                   eval_dataset=rows[:1], data_collator=CompletionCollator(0), callbacks=callbacks)

                first = trainer([InterruptAfterCheckpoint()])
                initial = first.evaluate(metric_key_prefix='initial')['initial_loss']
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    first.train()
                checkpoint = next((args.output / 'checkpoints').glob('checkpoint-*'))
                write_json(args.output / 'run.json', {'status': 'interrupted', 'contract': contract})
                args.resume_from_checkpoint = checkpoint
                check_run_directory(args, contract)
                resumed = trainer()
                resumed.train(resume_from_checkpoint=str(checkpoint))
                self.assertEqual(resumed.state.global_step, 3 if schedule == 'steps' else 8)
                self.assertTrue(resumed.state.best_model_checkpoint)
                self.assertTrue(math.isfinite(initial))
                self.assertTrue(math.isfinite(resumed.evaluate()['eval_loss']))
                resumed.model.save_pretrained(str(args.output / 'adapter'), safe_serialization=True)
                self.assertTrue((args.output / 'adapter/adapter_model.safetensors').is_file())
                self.assertLessEqual(len(list((args.output / 'checkpoints').glob('checkpoint-*'))), 2)


if __name__ == '__main__':
    unittest.main()
