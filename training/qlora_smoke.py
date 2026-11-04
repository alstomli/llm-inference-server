"""A small QLoRA pipeline check using the cached, original BF16 checkpoint.

This deliberately does not train a production behavior or update the source
checkpoint. Only the reviewed completion contributes to the training loss.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

BASE_ID = 'Qwen/Qwen3.8-27B'
BASE_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SYSTEM = 'You are Qwen, a helpful personal assistant. Answer clearly and directly. Use Markdown when helpful.'
SPECIAL_MARKERS = ('<|im_start|>', '<|im_end|>', '<think>', '</think>')


def checkpoint_identity(root):
    state = json.loads((root / 'model-manifest.json').read_text())
    if (state.get('model_id') != BASE_ID or state.get('revision') != BASE_REVISION
            or state.get('complete') is not True):
        raise ValueError('Expected the complete, pinned original Qwen BF16 checkpoint')
    if set(state.get('tensor_dtypes', {})) != {'BF16'}:
        raise ValueError('The source manifest must contain only BF16 weights')
    config_path = root / 'model/config.json'
    config = json.loads(config_path.read_text())
    if config.get('quantization_config') or config.get('model_type') != 'qwen3_5':
        raise ValueError('Expected the original, unquantized qwen3_5 checkpoint')
    shards = list((root / 'model').glob('*.safetensors'))
    if not shards:
        raise ValueError('No source weight shards found')
    for path in shards + [config_path]:
        record = state.get('files', {}).get(path.name)
        if not record or record.get('size') != path.stat().st_size:
            raise ValueError(f'Source file size does not match the verified manifest: {path.name}')
    return {'model_id': BASE_ID, 'revision': BASE_REVISION,
            'source_dtype': 'BF16', 'shards': len(shards),
            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}


def read_examples(path):
    examples, identifiers = [], set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f'Line {line_number}: expected a JSON object')
        identifier, prompt, completion = row.get('id'), row.get('prompt'), row.get('completion')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f'Line {line_number}: each example needs a unique nonempty id')
        if not isinstance(prompt, list) or not prompt or not isinstance(completion, str) or not completion.strip():
            raise ValueError(f'Line {line_number}: expected a prompt message list and a completion string')
        if any(not isinstance(message, dict) for message in prompt):
            raise ValueError(f'Line {line_number}: expected message objects')
        conversation = prompt[1:] if prompt[0].get('role') == 'system' else prompt
        expected = ['user' if i % 2 == 0 else 'assistant' for i in range(len(conversation))]
        if not conversation or [m.get('role') for m in conversation] != expected or expected[-1] != 'user':
            raise ValueError(f'Line {line_number}: prompt must alternate user/assistant and end with user')
        for message in prompt:
            if (message.get('role') not in ('system', 'user', 'assistant')
                    or not isinstance(message.get('content'), str) or not message['content'].strip()):
                raise ValueError(f'Line {line_number}: invalid prompt message')
        if any(marker in completion for marker in SPECIAL_MARKERS):
            raise ValueError(f'Line {line_number}: write the completion as plain answer text')
        examples.append({'id': identifier, 'prompt': prompt, 'completion': completion})
        identifiers.add(identifier)
    if not examples:
        raise ValueError('Dataset is empty')
    return examples


def tokenize_examples(tokenizer, examples, max_length):
    end_ids = tokenizer.encode('<|im_end|>', add_special_tokens=False)
    if len(end_ids) != 1:
        raise ValueError('Expected Qwen im_end to be a single token')
    rows, counts = [], []
    for example in examples:
        prompt = example['prompt']
        if prompt[0]['role'] != 'system':
            prompt = [{'role': 'system', 'content': SYSTEM}] + prompt
        # Match the gateway's non-thinking generation prefix. Tokenize the target
        # separately so the prompt token sequence is exactly the inference input.
        prefix = tokenizer.apply_chat_template(prompt, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
        answer_ids = tokenizer.encode(example['completion'], add_special_tokens=False) + end_ids
        input_ids = prompt_ids + answer_ids
        if len(input_ids) > max_length:
            raise ValueError(f'{example["id"]} has {len(input_ids)} tokens; no silent truncation is allowed')
        rows.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids),
                     'labels': [-100] * len(prompt_ids) + answer_ids})
        counts.append({'id': example['id'], 'prompt_tokens': len(prompt_ids),
                       'target_tokens': len(answer_ids), 'total_tokens': len(input_ids)})
    return rows, counts


class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        width = max(len(row['input_ids']) for row in rows)
        padding = {'input_ids': self.pad_id, 'attention_mask': 0, 'labels': -100}
        return {key: torch.tensor([row[key] + [value] * (width - len(row[key])) for row in rows],
                                  dtype=torch.long) for key, value in padding.items()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bf16-root', type=Path, default=Path('/workspace/qwen-bf16'))
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'examples/smoke.jsonl')
    parser.add_argument('--output', type=Path, default=Path('/workspace/qlora-smoke/runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--validate-only', action='store_true', help='Validate source and data on CPU; do not load model weights')
    args = parser.parse_args()
    if not 1 <= args.steps <= 32 or not 64 <= args.max_length <= 2048:
        parser.error('This smoke test allows 1–32 steps and 64–2048 tokens')
    identity = checkpoint_identity(args.bf16_root)
    examples = read_examples(args.data)
    if not args.validate_only:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        free, total = torch.cuda.mem_get_info()
        if free < 32 * 2**30:
            raise RuntimeError(f'Only {free / 2**30:.1f} GiB is free. Stop BF16 serving before training; do not kill it through the supervised shell.')
        # Patch training libraries before importing Transformers model classes.
        from unsloth import FastModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.bf16_root / 'model'), local_files_only=True,
                                              trust_remote_code=False)
    rows, counts = tokenize_examples(tokenizer, examples, args.max_length)
    report = {'schema_version': 1, 'purpose': 'pipeline-smoke-test-only', 'base': identity,
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'examples': counts, 'thinking': False, 'loss': 'completion-only',
              'max_length': args.max_length, 'steps': args.steps, 'seed': 42}
    if args.validate_only:
        print(json.dumps(report, indent=2))
        return

    from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if args.output.exists():
        raise ValueError('Use a new output directory; existing runs are never overwritten')
    args.output.mkdir(parents=True)
    report.update({'status': 'started', 'gpu': torch.cuda.get_device_name(),
                   'total_vram_gib': total / 2**30, 'initial_free_vram_gib': free / 2**30,
                   'quantization': {'storage': 'nf4', 'compute': 'bfloat16', 'double_quantization': True},
                   'lora': {'rank': 8, 'alpha': 16, 'dropout': 0},
                   'packages': {name: importlib.metadata.version(name) for name in
                                ['unsloth', 'unsloth-zoo', 'torch', 'transformers', 'peft', 'bitsandbytes']}})
    write_json(args.output / 'run.json', report)
    set_seed(42)
    torch.cuda.reset_peak_memory_stats()
    try:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model, _ = FastModel.from_pretrained(
            model_name=str(args.bf16_root / 'model'), use_exact_model_name=True,
            local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            load_in_4bit=True, full_finetuning=False, fast_inference=False,
            quantization_config=quantization, max_seq_length=args.max_length)
        import bitsandbytes as bnb
        quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        if not quantized or not all(m.compute_dtype == torch.bfloat16 for m in quantized):
            raise RuntimeError('Expected actual 4bit linear modules with BF16 computation')
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=8, lora_alpha=16, lora_dropout=0, bias='none',
            use_gradient_checkpointing='unsloth', random_state=42)
        trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not trainable or any('lora_' not in name for name, _ in trainable):
            raise RuntimeError('Only LoRA adapter parameters may be trainable')
        sample_name, sample = next((name, param) for name, param in trainable if '.lora_B.' in name)
        before = sample.detach().cpu().clone()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids('<|im_end|>')
        trainer = Trainer(
            model=model, train_dataset=rows, data_collator=CompletionCollator(pad_id),
            args=TrainingArguments(
                output_dir=str(args.output / 'checkpoints'), max_steps=args.steps,
                per_device_train_batch_size=1, gradient_accumulation_steps=1,
                learning_rate=2e-4, lr_scheduler_type='constant', warmup_steps=0,
                bf16=True, fp16=False, optim='adamw_8bit', logging_steps=1,
                save_strategy='no', report_to='none', seed=42, data_seed=42,
                remove_unused_columns=False, dataloader_num_workers=0,
                gradient_checkpointing=True))
        result = trainer.train()
        if not math.isfinite(result.training_loss) or torch.equal(before, sample.detach().cpu()):
            raise RuntimeError('Expected finite training loss and an actual adapter weight update')
        model.save_pretrained(str(args.output / 'adapter'), safe_serialization=True)
        tokenizer.save_pretrained(str(args.output / 'tokenizer'))
        if not (args.output / 'adapter/adapter_model.safetensors').is_file():
            raise RuntimeError('Adapter weights were not saved')
        report.update({'status': 'completed', 'global_steps': trainer.state.global_step,
                       'training_loss': result.training_loss, 'metrics': result.metrics,
                       'loss_history': trainer.state.log_history,
                       'trainable_parameters': sum(p.numel() for _, p in trainable),
                       'quantized_linear_modules': len(quantized), 'adapter_sample': sample_name,
                       'adapter_sample_changed': True,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30})
        print(json.dumps(report, indent=2))
    except Exception as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        write_json(args.output / 'run.json', report)


if __name__ == '__main__':
    main()
