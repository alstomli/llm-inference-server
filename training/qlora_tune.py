"""Train LoRA adapters on the cached original BF16 checkpoint.

Defaults to examples/train.json, with validation and resumable checkpoints.
--smoke-test retains the short pipeline check. Only completions contribute to
loss; the source checkpoint is never updated. Quality needs separate review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random

os.environ.setdefault('UNSLOTH_DISABLE_STATISTICS', '1')

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
    if not path.is_file():
        raise ValueError(f'Dataset not found: {path}. Create a JSON array of id/prompt/completion examples; see training/README.md')
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() == '.json':
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError('Expected a JSON array of training examples')
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    examples, identifiers = [], set()
    for line_number, row in enumerate(records, 1):
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


def prompt_key(example):
    messages = example['prompt']
    if messages[0]['role'] != 'system':
        messages = [{'role': 'system', 'content': SYSTEM}] + messages
    return json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def split_examples(examples, fraction, seed, evaluation=None):
    """Keep identical conversation prompts together to prevent exact-prompt leakage."""
    if evaluation is not None:
        if {e['id'] for e in examples} & {e['id'] for e in evaluation}:
            raise ValueError('Training and validation IDs must be disjoint')
        if {prompt_key(e) for e in examples} & {prompt_key(e) for e in evaluation}:
            raise ValueError('Training and validation prompts overlap; move all variants to one split')
        return examples, evaluation
    keys = sorted({prompt_key(e) for e in examples})
    if len(keys) < 2:
        raise ValueError('Need at least two distinct prompts for a validation split, or supply --eval-data')
    random.Random(seed).shuffle(keys)
    count = min(len(keys) - 1, max(1, math.ceil(len(keys) * fraction)))
    held_out = set(keys[:count])
    return ([e for e in examples if prompt_key(e) not in held_out],
            [e for e in examples if prompt_key(e) in held_out])


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
    # Keep the previous report intact if a write is interrupted.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bf16-root', type=Path, default=Path('/workspace/qwen-bf16'))
    parser.add_argument('--base-precision', choices=('nf4', 'bf16'), default='nf4',
                        help='Frozen base precision: nf4 for QLoRA, bf16 for LoRA')
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'examples/train.json')
    parser.add_argument('--eval-data', type=Path, help='Separate validation JSON/JSONL; otherwise split --data')
    parser.add_argument('--validation-fraction', type=float, default=0.1)
    parser.add_argument('--output', type=Path, help='New run directory; inferred from checkpoint when resuming')
    parser.add_argument('--resume-from-checkpoint', type=Path, help='A checkpoint from this script and the same run configuration')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps', type=int, help='Optional optimizer-update limit, overriding epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=int, help='LoRA scaling; defaults to twice the rank')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--save-steps', type=int, help='Save and evaluate every N updates; default: each epoch')
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--smoke-test', action='store_true', help='No validation/checkpoints; short pipeline test only')
    parser.add_argument('--validate-only', action='store_true', help='Check source/data on CPU without loading model weights')
    args = parser.parse_args(argv)
    for name in ('epochs', 'batch_size', 'gradient_accumulation_steps', 'rank', 'max_length'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    if args.save_steps is not None and args.save_steps < 1:
        parser.error('--save-steps must be positive')
    if args.save_total_limit < 2:
        parser.error('--save-total-limit must be at least 2 (best and latest checkpoints)')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive')
    if not 0 < args.validation_fraction < 1 or not 0 <= args.dropout < 1 or not 0 <= args.warmup_ratio <= 1:
        parser.error('Invalid validation fraction, dropout or warmup ratio')
    if not 64 <= args.max_length <= 40960:
        parser.error('--max-length must be between 64 and the deployed context limit of 40960')
    args.alpha = args.alpha if args.alpha is not None else 2 * args.rank
    if args.alpha < 1:
        parser.error('--alpha must be positive')
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    if args.smoke_test:
        if args.eval_data or args.resume_from_checkpoint:
            parser.error('--smoke-test does not use validation data or resumable checkpoints')
        args.steps = args.steps or 8
    if args.output is None:
        args.output = (args.resume_from_checkpoint.resolve().parent.parent if args.resume_from_checkpoint
                       else Path('/workspace/qlora-smoke/runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    return args


def run_contract(args, identity, training, evaluation):
    """Fields that must match when continuing an interrupted run."""
    return {
        'base': identity, 'bf16_root': str(args.bf16_root.resolve()),
        'base_precision': args.base_precision,
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'eval_data_sha256': hashlib.sha256(args.eval_data.read_bytes()).hexdigest() if args.eval_data else None,
        'training_ids': [e['id'] for e in training], 'validation_ids': [e['id'] for e in evaluation],
        'settings': {name: getattr(args, name) for name in (
            'epochs', 'steps', 'batch_size', 'gradient_accumulation_steps', 'learning_rate',
            'rank', 'alpha', 'dropout', 'warmup_ratio', 'max_length', 'seed',
            'save_steps', 'save_total_limit', 'validation_fraction', 'smoke_test')},
    }


def check_run_directory(args, contract):
    if not args.resume_from_checkpoint:
        if args.output.exists():
            raise ValueError('Use a new output directory; existing runs are never overwritten')
        return None
    checkpoint = args.resume_from_checkpoint.resolve()
    if checkpoint.parent != args.output.resolve() / 'checkpoints' or not checkpoint.name.startswith('checkpoint-'):
        raise ValueError('Resume checkpoint must belong to this output directory under checkpoints/checkpoint-N')
    for name in ('trainer_state.json', 'adapter_config.json', 'adapter_model.safetensors',
                 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Incomplete resume checkpoint: missing {name}')
    previous = json.loads((args.output / 'run.json').read_text(encoding='utf-8'))
    saved_contract = dict(previous.get('contract') or {})
    saved_contract.setdefault('base_precision', 'nf4')  # Runs before the flag used NF4.
    if saved_contract != contract:
        raise ValueError('Resume configuration/data do not match the saved run; use the original arguments and files')
    if previous.get('status') == 'completed':
        raise ValueError('This run is already completed; start a new run instead')
    return previous


def trainer_settings(args, training_count):
    updates_per_epoch = math.ceil(math.ceil(training_count / args.batch_size) / args.gradient_accumulation_steps)
    planned_steps = args.steps or args.epochs * updates_per_epoch
    settings = dict(
        output_dir=str(args.output / 'checkpoints'), max_steps=args.steps or -1,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, lr_scheduler_type='constant' if args.smoke_test else 'linear',
        warmup_steps=0 if args.smoke_test else min(planned_steps - 1, math.ceil(planned_steps * args.warmup_ratio)),
        bf16=True, fp16=False, optim='adamw_8bit', logging_steps=1,
        report_to='none', seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0, gradient_checkpointing=True,
    )
    if args.smoke_test:
        settings['save_strategy'] = 'no'
    else:
        strategy = 'steps' if args.save_steps else 'epoch'
        settings.update(
            eval_strategy=strategy, save_strategy=strategy,
            eval_steps=args.save_steps, save_steps=args.save_steps or 500,
            save_total_limit=args.save_total_limit, save_only_model=False,
            load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, prediction_loss_only=True,
            per_device_eval_batch_size=1, label_names=['labels'],
        )
    return settings


def main():
    args = parse_args()
    quantize_base = args.base_precision == 'nf4'
    environment = Path(os.environ.get('QLORA_ENV_ROOT', '/opt/qlora'))
    for name, value in {'HF_HOME': str(environment / 'hf-cache'),
                        'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                        'TORCHINDUCTOR_CACHE_DIR': str(environment / 'torch-cache'),
                        'TRITON_CACHE_DIR': str(environment / 'triton-cache')}.items():
        os.environ.setdefault(name, value)
    identity = checkpoint_identity(args.bf16_root)
    examples = read_examples(args.data)
    if args.smoke_test:
        training, evaluation = examples, []
    else:
        evaluation = read_examples(args.eval_data) if args.eval_data else None
        training, evaluation = split_examples(examples, args.validation_fraction, args.seed, evaluation)
    contract = run_contract(args, identity, training, evaluation)
    previous = check_run_directory(args, contract)
    if not args.validate_only:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A CUDA GPU with BF16 support is required')
        if torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('This runner supports one visible GPU and one process; select one GPU with CUDA_VISIBLE_DEVICES')
        free, total = torch.cuda.mem_get_info()
        minimum_free_gib = 32 if quantize_base else 64
        if free < minimum_free_gib * 2**30:
            raise RuntimeError(f'Only {free / 2**30:.1f} GiB is free; {args.base_precision} needs '
                               f'at least {minimum_free_gib} GiB to start. Use the training-only Pod entrypoint.')
        # Patch training libraries before importing Transformers model classes.
        from unsloth import FastModel
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.bf16_root / 'model'), local_files_only=True,
                                              trust_remote_code=False)
    rows, counts = tokenize_examples(tokenizer, training, args.max_length)
    eval_rows, eval_counts = tokenize_examples(tokenizer, evaluation, args.max_length)
    purpose = 'qlora-finetuning' if quantize_base else 'lora-finetuning'
    report = {'schema_version': 2, 'purpose': 'pipeline-smoke-test-only' if args.smoke_test else purpose,
              'base': identity, 'contract': contract, 'data_sha256': contract['data_sha256'],
              'examples': counts, 'validation_examples': eval_counts,
              'training_examples': len(rows), 'validation_example_count': len(eval_rows),
              'thinking': False, 'loss': 'completion-only', 'quality_improvement_evaluated': False,
              'max_length': args.max_length, 'steps': args.steps, 'seed': args.seed,
              'settings': contract['settings'], 'output': str(args.output),
              'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
              'warmup_steps': trainer_settings(args, len(rows))['warmup_steps']}
    if args.validate_only:
        print(json.dumps(report, indent=2))
        return

    from transformers import BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    if not args.resume_from_checkpoint:
        args.output.mkdir(parents=True)
        write_json(args.output / 'train.json', training)
        write_json(args.output / 'validation.json', evaluation)
    report['resume_history'] = (previous or {}).get('resume_history', [])
    if args.resume_from_checkpoint:
        report['resume_history'].append({'checkpoint': str(args.resume_from_checkpoint),
                                         'at': datetime.now(timezone.utc).isoformat()})
    report.update({'status': 'started', 'gpu': torch.cuda.get_device_name(),
                   'total_vram_gib': total / 2**30, 'initial_free_vram_gib': free / 2**30,
                   'quantization': {'storage': args.base_precision, 'compute': 'bfloat16',
                                    'double_quantization': quantize_base},
                   'lora': {'rank': args.rank, 'alpha': args.alpha, 'dropout': args.dropout},
                   'packages': {name: importlib.metadata.version(name) for name in
                                ['unsloth', 'unsloth-zoo', 'torch', 'transformers', 'peft', 'bitsandbytes']}})
    write_json(args.output / 'run.json', report)
    set_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    try:
        load_options = {'load_in_4bit': quantize_base}
        if quantize_base:
            load_options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            load_options['load_in_16bit'] = True
        model, _ = FastModel.from_pretrained(
            model_name=str(args.bf16_root / 'model'), use_exact_model_name=True,
            local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            full_finetuning=False, fast_inference=False, max_seq_length=args.max_length,
            **load_options)
        import bitsandbytes as bnb
        quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
        if quantize_base:
            if not quantized or not all(m.compute_dtype == torch.bfloat16 for m in quantized):
                raise RuntimeError('Expected actual 4bit linear modules with BF16 computation')
        elif (quantized or getattr(model, 'is_loaded_in_8bit', False)
              or model.get_input_embeddings().weight.dtype != torch.bfloat16):
            raise RuntimeError('Expected an unquantized BF16 base for LoRA')
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias='none',
            use_gradient_checkpointing='unsloth', random_state=args.seed)
        trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not trainable or any('lora_' not in name for name, _ in trainable):
            raise RuntimeError('Only LoRA adapter parameters may be trainable')
        sample_name, sample = next((name, param) for name, param in trainable if '.lora_B.' in name)
        before = sample.detach().cpu().clone()
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids('<|im_end|>')
        trainer = Trainer(
            model=model, train_dataset=rows, eval_dataset=eval_rows or None,
            data_collator=CompletionCollator(pad_id),
            args=TrainingArguments(**trainer_settings(args, len(rows))))
        # Evaluate the newly initialized adapter (equivalent to the frozen base)
        # before training, then compare with the selected trained checkpoint.
        if eval_rows:
            if previous and 'initial_validation_loss' in previous:
                report['initial_validation_loss'] = previous['initial_validation_loss']
            else:
                initial = trainer.evaluate(metric_key_prefix='initial')
                report['initial_validation_loss'] = initial['initial_loss']
                if not math.isfinite(report['initial_validation_loss']):
                    raise RuntimeError('Initial validation loss is not finite')
            write_json(args.output / 'run.json', report)
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        if not math.isfinite(result.training_loss) or torch.equal(before, sample.detach().cpu()):
            raise RuntimeError('Expected finite training loss and an actual adapter weight update')
        if eval_rows:
            if not trainer.state.best_model_checkpoint:
                raise RuntimeError('No evaluated checkpoint was saved; reduce --save-steps or use epoch checkpointing')
            final = trainer.evaluate()
            if not math.isfinite(final['eval_loss']):
                raise RuntimeError('Final validation loss is not finite')
            report.update({'validation_loss': final['eval_loss'],
                           'validation_loss_delta': final['eval_loss'] - report['initial_validation_loss'],
                           'best_checkpoint': trainer.state.best_model_checkpoint,
                           'best_validation_loss': trainer.state.best_metric})
        trainer.model.save_pretrained(str(args.output / 'adapter'), safe_serialization=True)
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
    except (Exception, KeyboardInterrupt) as exc:
        report.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    finally:
        if 'trainer' in locals():
            report.update({'global_steps': trainer.state.global_step,
                           'loss_history': trainer.state.log_history})
        report.update({'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                       'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30})
        write_json(args.output / 'run.json', report)


if __name__ == '__main__':
    main()
