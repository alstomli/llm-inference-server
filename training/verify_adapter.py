"""Reload a smoke adapter in a fresh process and check a forward/generation pass."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from qlora_tune import MIN_FREE_VRAM_GIB, checkpoint_identity, read_examples, tokenize_examples, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--bf16-root', type=Path, default=Path('/workspace/qwen-bf16'))
    args = parser.parse_args()
    report = json.loads((args.run / 'run.json').read_text())
    identity = checkpoint_identity(args.bf16_root)
    if report.get('status') != 'completed' or report['base'] != identity:
        raise ValueError('Expected a completed smoke run from this exact source checkpoint')
    from unsloth import FastModel
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from safetensors.torch import load_file
    if torch.cuda.mem_get_info()[0] < MIN_FREE_VRAM_GIB * 2**30:
        raise RuntimeError('Release the serving allocation before reloading the training model')
    tokenizer = AutoTokenizer.from_pretrained(str(args.run / 'tokenizer'), local_files_only=True)
    example = read_examples(Path(__file__).parent / 'examples/smoke.jsonl')[0]
    rows, counts = tokenize_examples(tokenizer, [example], report['max_length'])
    base, _ = FastModel.from_pretrained(
        model_name=str(args.bf16_root / 'model'), use_exact_model_name=True,
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
        fast_inference=False, max_seq_length=report['max_length'])
    model = PeftModel.from_pretrained(base, str(args.run / 'adapter'), is_trainable=False,
                                     local_files_only=True)
    if model.active_adapter != 'default':
        raise RuntimeError('The saved adapter is not active')
    weights = args.run / 'adapter/adapter_model.safetensors'
    saved = load_file(str(weights), device='cpu')
    live = dict(model.named_parameters())
    for name, tensor in saved.items():
        key = name.replace('.lora_A.weight', '.lora_A.default.weight').replace('.lora_B.weight', '.lora_B.default.weight')
        if key not in live or not torch.equal(live[key].detach().cpu(), tensor.to(live[key].dtype)):
            raise RuntimeError(f'Saved adapter tensor did not reload exactly: {name}')
    row = {key: torch.tensor([value], device='cuda') for key, value in rows[0].items()}
    model.eval()
    with torch.inference_mode():
        loss = float(model(**row).loss)
    if not math.isfinite(loss):
        raise RuntimeError('Reloaded adapter produced a non-finite completion loss')
    FastModel.for_inference(model)
    prompt_length = counts[0]['prompt_tokens']
    prompt = row['input_ids'][:, :prompt_length]
    end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    with torch.inference_mode():
        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                max_new_tokens=128, do_sample=False, use_cache=True,
                                eos_token_id=end_id, pad_token_id=end_id)
    text = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    if not text.strip():
        raise RuntimeError('Reloaded adapter produced an empty generation')
    result = {'status': 'completed', 'check': 'fresh-process-reload-only', 'base': identity,
              'adapter_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
              'adapter_tensors_matched': len(saved),
              'example_id': example['id'], 'completion_loss': loss,
              'generated_tokens': output.shape[-1] - prompt_length,
              'max_new_tokens': 128, 'sample_answer': text,
              'ended_with_eos': int(output[0, -1]) == end_id,
              'quality_improvement_evaluated': False}
    write_json(args.run / 'reload.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
