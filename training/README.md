# QLoRA training on the original BF16 checkpoint

`qlora_tune.py` supports a full training run and an explicit `--smoke-test` mode.
It trains separate language LoRA adapters;
it never changes the cached original checkpoint or activates an adapter in chat.
Use the existing `setup-qlora.sh` environment and the training-only Pod mode
described below. This is a single-GPU runner, not a distributed training setup.

## Prepare your dataset

The default input is `examples/train.json`, relative to this script, regardless
of your terminal's current directory. Create it on the Pod at:

```text
/workspace/qlora-smoke/source/training/examples/train.json
```

Use a **JSON array**, not JSONL or an OpenAI messages-only export:

```json
[
  {
    "id": "train-001",
    "prompt": [{"role": "user", "content": "What is TTFT?"}],
    "completion": "TTFT is the time from a request until the first generated token is returned. Specify the measurement boundary when comparing results."
  },
  {
    "id": "train-002",
    "prompt": [{"role": "user", "content": "What does the KV cache contain?"}],
    "completion": "The KV cache holds attention keys and values for earlier tokens so later generation can reuse them. It uses GPU memory and does not update model weights."
  }
]
```

These two records demonstrate the format, not a useful training dataset. Supply
reviewed answers for your chosen behavior. IDs must be unique. An optional
system message is followed by alternating user/assistant messages ending in a
user message. `completion` is the desired final answer as plain text, without
chat control tokens or thinking tags. Only that answer and its end token are
supervised; earlier messages are context. JSONL inputs remain supported with
`--data path/to/file.jsonl`.

The script does not create `train.json` or substitute the smoke examples when
it is missing. Local `train.json`, `validation.json` and run directories are
ignored by Git. Do not commit private training data or model artifacts.

## Validate and train

Validate before allocating model memory:

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py --validate-only
```

By default, a seeded split holds out 10% of distinct conversation prompts
(rounded up, with at least one prompt on each side). Identical prompts, with
the default system message normalized, remain together. You need at least two
distinct prompts. This prevents exact-prompt overlap; review related or
paraphrased examples yourself to prevent semantic leakage. A tiny validation
split gives noisy results.

Alternatively, use a separately reviewed validation JSON file of the same
format; all of `train.json` is then used for training:

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py \
  --eval-data /workspace/qlora-smoke/source/training/examples/validation.json \
  --validate-only
```

Explicit training/validation files must not share IDs or exact conversation
prompts. Keep an additional test set out of both training and model selection
for your final behavior assessment.

Once the GPU is released, start a run (this command actually trains):

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py
```

Defaults are three epochs, batch size 1, gradient accumulation 4 (effective
batch 4), learning rate `2e-4`, linear scheduling with 3% warmup (rounded up to
whole updates and capped below the total updates), rank 8, alpha 16, dropout 0,
and maximum sequence length 2048.
These are starting settings, not tuned recommendations for an arbitrary dataset.
`--steps N` optionally overrides epochs with an optimizer-update limit; there
is no longer a 32-step ceiling. For example:

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py \
  --epochs 3 --learning-rate 1e-4 --batch-size 1 \
  --gradient-accumulation-steps 4 --rank 8 --max-length 2048 \
  --output /workspace/qlora-smoke/runs/my-training-run
```

Both input and answer must fit the limit; examples are rejected rather than
truncated. Longer sequences, larger batches and larger ranks need more GPU
memory. The 32 GiB free-memory guard is not a guarantee that those settings fit.
The cached base is quantized to NF4 with BF16 computation; only adapters train.

## Validation, checkpoints and output

Before training, validation loss is measured with the new, initially zero-effect
adapter. Validation and checkpoint saving then happen after each epoch.
`--save-steps N` switches both to every N optimizer updates, including the final
update. By default, two checkpoints are retained, protecting the best and latest
as needed by Trainer. Use `--save-total-limit` to change the retention limit
(minimum 2). The final `adapter/` is exported from the checkpoint with the lowest
validation loss, which may differ from the latest resumable checkpoint.

```text
/workspace/qlora-smoke/runs/<timestamp>/
  run.json               # Configuration, hashes, IDs, loss history and memory
  train.json             # Exact parsed training split for this run
  validation.json        # Exact parsed validation split
  checkpoints/checkpoint-N/
                         # Adapter, optimizer, scheduler, RNG and Trainer state
  adapter/               # Selected adapter weights and configuration
  tokenizer/
```

`run.json` records initial/final validation losses and their difference
(`validation_loss_delta`; negative means lower loss), the best checkpoint,
training loss, settings, data hashes, base revision, package versions and
PyTorch peak memory. Validation loss is not an answer-quality score. Compare
generated answers from the base and adapter on your untouched test prompts
before choosing an adapter for chat. Serving deployment is a separate step.

An interrupted run can resume from its last complete checkpoint. Supply the
same data and training arguments as before, plus:

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py \
  --resume-from-checkpoint /workspace/qlora-smoke/runs/RUN_ID/checkpoints/checkpoint-N
```

The output directory is inferred from that checkpoint. If the original command
used nondefault flags, repeat them. Data, split and training configuration must
match the saved run. Completed runs are not resumed or overwritten. Checkpoints
are saved at the configured interval; interruption before the first checkpoint
requires starting a new run. Resume only your own trusted checkpoints, which
include serialized optimizer/RNG state. Resume restores training state, rather
than merely loading the final adapter. It does not extend a completed run.

## Original eight-step smoke test

The explicit smoke mode is an eight-step pipeline check, not evidence of
quality improvement. The eight hand-written inference explanations in
`examples/smoke.jsonl` are disposable test data. Choose a behavior and separate
training/validation/test examples before a useful training run.

The first real run and fresh-process reload succeeded. See
[SMOKE_RESULTS.md](SMOKE_RESULTS.md) for measured memory, timings and limits.

## What it does

- Reads the cached `Qwen/Qwen3.8-27B` BF16 checkpoint at the deployment's exact
  revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Quantizes supported linear weights to NF4 during loading, with double
  quantization and BF16 computation. It does not use the AWQ serving artifact or
  download an interchangeable base checkpoint.
- Trains rank-8 language adapters only. Source weights and vision layers stay
  frozen. Prompts and earlier conversation replies have labels masked to -100;
  only the selected completion and its end token contribute to loss.
- Uses the original tokenizer/template with `enable_thinking=False`, matching
  the chat gateway. Examples exceeding the sequence limit fail rather than
  silently losing their questions or targets.
- Saves an adapter, tokenizer, exact base identity, data hash, package versions,
  loss history and PyTorch peak allocated/reserved GPU memory under a new run
  directory. These GPU figures do not include all driver/process overhead.
- Checks actual 4bit modules, BF16 computation, finite loss and a changed adapter
  weight. `verify_adapter.py` checks a fresh-process reload; held-out quality
  evaluation remains necessary before any serving deployment.

## Prepare on the existing BF16 Pod

Upload this directory to `/workspace/qlora-smoke/source/training`. The source
checkpoint remains under `/workspace/qwen-bf16/model`; never overwrite it.

```bash
bash /workspace/qlora-smoke/source/training/setup-qlora.sh
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_tune.py --smoke-test --data /workspace/qlora-smoke/source/training/examples/smoke.jsonl --validate-only
```

The virtual environment reuses the image's PyTorch binaries and pins their
versions; other training packages are installed inside the venv. It
does not modify the serving installation. `/opt` is container-local and may be
lost after a Pod reset, so rerun setup if necessary. Code and runs under
`/workspace` persist on the existing global volume.

The serving image already overrides PyTorch's declared NCCL dependency. The
training venv installs the matching NCCL version in its own directory. Pip may
also report that the inherited vLLM package requires a newer Transformers
version; vLLM is not used by this training process (`fast_inference=False`). The
original serving interpreter retains its original packages.

## Release the GPU safely

The serving entrypoint uses `wait -n`: killing vLLM also shuts down Jupyter and
the other supervised services. Do not kill vLLM from the current Jupyter session.

For a temporary training window, set the existing BF16 Pod's **Container Start
Command** to the following and apply the Pod restart:

```json
{"entrypoint":["python3","/usr/local/lib/start-jupyter.py"],"cmd":[]}
```

Use the explicit entrypoint object: a plain shell command in this field only
overrides Docker CMD, which the serving entrypoint does not execute.

Keep the official image, persistent volume, HTTP ports and existing environment
unchanged. Jupyter retains its current authentication and port 8888. BF16 chat
and metrics are unavailable during this window; the independent 4bit Pod keeps
serving. Do not create a new Pod merely to run this check.

## Run after the training-only restart

In the existing authenticated Jupyter terminal:

```bash
bash /workspace/qlora-smoke/source/training/setup-qlora.sh
bash /workspace/qlora-smoke/source/training/run-smoke.sh
```

The smoke wrapper refuses to start if less than 32 GiB of GPU memory is free. Its
settings are batch size 1, 2,048 maximum tokens, rank 8, and eight optimizer
steps. This is a free-memory guard, not a promise that every dataset fits.

Monitor its terminal output and `/workspace/qlora-smoke/runs/*/run.json`.
`status: completed` requires finite loss and a changed adapter parameter. Keep
the adapter for the experiment; do not automatically enable it for users.

After that process exits, verify the saved adapter in a fresh process:

```bash
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/verify_adapter.py /workspace/qlora-smoke/runs/RUN_ID
```

This reloads the same base plus saved adapter, checks a finite completion loss,
and generates a short smoke answer. `reload.json` records the adapter hash and
result. It is a reload check using a training example, not a quality evaluation.

## Restore BF16 serving

After the training process exits, clear **Container Start Command** and apply
the restart to restore the official image entrypoint. Keep `CPU_OFFLOAD_GB=0`
and the existing environment. Verify chat readiness, an actual reply and the
metrics target before declaring serving restored. No frontend routing change
is needed. The original checkpoint and saved adapter are separate directories.

## Local checks

```bash
python -m unittest discover -s training -p 'test_*.py' -v
```

The installed-stack test is skipped when Torch/Transformers/PEFT are absent.
In the QLoRA venv it uses a tiny, randomly initialized CPU model to exercise
epoch/step evaluation, checkpoint interruption/resume, retention and adapter
export. It does not load Qwen or establish GPU/Unsloth training performance.

Sources: [Unsloth Qwen3.8 training guide](https://unsloth.ai/docs/models/qwen3.8/train),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer),
[Unsloth checkpoint recovery](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint).
