# QLoRA smoke test on the original BF16 checkpoint

This is an eight-step pipeline check, not a production fine-tune or evidence of
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
/opt/qlora/venv/bin/python /workspace/qlora-smoke/source/training/qlora_smoke.py --validate-only
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

The run refuses to start if less than 32 GiB of GPU memory is free. Default
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

Sources: [Unsloth Qwen3.8 training guide](https://unsloth.ai/docs/models/qwen3.8/train),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization).
