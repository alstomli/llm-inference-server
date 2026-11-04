# Preparation record — November 3, 2026

Prepared on the existing BF16 Pod `o6f2qoctk798tq`, through its authenticated
Jupyter service. No serving process was stopped and no Pod configuration was
changed during preparation.

- GPU: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB total; 87,877 MiB used
  by the serving deployment at inspection time.
- Source checkpoint: original Qwen3.8-27B revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, 18 shards,
  55,563,006,776 bytes of weights; manifest records 1,199 BF16 tensors.
- Separate environment installed at `/opt/qlora/venv`; PyTorch remained
  `2.13.0+cu130`. Unsloth `2026.11.1` imported successfully. An import check is
  not proof that weight loading, all kernels or backpropagation work.
- CPU validation using the real tokenizer passed for all eight smoke examples:
  76–89 total tokens each. The source config and shard sizes match the existing
  verified manifest. This did not rehash all 55 GB of checkpoint weights.
- Local supervision tests passed: only the selected completion is labelled,
  earlier assistant replies are masked, overlength examples are rejected,
  invalid examples are rejected, and the base revision must match.
- `bash -n` passed for both training shell scripts on the Pod.

These were preflight checks, before GPU training. The user subsequently
approved the temporary Pod reset, and the eight-step training run plus a
fresh-process adapter reload both completed successfully. The actual results,
measured memory and limitations are in [SMOKE_RESULTS.md](SMOKE_RESULTS.md).

The staged source is `/workspace/qlora-smoke/source/training`. CPU validation and
import logs from preparation are at `/workspace/qlora-smoke/validation.json` and
`/workspace/qlora-smoke/import-check.log`. These checks do not advance the
learning path's fine-tuning completion status.
