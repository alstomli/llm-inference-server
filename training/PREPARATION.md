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

**GPU training has not run. No trained adapter or measured training memory peak
exists yet.** The next action is the temporary training-only startup described
in [README.md](README.md). Initial browser automation could not verify Chrome's
URL. Access subsequently worked through the user's logged-in inner browser.
Saving the prepared edit was then rejected by automatic approval review because
it resets the running service and may erase container-local files. Explicit
approval of that reset is pending; no Pod startup change has been applied.

The staged source is `/workspace/qlora-smoke/source/training`. CPU validation and
import logs are at `/workspace/qlora-smoke/validation.json` and
`/workspace/qlora-smoke/import-check.log`. These checks do not advance the
learning path's fine-tuning completion status.
