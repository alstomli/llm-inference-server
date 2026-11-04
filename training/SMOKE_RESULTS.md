# QLoRA pipeline test — November 3, 2026

Training and a fresh-process adapter reload both succeeded on the existing
BF16 Pod. This establishes a working training pipeline; quality improvement
has not been evaluated. Serving restoration is still being verified.

## Configuration

| Item | Value |
| --- | --- |
| Source | Original `Qwen/Qwen3.8-27B` BF16 checkpoint |
| Source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPU | RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported total |
| Training base | Frozen NF4 weights, double quantization, BF16 computation |
| Adapter | Rank 8, alpha 16, dropout 0; saved tensors are FP32 |
| Trainable parameters | 58,363,904, about 0.21% of the reported model parameters |
| Data | Eight hand-written, English pipeline examples; 76–89 total tokens each |
| Supervision | Completion only, including end token; question/history masked |
| Updates | Eight steps, batch size 1, one pass over the examples |
| Learning rate | 0.0002, constant |
| Sequence cap | 2,048 tokens; examples were not padded to that length |
| Thinking | Disabled, matching the chat gateway |

The original checkpoint was quantized while loading. The AWQ serving artifact
was not used. Unsloth, PEFT and Transformers Trainer performed the updates.
The original BF16 files were not overwritten.

## Observed results

| Measurement | Result |
| --- | --- |
| Completed optimizer steps | 8 |
| Mean training loss | 2.3368, finite |
| Adapter update | A monitored LoRA B tensor changed |
| Actual 4bit linear modules | 606 |
| PyTorch peak allocated memory | 17.49 GiB |
| PyTorch peak reserved memory | 17.95 GiB |
| Observed whole-GPU peak | 19,431 MiB, approximately 18.98 GiB |
| Trainer runtime | 73.57 seconds, including initial compilation work |
| Complete training job | 470.57 seconds, including imports, loading and saving |
| Adapter weight file | 233,605,480 bytes, about 222.8 MiB |
| Fresh-process reload | Passed; all 992 saved tensors matched exactly |
| Reloaded completion loss | 1.5945 on one training example |
| Reloaded generation | 17 tokens, ended with the expected end token |
| Adapter after serving reset | Same SHA-256 checksum; persisted successfully |

The GPU peak was sampled once per second with `nvidia-smi`; brief spikes can
fall between samples. PyTorch's counters cover its allocator, not every driver
or process allocation. These are total training-process observations, not
memory to add to a simultaneously running BF16 server.

The generation check answered “What is TTFT?” with:

> TTFT is the time from a request until the first generated token is returned.

This checks that saved weights can be used after a fresh load. The question
was in the training data; it is not an independent quality evaluation.

## What this establishes

- The original checkpoint can be loaded for QLoRA on this GPU, perform adapter
  updates, save the adapter and reload it successfully.
- The measured memory footprint is comfortable on this 96 GB GPU for the
  tested short examples. It does not establish memory use for 2,048-token
  examples, long history, larger batches or higher adapter ranks.
- Startup/loading dominated this tiny experiment. Trainer time is not an
  estimate of a full dataset's training duration.
- Different steps used different examples. Their loss values do not establish
  a before/after improvement, and training loss does not replace held-out tests.

The optional xFormers wheel did not match the image's PyTorch build; Unsloth
fell back to PyTorch attention. This run is not a training-speed benchmark.

## Artifacts and next step

The adapter and tokenizer are on the attached persistent volume under:

```text
/workspace/qlora-smoke/runs/20261104T042842Z/adapter
/workspace/qlora-smoke/runs/20261104T042842Z/tokenizer
```

The adapter hash is
`6146939b60d9d3837bc887c3d8216c17fa57c2884e9dbd7f158daa62ca9eb0fe`.
Weights remain on the Pod; only metadata is included in this repository.

Raw evidence: [training](results/2026-11-03-smoke/train.json),
[job and GPU peak](results/2026-11-03-smoke/job.json),
[reload](results/2026-11-03-smoke/reload.json),
[post-reset artifact/config check](results/2026-11-03-smoke/post-restore.json).
The full training log and one-second memory trace remain under
`/workspace/qlora-smoke/jobs/20261104T042842Z` and in the local tools archive.

Next, choose one desired chat behavior, prepare reviewed train/validation/test
examples, and compare against a strong prompted baseline. This pipeline test
does not mark Stage 5 complete or automatically enable the adapter for chat.
