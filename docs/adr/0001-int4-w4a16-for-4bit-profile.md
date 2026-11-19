---
status: accepted
---

# INT4 W4A16 (RedHat) for the 4-bit profile on an RTX PRO 4500

The `bf16` profile needs an RTX PRO 6000 (96 GB), which costs more credits than this learning project can sustain. We are adding a `4bit` profile on a single RTX PRO 4500 (32 GB, Blackwell SM120) using `RedHatAI/Qwen3.8-27B-INT4` pinned to a revision. It is GPTQ with AWQ smoothing, W4A16, and keeps the vision tower in BF16. The `4bit` profile uses an FP8 KV cache and has no adapter. Answer quality on hard questions is explicitly not a goal; the BF16 Pod is stopped, not deleted, and remains the fallback.

## Considered Options

- **NVFP4** (the card supports FP4 natively). Rejected: no first-party checkpoint exists, and vLLM's NVFP4 path on SM120 frequently falls back to Marlin W4A16 kernels anyway, so we would take on the risk without the speed-up.
- **`avyukth/Qwen3.8-27B-AWQ-INT4`** (previously served here until commit `5ee7d8c`). Rejected: same size and speed, but 48 linear-attention layers are round-to-nearest only and its only quality evidence is 40 GSM8K questions.
- **Keeping BF16 with CPU weight offload on 32 GB.** Rejected: already tried; it took 85 s for a single cold request at 4k context.

## Consequences

- Concurrency on the `4bit` profile is whatever passes the near-context acceptance wave at 40,960 context (minimum 2), not the `bf16` profile's 8.
- MTP speculative decoding is left off at first because its memory competes with the KV cache. Turning it on later is a separate experiment.
