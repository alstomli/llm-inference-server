# LLM Inference Server

Self-hosted Qwen3.8-27B chat serving on RunPod, kept as a learning project. One codebase serves the model in more than one precision so cost and quality can be traded against each other.

## Language

### Models

**Profile**:
A named serving configuration of the model: `bf16` or `4bit`. Each profile has its own Pod, aliases and limits.
_Avoid_: backend, variant, mode, engine

**BF16 base**:
The unquantized Qwen3.8-27B weights served by the `bf16` profile.
_Avoid_: full model, original model, 16-bit model

**4-bit base**:
The INT4-quantized Qwen3.8-27B weights served by the `4bit` profile.
_Avoid_: quantized model, AWQ model, small model, 4-bit model

**Adapter**:
The LoRA weights layered on top of the BF16 base, and nothing else. Only the `bf16` profile has an adapter.
_Avoid_: fine-tuned model, LoRA model, tuned base

**Alias**:
The model name a client asks for, mapping to one profile's base or base-plus-adapter.
_Avoid_: model ID, model name, served name

### Operations

**Acceptance**:
The fixed set of known-answer request waves a profile must pass on its Pod before it takes user traffic.
_Avoid_: smoke test, QA, verification, benchmark

**Wave**:
One batch of concurrent acceptance requests sharing an input shape (short, medium, mixed or near-context).
_Avoid_: round, batch, test case
