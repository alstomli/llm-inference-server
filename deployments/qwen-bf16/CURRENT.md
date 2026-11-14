# BF16 runtime snapshot

Observed November 13, 2026 (America/New_York) from the configured Runpod service state during repository cleanup. This is a dated configuration record, not a deployment manifest or a fresh engine/throughput verification.

| Item | Configured value |
| --- | --- |
| Pod | `66j3wfha6suokg` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Image | `zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d` |
| `MODEL_PROFILE` | BF16 |
| `MAX_NUM_SEQS` | 2 |
| `MAX_MODEL_LEN` | 40,960 |
| `GPU_MEMORY_UTILIZATION` | 0.90 |
| `CPU_OFFLOAD_GB` | 0 (weight offloading disabled) |
| `KV_OFFLOAD_CPU_GIB` | 64 |
| `CUDA_DECODE_GRAPHS` | `true` |

CPU KV-cache offloading and CPU weight offloading are separate settings. The current configuration requests CPU KV residency while keeping model weights on the GPU. Configured CUDA decode graphs also differ from the eager-only source defaults on `main` at `6685b0e`.

The running image contains changes under review; this record does not claim those changes are merged to `main`. The deployed implementation is recorded in open PR [#37](https://github.com/alstomli/llm-inference-server/pull/37): image implementation commit `7b10046`, with the dated report at `82fe9a9`. PR [#36](https://github.com/alstomli/llm-inference-server/pull/36) is an earlier KV-residency experiment that restored offload-disabled settings at its date; it is not the deployed implementation. Confirm the image provenance and actual engine startup arguments before rebuilding or replacing the service from a source branch. Configured environment values alone do not establish that an engine option is effective. Consult the dated validation evidence in those PRs; cleanup does not rerun GPU benchmarks.

## Operational entry points

- [Repository operations](../../README.md): releases, deployment, metrics and limits.
- [Go gateway](../../docs/GATEWAY-GO.md): current default and supported explicit Python fallback.
- [Checkpoint staging](prepare_model.py): validates pinned weights; `config.json` supplies its model ID, while launcher environment owns serving settings.
- [First official image record](VLLM-IMAGE.md), [earlier RTX PRO 6000 migration](RTX6000-MIGRATION.md), and [original PyTorch-template measurements](README.md): historical evidence, not current endpoints or rollback selections.
- [LoRA serving configuration](LORA-SERVING.md): adapter setup and validation requirements.

Use the current service bindings kept privately in deployment environments rather than copying old Pod origins. Deployment and rollback use a reviewed official image digest with the current volume, credentials and settings. No Pod, image, database or frontend deployment was changed by this cleanup.
