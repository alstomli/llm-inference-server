# Qwen3.8-27B original BF16 deployment

This is the historical November 2 PyTorch-template deployment record. See the [dated runtime snapshot](CURRENT.md) for the later configured service and the [official image record](VLLM-IMAGE.md) for the subsequent image migration. Historical Pod IDs and measured values below must not be used as current deployment targets.

Deployment date: November 2, 2026 (America/New_York).

This directory records the deployment on Runpod Pod `ftntkktzid9f4p`
(`ideological_plum_ferret`) and records its retired startup flow. Full runtime
verification results are saved on the global volume at
`/workspace/qwen-bf16/verification.json`; the observed results are summarized below.

## Completed verification

Verified November 2, 2026 at 5:48 PM America/New_York
(`2026-11-02T22:48:47.167841+00:00`).

| Check | Observed result |
| --- | --- |
| `/health` | HTTP 200 |
| `/v1/models` | `qwen-bf16`, context ceiling 4,096 |
| Original checkpoint revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Original weight shards | 18, with SHA-256 checks against Hugging Face metadata |
| Checkpoint data stored | 55,586,048,281 bytes including supporting files |
| Weight tensor headers | 1,199 BF16 tensors |
| Real chat completion | `Hello! Yes, I can respond. How can I help you today?` |
| Completion status | `stop`; 23 prompt tokens and 16 completion tokens |
| Single cold request elapsed time | 85.225 seconds; not a throughput benchmark |
| GPU memory after generation | 27,323 MiB used of 32,623 MiB |
| Container memory after generation | Approximately 81.45 GiB, including filesystem caches |
| Installed packages | vLLM 0.32.0; Transformers 5.19.0; PyTorch 2.13.0; huggingface-hub 2.4.0; safetensors 0.8.0 |
| Startup persistence | The approved Pod edit reset the container; Jupyter returned and the model bootstrap started automatically |

The engine reported approximately 22.91 GiB for model loading on the GPU before
its runtime cache allocations. It used a fallback sampler because the template's
CUDA toolkit did not satisfy the FlashInfer sampler's SM12.x requirement. This
was a warning, and the real generation completed successfully. The 4,096-token
ceiling is configured; maximum-context, image-input, and concurrent-load behavior
have not been benchmarked. The existing frontend and production Pod were not
migrated as part of this standalone deployment.

Local validation passed Python compilation, shell syntax validation on the Pod,
and three offline tests covering pinned restart reuse, corrupted-weight
rejection, and unsafe checkpoint paths. The current CI retains checkpoint staging tests; the obsolete template bootstrap
and its syntax check have been removed.

## Pod and storage

| Item | Configuration |
| --- | --- |
| GPU | One NVIDIA RTX PRO 4500 Blackwell Server Edition, 32,623 MiB VRAM |
| System RAM | Container limit 93,999,996,928 bytes, approximately 87.54 GiB |
| Data center | US-KS-2 |
| Container image | `runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404` |
| Container disk | 50 GB |
| Persistent storage | Global volume `spicy_coffee_meerkat`, mounted at `/workspace` |
| Existing HTTP/TCP ports | 8888 / 22 |
| Model API | `127.0.0.1:8000`, accessible from inside the Pod |

Runpod global volumes are object-backed and do not provide file locking,
atomic rename, hard links, or POSIX permission bits. The Python environment,
download locks, temporary downloads, compilation caches, and runtime log stay
on the container disk. Completed model files and deployment configuration live
on the global volume. No Python environment is installed on the global volume.

## Model settings

The historical template configuration was:

- Original repository: `Qwen/Qwen3.8-27B`, pinned to a commit at first download.
- Model weights and execution dtype: BF16; no quantization option.
- Partial CPU weight offloading: up to 30 GiB per GPU.
- Combined prompt/output context ceiling: 4,096 tokens.
- Maximum concurrent sequences: one.
- GPU memory utilization budget: 0.90.
- Eager execution, with compilation and CUDA graphs disabled.
- Eager safetensors loading, suited to network/object-backed model storage.
- Reasoning parser: `qwen3`.
- Initial multimodal limits: one image per prompt, video input disabled.
  The original checkpoint, including its vision weights, is retained.
- Served API model name: `qwen-bf16`.

CPU offloading stores part of the weights in system RAM while GPU computation
continues to use the full model. The setting is a budget, not an exact tensor
placement guarantee. Reducing it uses more VRAM; increasing it uses more system
RAM and typically increases transfer overhead. Context, concurrency, and working
memory also consume VRAM.

## Retired startup and monitoring

The PyTorch-template startup scripts and duplicate monitoring procedure were removed after the official image launcher replaced them. Their immutable [historical source](https://github.com/alstomli/llm-inference-server/tree/6685b0e/deployments/qwen-bf16) remains available for interpreting these measurements. Do not run that startup procedure on the current service.

Monitoring verification on the original November 2 Pod passed: missing token returned 401, wrong token returned 401, correct token returned 200 with vLLM/GPU metrics, and an authenticated model route returned 404. Its relay used port 8889 while Jupyter used 8888. These are historical access checks, not a current health assertion.

The active image uses `scripts/start-server.sh` for supervised startup, the Go gateway by default, and the shared `scripts/metrics-relay.py`. `prepare_model.py` and its tests remain active; `config.json` supplies the model ID to checkpoint preparation, but its old context/offload fields do not override the launcher's environment. Current release and rollback steps live in the [repository README](../../README.md).

## References

- [Original model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [vLLM model recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM engine and offloading settings](https://docs.vllm.ai/en/v0.32.0/configuration/engine_args/)
- [Runpod global-volume limitations](https://docs.runpod.io/storage/globalvolume/overview#limitations)
- [Runpod Pod updates](https://docs.runpod.io/pods/manage-pods#update-a-pod)
