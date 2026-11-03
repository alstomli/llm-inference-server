# Qwen3.8-27B original BF16 deployment

**Current deployment: verified official vLLM image 0.10.1.** See the
[current deployment and verification guide](VLLM-IMAGE.md) for its configuration,
startup, and rollback procedure. The body below preserves the earlier
PyTorch-template deployment record; its `bootstrap.sh` and start commands are
not the current Pod startup.

Deployment date: November 2, 2026 (America/New_York).

This directory records the deployment on Runpod Pod `ftntkktzid9f4p`
(`ideological_plum_ferret`) and contains its startup files. Full runtime
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
rejection, and unsafe checkpoint paths. CI runs those staging tests and the new
bootstrap syntax check as part of the existing image-check workflow.

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

`config.json` is the deployed configuration:

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

## Files and startup

Persistent files:

```text
/workspace/qwen-bf16/
  bootstrap.sh
  config.json
  prepare_model.py
  serve.py
  verify.py
  README.md
  model/                   # original checkpoint files
  model-manifest.json      # pinned commit, file hashes/sizes, tensor dtypes
  verification.json        # actual API/generation/memory checks
  requirements-resolved.txt
```

Local, ephemeral files:

```text
/opt/qwen-bf16/venv/        # isolated vLLM environment
/opt/qwen-bf16/download/    # one temporary checkpoint file at a time
/opt/qwen-bf16/hf-cache/
/opt/qwen-bf16/vllm-cache/
/tmp/qwen-bf16.lock         # prevents duplicate deployments
/tmp/qwen-bf16-bootstrap.log
```

The Pod's original container start-command override was empty. It was changed to:

```bash
bash -c 'bash /workspace/qwen-bf16/bootstrap.sh >>/tmp/qwen-bf16-bootstrap.log 2>&1 & exec /start.sh'
```

This runs the model bootstrap in the background and retains the template's
normal `/start.sh`, which starts Jupyter and SSH. The Pod image, exposed ports,
and attached volume are preserved. Editing the Pod resets its container;
the global volume retains the deployment files and model.

The bootstrap installs `vllm==0.32.0` and `transformers>=5.8.0,<6` in an isolated
virtual environment. It recreates this environment after container replacement.
The live verification saves the exact installed package versions in
`requirements-resolved.txt`; that file can be used with `pip install -r` to
reproduce the verified environment instead of resolving the version range again.
Model downloads use local staging and locks, then copy one verified file at a
time onto the global volume, so the complete checkpoint need not fit on the
50 GB container disk. Large-file SHA-256 values are checked against Hugging Face
metadata. The manifest pins the checkpoint commit and permits completed files
to be reused on restart. Incomplete transfers are not marked complete. A corrupt
manifest stops the deployment for inspection rather than silently selecting a
different revision.

## Inspect and use the deployment

Run commands in the Pod's Jupyter terminal:

```bash
tail -n 60 /tmp/qwen-bf16-bootstrap.log
nvidia-smi
cat /sys/fs/cgroup/memory.current
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

Example generation:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-bf16","messages":[{"role":"user","content":"Reply with a short greeting."}],"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}'
```

The model API is private to the Pod. Use an SSH tunnel after configuring SSH
access, or run API clients in the Pod. Jupyter retains its existing authentication.
Publishing a model endpoint requires a separate authenticated ingress setup.

## Restart, tuning, and rollback

- A Pod restart runs the saved start command automatically. If the container
  filesystem was replaced, package installation repeats; completed model files
  are reused from the global volume.
- If the model exits while the Pod/Jupyter remain running, inspect the log,
  correct the cause, and launch `nohup bash /workspace/qwen-bf16/bootstrap.sh
  >>/tmp/qwen-bf16-bootstrap.log 2>&1 </dev/null &`. The local lock prevents a
  second live model instance. This launcher is not an automatic crash supervisor.
- To tune offloading, context, or concurrency, edit the corresponding value in
  `/workspace/qwen-bf16/config.json` and restart the model/Pod. Start with small
  prompts and check actual generation and memory use after each change.
- To roll back the startup integration, clear the container start-command
  override in Edit Pod and save. This resets the container and restores the
  original PyTorch/Jupyter startup. Keep the global volume attached to retain
  the model and deployment record.

Do not store credentials in these files or publish Jupyter token URLs.

## References

- [Original model](https://huggingface.co/Qwen/Qwen3.8-27B)
- [vLLM model recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [vLLM engine and offloading settings](https://docs.vllm.ai/en/v0.32.0/configuration/engine_args/)
- [Runpod global-volume limitations](https://docs.runpod.io/storage/globalvolume/overview#limitations)
- [Runpod Pod updates](https://docs.runpod.io/pods/manage-pods#update-a-pod)
