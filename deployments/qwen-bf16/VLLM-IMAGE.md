# Official BF16 image deployment

The BF16 Pod's first official release uses the same `zihan000/qwen-inference`
image and versioned-image/digest workflow as the 4bit Pod. Its image starts from
`vllm/vllm-openai:v0.32.0`; application code and Jupyter dependencies are installed
at image build time. Container startup does not install vLLM or execute the old
volume bootstrap. The original BF16 weights stay on global volume
`spicy_coffee_meerkat` at `/workspace/qwen-bf16/model`.

## Shared configuration

Both profiles use the same configuration keys and defaults:

| Setting | Shared default |
| --- | --- |
| `MAX_MODEL_LEN` | `40960` |
| `MAX_NUM_SEQS` | `2` |
| `GPU_MEMORY_UTILIZATION` | `0.90` |
| `VLLM_HOST`, `VLLM_PORT` | `127.0.0.1`, `8000` |
| `CHAT_PORT` | `8001` |

`MODEL_PROFILE=4bit` is the default and retains the AWQ checkpoint and `qwen`
served alias. `MODEL_PROFILE=bf16` selects the original local checkpoint, BF16
dtype, `qwen-bf16` alias, eager execution/loading, and `CPU_OFFLOAD_GB=30`.
The offloading budget is configurable. Both profiles explicitly enable prefix
caching and retain the same gateway API, request limits, and metrics collection.

These are requested serving ceilings. Image tests check configuration, not GPU
capacity. Actual BF16 readiness and generation must pass before this release is
called verified. Two configured sequences do not establish that two simultaneous
maximum-context requests fit; record the exact GPU workload tested. Do not
silently reduce the requested context or concurrency if the trial cannot fit.

The image's environment is authoritative for serving settings. The older
`config.json`, `serve.py`, and `bootstrap.sh` describe the preceding script-based
setup; the official image uses only the checkpoint staging data from the volume.
`prepare_model.py` runs from the image, verifies the pinned checkpoint metadata,
and reuses staged files. Locks, temporary downloads, and caches remain local
because the global volume does not provide POSIX locking/atomic rename.

## One-time migration

Before changing the Pod, inspect its current settings and checkpoint, and retain
a private read-only settings snapshot for migration diagnosis. Preserve the
existing global-volume attachment, 50 GB container disk, and all existing
nonconflicting environment values. Stage a separate chat token at
`/workspace/chat/chat-token`; do not reuse the monitoring credential.

Set these non-secret environment values on BF16 Pod `ftntkktzid9f4p`:

```text
MODEL_PROFILE=bf16
MAX_MODEL_LEN=40960
MAX_NUM_SEQS=2
CPU_OFFLOAD_GB=30
GPU_MEMORY_UTILIZATION=0.90
SERVED_MODEL_NAME=qwen-bf16
BF16_ROOT=/workspace/qwen-bf16
CHAT_ENABLED=true
CHAT_HOST=0.0.0.0
CHAT_PORT=8001
CHAT_TOKEN_FILE=/workspace/chat/chat-token
METRICS_ENABLED=true
METRICS_HOST=0.0.0.0
METRICS_PORT=8889
METRICS_TOKEN_FILE=/workspace/qwen-bf16/metrics-token
JUPYTER_ENABLED=true
JUPYTER_PORT=8888
JUPYTER_SERVER_ROOT=/workspace
```

Retain the existing private `JUPYTER_PASSWORD`: it is Jupyter's existing token,
not a new password. The image uses it in process configuration and suppresses
token-bearing startup URLs. Do not print it or place it in Git. Jupyter uses the
same root directory and authentication; its installed versions are pinned to the
prior template's JupyterLab 4.6.4, Jupyter Server 2.21.1, and Notebook 7.6.3. The
4bit profile does not start Jupyter unless explicitly enabled. SSH was not
listening on the prior BF16 Pod, and this migration does not enable it.

Apply the published image digest, environment, empty container start-command
override, and HTTP ports `8888,8889,8001` in one Pod update. Retain TCP port 22.
An old start-command override would run the legacy volume bootstrap instead of
the image entrypoint and must be removed. This one-time migration configuration
is separate from normal official release rollback. Do not perform an env-only
reset followed by an image reset, or trigger another same-image deployment just
to check readiness.

The update resets the container and reloads BF16. The observed earlier global-
volume reload took about 32 minutes, so readiness verification allows 60 minutes.
The authenticated gateway starts early but reports unavailable until vLLM is
ready. Jupyter remains available during model loading. The model API stays local;
port 8001 publishes only the token-protected chat gateway. Metrics stay on 8889,
so Prometheus target identities and historical series are preserved.

## Verification

Use the existing chat checker with a privately supplied BF16 chat credential:

```sh
RUNPOD_CHAT_URL=https://ftntkktzid9f4p-8001.proxy.runpod.net \
python scripts/check-chat-deployment.py --expected-model qwen-bf16 \
  --wait 3600 --read-timeout 180
```

Also require missing/wrong chat tokens to fail, actual context to be 40,960,
reported admission capacity to be two active/six waiting, a real BF16 response,
concurrent short requests, cancellation, correct context rejection, Jupyter
login protection/access, and both Prometheus targets/GPU exporters healthy.
A short generation test does not benchmark two maximum-context generations.

## Release and rollback

The existing `prod` GitHub environment keeps its 4bit bindings
`RUNPOD_POD_ID` and `RUNPOD_CHAT_TOKEN`. Add `RUNPOD_BF16_POD_ID` and the separate
secret `RUNPOD_BF16_CHAT_TOKEN`; reuse the existing `RUNPOD_API_KEY` account
credential. BF16 selection fails closed if either binding is absent or its Pod
ID equals the 4bit Pod. Secrets never enter the image.

Build/publish a new image version on `main` through **Build and publish release**.
The first successful BF16 migration/verification establishes its first official
image digest. No earlier PyTorch image is an official BF16 release.

For later deploys or rollbacks, use **Deploy or roll back Runpod image**, choose
`target=prod`, `model_profile=bf16`, and the intended published image digest.
Leave `first_official_release=false`. The normal versioned-image rollback is the
same as the 4bit workflow: a failed update restores the previous official image
and verifies it. Stable Pod profile, volume, entrypoint, and ports stay configured.

If retrying a failed first official release through the workflow, set
`first_official_release=true` and keep it enabled until verification succeeds.
This explicit mode permits retry of the same configured digest, reports that no
previous official BF16 release exists on failure, and does not automatically
restore a PyTorch-era deployment. It is an operator assertion, not release history
inferred from an image name. Diagnose initial migration failures using the saved
settings and logs. Subsequent successful releases use normal mode.
