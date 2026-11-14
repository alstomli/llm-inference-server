# Native CPU KV offload and decode CUDA graphs

The [November 13 enablement](KV-GRAPH-ENABLEMENT-2026-11-13.md) passed and
left both features active on the pinned production runtime.

The launcher supports two independent, default-off settings on the pinned
vLLM 0.32.0 BF16 stack:

| Setting | Default | Approved opt-in |
| --- | --- | --- |
| `KV_OFFLOAD_CPU_GIB` | `0` | `64` |
| `CUDA_DECODE_GRAPHS` | `false` | `true` |

The host tier uses the installed `OffloadingConnector`, `kv_both`, one GPU
block per CPU chunk, and `offload_prompt_only=false`. Prefix caching stays on
and Mamba cache mode becomes `align`. The 64 GiB value is a budget; actual
mapped bytes round down to whole native chunks. Launch fails if the projected
allocation would leave less than 32 GiB cgroup RAM or 8 GiB shared memory.

Decode graphs use compilation mode zero, `FULL_DECODE_ONLY`, and capture
sizes `[1,2]` for two sequences or `[1,2,4,8]` for eight sequences. Intermediate active sizes pad to the next captured bucket; eight is not a request to capture every size. They require zero CPU **weight** offload and the existing
supported two- or eight-sequence profile. Eager prefill remains available through native dispatch.
The settings do not override the GPU cache byte budget, batched-token budget,
context length, LoRA/vision allocations, or gateway admission limits.

## Deployment decision

Normal CI and releases use `Dockerfile`, which includes the launcher and both
runtime helpers. The historical enablement image derived from the then-running
inference digest and copied only those files; its build definition and dated
image provenance remain available in Git history.
The temporary feature-branch release exception used for enablement has been
removed: publication is restricted to `main` and the ordinary full image build.
Deployment continues to pin a reviewed published digest. Merging this source
does not publish an image, activate either opt-in, or change the running Pod.

## Bounded activation proof

Proof uses `KV_RUNTIME_OBSERVE=true` and `KV_RUNTIME_PROOF=true` only on the
first startup. Ordinary final serving omits both variables and therefore has
the native vendor worker, no middleware, no replay callback and no control
socket. The temporary observer records completed native graph replay calls
and cache allocation geometry without prompts, tensors, storage addresses or
credentials. Request-time before/after deltas are required; startup capture
or dispatch logs alone do not establish replay.

Proof middleware keeps ordinary inference in maintenance while allowing the
private verifier. Its sole administrative operation is a Unix socket under
the private `/tmp/qwen-kv-proof` directory, with mode 0600 and a ten-minute
socket lifetime. A mode-0600 regular, owner-held, nonsymlink capability file
provides a single-use token with at most ten minutes of validity. The fixed
operation checks unfinished requests, engine queues and native transfer
gauges, then calls the installed asynchronous prefix reset with
`reset_running_requests=False, reset_connector=False`. It must return true.
Reset consumption never releases maintenance. There is no HTTP admin route
and server development mode is never enabled.

Prime, GPU-hit and CPU-hit requests use the same small base-model text prefix
and cache salt. The native idle GPU-only reset between hits preserves CPU
connector state; positive host-hit tokens and load bytes plus the exact
expected answer establish CPU restoration without filling the GPU pool.
Different-salt isolation is checked separately. No adapter/image/web inference
or full capacity load is part of this enablement.

An independent local watchdog holds a fresh private snapshot and an absolute
120-minute lease. It starts rollback at minute 90, preserving 30 minutes for
the original service to load. First and final startup each have a 35-minute
cap. Activation changes only the image and explicit feature environment;
unrelated configuration must match the snapshot. CPU proof is limited to ten
minutes. Final restart is allowed only when the remaining lease can fit its
startup, validation and recovery reserve. Correctness failure, OOM/swap growth,
or RAM/shared-memory reserve failure triggers rollback.

After proof, a clean restart removes both proof variables while retaining
offload and graph settings. Final verification requires the same image and
resolved serving controls, base-only inference, authenticated health and
metrics, empty queues, safe memory, native allocation and graph capture/dispatch
evidence, and absence of proof processes/files/interfaces. Only then is the
watchdog disarmed and the two features left enabled. This is an enablement
check, not a throughput or latency conclusion.
