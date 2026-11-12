# KV-cache CPU offload attempt — November 12, 2026

**The attempt was blocked before measurements. CPU KV offloading was never
activated, and there are zero accepted A/B latency samples.** There is no TTFT,
ITL, TPOT, end-to-end latency, overhead or throughput comparison from this run.
This is not evidence that native offloading performs poorly or is unsupported.
The original production configuration and service were restored and verified.

The reviewed tooling is a draft. It must not be reused unchanged: runtime
control, status and logs need independence from global-volume I/O before a
future attempt. The unexecuted local-staging redesign is excluded from this PR.

## Intended comparison

The target was the existing Qwen3.8-27B BF16 model on its existing RTX PRO 6000
Blackwell Server GPU, using vLLM 0.32.0 and the pinned image/model revision.
Arm A omitted KV transfer; B would add native `OffloadingConnector` with a 64 GiB
CPU tier and decode saving. CPU weight offloading remained zero. Both arms
would retain eager execution, prefix caching, aligned Mamba caches, the hybrid
cache manager, loaded LoRA/vision allocation, context 40960, max sequences 256
and batched tokens 8192. Performance traffic was to be base-model text only.

The planned long workload was the same 8192 input / 512 output shape as the earlier
capacity test, with identical seeded tokenizer-generated input pools. The
generator is unchanged; full token-ID pool hashes and template hash are retained
in the [numeric report](KV-OFFLOAD-ATTEMPT-2026-11-12.json). No prompt bodies or
generated model output are included.

## What happened

All times below are UTC (Eastern time was five hours earlier).

| Time | Observed event |
|---|---|
|19:18:17|Independent watchdog armed; restoration 21:48:17, exclusive end 22:18:17.|
|19:19:12|Original pod args temporarily overridden; image/environment/mounts preserved.|
|19:19:18|Supervisor failed before model launch because a local variable shadowed `http.server`; safety path executed the stock launcher.|
|19:24:36|Reviewed correction retried within the same lease, after expanded mocked safety tests.|
|19:41:27–28|A weights took 947.95 s; model loading took 951.55 s and 51.14 GiB of GPU memory.|
|19:42:05|A cache allocation was logged.|
|19:42:53|Engine initialization logged 84.87 s; API startup completion also appears in recovered private logs.|
|19:43:27|Live control readiness remained unconfirmed, so original pod args were explicitly restored through the control API.|
|20:05:00|The conditional retry's firm recovery gate was missed; no further experimental activation or staging occurred.|
|20:06:12|Authenticated original production readiness confirmed.|
|20:12:04|Full runtime restoration verification passed.|
|20:14:09|Backed-up owned remote session removed; final health/metrics/queues checked again afterward.|

A did initialize and allocate its cache, as discovered afterward. Its supervisor
status remained `starting`; the live control path did not confirm readiness or
start accepted latency scenarios. No B engine was started.

## Partial verified allocation finding

The observation-only A worker captured one deduplicated GPU backing storage:

| Property | A startup observation |
|---|---:|
|Actual GPU cache bytes|34,270,609,408 (~31.917 GiB)|
|Cache blocks|667|
|Cache groups|3 Mamba groups + 1 full-attention group|
|Tensor views|64|
|Layout|LBNHC|
|Hash block / cache-hit alignment|784 / 784 tokens|

The layout fingerprint is
`420dc40d28b0e038d171ae3f93f8dbb8a31051da7faa97a07993ec848b7d768e`.
The private `_kernel_block_sizes` attribute was absent, so that optional field
is null; concrete spec, shape, stride, alias and block metadata were captured.
These are startup/allocation observations, not response-time measurements or
an A/B equality check.

## Diagnosis and limits

There are two concrete findings. First, the cold weight load was substantially
longer than the previous 9–12 minute planning observation, leaving roughly 87 s
before the 18 minute startup limit. Second, after recovery a bounded independent
reader received `OSError` / `errno 5` (input/output error) from the owned
`session.events.jsonl` on `/workspace` (`fuse.geesefs`). Other private startup
logs and the allocation file were readable. Jupyter status/content requests had
timed out during the attempt while the separate maintenance endpoint responded.

This establishes a control-log I/O failure and a slow cold load. It does not
establish that the model or observer failed to initialize, or explain every
monitoring stall. The complete event object could not be backed up; its earlier
partial local copy and the explicit read error were preserved. Readable startup
logs and fresh private restoration snapshots remain outside Git.

The remote supervisor's deadline checks can themselves be delayed by global
filesystem operations. They were not observed firing reliably here; the
independent external control-plane watchdog and explicit original-args restore
were essential. The retained best-effort logging fix handles write exceptions,
not indefinitely blocked writes. That fix was added after the live attempt and
was tested locally and in CI; it was not deployed to the pod.

## Restoration and validation

Exact original pod args, environment, image, ports, mounts, disk and registry
configuration matched the fresh snapshot. The original model argv matched
exactly, with max sequences 2. Engine and gateway active/waiting queues were zero;
no benchmark process remained. Authenticated health/metrics and the GPU exporter
passed. Base and adapter smoke streams both completed. The first base stream
had the expected cold delay (~65 s); these smoke timings are not benchmark data.

Original limits and capability schema were preserved: context 40960, global
2 active / 6 waiting, account quota enforcement, adapter toggle and image capability.
The base web profile remained ready; the adapter web profile remained explicitly
unverified, matching the original behavior. No image/video or web-search
inference was added. Shared-memory availability returned exactly to its
93,999,996,928 byte baseline, with swap 0. Because B never allocated, this proves
no residual allocation; it does not test reclamation of a 64 GiB offload buffer.

Local validation passed 12 experiment safety tests and 16 existing benchmark tests.
The final tooling source is `d73affd62d742b0465ddc75eac468eca3501150c`; its
[image CI run](https://github.com/alstomli/llm-inference-server/actions/runs/43297558568)
passed the full image build and image suites. CI/local tests do not establish
live CPU cache transfer correctness or performance.

No latency charts were created. The CPU store/load, preemption-resume,
correctness/isolation, memory-under-offload and performance acceptance gates
remain untested. Offloading remains disabled.

## Before another attempt

A future plan needs a separately reviewed local control/status/log path,
fail-path coverage for blocked/erroring storage, observed startup budgets, and
enough time for matched measurements plus restoration. A byte-verified local
checkpoint may reduce repeated global reads, but must be reviewed for disk,
duplicate page-cache and 64 GiB tier memory pressure. Read-only feasibility found
~149.98 GiB local disk free and 55,586,048,281 bytes of regular model files; no
checkpoint copy or file-cache advice was performed. No staging implementation
or staging validation is included in this PR.
