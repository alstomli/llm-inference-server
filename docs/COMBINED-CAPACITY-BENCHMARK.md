# B-only combined KV offload and decode-graph capacity run

This tooling runs the fixed historical suite on the existing, enabled BF16
deployment. It does not rerun the historical eager arm or change production
limits permanently. The tester owns activation, snapshots, the independent
watchdog, backup and restoration; the developer supplies this reviewed tooling.

## Frozen profile and historical cases

The exact image is `zihan000/qwen-inference@sha256:59ff84b6a9c8ceca9aa5e568d726b5a5f3f72959229cacc622cf83257757f50d`
(0.17.0, vLLM 0.32.0). The temporary profile uses sequence ceiling 256, batched
tokens 8192, context 40960, GPU utilization 0.90, BF16 and zero weight offload.
Native CPU KV remains 64 GiB, `kv_both`, one block per chunk and
`offload_prompt_only=false`, with prefix caching and Mamba `align`. Decode
graphs use compilation mode zero, `FULL_DECODE_ONLY`, and capture sizes
`[1,2,4,8,16,32,64,128,256]`. Model, LoRA and vision allocation settings remain
unchanged. Every submitted inference uses `qwen-bf16-base`.

[The canonical manifest](COMBINED-CAPACITY-PLAN.json) contains 29 historical
cases, in their original order. The two accidental cold retries at medium C2
and C4 are excluded. Their remaining historical elapsed time is 48.56057
minutes. The benchmark foundation comes from unmerged capacity PR #34,
commit `ae82a1c`; its prompt generator is unchanged.

The prompt pool is explicitly **1024 per workload**, seed 12345. The generator's
single RNG spans short, medium and long, so a different pool size would change
the corpus. Model-config and tokenizer-template suffix hashes are checked;
new corpus hashes are recorded without saving prompt contents.

The four open-loop cases replay each historical `(id, scheduled_s, workload)`
record sorted by request ID, rather than drawing new arrivals. Their historical
rates are 0.6616102834535516, 1.0585764535256825, 1.323220566907103 and
1.5878646802885237 requests/s, with 80, 121, 147 and 181 scheduled arrivals.
Long C32/C64 explicitly cap requests at 32/64 to preserve their historical
single wave. Short C32/C64 retain continuous 75-second windows. The C8
validations retain their original durations/walls and maximum 100 requests.
The 180-second per-request deadline includes client queue time for all cases.

Capacity qualification requires at least 100 actual outcomes, 99% success,
TTFT p95 at most five seconds and legacy approximate TPOT p95 at most 100 ms.
Smaller samples and censored cases remain visible but cannot establish a new
capacity claim. Failed or incomplete streams record observed usage when
available and explicitly unknown/censored tokens otherwise; chunks are never
converted into token counts.

## Temporary launcher and safety boundaries

The stock image launcher still sees its approved production settings, including
its two-sequence graph guard. A private PATH `vllm` shim validates the complete
fresh production argv contract, rejects duplicate options or unrelated drift,
then changes only the approved capacity flags, ITL buckets and temporary worker
observer before executing the absolute vendor `vllm` executable. Existing
histogram bounds are preserved and fine ITL bins are added. Secrets remain in
the unchanged container environment; the bundle stores only an API-key
environment placeholder.

The native launcher provides authenticated Jupyter and metrics. Only its
temporary child environment disables the normal chat gateway and points
Jupyter at the owned run directory directly under `/tmp`. The supervisor serves
maintenance on port 8001. No control, status, log or result file relies on the
global-volume filesystem. The pod image and configured environment are not
modified by this temporary args override.

Freeze source and the manifest first. Then arm a fresh external lease, build
the payload with that lease and the fresh scrubbed argv contract, seal the
final args digest, verify the external watchdog process, and activate within
its freshness guard. The fixed lease stops testing at minute 110, permits
backup until minute 120, starts restoration no later than minute 120 and ends
at minute 150. Startup is capped at 35 minutes. OOM, swap growth, unsafe RAM or
shared-memory reserve, or engine/sidecar failure stops the test. A healthy
engine after ordinary runner failure remains in maintenance with `/tmp`
Jupyter available for bounded backup. There is no automatic reduction of the
test profile and no adaptive rate or scenario selection.

The restoration target is the **fresh enabled production snapshot**: native
64 GiB offload, decode capture sizes `[1,2]`, maximum two sequences and the same
pinned 0.17.0 image. It is not an older eager/offload-disabled snapshot.

## Replay, metrics and completion feed

Canonical warm-short, warm-medium and warm-long run first, preserving the cold
warm-short ordering. A temporary observer records native replay counts before
and after each of these already-excluded warmups. Their positive request-time
deltas are backed separately from startup capture counts. A named excluded
`graph-observer-disable-flush` scores a simple base-model answer, restores the
native replay method and requires a positive inactive acknowledgment before
natural-stop validation or any capacity measurement. Standard graph dispatch
telemetry remains enabled afterward. Warmup timings include diagnostic
overhead and are operational evidence.

Each case drains before and after measurement and waits for published counters.
Raw Prometheus snapshots, labeled counter/histogram deltas and one-second
samples are retained. Native ITL is called per-token only when its count equals
the sum of successful uncensored `(completion_tokens - 1)` intervals; otherwise
it is labeled event/partial. Historical client approximate TPOT comparisons
remain separate; historical server ITL is not compared without compatible
histogram evidence. Unique per-request salts plus preemption, host-hit and load
counters can establish consistent reload evidence, without per-request causal
attribution or conclusions from GPU utilization alone. If reload is not
exercised, the report says so.

The CPU usage/read/write gauges measure **active transfer pins**, not retained
host-cache occupancy (`CPUOffloadingSpec.build_metric_definitions`, installed
`vllm/v1/kv_offload/cpu/spec.py`). All five queue/transfer gauges must drain to
zero. GPU busy percentages, memory, power and clocks are sampled; unavailable
SM-active/DRAM profiling counters are identified explicitly.

`status.json` has one supervisor writer and always includes run/lease identity.
The runner writes immutable `completions/NNNN-case.json` files and an atomic
`completion-index.json` only after referenced case artifacts are finalized.
The expected complete feed has **30 events: 29 canonical cases plus the excluded
flush**. Each event has actual start/end Unix milliseconds and UTC, category,
cold/warm status, outcome counts, source/manifest hashes and artifact digests.
Its Grafana window is `[S-30s,E+30s]`, eligible after an additional scrape lag.
The independent screenshot worker queues four panel images asynchronously;
browser work never delays the load loop. Images stay local under the project's
`images-in-chat` run folder and are not uploaded with public results.

SIGTERM and the absolute testing deadline preserve scheduled partial outcomes,
write exactly one terminal case event and prohibit another case from starting.
Final run metadata is retained even on cancellation. After backup, the tester
restores and verifies enabled production, then disarms its external safety
processes. Pending historical Grafana captures may finish after restoration.

`combined_capacity_compare.py` produces a numerical historical/B comparison
from the frozen manifest and backed result files. It reports missing/censored
cases, legacy client TPOT and current native ITL separately, and treats this as
private base-engine capacity rather than public gateway capacity.
