# Focused CPU-restored versus GPU-resident prefix test

The [completed 20-pair results](KV-RESIDENCY-RESULTS-2026-11-12.md) include
the raw numerical archive, verified residency counters and full restoration.

This diagnostic uses one fixed native-offload engine: 4 GiB GPU KV budget,
8 GiB CPU tier, original BF16 model/image, context 40960, sequence ceiling 2,
eager execution, loaded LoRA/vision allocation, prefix caching and aligned Mamba
caches. Weight offloading remains zero. It measures residency effects rather
than production capacity, concurrent traffic or decode preemption.

The exact target is 3200 input IDs: a 3136-token reusable prefix and identical
64-token uncached suffix, followed by 128 forced output tokens. Each fresh trial
salt is shared by its prime and both measurements. A known-answer fact is near
the prefix beginning and its question is in the suffix. Prompt IDs and output
text remain private; only hashes, counters and correctness booleans are stored.

After actual 83-block allocation is verified, 22 sequential distinct 3200/1
fillers with `max_offload_tokens=0` evict the GPU prefix without populating CPU.
Every target uses `max_offload_tokens=3136`. Acceptance requires exactly 3136
device-hit tokens and no host hit/load for GPU residency, or zero device hits,
3136 host-hit tokens and positive completed CPU-load bytes for restored residency.
Partial/mixed hits are rejected. Fillers must produce zero CPU store/load bytes.
Measured targets must not store new CPU data.

After a proof-gated warm pair, 20 paired trials use seeded balanced order (10
GPU→CPU and 10 CPU→GPU). All requests are sequential. Finalized idle queues,
zero active transfers, a stable publication interval and one quiet second are
required around measurements. Each accepted target must have exactly one
finished request, 3200 prompt tokens, 64 computed prefill tokens, 128 output
tokens, 127 ITL observations and zero preemptions.

Server TTFT histogram sum/count is the primary paired timing. Loopback first-text
TTFT and client content-chunk gaps are separate. ITL histogram counts are verified
against token counts; means use sum/count and quantiles are bucket estimates.
The engine preserves TTFT buckets and adds 1 ms ITL buckets through 200 ms plus
250/500/1000/2000/5000 ms tails. Uncertainty resamples paired requests, not token
intervals. Native CUDA transfer time supports the diagnosis and is not TTFT.

The inline frozen bootstrap contains no credential values and materializes only
an owned `/tmp` directory. Control/status/log/results stay local. Normal ingress
returns maintenance 503; an authenticated bounded status/backup endpoint uses
the existing password. A fresh external watchdog independently restores original
pod args at minute 150, with completion verification due by minute 180. Startup
has a bounded 35-minute allowance. Correctness/resource failures restore;
recoverable classification failures can hold an idle engine for at most ten
minutes while preserving all failures and previously accepted pairs.

Original configuration, model argv, health/metrics, base/adapter streams, quotas,
queues, process cleanup and actual CPU-buffer/shared-memory reclamation must be
verified afterward. No permanent offload enablement or software/image upgrade is
part of this test. It builds on the unmerged infrastructure in PRs #34 and #35.
