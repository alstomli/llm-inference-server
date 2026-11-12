# Native CPU KV-cache experiment

**Draft tooling; do not repeat unchanged.** The
[November 12 attempt](KV-OFFLOAD-ATTEMPT-2026-11-12.md) was blocked before latency
measurements. Offloading was never activated. Runtime control/status/log state
must be independent of global-volume I/O before another attempt.

This experiment compares the existing BF16 Qwen3.8-27B engine with and without
vLLM 0.32.0's native `OffloadingConnector`. It does not change production gateway
quotas or enable offloading permanently. CPU **weight** offloading remains zero.

The new branch builds on the benchmark infrastructure in the still-open
[capacity PR #34](https://github.com/alstomli/llm-inference-server/pull/34).
Until that PR is merged, a PR from this branch to `main` includes its foundation.

## Decision

Use the native connector already installed in the production image, with a
64 GiB host tier, `blocks_per_chunk=1`, and `offload_prompt_only=false`. Native
source handles the model's full-attention and recurrent cache groups, including
group alignment and preemption transfers. This avoids another server or package
upgrade. Runtime retrieval, correctness, isolation and matched physical GPU
allocation remain mandatory gates; source support alone does not establish a
working or beneficial configuration.

## Controls

Both arms retain the same model revision, image, BF16 dtype, context 40960,
sequence ceiling 256, batched-token ceiling 8192, prefix caching, Mamba `align`
mode, hybrid cache manager, eager execution, loaded LoRA and vision allocation.
Performance requests are base-model text only, temperature zero, thinking off.
Image/video offload behavior is unvalidated.

`kv-offload-observer.py` subclasses the installed GPU worker and observes the
completed standard cache allocation. It deduplicates backing storage, records
concrete layouts, spec types, pages, block counts and alias relationships, and
does not serialize tensor contents or addresses. Both arms use this observer.
A measures the exact physical GPU-cache bytes. B and A2 request those bytes and
must match A's complete layout fingerprint. A mismatch stops the experiment.

`kv-offload-supervisor.py` is used only as a temporary pod-args override. The
original image, environment and mounts stay intact. It owns its process groups,
serves maintenance responses, verifies worker release before starting another
engine, and stops on engine failure, insufficient RAM/shared memory, OOM or swap
growth. There is no automatic reduction of concurrency or cache capacity.

An independently detached local watchdog restores the fresh original pod args
at minute 150, leaving 30 minutes for the original production launcher to load
and pass verification. The remote supervisor checks the same absolute deadline,
but its filesystem operations can block; the independent external watchdog is
essential, and the remote deadline was not observed firing reliably in this run.
B must end by minute 125; measured B stages stop earlier to allow transition.
All work uses the existing paid pod. Private access and snapshots are outside
the repository and must never be committed.

## Measurements

The tokenizer and seeded corpus generator match the preceding capacity test.
Input-ID hashes are recorded, and matched A/B scenarios use the same inputs,
offsets, seeds, offered concurrency and immediate-refill policy. Each unique
performance request has its own early prefix and engine cache salt.

Priority measurements are C8 short 512/128 and medium 2048/256 (34/33/33 trials),
three long 8192/512 burst waves at each of C32 and C64, and twenty requests each
at C2 for transfer overhead. Startup and each new shape are warmed separately.
C8 conclusions pool all 100 raw records; percentiles are not averaged across
trials. Long waves are bursts, not steady traffic measurements.

All performance requests have the same 300-second scheduled-arrival deadline.
The report also counts successful completion within 180 seconds over **all**
outcomes. Goodput includes scenario drain, and timeouts/cancellations remain in
the denominator.

Client streaming chunks can contain multiple tokens. Approximate TPOT and
content-chunk gaps are therefore separate from server ITL. Drained pre/post
server histogram deltas are used for ITL only when their count equals the exact
number of generated decode-token intervals. Otherwise they are labeled engine
output-event gaps. Histogram quantiles are bucket estimates. Counter resets
invalidate a delta; nonfinite numerical evidence is rejected.

CPU retrieval is checked by priming an answer-bearing 8192-token prefix,
evicting it from GPU with distinct requests while remaining within host cache
capacity, and replaying it with the same salt. The answer fact is near the
beginning, so the replay must recover state from the restored prefix. Positive
host-hit tokens and transfer bytes are required, followed by different-salt and
base-versus-LoRA identity isolation. Unique-request preemption/resume is a
separate gate: long-wave preemptions and CPU retrieval must coincide under
unique request salts. Prefix reuse alone does not prove resumed decode works.

Proposed practical acceptance thresholds are C8 success at least 99%, TTFT p95
at most 5 seconds, approximate TPOT p95 at most 100 ms, ordinary-traffic p95
regression at most 10%, and long C64 p95 end-to-end improvement at least 10%
without more than 5% successful-throughput loss. They are workload-specific,
not universal limits. An inconclusive or unhelpful result leaves offload off.

## Verification

The image CI runs the existing benchmark tests and `test_kv_offload.py` without
a GPU. The latter covers deadline accounting, histogram resets, strict answer
scoring, mocked startup and A/B/A2 transitions, expiry and memory-reclamation
gates, and cleanup before the original launcher is executed.

Numerical artifacts are stored in the ignored `benchmark-results` directory.
Final restoration verifies exact original pod configuration and model argv,
gateway limits, authenticated health and metrics, base/adapter streams, empty
queues, absence of benchmark processes and release of the host buffer. The
owned remote directory is backed up before removal.
