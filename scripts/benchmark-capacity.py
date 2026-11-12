"""Bounded continuous vLLM load generator. Never persists prompts, text or secrets.

Run on the inference host against its private loopback endpoint. This measures the
engine, not public gateway capacity. Requires aiohttp; transformers only for CLI.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time
import uuid
from urllib.parse import urlsplit

import aiohttp

WORKLOADS = {"short": (512, 128), "medium": (2048, 256), "long": (8192, 512)}
METRICS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
           "gpu_cache_usage_perc", "num_preemptions_total", "prompt_tokens_total",
           "generation_tokens_total", "request_prefill_time_seconds_count",
           "request_decode_time_seconds_count")


@dataclass
class Config:
    name: str = "capacity"
    url: str = "http://127.0.0.1:8000"
    model: str = "qwen-bf16-base"
    workload: str = "medium"
    concurrency: int = 1
    duration: float = 60
    wall_seconds: float = 240
    request_deadline: float = 180
    max_requests: int = 0
    arrival_rate: float = 0
    max_backlog: int = 1024
    seed: int = 12345
    force_length: bool = True
    metrics_interval: float = 1
    collect_gpu: bool = True
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    runtime_fingerprint: str = "unknown"
    scenario_kind: str = "capacity"
    cache_namespace: str = "auto"

    def validate(self):
        for key in ("duration", "wall_seconds", "request_deadline", "arrival_rate", "metrics_interval"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError("Timing and rate values must be finite")
        if self.metrics_interval < .05:
            raise ValueError("Metrics sampling interval must be at least50ms")
        parsed = urlsplit(self.url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Only private loopback HTTP targets are allowed")
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("Target must be a credential-free loopback origin")
        if self.workload not in (*WORKLOADS, "mixed") or not 1 <= self.concurrency <= 1024:
            raise ValueError("Invalid workload or concurrency (hard maximum 1024)")
        if not 1 <= self.max_backlog <= 1024 or self.duration <= 0 or self.wall_seconds <= 0:
            raise ValueError("Invalid queue or stage duration")
        if not 0 < self.request_deadline <= 180 or self.arrival_rate < 0 or self.max_requests < 0:
            raise ValueError("Invalid arrival or deadline limit")
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in self.name):
            raise ValueError("Name must be a safe filename component")
        if self.scenario_kind not in ("capacity", "warmup", "natural_stop", "stagger", "sustained", "tuning"):
            raise ValueError("Invalid scenario kind")


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


class SSE:
    """Incremental byte parser; fragmented UTF-8, CRLF and multiline data work."""
    def __init__(self):
        self.buffer = b""

    def feed(self, data):
        self.buffer += data
        if len(self.buffer) > 2 * 1024 * 1024:
            raise ValueError("Oversized SSE frame")
        self.buffer = self.buffer.replace(b"\r\n", b"\n")
        frames = []
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            lines = [line[5:].lstrip(b" ") for line in frame.split(b"\n") if line.startswith(b"data:")]
            if lines:
                frames.append(b"\n".join(lines).decode("utf-8"))
        return frames


class Prompts:
    """Precompute exact tokenizer-verified chat-template IDs before timing starts.

    Unique random user prefixes and per-request engine cache salts prevent
    inter-request prefix-cache reuse. No corpus is persisted by the runner.
    """
    def __init__(self, tokenizer, seed, size=2048):
        self.seed, self.size = seed, size
        self.items = {}
        rng = random.Random(seed)
        words = ["river", "window", "garden", "planet", "yellow", "forest", "paper", "stone",
                 "silver", "cloud", "orange", "bridge", "candle", "simple", "number", "table"]
        template = tokenizer.apply_chat_template([{"role":"user", "content":"CAPACITY_SENTINEL"}],
                                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix, suffix = template.split("CAPACITY_SENTINEL")
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        assert tokenizer.decode(suffix_ids, skip_special_tokens=False) == suffix
        self.template_suffix_sha256 = hashlib.sha256(suffix.encode()).hexdigest()
        for name, (length, output) in WORKLOADS.items():
            corpus = []
            for i in range(size):
                # UUID-like prefix occurs in the first user-content block.
                text = f"{rng.getrandbits(128):032x} " + " ".join(rng.choices(words, k=length + 32))
                body = tokenizer.encode(text, add_special_tokens=False)
                body_length = length - len(prefix_ids) - len(suffix_ids)
                ids = prefix_ids + body[:body_length] + suffix_ids
                assert len(ids) == length and ids[-len(suffix_ids):] == suffix_ids
                corpus.append((ids, output))
            self.items[name] = corpus

    def get(self, workload, request_id):
        ids, output = self.items[workload][request_id % self.size]
        # Reuse is forbidden within a scenario; caller bounds request counts.
        if request_id >= self.size:
            raise ValueError("Prompt pool exhausted; increase corpus size before timing")
        return ids, output


class Writer:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.requests = (self.root / (config.name + ".requests.jsonl")).open("x", encoding="utf-8")
        self.samples = (self.root / (config.name + ".metrics.jsonl")).open("x", encoding="utf-8")

    def request(self, row):
        self.requests.write(json.dumps(row, allow_nan=False) + "\n")
        self.requests.flush()

    def sample(self, row):
        self.samples.write(json.dumps(row, allow_nan=False) + "\n")
        self.samples.flush()

    def close(self, summary):
        self.requests.close()
        self.samples.close()
        destination = self.root / (self.config.name + ".summary.json")
        temp = destination.with_suffix(".tmp")
        temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(destination)


def aggregate(config, records, elapsed, peak_outstanding=0, peak_backlog=0, stopped=None):
    good = [r for r in records if r["outcome"] == "success"]
    result = {"config": asdict(config), "elapsed_s": elapsed, "outcomes": len(records),
              "successful": len(good), "success_rate": len(good) / len(records) if records else 0,
              "completed_requests_s": len(good) / elapsed if elapsed else 0,
              "output_tokens_s": sum(r.get("completion_tokens", 0) for r in good) / elapsed if elapsed else 0,
              "total_output_tokens": sum(r.get("completion_tokens", 0) for r in good),
              "peak_outstanding": peak_outstanding, "peak_client_backlog": peak_backlog,
              "stopped": stopped, "outcome_counts": {}}
    result["avg_tpot_definition"] = "Approximate streaming-derived (last content chunk - first content chunk)/(usage output tokens - 1); chunks may contain multiple tokens"
    for r in records:
        result["outcome_counts"][r["outcome"]] = result["outcome_counts"].get(r["outcome"], 0) + 1
    for key in ("ttft_s", "e2e_s", "avg_tpot_s", "client_wait_s", "dispatch_lag_s", "chunk_gap_p95_s"):
        values = [r[key] for r in good if r.get(key) is not None]
        result[key] = {"p50": percentile(values, 50), "p95": percentile(values, 95), "p99": percentile(values, 99)}
    result["chat_slo_pass"] = (stopped is None and config.scenario_kind == "capacity" and config.force_length
                               and not config.arrival_rate and len(records) >= 100 and result["success_rate"] >= .99
                               and result["ttft_s"]["p95"] is not None and result["ttft_s"]["p95"] <= 5
                               and result["avg_tpot_s"]["p95"] is not None and result["avg_tpot_s"]["p95"] <= .1)
    result["chat_slo_provisional"] = len(records) < 100
    return result


def completed_scenario_matches(summary, expected, cache_namespace):
    """Reuse only complete, successful evidence from the exact same experiment."""
    keys = ("model", "workload", "concurrency", "duration", "request_deadline", "max_requests",
            "arrival_rate", "max_backlog", "seed", "force_length", "max_num_seqs",
            "max_num_batched_tokens", "runtime_fingerprint", "scenario_kind")
    previous = summary.get("config", {})
    return (all(previous.get(key) == getattr(expected, key) for key in keys)
            and summary.get("all_outcomes_accounted") is True and not summary.get("stopped")
            and summary.get("outcomes", 0) > 0 and summary.get("successful") == summary.get("outcomes")
            and summary.get("cache_namespace") == cache_namespace
            and summary.get("cache_isolation", "").startswith("unique session/scenario/request"))


class Runner:
    def __init__(self, config, prompts, writer, *, api_key=None):
        config.validate()
        self.config, self.prompts, self.writer = config, prompts, writer
        self.headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.records, self.tasks = [], set()
        self.outstanding = self.peak_outstanding = self.peak_backlog = 0
        self.stop = asyncio.Event()
        self.stop_reason = None
        self.next_id = 0
        self.rng = random.Random(config.seed)
        self.cache_namespace = uuid.uuid4().hex if config.cache_namespace == "auto" else config.cache_namespace

    def workload(self):
        return self.rng.choices(list(WORKLOADS), weights=[50, 35, 15])[0] if self.config.workload == "mixed" else self.config.workload

    def record(self, row):
        self.records.append(row)
        self.writer.request(row)
        if len(self.records) >= 20:
            failures = sum(r["outcome"] not in ("success", "stage_cancel") for r in self.records)
            if failures / len(self.records) > .05:
                self.stop_reason = "failure_rate_exceeded_5_percent"
                self.stop.set()

    def new_row(self, scheduled):
        row = {"id": self.next_id, "workload": self.workload(), "scheduled_s": scheduled - self.started,
               "enqueued_s": time.monotonic() - self.started, "submitted_s": None,
               "outcome": "pending", "forced_length": self.config.force_length}
        row["scheduler_lag_s"] = row["enqueued_s"] - row["scheduled_s"]
        self.next_id += 1
        return row

    async def request(self, session, row):
        scheduled = self.started + row["scheduled_s"]
        deadline = scheduled + self.config.request_deadline
        first = last = None
        gaps, usage, finished, done = [], None, False, False
        try:
            async with asyncio.timeout_at(deadline):
                ids, output = self.prompts.get(row["workload"], row["id"])
                row.update(requested_prompt_tokens=len(ids), requested_output_tokens=output,
                           submitted_s=time.monotonic() - self.started)
                row["client_wait_s"] = row["submitted_s"] - row["scheduled_s"]
                row["dispatch_lag_s"] = row["client_wait_s"]
                row["client_queue_wait_s"] = row["submitted_s"] - row["enqueued_s"]
                payload = {"model": self.config.model, "prompt": ids, "max_tokens": output,
                           "temperature": 0, "seed": self.config.seed + row["id"], "stream": True,
                           "stream_options": {"include_usage": True}, "ignore_eos": self.config.force_length,
                           "cache_salt": hashlib.sha256(f"{self.cache_namespace}/{self.config.seed}/{self.config.name}/{row['id']}".encode()).hexdigest()}
                if self.config.force_length:
                    payload["min_tokens"] = output
                self.outstanding += 1
                self.peak_outstanding = max(self.peak_outstanding, self.outstanding)
                try:
                    async with session.post(self.config.url.rstrip("/") + "/v1/completions", json=payload,
                                            headers=self.headers) as response:
                        row["http_status"] = response.status
                        if response.status != 200:
                            row["outcome"] = "http_error"
                            return
                        decoder = SSE()
                        async for block in response.content.iter_any():
                            for frame in decoder.feed(block):
                                now = time.monotonic()
                                if frame == "[DONE]":
                                    done = True
                                    continue
                                event = json.loads(frame)
                                if event.get("error"):
                                    row["outcome"] = "sse_error"
                                    return
                                if event.get("usage"):
                                    usage = event["usage"]
                                for choice in event.get("choices", []):
                                    if choice.get("finish_reason"):
                                        finished = choice["finish_reason"] in ("stop", "length")
                                        row["finish_reason"] = choice["finish_reason"] if choice["finish_reason"] in ("stop", "length", "abort") else "other"
                                    if choice.get("text") or choice.get("delta", {}).get("content"):
                                        if first is None:
                                            first = now
                                        if last is not None:
                                            gaps.append(now - last)
                                        last = now
                        if decoder.buffer.strip():
                            row["outcome"] = "truncated_sse"
                            return
                finally:
                    self.outstanding -= 1
                if not (done and finished and first is not None and usage
                        and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] > 0
                        and type(usage.get("prompt_tokens")) is int):
                    row["outcome"] = "incomplete_stream"
                    return
                row.update(prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage["completion_tokens"])
                if row["prompt_tokens"] != len(ids) or (self.config.force_length and row["completion_tokens"] != output):
                    row["outcome"] = "token_count_mismatch"
                    return
                row["outcome"] = "success"
        except TimeoutError:
            row["outcome"] = "deadline"
        except asyncio.CancelledError:
            row["outcome"] = "stage_cancel"
            raise
        except (aiohttp.ClientError, OSError):
            row["outcome"] = "transport_error"
        except (ValueError, KeyError, TypeError):
            row["outcome"] = "protocol_error"
        finally:
            ended = time.monotonic()
            row["ended_s"] = ended - self.started
            row["e2e_s"] = ended - scheduled
            row["ttft_s"] = first - scheduled if first is not None else None
            row["first_content_s"] = first - self.started if first is not None else None
            row["last_content_s"] = last - self.started if last is not None else None
            row["engine_ttft_s"] = first - self.started - row["submitted_s"] if first is not None else None
            row["avg_tpot_s"] = ((last - first) / (row["completion_tokens"] - 1)
                                 if first is not None and last is not None and row.get("completion_tokens", 0) > 1 else None)
            row["chunk_count"] = len(gaps) + (1 if first is not None else 0)
            row["chunk_gap_p95_s"] = percentile(gaps, 95)
            self.record(row)

    async def metrics(self, session):
        previous_cpu = None
        previous_process_cpu, previous_wall = time.process_time(), time.monotonic()
        expected = previous_wall
        while not self.stop.is_set():
            proc = None
            now, process_cpu = time.monotonic(), time.process_time()
            sample = {"t_s": now - self.started, "client_outstanding": self.outstanding,
                      "client_backlog": getattr(self, "queue", None).qsize() if hasattr(self, "queue") else 0}
            sample["sampler_lag_s"] = max(0, now - expected)
            sample["loadgen_cpu_percent_one_core"] = 100 * (process_cpu-previous_process_cpu) / max(.000001,now-previous_wall)
            previous_process_cpu, previous_wall = process_cpu, now
            try:
                async with session.get(self.config.url.rstrip("/") + "/metrics", headers=self.headers,
                                       timeout=aiohttp.ClientTimeout(total=3)) as response:
                    if response.status == 200:
                        text = await response.text()
                        for line in text.splitlines():
                            if line.startswith("#"):
                                continue
                            parts = line.rsplit(" ", 1)
                            if len(parts) != 2:
                                continue
                            name = parts[0].split("{", 1)[0].removeprefix("vllm:")
                            if name in METRICS:
                                sample[name] = sample.get(name, 0) + float(parts[1])
                if Path("/proc/stat").exists():
                    cpu = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
                    if previous_cpu:
                        total = sum(cpu) - sum(previous_cpu)
                        sample["host_cpu_percent"] = 100 * (1 - (cpu[3] - previous_cpu[3]) / total) if total else 0
                    previous_cpu = cpu
                if not self.config.collect_gpu:
                    self.writer.sample(sample)
                    expected = time.monotonic() + self.config.metrics_interval
                    await asyncio.sleep(self.config.metrics_interval)
                    continue
                proc = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                                            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.DEVNULL)
                raw, _ = await asyncio.wait_for(proc.communicate(), 3)
                values = raw.decode().strip().split(",")
                sample.update(gpu_util_percent=float(values[0]), gpu_memory_mib=float(values[1]), gpu_total_mib=float(values[2]))
            except (aiohttp.ClientError, OSError, ValueError, TimeoutError):
                sample["sample_error"] = True
            finally:
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            self.writer.sample(sample)
            expected = time.monotonic() + self.config.metrics_interval
            await asyncio.sleep(self.config.metrics_interval)

    async def closed_worker(self, session):
        while not self.stop.is_set() and time.monotonic() < self.arrivals_end:
            if self.config.max_requests and self.next_id >= self.config.max_requests:
                return
            row = self.new_row(time.monotonic())
            await self.request(session, row)

    async def open_worker(self, session):
        while True:
            row = await self.queue.get()
            try:
                if row is None:
                    return
                if self.stop.is_set():
                    row.update(outcome="safety_cancel", ended_s=time.monotonic() - self.started)
                    self.record(row)
                    continue
                if time.monotonic() >= self.started + row["scheduled_s"] + self.config.request_deadline:
                    row.update(outcome="deadline", ended_s=time.monotonic() - self.started)
                    self.record(row)
                else:
                    await self.request(session, row)
            finally:
                self.queue.task_done()

    async def arrivals(self):
        scheduled = self.started
        while scheduled < self.arrivals_end and not self.stop.is_set():
            if self.config.max_requests and self.next_id >= self.config.max_requests:
                break
            await asyncio.sleep(max(0, scheduled - time.monotonic()))
            row = self.new_row(scheduled)
            try:
                self.queue.put_nowait(row)
                self.peak_backlog = max(self.peak_backlog, self.queue.qsize())
            except asyncio.QueueFull:
                row.update(outcome="client_drop", ended_s=time.monotonic() - self.started,
                           dispatch_lag_s=time.monotonic() - scheduled)
                self.record(row)
            scheduled += self.rng.expovariate(self.config.arrival_rate)

    async def run(self):
        self.started = time.monotonic()
        self.arrivals_end = self.started + self.config.duration
        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(limit=self.config.concurrency + 4, limit_per_host=0)
        sampler = None
        async def execute(session):
            if self.config.arrival_rate:
                # Queue bound is separate from the <=1024 submitted bound.
                self.queue = asyncio.Queue(maxsize=self.config.max_backlog)
                workers = [asyncio.create_task(self.open_worker(session)) for _ in range(self.config.concurrency)]
                self.tasks.update(workers)
                await self.arrivals()
                await self.queue.join()
                for _ in workers:
                    await self.queue.put(None)
                await asyncio.gather(*workers)
            else:
                workers = [asyncio.create_task(self.closed_worker(session)) for _ in range(self.config.concurrency)]
                self.tasks.update(workers)
                await asyncio.gather(*workers)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
            sampler = asyncio.create_task(self.metrics(session))
            execution = asyncio.create_task(execute(session))
            stopper = asyncio.create_task(self.stop.wait())
            try:
                async with asyncio.timeout(self.config.wall_seconds):
                    done, _ = await asyncio.wait([execution, stopper], return_when=asyncio.FIRST_COMPLETED)
                    if execution in done:
                        await execution
            except TimeoutError:
                self.stop_reason = "stage_wall_deadline"
            except asyncio.CancelledError:
                self.stop_reason = "controller_cancelled"
            finally:
                self.stop.set()
                execution.cancel()
                stopper.cancel()
                for task in self.tasks:
                    task.cancel()
                await asyncio.gather(execution, stopper, *self.tasks, return_exceptions=True)
                sampler.cancel()
                await asyncio.gather(sampler, return_exceptions=True)
        try:
            self.stop.set()
            if hasattr(self, "queue"):
                while not self.queue.empty():
                    row = self.queue.get_nowait()
                    if row is not None:
                        row.update(outcome="stage_cancel", ended_s=time.monotonic() - self.started)
                        self.record(row)
                    self.queue.task_done()
            result = aggregate(self.config, self.records, time.monotonic() - self.started,
                               self.peak_outstanding, self.peak_backlog, self.stop_reason)
            result["scheduled_requests"] = self.next_id
            result["cache_isolation"] = "unique session/scenario/request prefix-cache salts plus unique early prompt prefixes"
            result["cache_namespace"] = self.cache_namespace
            result["all_outcomes_accounted"] = self.next_id == len(self.records)
            result["scheduled_arrival_span_s"] = max((r["scheduled_s"] for r in self.records), default=0)
            result["arrival_window_output_tokens_s"] = sum(r.get("completion_tokens",0) for r in self.records
                    if r["outcome"] == "success" and r.get("ended_s", 1e99) <= self.config.duration) / self.config.duration
        finally:
            self.writer.close(result)
        return result


def analyze(root):
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(root).glob("*.summary.json"))]
    if not rows:
        return {"scenarios": [], "knee": {}, "chat_capacity": None}
    classification_path = Path(root) / "scenario-classifications.json"
    classifications = json.loads(classification_path.read_text()) if classification_path.exists() else {}
    for row in rows:
        row["measurement_state"] = classifications.get(row["config"]["name"], {}).get("measurement_state", "steady")
    groups = {}
    for row in rows:
        c = row["config"]
        key = f"{c['runtime_fingerprint']}|seq={c['max_num_seqs']}|batch={c['max_num_batched_tokens']}"
        groups.setdefault(key, []).append(row)
    group_results = {}
    for group, scenarios in groups.items():
        measured = [r for r in scenarios if r["config"]["scenario_kind"] == "capacity"
                    and r["config"]["force_length"] and not r["config"]["arrival_rate"]
                    and r["measurement_state"] == "steady" and not r.get("stopped")]
        knee = {}
        for workload in WORKLOADS:
            eligible = [r for r in measured if r["config"]["workload"] == workload and r["success_rate"] >= .99]
            if eligible:
                peak = max(r["output_tokens_s"] for r in eligible)
                knee[workload] = min(r["config"]["concurrency"] for r in eligible if r["output_tokens_s"] >= .95 * peak)
        short = {r["config"]["concurrency"] for r in measured if r["config"]["workload"] == "short" and r["chat_slo_pass"]}
        medium = {r["config"]["concurrency"] for r in measured if r["config"]["workload"] == "medium" and r["chat_slo_pass"]}
        group_results[group] = {"knee":knee, "chat_capacity":max(short & medium) if short & medium else None}
    only = next(iter(group_results.values())) if len(group_results) == 1 else {"knee":{}, "chat_capacity":None}
    result = {"scenarios": rows, **only, "configuration_groups":group_results,
              "scope": "Private base-text vLLM; forced-length synthetic; public gateway capacity excluded"}
    (Path(root) / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (Path(root) / "aggregate.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ["name", "workload", "concurrency", "arrival_rate", "outcomes", "success_rate", "output_tokens_s",
                  "completed_requests_s", "ttft_p95_s", "tpot_p95_s", "e2e_p95_s", "chat_slo_pass"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r["config"][k] for k in fields[:4]}, **{k:r[k] for k in fields[4:8]},
                             "ttft_p95_s":r["ttft_s"]["p95"], "tpot_p95_s":r["avg_tpot_s"]["p95"],
                             "e2e_p95_s":r["e2e_s"]["p95"], "chat_slo_pass":r["chat_slo_pass"]})
    return result


async def main_async(args):
    if args.analyze:
        result = analyze(args.output)
        print(json.dumps({"scenarios":len(result["scenarios"]), "knee":result["knee"], "chat_capacity":result["chat_capacity"]}))
        return
    from transformers import AutoTokenizer
    config = Config(**{key:getattr(args, key) for key in Config.__dataclass_fields__})
    config.validate()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    prompts = Prompts(tokenizer, config.seed, args.corpus_size)
    writer = Writer(args.output, config)
    runner = Runner(config, prompts, writer, api_key=os.getenv("VLLM_API_KEY"))
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, main_task.cancel)
        except (NotImplementedError, RuntimeError):
            pass
    result = await runner.run()
    print(json.dumps(result), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["continuous"], default="continuous")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--output", type=Path, default=Path("benchmark-results/capacity"))
    p.add_argument("--tokenizer", default="/workspace/qwen-bf16/model")
    p.add_argument("--corpus-size", type=int, default=2048)
    for key, field in Config.__dataclass_fields__.items():
        if key == "force_length":
            p.add_argument("--natural-stop", dest=key, action="store_false", default=True)
        elif type(field.default) is bool:
            p.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction, default=field.default)
        else:
            kind = float if key in ("duration", "wall_seconds", "request_deadline", "arrival_rate", "metrics_interval") else type(field.default)
            p.add_argument("--" + key.replace("_", "-"), type=kind, default=field.default)
    return p


if __name__ == "__main__":
    asyncio.run(main_async(parser().parse_args()))
