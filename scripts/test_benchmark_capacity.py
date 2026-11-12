import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from aiohttp import web

SPEC = importlib.util.spec_from_file_location("capacity", Path(__file__).with_name("benchmark-capacity.py"))
capacity = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capacity
SPEC.loader.exec_module(capacity)


class Prompts:
    def get(self, workload, request_id):
        return [10, 11, 12], 4


class CapacityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.active = self.peak = 0
        self.arrived, self.ended = [], []
        self.delay, self.behavior, self.slow_first = .015, "ok", False
        self.slow_ended = None
        self.case_number = 0
        self.salts = []
        app = web.Application()
        app.router.add_post("/v1/completions", self.respond)
        app.router.add_get("/metrics", lambda _: web.Response(text="vllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0.2\n"))
        self.server = web.AppRunner(app)
        await self.server.setup()
        self.site = web.TCPSite(self.server, "127.0.0.1", 0)
        await self.site.start()
        self.url = "http://127.0.0.1:" + str(self.site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.server.cleanup()
        self.tmp.cleanup()

    async def respond(self, request):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.arrived.append(time.monotonic())
        try:
            payload = await request.json()
            self.salts.append(payload.get('cache_salt'))
            if self.behavior == "http":
                return web.Response(status=503, text="private-error-body-never-persist")
            response = web.StreamResponse(headers={"Content-Type":"text/event-stream"})
            await response.prepare(request)
            slow = self.slow_first and payload.get("seed") == 12345
            await asyncio.sleep(.25 if slow else self.delay)
            if self.behavior == "sse_error":
                await response.write(b'data: {"error":{"message":"secret-value"}}\n\n')
            else:
                frames = [json.dumps({"choices":[{"text":"private-output-\u96ea", "finish_reason":None}]}),
                          json.dumps({"choices":[{"text":"second-chunk", "finish_reason":"length"}]}),
                          json.dumps({"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4}}), "[DONE]"]
                if self.behavior == "missing_usage":
                    frames.pop(2)
                if self.behavior == "truncated":
                    frames.pop()
                if self.behavior == "abort":
                    frames[1] = json.dumps({"choices":[{"text":"second-chunk", "finish_reason":"abort"}]})
                if self.behavior == "no_content":
                    frames[0] = json.dumps({"choices":[{"text":"", "finish_reason":None}]})
                    frames[1] = json.dumps({"choices":[{"text":"", "finish_reason":"length"}]})
                for frame in frames:
                    block = ("data: " + frame + "\r\n\r\n").encode()
                    if self.behavior == "fragmented":
                        for i in range(0, len(block), 7):
                            await response.write(block[i:i+7])
                    else:
                        await response.write(block)
                    await asyncio.sleep(.002)
            await response.write_eof()
            return response
        except (ConnectionResetError, asyncio.CancelledError):
            return web.Response()
        finally:
            self.active -= 1
            self.ended.append(time.monotonic())
            if self.slow_first and payload.get("seed") == 12345:
                self.slow_ended = time.monotonic()

    async def run_case(self, **overrides):
        self.case_number += 1
        config = capacity.Config(name="case" + str(self.case_number), url=self.url, duration=2,
                                 wall_seconds=5, collect_gpu=False, **overrides)
        writer = capacity.Writer(self.tmp.name, config)
        runner = capacity.Runner(config, Prompts(), writer, api_key="super-secret-credential")
        async def no_metrics(_):
            await runner.stop.wait()
        runner.metrics = no_metrics
        result = await runner.run()
        return result, runner

    async def test_continuous_refill_and_actual_usage_not_chunks(self):
        self.slow_first = True
        result, runner = await self.run_case(concurrency=3, max_requests=12)
        self.assertEqual(result["successful"], 12)
        self.assertEqual(result["total_output_tokens"], 48)
        self.assertEqual(result["peak_outstanding"], 3)
        self.assertGreaterEqual(self.peak, 2)
        self.assertLess(self.arrived[3], self.slow_ended)
        self.assertTrue(all(r["completion_tokens"] == 4 and r["chunk_count"] == 2 for r in runner.records))
        self.assertTrue(result["all_outcomes_accounted"])
        self.assertEqual(len(set(self.salts)), 12)
        self.assertTrue(all(isinstance(salt,str) and len(salt)==64 for salt in self.salts))

    async def test_open_loop_queue_bound_and_explicit_drops(self):
        self.delay = .12
        result, runner = await self.run_case(concurrency=2, arrival_rate=1000, max_backlog=3, request_deadline=.05)
        self.assertLessEqual(result["peak_outstanding"], 2)
        self.assertLessEqual(result["peak_client_backlog"], 3)
        self.assertGreater(result["outcome_counts"].get("client_drop", 0), 0)
        self.assertTrue(result["all_outcomes_accounted"])

    async def test_open_loop_deadline_includes_time_in_client_queue(self):
        self.delay = .12
        result, runner = await self.run_case(concurrency=1, arrival_rate=50, max_backlog=3,
                                             request_deadline=.05, max_requests=12)
        self.assertGreater(result["outcome_counts"].get("deadline", 0), 0)
        self.assertTrue(result["all_outcomes_accounted"])
        self.assertEqual(result['outcomes'], 12)
        for row in runner.records:
            if row["outcome"] == "deadline":
                self.assertLess(row["ended_s"] - row["scheduled_s"], .3)

    async def test_wall_limit_preserves_partial_and_cancels_all(self):
        self.delay = 1.5
        config = capacity.Config(url=self.url, concurrency=4, duration=2, wall_seconds=.1, collect_gpu=False)
        runner = capacity.Runner(config, Prompts(), capacity.Writer(self.tmp.name, config))
        result = await runner.run()
        self.assertLess(result["elapsed_s"], 1)
        self.assertEqual(result["outcomes"], 4)
        self.assertTrue(result["all_outcomes_accounted"])
        self.assertEqual(result["stopped"], "stage_wall_deadline")
        self.assertEqual(json.loads((Path(self.tmp.name) / "capacity.summary.json").read_text())["outcomes"], 4)

    async def test_protocol_failures_never_count_as_success(self):
        for behavior, outcome in [("http", "http_error"), ("sse_error", "sse_error"),
                                  ("missing_usage", "incomplete_stream"), ("truncated", "incomplete_stream"),
                                  ("abort", "incomplete_stream"), ("no_content", "incomplete_stream")]:
            with self.subTest(behavior=behavior):
                self.behavior = behavior
                result, _ = await self.run_case(max_requests=1)
                self.assertEqual(result["success_rate"], 0)
                self.assertEqual(result["outcome_counts"], {outcome:1})

    async def test_fragmented_network_sse_reports_actual_tokens(self):
        self.behavior = "fragmented"
        result, _ = await self.run_case(max_requests=1)
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["total_output_tokens"], 4)

    async def test_artifacts_omit_credentials_prompts_and_responses(self):
        await self.run_case(max_requests=1)
        content = "".join(p.read_text() for p in Path(self.tmp.name).iterdir())
        for private in ("super-secret-credential", "private-output", "second-chunk", "Authorization", "Bearer"):
            self.assertNotIn(private, content)

    async def test_controller_cancellation_still_writes_summary(self):
        self.delay = .3
        config = capacity.Config(url=self.url, concurrency=2, duration=1, collect_gpu=False)
        runner = capacity.Runner(config, Prompts(), capacity.Writer(self.tmp.name, config))
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(.03)
        task.cancel()
        result = await task
        self.assertEqual(result["stopped"], "controller_cancelled")
        self.assertTrue(result["all_outcomes_accounted"])

    def test_fragmented_multiline_sse_and_oversized_limit(self):
        decoder = capacity.SSE()
        events = []
        for b in 'data: {"text":"\u96ea"}\r\n\r\n'.encode():
            events.extend(decoder.feed(bytes([b])))
        self.assertEqual(json.loads(events[0]), {"text":"\u96ea"})
        self.assertEqual(decoder.feed(b"data: one\ndata: two\n\n"), ["one\ntwo"])
        with self.assertRaises(ValueError):
            decoder.feed(b"x" * (2 * 1024 * 1024 + 1))

    def test_slo_requires_all_outcomes_and_100_requests(self):
        config = capacity.Config()
        good = {"outcome":"success", "completion_tokens":128, "ttft_s":1, "e2e_s":4, "avg_tpot_s":.03}
        rows = [dict(good) for _ in range(100)]
        self.assertTrue(capacity.aggregate(config, rows, 10)["chat_slo_pass"])
        self.assertFalse(capacity.aggregate(config, rows[:99], 10)["chat_slo_pass"])
        self.assertFalse(capacity.aggregate(config, rows, 10, stopped='stage_wall_deadline')["chat_slo_pass"])
        self.assertFalse(capacity.aggregate(config, rows + [{"outcome":"deadline"}] * 2, 10)["chat_slo_pass"])

    async def test_safety_stop_prevents_continued_submission(self):
        self.behavior = "http"
        result, _ = await self.run_case(concurrency=4, max_requests=100)
        self.assertEqual(result["stopped"], "failure_rate_exceeded_5_percent")
        self.assertLessEqual(len(self.arrived), 23)
        self.assertTrue(result["all_outcomes_accounted"])

    def test_analysis_separates_configs_and_excludes_warmup(self):
        good = {"outcome":"success", "completion_tokens":128, "ttft_s":1, "e2e_s":4, "avg_tpot_s":.03}
        for name, workload, seq, c, kind, elapsed in [
            ("short", "short", 256, 8, "capacity", 10),
            ("medium", "medium", 128, 8, "capacity", 10),
            ("warm", "short", 256, 1, "warmup", .01),
        ]:
            config = capacity.Config(name=name, workload=workload, max_num_seqs=seq,
                                     concurrency=c, scenario_kind=kind)
            summary = capacity.aggregate(config, [dict(good) for _ in range(100)], elapsed)
            (Path(self.tmp.name) / (name + ".summary.json")).write_text(json.dumps(summary))
        analysis = capacity.analyze(self.tmp.name)
        self.assertIsNone(analysis["chat_capacity"])
        self.assertEqual(len(analysis["configuration_groups"]), 2)
        self.assertTrue(all(r["chat_capacity"] is None for r in analysis["configuration_groups"].values()))
        self.assertEqual(analysis["configuration_groups"]["unknown|seq=256|batch=0"]["knee"]["short"], 8)
        self.assertTrue((Path(self.tmp.name) / "aggregate.csv").exists())
        (Path(self.tmp.name) / 'scenario-classifications.json').write_text(json.dumps({'short':{'measurement_state':'cold_mixed'}}))
        classified=capacity.analyze(self.tmp.name)
        self.assertNotIn('short',classified['configuration_groups']['unknown|seq=256|batch=0']['knee'])

    def test_writer_refuses_to_overwrite_existing_evidence(self):
        config = capacity.Config()
        writer = capacity.Writer(self.tmp.name, config)
        writer.close({"partial":True})
        with self.assertRaises(FileExistsError):
            capacity.Writer(self.tmp.name, config)

    def test_resume_requires_exact_completed_successful_isolated_evidence(self):
        config = capacity.Config(cache_namespace='session-one')
        summary = {'config':capacity.asdict(config),'outcomes':100,'successful':100,
                   'all_outcomes_accounted':True,'stopped':None,'cache_namespace':'session-one',
                   'cache_isolation':'unique session/scenario/request prefix-cache salts'}
        self.assertTrue(capacity.completed_scenario_matches(summary,config,'session-one'))
        for key,value in [('max_num_seqs',256),('max_num_batched_tokens',8192),('runtime_fingerprint','other'),
                          ('force_length',False),('concurrency',2),('duration',61)]:
            changed=json.loads(json.dumps(summary));changed['config'][key]=value
            self.assertFalse(capacity.completed_scenario_matches(changed,config,'session-one'))
        for key,value in [('all_outcomes_accounted',False),('stopped','stage_wall_deadline'),('successful',99),
                          ('cache_namespace','other-session'),('cache_isolation','unknown')]:
            changed={**summary,key:value}
            self.assertFalse(capacity.completed_scenario_matches(changed,config,'session-one'))

    def test_validation_excludes_public_targets_and_unbounded_outstanding(self):
        for config in (capacity.Config(url="https://public.invalid"), capacity.Config(concurrency=1025),
                       capacity.Config(url="http://secret@127.0.0.1"), capacity.Config(request_deadline=181)):
            with self.assertRaises(ValueError):
                config.validate()

    def test_cli_accepts_fractional_rates_and_rejects_nonfinite_values(self):
        args = capacity.parser().parse_args(["--arrival-rate", ".75", "--duration", "1.5"])
        self.assertEqual(args.arrival_rate, .75)
        self.assertEqual(args.duration, 1.5)
        for config in (capacity.Config(arrival_rate=float('nan')), capacity.Config(duration=float('inf')),
                       capacity.Config(metrics_interval=0)):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
