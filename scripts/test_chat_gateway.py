"""Exercise the real gateway with a controlled vLLM-compatible upstream."""
import asyncio
import importlib.util
import json
import re
from pathlib import Path
import unittest
import httpx

spec = importlib.util.spec_from_file_location('gateway', Path(__file__).with_name('chat-gateway.py'))
gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gateway)
TOKEN = 'test-token-' + 'x' * 40
AUTH = {'Authorization': 'Bearer ' + TOKEN}
SSE = b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.forwarded = []
        self.tokenizer_requests = 0
        self.tokenizer_models = []
        self.tokenizer_failure = False
        self.tokenizer_invalid = False
        self.model_failure = False
        self.oversized = False
        self.token_count = None
        self.model_context = 40960
        self.model_cards = None
        self.tokenize_delay = 0
        self.reply_delay = 0
        self.stream_parts = None
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.blocked_prompts = set()
        self.started_prompts = []
        self.finished_prompts = set()
        self.prompt_releases = {}
        owner = self

        class ControlledStream(httpx.AsyncByteStream):
            def __init__(self, prompt):
                self.prompt = prompt

            async def __aiter__(self):
                owner.started_prompts.append(self.prompt)
                await owner.prompt_releases.setdefault(self.prompt, asyncio.Event()).wait()
                yield SSE

            async def aclose(self):
                owner.finished_prompts.add(self.prompt)

        class BlockedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                owner.entered.set()
                if owner.reply_delay:
                    await asyncio.sleep(owner.reply_delay)
                    yield SSE
                    return
                await owner.release.wait()
                yield SSE

            async def aclose(self):
                owner.closed = True

        class FragmentedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for part in owner.stream_parts:
                    await asyncio.sleep(0.01)
                    yield part

        async def upstream(request):
            if request.url.path == '/health':
                return httpx.Response(503 if self.model_failure else 200)
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': self.model_cards if self.model_cards is not None
                                                 else [{'id': 'qwen', 'max_model_len': self.model_context}]})
            body = json.loads(request.content)
            if request.url.path == '/tokenize':
                self.tokenizer_requests += 1
                self.tokenizer_models.append(body['model'])
                await asyncio.sleep(self.tokenize_delay)
                if self.tokenizer_failure:
                    return httpx.Response(503)
                if self.tokenizer_invalid:
                    return httpx.Response(200, json={'count': 'invalid'})
                count = 50000 if self.oversized or len(body['messages']) > 2 else 150
                if self.token_count is not None:
                    count = self.token_count
                return httpx.Response(200, json={'count': count})
            self.forwarded.append(body)
            if self.model_failure:
                return httpx.Response(500, text='internal details must not be exposed')
            prompt = body['messages'][-1]['content']
            stream = ControlledStream(prompt) if prompt in self.blocked_prompts else FragmentedStream() if self.stream_parts is not None else BlockedStream() if self.block or self.reply_delay else httpx.ByteStream(SSE)
            return httpx.Response(200, stream=stream, headers={'content-type': 'text/event-stream'})

        self.transport = httpx.MockTransport(upstream)
        self.app = gateway.create_app(TOKEN, transport=self.transport, diagnostics_enabled=False)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.app.state.client.aclose()

    async def send(self, messages=None, **extra):
        return await self.client.post('/v1/chat/completions', headers=AUTH, json={
            'messages': messages or [{'role': 'user', 'content': 'Hi'}], 'stream': True, **extra,
        })

    async def enable_diagnostics(self):
        await self.client.aclose()
        await self.app.state.client.aclose()
        self.app = gateway.create_app(TOKEN, transport=self.transport, diagnostics_enabled=True)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def diagnostic_send(self, messages=None, **extra):
        return await self.client.post('/v1/chat/completions', headers={**AUTH, 'X-Chat-Diagnostics': '1'}, json={
            'messages': messages or [{'role': 'user', 'content': 'Hi'}], 'stream': True, **extra,
        })

    async def test_diagnostics_require_server_opt_in_and_authentication(self):
        self.assertEqual((await self.diagnostic_send()).status_code, 403)
        self.assertEqual(self.forwarded, [])
        self.assertEqual((await self.client.get('/health', headers=AUTH)).json()['diagnosticsVersion'], 0)
        await self.enable_diagnostics()
        denied = await self.client.post('/v1/chat/completions', headers={'X-Chat-Diagnostics': '1'}, json={})
        self.assertEqual(denied.status_code, 401)
        self.assertNotIn('X-Chat-Diagnostics', denied.headers)
        self.assertEqual((await self.client.get('/health', headers=AUTH)).json()['diagnosticsVersion'], 1)

    async def test_diagnostics_reuse_exact_counts_and_normal_stream_is_unchanged(self):
        await self.enable_diagnostics()
        plain = await self.send()
        self.assertNotIn('X-Chat-Diagnostics', plain.headers)
        self.assertNotIn('stream_options', self.forwarded[-1])
        self.assertEqual(self.tokenizer_requests, 1)
        response = await self.diagnostic_send()
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual(record['inputTokensBefore'], 150)
        self.assertEqual(record['inputTokensAfter'], 150)
        self.assertEqual(record['gatewayTrimmedMessages'], 0)
        self.assertEqual(self.tokenizer_requests, 2)
        self.assertEqual(self.forwarded[-1]['stream_options'], {'include_usage': True})
        self.assertEqual(re.sub(rb'event: chat-timing\ndata: [^\n]+\n\n', b'', response.content), plain.content)
        self.assertNotIn(b'chat-timing', plain.content)
        self.assertNotIn('messages', record)
        self.assertNotIn('Hi', response.headers['X-Chat-Diagnostics'])
        self.assertNotIn(TOKEN, response.headers['X-Chat-Diagnostics'])

    async def test_timing_observes_content_events_across_utf8_and_network_boundaries(self):
        await self.enable_diagnostics()
        self.tokenize_delay = 0.025
        content = 'data: {"choices":[{"delta":{"content":"你好"}}]}\r\n\r\n'.encode()
        split = content.index('你'.encode()) + 1
        self.stream_parts = [b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
                             content[:split], content[split:],
                             b'data: {"choices":[],"usage":{"prompt_tokens":150,"completion_tokens":2}}\n\n',
                             b'data: [DO', b'NE]\n\n']
        response = await self.diagnostic_send([
            {'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}])
        event = json.loads(re.search(rb'event: chat-timing\ndata: ([^\n]+)', response.content)[1])
        self.assertEqual(event['requestId'], json.loads(response.headers['X-Chat-Diagnostics'])['requestId'])
        timing = event['timings']
        self.assertGreaterEqual(timing['contextFitMs'], 40)  # Both tokenizer calls, including trimming.
        self.assertGreaterEqual(timing['upstreamFirstContentMs'], 20)  # Ignore role/empty content.
        self.assertGreater(timing['gatewayFirstContentMs'], timing['upstreamFirstContentMs'])
        self.assertGreater(timing['gatewayTotalMs'], timing['gatewayFirstContentMs'])
        self.assertEqual(response.content.count(b'event: chat-timing'), 1)
        self.assertLess(response.content.index(b'event: chat-timing'), response.content.index(b'data: [DONE]'))
        self.assertEqual(re.sub(rb'event: chat-timing\ndata: [^\n]+\n\n', b'', response.content), b''.join(self.stream_parts))

    async def test_timing_keeps_missing_first_content_unavailable_on_timeout(self):
        await self.enable_diagnostics()
        self.reply_delay = 1.2
        response = await self.diagnostic_send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1})
        event = json.loads(re.search(rb'event: chat-timing\ndata: ([^\n]+)', response.content)[1])
        self.assertIsNone(event['timings']['upstreamFirstContentMs'])
        self.assertIsNone(event['timings']['gatewayFirstContentMs'])
        self.assertGreaterEqual(event['timings']['gatewayTotalMs'], 900)
        self.assertIn(b'interrupted', response.content)
        self.assertTrue(self.closed)
        self.reply_delay = 0
        self.assertEqual((await self.send()).status_code, 200)

    async def test_diagnostics_report_trimmed_and_rejected_prompts(self):
        await self.enable_diagnostics()
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}]
        response = await self.diagnostic_send(messages)
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual((record['inputTokensBefore'], record['inputTokensAfter']), (50000, 150))
        self.assertEqual(record['gatewayTrimmedMessages'], 2)
        self.assertEqual(self.tokenizer_requests, 2)
        response = await self.diagnostic_send(messages[:2] + messages)
        self.assertEqual(json.loads(response.headers['X-Chat-Diagnostics'])['gatewayTrimmedMessages'], 4)
        self.assertEqual(self.tokenizer_requests, 5)
        self.oversized = True
        response = await self.diagnostic_send()
        self.assertEqual(response.status_code, 400)
        record = json.loads(response.headers['X-Chat-Diagnostics'])
        self.assertEqual(record['inputTokensBefore'], 50000)
        self.assertEqual(record['lastCountedInputTokens'], 50000)
        self.assertIsNone(record['inputTokensAfter'])
        self.assertEqual(len(self.forwarded), 2)
        self.oversized = False; self.tokenizer_failure = True
        response = await self.diagnostic_send()
        self.assertEqual(response.status_code, 503)
        self.assertIsNone(json.loads(response.headers['X-Chat-Diagnostics'])['inputTokensBefore'])

    async def test_authentication_and_route_isolation(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            self.assertEqual((await self.client.get('/health', headers=headers)).status_code, 401)
        for path in ['/metrics', '/docs', '/openapi.json', '/v1/models']:
            self.assertEqual((await self.client.get(path, headers=AUTH)).status_code, 404)
        self.assertEqual((await self.client.get('/health', headers=AUTH)).status_code, 200)

    async def test_adapter_inherits_only_its_named_base_capacity_and_routes_to_adapter(self):
        self.model_cards = [{'id': 'unrelated', 'max_model_len': 40960},
                            {'id': 'qwen-base', 'max_model_len': 8192},
                            {'id': 'qwen', 'parent': 'qwen-base', 'max_model_len': None}]
        health = (await self.client.get('/health', headers=AUTH)).json()
        self.assertTrue(health['ready'])
        self.assertEqual(health['model'], 'qwen')
        self.assertEqual(health['limits']['contextTokens'], 8192)
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 8192, 'outputTokens': 128}
        self.assertEqual((await self.send(settings=settings, max_tokens=128)).status_code, 200)
        self.assertEqual(self.forwarded[-1]['model'], 'qwen')
        self.assertEqual(self.tokenizer_models[-1], 'qwen')
        self.assertEqual((await self.send()).status_code, 400)

    async def test_adapter_missing_or_invalid_base_capacity_fails_closed(self):
        for cards in ([{'id': 'other', 'max_model_len': 40960}],
                      [{'id': 'qwen', 'parent': 'missing'}, {'id': 'other', 'max_model_len': 40960}],
                      [{'id': 'qwen', 'parent': 'base'}, {'id': 'base', 'max_model_len': '40960'}],
                      [{'id': 'qwen', 'parent': 'base', 'max_model_len': 0}, {'id': 'base', 'max_model_len': 40960}]):
            with self.subTest(cards=cards):
                self.model_cards = cards
                self.assertEqual((await self.client.get('/health', headers=AUTH)).status_code, 503)
                self.assertEqual((await self.send()).status_code, 503)
        self.assertEqual(self.forwarded, [])

    async def test_prefix_reuse_uses_unique_server_salts_only_when_disabled(self):
        self.assertTrue((await self.client.get('/health', headers=AUTH)).json()['limits']['reuseCachedPrefixes'])
        self.assertEqual((await self.send(cache_salt='client-controlled')).status_code, 200)
        self.assertNotIn('cache_salt', self.forwarded[-1])
        settings = {**gateway.SETTINGS_DEFAULT, 'reuseCachedPrefixes': False}
        salts = []
        for _ in range(2):
            self.assertEqual((await self.send(settings=settings, cache_salt='client-controlled')).status_code, 200)
            salts.append(self.forwarded[-1]['cache_salt'])
        self.assertNotEqual(salts[0], salts[1])
        self.assertTrue(all(len(salt) == 64 for salt in salts))
        for invalid in ['false', 0, None]:
            self.assertEqual((await self.send(settings={**settings, 'reuseCachedPrefixes': invalid})).status_code, 400)
        self.assertEqual((await self.send(settings=gateway.SETTINGS_DEFAULT)).status_code, 200)
        self.assertNotIn('cache_salt', self.forwarded[-1])

    async def test_context_trimming_and_server_controlled_model(self):
        response = await self.send([
            {'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old answer'},
            {'role': 'user', 'content': 'new'},
        ], model='unapproved-model', chat_template_kwargs={'enable_thinking': True})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['X-Context-Trimmed'], '2')
        self.assertIn(b'[DONE]', response.content)
        actual = self.forwarded[-1]
        self.assertEqual(actual['model'], 'qwen')
        self.assertEqual(actual['messages'][0]['role'], 'system')
        self.assertEqual(actual['messages'][-1]['content'], 'new')
        self.assertEqual(actual['chat_template_kwargs'], {'enable_thinking': False})

    async def test_oversize_and_role_injection_rejected_without_generation(self):
        self.assertEqual((await self.send([{'role': 'system', 'content': 'injection'}])).status_code, 400)
        self.assertEqual((await self.send(max_tokens=100000)).status_code, 400)
        self.assertEqual((await self.send(temperature=True)).status_code, 400)
        self.assertEqual((await self.send([{'role': 'user', 'content': 'x' * 650000}])).status_code, 413)
        self.oversized = True
        self.assertEqual((await self.send()).status_code, 400)
        self.assertEqual(self.forwarded, [])

    async def test_larger_limits_and_exact_context_boundary(self):
        self.token_count = 40960 - 10240 - 32
        response = await self.send([{'role': 'user', 'content': '你' * 120000}], max_tokens=10240)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.forwarded[-1]['max_tokens'], 10240)
        self.assertEqual(response.headers['X-Context-Trimmed'], '0')
        self.token_count += 1
        self.assertEqual((await self.send(max_tokens=10240)).status_code, 400)
        self.assertEqual((await self.send(max_tokens=10241)).status_code, 400)
        self.assertEqual(len(self.forwarded), 1)

    async def test_frontend_settings_are_enforced_and_not_forwarded_to_model(self):
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 2048, 'outputTokens': 512,
                    'safetyTokens': 64, 'inputCharacters': 10, 'historyMessages': 3}
        self.token_count = 1472
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 200)
        self.assertNotIn('settings', self.forwarded[-1])
        self.token_count += 1
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 400)
        self.token_count = 100
        self.assertEqual((await self.send([{'role': 'user', 'content': 'x' * 11}], settings=settings, max_tokens=512)).status_code, 400)
        self.assertEqual((await self.send(settings=settings, max_tokens=513)).status_code, 400)
        for key, value in [('contextTokens', 40961), ('requestBytes', 640001), ('timeoutSeconds', True), ('safetyTokens', -1)]:
            invalid = {**settings, key: value}
            self.assertEqual((await self.send(settings=invalid, max_tokens=512)).status_code, 400)

    async def test_request_and_history_limits_cannot_be_bypassed(self):
        for key, value, status in [('historyBytes', 128, 400), ('requestBytes', 1024, 413)]:
            settings = {**gateway.SETTINGS_DEFAULT, key: value}
            response = await self.send([{'role': 'user', 'content': 'x' * 1500}], settings=settings)
            self.assertEqual(response.status_code, status)
        settings = {**gateway.SETTINGS_DEFAULT, 'historyMessages': 1}
        messages = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'answer'}, {'role': 'user', 'content': 'new'}]
        self.assertEqual((await self.send(messages, settings=settings)).status_code, 400)
        self.assertEqual(self.forwarded, [])

    async def test_context_ceiling_comes_from_running_model(self):
        self.model_context = 4096
        health = await self.client.get('/health', headers=AUTH)
        self.assertEqual(health.json()['limits']['contextTokens'], 4096)
        self.assertEqual((await self.send(settings=gateway.SETTINGS_DEFAULT)).status_code, 400)
        settings = {**gateway.SETTINGS_DEFAULT, 'contextTokens': 4096, 'outputTokens': 512}
        self.assertEqual((await self.send(settings=settings, max_tokens=512)).status_code, 200)

    async def test_configured_timeout_covers_prefill_and_streaming_and_releases_capacity(self):
        settings = {**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1}
        self.tokenize_delay = 1.2
        self.assertEqual((await self.send(settings=settings)).status_code, 503)
        self.tokenize_delay = 0
        self.reply_delay = 1.2
        response = await self.send(settings=settings)
        self.assertIn(b'interrupted', response.content)
        self.assertTrue(self.closed)
        self.reply_delay = 0
        self.assertEqual((await self.send(settings=settings)).status_code, 200)

    async def test_failures_release_capacity_and_hide_upstream_details(self):
        self.tokenizer_failure = True
        self.assertEqual((await self.send()).status_code, 503)
        self.tokenizer_failure = False; self.tokenizer_invalid = True
        self.assertEqual((await self.send()).status_code, 503)
        self.tokenizer_invalid = False; self.model_failure = True
        response = await self.send()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('internal details', response.text)
        self.model_failure = False
        self.assertEqual((await self.send()).status_code, 200)

    async def until(self, predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(poll(), 3)

    async def controlled(self, prompt, **extra):
        self.blocked_prompts.add(prompt)
        return asyncio.create_task(self.send([{'role': 'user', 'content': prompt}], **extra))

    async def stop_tasks(self, tasks):
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def test_two_streams_run_without_waiting_for_a_pair(self):
        tasks = []
        try:
            tasks.append(await self.controlled('first'))
            await self.until(lambda: self.started_prompts == ['first'])
            # The first stream is already running before the second is submitted.
            tasks.append(await self.controlled('second'))
            await self.until(lambda: self.started_prompts == ['first', 'second'])
            self.assertEqual(len(self.app.state.admission.active), 2)
            self.assertFalse(any(task.done() for task in tasks))
            for event in self.prompt_releases.values():
                event.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 3)
            self.assertTrue(all(result.status_code == 200 and b'[DONE]' in result.content for result in results))
        finally:
            await self.stop_tasks(tasks)

    async def test_six_waiters_are_fifo_and_ninth_request_is_rejected(self):
        tasks = []
        try:
            for index in range(8):
                tasks.append(await self.controlled(str(index)))
                await self.until(lambda: len(self.app.state.admission.active) + len(self.app.state.admission.waiting) == len(tasks))
            await self.until(lambda: len(self.started_prompts) == 2)
            self.assertEqual(self.started_prompts, ['0', '1'])
            self.assertEqual(len(self.app.state.admission.waiting), 6)
            response = await self.send()
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers['X-Chat-Rejection'], 'queue_full')
            self.assertEqual(len(self.forwarded), 2)
            for index in range(8):
                await self.until(lambda: str(index) in self.prompt_releases)
                self.prompt_releases[str(index)].set()
                self.assertEqual((await asyncio.wait_for(tasks[index], 3)).status_code, 200)
            self.assertEqual(self.started_prompts, [str(index) for index in range(8)])
            self.assertEqual(len(self.app.state.admission.active), 0)
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            # Eight reservations plus twelve more reach the rate limit; overflow did not count.
            for _ in range(12):
                self.assertEqual((await self.send()).status_code, 200)
            self.assertEqual((await self.send()).headers['X-Chat-Rejection'], 'rate_limit')
        finally:
            await self.stop_tasks(tasks)

    async def test_cancelled_waiter_is_removed_and_never_reaches_model(self):
        tasks = []
        try:
            for prompt in ['first', 'second', 'cancelled']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            tasks[2].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[2]
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            self.assertEqual(len(self.forwarded), 2)
            tasks.append(await self.controlled('replacement'))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            self.prompt_releases['first'].set()
            await tasks[0]
            await self.until(lambda: 'replacement' in self.started_prompts)
            self.assertNotIn('cancelled', self.started_prompts)
        finally:
            await self.stop_tasks(tasks)

    async def test_queue_wait_counts_toward_timeout_and_frees_waiting_slot(self):
        tasks = []
        try:
            for prompt in ['first', 'second']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2)
            response = await asyncio.wait_for(self.send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1}), 3)
            self.assertEqual(response.status_code, 504)
            self.assertEqual(response.headers['X-Chat-Rejection'], 'queue_timeout')
            self.assertEqual(len(self.app.state.admission.waiting), 0)
            self.assertEqual(len(self.forwarded), 2)
        finally:
            await self.stop_tasks(tasks)

    async def test_waiting_consumes_the_same_deadline_as_tokenization(self):
        tasks = []
        try:
            for prompt in ['first', 'second']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2)
            tasks.append(asyncio.create_task(self.send(settings={**gateway.SETTINGS_DEFAULT, 'timeoutSeconds': 1})))
            await self.until(lambda: len(self.app.state.admission.waiting) == 1)
            await asyncio.sleep(0.6)
            self.tokenize_delay = 0.6
            self.prompt_releases['first'].set()
            response = await asyncio.wait_for(tasks[2], 2)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(len(self.forwarded), 2)  # Queue wait did not reset the generation deadline.
        finally:
            await self.stop_tasks(tasks)

    async def test_rate_limit_is_global_even_when_slots_are_free(self):
        for _ in range(20):
            self.assertEqual((await self.send()).status_code, 200)
        response = await self.send()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['X-Chat-Rejection'], 'rate_limit')
        self.assertEqual(len(self.app.state.admission.active), 0)

    async def test_cancelling_one_active_stream_leaves_the_other_running(self):
        tasks = []
        try:
            for prompt in ['first', 'second', 'third']:
                tasks.append(await self.controlled(prompt))
            await self.until(lambda: len(self.started_prompts) == 2 and len(self.app.state.admission.waiting) == 1)
            tasks[0].cancel()
            with self.assertRaises(asyncio.CancelledError):
                await tasks[0]
            await self.until(lambda: 'third' in self.started_prompts)
            self.assertIn('first', self.finished_prompts)
            self.assertNotIn('second', self.finished_prompts)
            self.assertFalse(tasks[1].done())
            self.assertEqual(len(self.app.state.admission.active), 2)
        finally:
            await self.stop_tasks(tasks)

    async def test_active_stream_cancellation_releases_upstream(self):
        self.block = True
        first = asyncio.create_task(self.send())
        await asyncio.wait_for(self.entered.wait(), 3)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.closed)
        self.block = False
        self.assertEqual((await self.send()).status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
