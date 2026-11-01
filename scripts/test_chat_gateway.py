"""Exercise the real gateway with a controlled vLLM-compatible upstream."""
import asyncio
import importlib.util
import json
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
        self.tokenizer_failure = False
        self.tokenizer_invalid = False
        self.model_failure = False
        self.oversized = False
        self.token_count = None
        self.model_context = 40960
        self.tokenize_delay = 0
        self.reply_delay = 0
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        owner = self

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

        async def upstream(request):
            if request.url.path == '/health':
                return httpx.Response(503 if self.model_failure else 200)
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'qwen', 'max_model_len': self.model_context}]})
            body = json.loads(request.content)
            if request.url.path == '/tokenize':
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
            return httpx.Response(200, stream=BlockedStream() if self.block or self.reply_delay else httpx.ByteStream(SSE), headers={'content-type': 'text/event-stream'})

        self.app = gateway.create_app(TOKEN, transport=httpx.MockTransport(upstream))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.app.state.client.aclose()

    async def send(self, messages=None, **extra):
        return await self.client.post('/v1/chat/completions', headers=AUTH, json={
            'messages': messages or [{'role': 'user', 'content': 'Hi'}], 'stream': True, **extra,
        })

    async def test_authentication_and_route_isolation(self):
        for headers in ({}, {'Authorization': 'Bearer wrong'}):
            self.assertEqual((await self.client.get('/health', headers=headers)).status_code, 401)
        for path in ['/metrics', '/docs', '/openapi.json', '/v1/models']:
            self.assertEqual((await self.client.get(path, headers=AUTH)).status_code, 404)
        self.assertEqual((await self.client.get('/health', headers=AUTH)).status_code, 200)

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

    async def test_busy_generation_rejected_and_cancellation_releases_upstream(self):
        self.block = True
        first = asyncio.create_task(self.send())
        await asyncio.wait_for(self.entered.wait(), 3)
        self.assertEqual((await self.send()).status_code, 429)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertTrue(self.closed)
        self.block = False
        self.assertEqual((await self.send()).status_code, 200)


if __name__ == '__main__':
    unittest.main(verbosity=2)
