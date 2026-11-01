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
        self.block = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        owner = self

        class BlockedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                owner.entered.set()
                await owner.release.wait()
                yield SSE

            async def aclose(self):
                owner.closed = True

        def upstream(request):
            if request.url.path == '/health':
                return httpx.Response(503 if self.model_failure else 200)
            body = json.loads(request.content)
            if request.url.path == '/tokenize':
                if self.tokenizer_failure:
                    return httpx.Response(503)
                if self.tokenizer_invalid:
                    return httpx.Response(200, json={'count': 'invalid'})
                count = 5000 if self.oversized or len(body['messages']) > 2 else 150
                return httpx.Response(200, json={'count': count})
            self.forwarded.append(body)
            if self.model_failure:
                return httpx.Response(500, text='internal details must not be exposed')
            return httpx.Response(200, stream=BlockedStream() if self.block else httpx.ByteStream(SSE), headers={'content-type': 'text/event-stream'})

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
        self.assertEqual((await self.send([{'role': 'user', 'content': 'x' * 65000}])).status_code, 413)
        self.oversized = True
        self.assertEqual((await self.send()).status_code, 400)
        self.assertEqual(self.forwarded, [])

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
