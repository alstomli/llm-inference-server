#!/usr/bin/env python3
"""Authenticated, streaming chat-only ingress for the private vLLM server."""
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MAX_BODY = 64_000
SYSTEM = 'You are Qwen, a helpful personal assistant. Answer clearly and directly. Use Markdown when helpful.'


def validate(value):
    if not isinstance(value, dict):
        raise ValueError('Invalid request')
    messages = value.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 80 or len(messages) % 2 != 1:
        raise ValueError('Invalid conversation')
    cleaned = []
    for index, message in enumerate(messages):
        role = 'user' if index % 2 == 0 else 'assistant'
        if not isinstance(message, dict) or message.get('role') != role or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Invalid conversation')
        cleaned.append({'role': role, 'content': message['content']})
    maximum = value.get('max_tokens', 768)
    temperature = value.get('temperature', 0.7)
    if type(maximum) is not int or not 128 <= maximum <= 1024:
        raise ValueError('Invalid response length')
    if type(temperature) not in (float, int) or not 0 <= temperature <= 1.5:
        raise ValueError('Invalid temperature')
    if value.get('stream') is not True:
        raise ValueError('Streaming is required')
    return cleaned, maximum, temperature


def create_app(token, upstream='http://127.0.0.1:8000', model='qwen', context=4096, transport=None):
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError('A chat token of at least 32 characters is required')
    headers = {}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    client = httpx.AsyncClient(base_url=upstream, headers=headers, transport=transport,
                               timeout=httpx.Timeout(60, connect=5), follow_redirects=False)
    lock = asyncio.Lock()
    recent = []

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.client = client

    def error(message, code):
        return JSONResponse({'error': message}, status_code=code, headers={'Cache-Control': 'no-store'})

    @app.middleware('http')
    async def authenticate(request: Request, call_next):
        supplied = request.headers.get('authorization', '')
        if not hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
            return error('Unauthorized', 401)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/health')
    async def health():
        try:
            response = await client.get('/health', timeout=5)
            if response.status_code == 200:
                return {'ready': True, 'model': model}
        except httpx.HTTPError:
            pass
        return error('Model unavailable', 503)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        if not request.headers.get('content-type', '').startswith('application/json'):
            return error('JSON required', 415)
        try:
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > MAX_BODY:
                    return error('Conversation too large', 413)
            messages, maximum, temperature = validate(json.loads(body))
        except (ValueError, UnicodeError):
            return error('Invalid chat request', 400)
        now = time.monotonic()
        recent[:] = [stamp for stamp in recent if stamp > now - 60]
        if lock.locked() or len(recent) >= 20:
            return error('Model busy; retry shortly', 429)
        await lock.acquire()
        recent.append(now)
        response = None
        try:
            trimmed = 0
            while True:
                conversation = [{'role': 'system', 'content': SYSTEM}] + messages
                counts = await client.post('/tokenize', json={
                    'model': model, 'messages': conversation, 'add_generation_prompt': True,
                    'chat_template_kwargs': {'enable_thinking': False},
                })
                if counts.status_code != 200:
                    lock.release()
                    return error('Model tokenizer unavailable', 503)
                payload = counts.json()
                count = payload.get('count') if isinstance(payload, dict) else None
                if type(count) is not int or count < 0:
                    lock.release()
                    return error('Invalid tokenizer response', 503)
                if count + maximum + 32 <= context:
                    break
                if len(messages) <= 1:
                    lock.release()
                    return error('Message exceeds model context', 400)
                messages = messages[2:]
                trimmed += 2
            response = await client.send(client.build_request('POST', '/v1/chat/completions', json={
                'model': model, 'messages': conversation, 'stream': True,
                'max_tokens': maximum, 'temperature': temperature,
                'chat_template_kwargs': {'enable_thinking': False},
            }), stream=True)
            if response.status_code != 200:
                await response.aclose()
                lock.release()
                return error('Model unavailable', 503)
        except (httpx.HTTPError, ValueError, TypeError):
            if response is not None:
                await response.aclose()
            lock.release()
            return error('Model unavailable', 503)
        except BaseException:
            if response is not None:
                await response.aclose()
            lock.release()
            raise

        async def generate():
            try:
                async with asyncio.timeout(240):
                    async for part in response.aiter_bytes():
                        yield part
            except (TimeoutError, httpx.HTTPError):
                yield b'data: {"error":{"message":"The model response was interrupted. Please retry."}}\n\ndata: [DONE]\n\n'
            finally:
                try:
                    await response.aclose()
                finally:
                    lock.release()

        return StreamingResponse(generate(), media_type='text/event-stream', headers={
            'X-Context-Trimmed': str(trimmed), 'X-Accel-Buffering': 'no',
        })

    return app


if __name__ == '__main__':
    import uvicorn
    secret = Path(os.environ.get('CHAT_TOKEN_FILE', '/workspace/chat/chat-token')).read_text().strip()
    app = create_app(secret, model=os.environ.get('SERVED_MODEL_NAME', 'qwen'),
                     upstream='http://127.0.0.1:' + os.environ.get('VLLM_PORT', '8000'),
                     context=int(os.environ.get('MAX_MODEL_LEN', '4096')))
    uvicorn.run(app, host=os.environ.get('CHAT_HOST', '127.0.0.1'),
                port=int(os.environ.get('CHAT_PORT', '8001')), access_log=False)
