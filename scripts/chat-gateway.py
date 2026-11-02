#!/usr/bin/env python3
"""Authenticated, streaming chat-only ingress for the private vLLM server."""
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import hmac
import json
import os
import re
import secrets
from pathlib import Path
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MAX_BODY = 640_000
SETTINGS_MAX = {
    'inputCharacters': 120000, 'outputTokens': 10240, 'contextTokens': 40960,
    'historyMessages': 799, 'historyBytes': 580000, 'requestBytes': MAX_BODY,
    'timeoutSeconds': 2400, 'safetyTokens': 4096,
}
SETTINGS_DEFAULT = {**SETTINGS_MAX, 'outputTokens': 7680, 'safetyTokens': 32, 'reuseCachedPrefixes': True}
SETTINGS_MIN = {
    'inputCharacters': 1, 'outputTokens': 128, 'contextTokens': 256,
    'historyMessages': 1, 'historyBytes': 128, 'requestBytes': 1024,
    'timeoutSeconds': 1, 'safetyTokens': 0,
}
SYSTEM = 'You are Qwen, a helpful personal assistant. Answer clearly and directly. Use Markdown when helpful.'


class AdmissionQueue:
    """FIFO reservations owned by one event loop; granted slots include stream lifetime."""
    def __init__(self, concurrency=2, queue_limit=6):
        if type(concurrency) is not int or concurrency < 1 or type(queue_limit) is not int or queue_limit < 0:
            raise ValueError('Invalid admission limits')
        self.concurrency = concurrency
        self.queue_limit = queue_limit
        self.active = set()
        self.waiting = deque()

    def reserve(self):
        ticket = asyncio.get_running_loop().create_future()
        if len(self.active) < self.concurrency and not self.waiting:
            self.active.add(ticket)
            ticket.set_result(None)
        elif len(self.waiting) < self.queue_limit:
            self.waiting.append(ticket)
        else:
            return None
        return ticket

    def release(self, ticket):
        if ticket in self.active:
            self.active.remove(ticket)
        elif ticket in self.waiting:
            self.waiting.remove(ticket)
            ticket.cancel()
        else:
            return  # Cleanup can race with cancellation; release only once.
        while self.waiting and len(self.active) < self.concurrency:
            next_ticket = self.waiting.popleft()
            self.active.add(next_ticket)
            next_ticket.set_result(None)


def validate_settings(value, context):
    if value is None:
        settings = {**SETTINGS_DEFAULT, 'contextTokens': min(context, SETTINGS_MAX['contextTokens'])}
        settings['outputTokens'] = min(settings['outputTokens'], settings['contextTokens'] - settings['safetyTokens'] - 1)
    else:
        if not isinstance(value, dict) or set(value) - set(SETTINGS_DEFAULT):
            raise ValueError('Invalid chat settings')
        settings = {**SETTINGS_DEFAULT, **value}
    if type(settings['reuseCachedPrefixes']) is not bool:
        raise ValueError('Invalid prefix reuse setting')
    for key, maximum in SETTINGS_MAX.items():
        if key == 'contextTokens':
            maximum = min(context, maximum)
        if type(settings[key]) is not int or not SETTINGS_MIN[key] <= settings[key] <= maximum:
            raise ValueError('Invalid chat setting: ' + key)
    if settings['outputTokens'] + settings['safetyTokens'] >= settings['contextTokens']:
        raise ValueError('Context budget must leave room for input')
    return settings


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def validate(value):
    if not isinstance(value, dict):
        raise ValueError('Invalid request')
    messages = value.get('messages')
    if not isinstance(messages, list) or not 1 <= len(messages) <= 800 or len(messages) % 2 != 1:
        raise ValueError('Invalid conversation')
    cleaned = []
    for index, message in enumerate(messages):
        role = 'user' if index % 2 == 0 else 'assistant'
        if not isinstance(message, dict) or message.get('role') != role or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Invalid conversation')
        cleaned.append({'role': role, 'content': message['content']})
    maximum = value.get('max_tokens', 7680)
    temperature = value.get('temperature', 0.7)
    if type(maximum) is not int or not 128 <= maximum <= 10240:
        raise ValueError('Invalid response length')
    if type(temperature) not in (float, int) or not 0 <= temperature <= 1.5:
        raise ValueError('Invalid temperature')
    if value.get('stream') is not True:
        raise ValueError('Streaming is required')
    return cleaned, maximum, temperature


def create_app(token, upstream='http://127.0.0.1:8000', model='qwen', context=40960, transport=None, diagnostics_enabled=None, concurrency=2):
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError('A chat token of at least 32 characters is required')
    if diagnostics_enabled is None:
        diagnostics_enabled = os.environ.get('CHAT_DIAGNOSTICS_ENABLED') == 'true'
    headers = {}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    client = httpx.AsyncClient(base_url=upstream, headers=headers, transport=transport,
                               timeout=httpx.Timeout(60, connect=5), follow_redirects=False)
    admission = AdmissionQueue(concurrency=concurrency)
    recent = []

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.client = client
    app.state.admission = admission

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

    async def serving_context():
        # Read vLLM's actual serving capacity rather than trusting a browser value.
        response = await client.get('/v1/models', timeout=5)
        response.raise_for_status()
        models = response.json().get('data', [])
        capacity = next((item.get('max_model_len') for item in models if item.get('id') == model), None)
        if type(capacity) is not int or capacity < SETTINGS_MIN['contextTokens']:
            raise ValueError('Serving context unavailable')
        return min(context, capacity)

    @app.get('/health')
    async def health():
        try:
            response = await client.get('/health', timeout=5)
            if response.status_code == 200:
                capacity = await serving_context()
                return {'ready': True, 'model': model, 'settingsVersion': 1,
                        'diagnosticsVersion': 1 if diagnostics_enabled else 0,
                        'admission': {'activeRequests': len(admission.active), 'waitingRequests': len(admission.waiting),
                                      'maxConcurrentRequests': admission.concurrency, 'maxWaitingRequests': admission.queue_limit},
                        'limits': {**SETTINGS_MAX, 'contextTokens': min(context, capacity, SETTINGS_MAX['contextTokens']), 'reuseCachedPrefixes': True}}
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            pass
        return error('Model unavailable', 503)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        request_started = time.monotonic()
        requested = request.headers.get('x-chat-diagnostics') == '1'
        if requested and not diagnostics_enabled:
            return error('Request diagnostics are disabled', 403)
        diagnostic = {
            'version': 1, 'requestId': secrets.token_hex(16), 'model': model,
            'inputTokensBefore': None, 'inputTokensAfter': None,
            'lastCountedInputTokens': None, 'outputBudget': None,
            'safetyReserve': None, 'contextBudget': None,
            'gatewayTrimmedMessages': 0, 'reuseCachedPrefixes': None,
        } if requested else None

        def diagnostic_headers():
            return {'X-Chat-Diagnostics': json.dumps(diagnostic, separators=(',', ':'))} if diagnostic is not None else {}

        def chat_error(message, code, rejection=None):
            response = error(message, code)
            response.headers.update(diagnostic_headers())
            if rejection is not None:
                response.headers['X-Chat-Rejection'] = rejection
            return response

        if not request.headers.get('content-type', '').startswith('application/json'):
            return chat_error('JSON required', 415)
        try:
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > MAX_BODY:
                    return chat_error('Conversation too large', 413)
            value = json.loads(body)
            messages, maximum, temperature = validate(value)
            # The fixed body cap applies before parsing; browser limits may only lower it.
            if 'settings' in value and value['settings'] is None:
                raise ValueError('Invalid chat settings')
            settings = validate_settings(value.get('settings'), context)
            if value.get('settings') is not None and maximum != settings['outputTokens']:
                raise ValueError('Conflicting output budget')
            settings['outputTokens'] = maximum
            if diagnostic is not None:
                diagnostic.update(outputBudget=maximum, safetyReserve=settings['safetyTokens'],
                                  contextBudget=settings['contextTokens'], reuseCachedPrefixes=settings['reuseCachedPrefixes'])
            if len(body) > settings['requestBytes']:
                return chat_error('Conversation exceeds request byte limit', 413)
            if len(messages[-1]['content'].encode('utf-16-le')) // 2 > settings['inputCharacters']:
                raise ValueError('Input character limit exceeded')
            if len(messages) > settings['historyMessages'] or json_bytes(messages) > settings['historyBytes']:
                raise ValueError('History limit exceeded')
            if maximum + settings['safetyTokens'] >= settings['contextTokens']:
                raise ValueError('Context budget must leave room for input')
        except (ValueError, UnicodeError):
            return chat_error('Invalid chat request', 400)
        deadline = time.monotonic() + settings['timeoutSeconds']
        now = time.monotonic()
        recent[:] = [stamp for stamp in recent if stamp > now - 60]
        if len(recent) >= 20:
            return chat_error('Chat rate limit reached; retry shortly', 429, 'rate_limit')
        ticket = admission.reserve()
        if ticket is None:
            return chat_error('Chat queue is full; retry shortly', 429, 'queue_full')
        recent.append(now)

        async def disconnected():
            # Await receive normally: is_disconnected's cancelled polling scope can
            # miss disconnects behind BaseHTTPMiddleware's receive channel.
            while (await request.receive())['type'] != 'http.disconnect':
                pass

        disconnect = asyncio.create_task(disconnected())
        try:
            ready, _ = await asyncio.wait({ticket, disconnect}, timeout=max(0, deadline - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
            if disconnect in ready:
                admission.release(ticket)
                return chat_error('Request cancelled', 499)
            if ticket not in ready:
                admission.release(ticket)
                return chat_error('Request timed out waiting for the model', 504, 'queue_timeout')
        except BaseException:
            admission.release(ticket)
            raise
        finally:
            disconnect.cancel()
            try:
                await asyncio.gather(disconnect, return_exceptions=True)
            except BaseException:
                admission.release(ticket)
                raise
        try:
            capacity = await asyncio.wait_for(serving_context(), timeout=max(0.001, deadline - time.monotonic()))
            if settings['contextTokens'] > capacity:
                admission.release(ticket)
                return chat_error('Context budget exceeds the running model capacity', 400)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, TimeoutError):
            admission.release(ticket)
            return chat_error('Serving context unavailable', 503)
        except BaseException:
            admission.release(ticket)
            raise
        response = None
        context_started = time.monotonic()
        timing = {'contextFitMs': None, 'upstreamFirstContentMs': None,
                  'gatewayFirstContentMs': None, 'gatewayTotalMs': None}
        try:
            trimmed = 0
            while True:
                conversation = [{'role': 'system', 'content': SYSTEM}] + messages
                counts = await asyncio.wait_for(client.post('/tokenize', timeout=settings['timeoutSeconds'], json={
                    'model': model, 'messages': conversation, 'add_generation_prompt': True,
                    'chat_template_kwargs': {'enable_thinking': False},
                }), timeout=max(0.001, deadline - time.monotonic()))
                if counts.status_code != 200:
                    admission.release(ticket)
                    return chat_error('Model tokenizer unavailable', 503)
                payload = counts.json()
                count = payload.get('count') if isinstance(payload, dict) else None
                if type(count) is not int or count < 0:
                    admission.release(ticket)
                    return chat_error('Invalid tokenizer response', 503)
                if diagnostic is not None:
                    if diagnostic['inputTokensBefore'] is None:
                        diagnostic['inputTokensBefore'] = count
                    diagnostic['lastCountedInputTokens'] = count
                if count + maximum + settings['safetyTokens'] <= settings['contextTokens']:
                    if diagnostic is not None:
                        diagnostic['inputTokensAfter'] = count
                    break
                if len(messages) <= 1:
                    admission.release(ticket)
                    return chat_error('Message exceeds model context', 400)
                messages = messages[2:]
                trimmed += 2
                if diagnostic is not None:
                    diagnostic['gatewayTrimmedMessages'] = trimmed
            timing['contextFitMs'] = (time.monotonic() - context_started) * 1000
            upstream_started = time.monotonic()
            response = await asyncio.wait_for(client.send(client.build_request('POST', '/v1/chat/completions', timeout=settings['timeoutSeconds'], json={
                'model': model, 'messages': conversation, 'stream': True,
                'max_tokens': maximum, 'temperature': temperature,
                **({'stream_options': {'include_usage': True}} if diagnostic is not None else {}),
                'chat_template_kwargs': {'enable_thinking': False},
                **({} if settings['reuseCachedPrefixes'] else {'cache_salt': secrets.token_hex(32)}),
            }), stream=True), timeout=max(0.001, deadline - time.monotonic()))
            if response.status_code != 200:
                await response.aclose()
                admission.release(ticket)
                return chat_error('Model unavailable', 503)
        except (httpx.HTTPError, ValueError, TypeError, TimeoutError):
            try:
                if response is not None:
                    await response.aclose()
            finally:
                admission.release(ticket)
            return chat_error('Model unavailable', 503)
        except BaseException:
            try:
                if response is not None:
                    await response.aclose()
            finally:
                admission.release(ticket)
            raise

        async def generate():
            buffer = b''
            timing_sent = False

            def timing_event():
                timing['gatewayTotalMs'] = (time.monotonic() - request_started) * 1000
                data = {'version': 1, 'requestId': diagnostic['requestId'],
                        'timings': {key: round(value, 3) if value is not None else None
                                    for key, value in timing.items()}}
                return ('event: chat-timing\ndata: ' + json.dumps(data, separators=(',', ':')) + '\n\n').encode()

            try:
                async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                    async for part in response.aiter_bytes():
                        if diagnostic is None:
                            yield part
                            continue
                        # Observe complete SSE events, not transport chunks. Preserve their bytes.
                        buffer += part
                        while (boundary := re.search(rb'\r?\n\r?\n', buffer)) is not None:
                            frame, buffer = buffer[:boundary.end()], buffer[boundary.end():]
                            data = b'\n'.join(line[5:].lstrip(b' ') for line in frame.splitlines() if line.startswith(b'data:'))
                            if data == b'[DONE]' and not timing_sent:
                                yield timing_event()
                                timing_sent = True
                            elif timing['upstreamFirstContentMs'] is None:
                                try:
                                    item = json.loads(data)
                                    content = item.get('choices', [{}])[0].get('delta', {}).get('content')
                                    if isinstance(content, str) and content:
                                        observed = time.monotonic()
                                        timing['upstreamFirstContentMs'] = (observed - upstream_started) * 1000
                                        timing['gatewayFirstContentMs'] = (observed - request_started) * 1000
                                except (ValueError, IndexError, AttributeError, TypeError):
                                    pass
                            yield frame
                        if len(buffer) > 1_048_576:
                            raise ValueError('Upstream SSE event too large')
                    if buffer:
                        yield buffer + b'\n\n'
                    if diagnostic is not None and not timing_sent:
                        yield timing_event()
            except (TimeoutError, httpx.HTTPError, ValueError):
                if diagnostic is not None and not timing_sent:
                    yield timing_event()
                yield b'data: {"error":{"message":"The model response was interrupted. Please retry."}}\n\ndata: [DONE]\n\n'
            finally:
                try:
                    await response.aclose()
                finally:
                    admission.release(ticket)

        return StreamingResponse(generate(), media_type='text/event-stream', headers={
            'X-Context-Trimmed': str(trimmed), 'X-Accel-Buffering': 'no',
            **diagnostic_headers(),
        })

    return app


if __name__ == '__main__':
    import uvicorn
    secret = Path(os.environ.get('CHAT_TOKEN_FILE', '/workspace/chat/chat-token')).read_text().strip()
    app = create_app(secret, model=os.environ.get('SERVED_MODEL_NAME', 'qwen'),
                     upstream='http://127.0.0.1:' + os.environ.get('VLLM_PORT', '8000'),
                     context=int(os.environ.get('MAX_MODEL_LEN', '40960')),
                     concurrency=min(2, int(os.environ.get('MAX_NUM_SEQS', '2'))))
    uvicorn.run(app, host=os.environ.get('CHAT_HOST', '127.0.0.1'),
                port=int(os.environ.get('CHAT_PORT', '8001')), access_log=False)
