#!/usr/bin/env python3
"""Send a 100-word example prompt to the API inside the Runpod Pod."""
import json
import time
from urllib.request import Request, urlopen

PROMPT = (
    "Explain how an LLM inference server processes a request, starting when a user submits "
    "a prompt and ending when the generated answer reaches the client. Describe tokenization, "
    "loading model weights, GPU computation, prefill, decoding, and the KV cache in language "
    "that a beginner can understand. Then explain how continuous batching helps multiple users "
    "share one GPU. Finally, describe what time to first token, output tokens per second, GPU "
    "utilization, and memory usage reveal about performance. Use our example of Qwen running "
    "through vLLM on a Runpod GPU. Give a clear, concise explanation with one useful analogy "
    "and no unnecessary jargon."
)

payload = {
    'model': 'qwen-bf16',
    'messages': [{'role': 'user', 'content': PROMPT}],
    'max_tokens': 512,
    'temperature': 0.3,
    'chat_template_kwargs': {'enable_thinking': False},
    'stream': False,
}
print(f'PROMPT ({len(PROMPT.split())} words):\n{PROMPT}\n', flush=True)
request = Request('http://127.0.0.1:8000/v1/chat/completions',
                  data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'})
start = time.perf_counter()
with urlopen(request, timeout=180) as response:
    result = json.load(response)
    print('HTTP:', response.status)
elapsed = time.perf_counter() - start
print('\nRESPONSE:\n' + result['choices'][0]['message']['content'])
print('\nFinish reason:', result['choices'][0]['finish_reason'])
print('Usage:', json.dumps(result.get('usage')))
print(f'Total request time: {elapsed:.3f} seconds')
