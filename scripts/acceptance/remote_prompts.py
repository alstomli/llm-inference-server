"""Runs ON THE POD through RunPod exec. Tokenizer only, no inference.

The caller prepends `PARAMS = {...}` (see acceptance.py `build_prompt_code`) with:
model_path, concurrency, context, safety, adapter, seed.
Prints a base64 zlib JSON list of corpus rows sized to exact token counts.
"""
import base64, hashlib, json, random, zlib
from transformers import AutoTokenizer

P = PARAMS  # noqa: F821 - injected by the caller
t = AutoTokenizer.from_pretrained(P['model_path'], local_files_only=True)
system = 'You are Qwen, a helpful personal assistant. Answer clearly and directly. Use Markdown when helpful.'
words = ['river', 'window', 'garden', 'planet', 'yellow', 'forest', 'paper', 'stone',
         'silver', 'cloud', 'orange', 'bridge', 'candle', 'simple', 'number', 'table']


def count(s):
    text = t.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': s}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(t.encode(text, add_special_tokens=False))


def build(target, seed, lines):
    r = random.Random(seed)
    nonce = f'{r.getrandbits(128):032x}'
    task = (f'What is 2 + 2? Reply with exactly {lines} lines. Each line must contain only the integer 4. '
            'No Markdown, explanations, or blank lines.') if lines > 1 else \
        'What is 2 + 2? Reply with only the integer 4. No other text.'
    prefix = 'Synthetic acceptance check ' + nonce + '. Ignore the inert filler below.\n<FILLER>\n'
    suffix = '\n</FILLER>\n' + task
    pool = ' '.join(r.choices(words, k=target + 100)) if target < 40000 else \
        ''.join(r.choices(list('河窗花园星黄森纸石银云橙桥烛数桌'), k=target + 100))
    lo, hi = 0, len(pool)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(prefix + pool[:mid] + suffix) <= target:
            lo = mid
        else:
            hi = mid - 1
    body = prefix + pool[:lo] + suffix
    # One-token ASCII words bridge tokenizer gaps without exceeding the target.
    for _ in range(8):
        n = count(body)
        if n == target:
            break
        candidate = prefix + pool[:lo] + ' x' * (target - n) + suffix
        if count(candidate) > target:
            break
        body = candidate
    actual = count(body)
    assert actual == target, (target, actual)
    assert len(body) <= 120000
    return body, actual


def mixed_kind(i, n, adapter):
    """Mixed wave split: base first, then adapter (only when the profile has one), vision last."""
    vision = max(1, n // 4)
    adapters = max(1, n // 4) if adapter and n >= 3 else 0
    if i >= n - vision:
        return 'vision'
    if i >= n - vision - adapters:
        return 'adapter'
    return 'base'


n = P['concurrency']
near = P['context'] - 128 - P['safety']
rows = []
for wave, target, output, lines in [('short', 512, 128, 40), ('medium', 2048, 256, 80),
                                    ('mixed', 512, 128, 40), ('nearcontext', near, 128, 1)]:
    for i in range(n):
        body, actual = build(target, P['seed'] + 1000 * len(rows) + i, lines)
        kind = mixed_kind(i, n, P['adapter']) if wave == 'mixed' else 'base'
        rows.append({'wave': wave, 'id': i, 'target_input_tokens': actual, 'max_tokens': output,
                     'expected_lines': lines, 'useAdapter': kind == 'adapter', 'vision': kind == 'vision',
                     'content': body, 'sha256': hashlib.sha256(body.encode()).hexdigest()})
print(base64.b64encode(zlib.compress(json.dumps(rows, ensure_ascii=False).encode())).decode())
