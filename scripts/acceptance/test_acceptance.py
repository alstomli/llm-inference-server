"""Offline checks for the profile acceptance harness. No network, no Pod."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acceptance as acc  # noqa: E402

GIB = 1024**3


def cfg(profile='4bit', n=4):
    return acc.parse(['--profile', profile, '--pod', 'pod123', '--concurrency', str(n), 'status'])[0]


def status_row(c):
    spec = c.spec
    args = ['vllm', 'serve', c.model_path, '--served-model-name', spec['base_alias'],
            '--max-num-seqs', str(c.concurrency), '--max-model-len', str(c.context)]
    if spec['kv_cache_dtype']:
        args += ['--kv-cache-dtype', spec['kv_cache_dtype']]
    if spec['adapter_alias']:
        args += ['--enable-lora']
    return {'processes': [{'pid': 1, 'args': args}], 'health_status': 200, 'gateway_http': 200,
            'gateway_health': {'ready': True, 'imageInputVersion': 1, 'limits': {'contextTokens': c.context},
                               'admission': {'maxConcurrentRequests': c.concurrency, 'maxWaitingRequests': c.waiting,
                                             'activeRequests': 0, 'waitingRequests': 0}},
            'idle_counts': {g: 0 for g in acc.IDLE_GAUGES},
            'memory': {'memory.max': 100 * GIB, 'memory.current': 40 * GIB, 'memory.swap.current': 0,
                       'memory.events': {'oom': 0, 'oom_kill': 0}, 'shm_available_bytes': 16 * GIB}}


def remote_mixed_kind():
    """Load mixed_kind from remote_prompts.py without running its tokenizer code."""
    tree = ast.parse((Path(acc.HERE) / 'remote_prompts.py').read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'mixed_kind')
    scope = {}
    exec(compile(ast.Module([fn], []), 'remote_prompts', 'exec'), scope)
    return scope['mixed_kind']


def corpus(c):
    kind = remote_mixed_kind()
    rows = []
    for wave in acc.WAVES:
        for i in range(c.concurrency):
            k = kind(i, c.concurrency, bool(c.spec['adapter_alias'])) if wave == 'mixed' else 'base'
            tokens = c.context - 128 - acc.SAFETY_TOKENS if wave == 'nearcontext' else 512
            rows.append({'wave': wave, 'id': i, 'target_input_tokens': tokens, 'max_tokens': 128,
                         'useAdapter': k == 'adapter', 'vision': k == 'vision'})
    return rows


class ParseTests(unittest.TestCase):
    def test_profile_defaults(self):
        c = cfg('4bit', 4)
        self.assertEqual((c.waiting, c.context), (12, 40960))
        self.assertTrue(str(c.token_file).endswith('4bit-chat-token'))
        self.assertEqual(c.host, 'pod123-8001.proxy.runpod.net')
        self.assertIn('acceptance-results', str(c.run_dir))

    def test_rejects_zero_concurrency(self):
        with self.assertRaises(SystemExit):
            acc.parse(['--profile', '4bit', '--pod', 'p', '--concurrency', '0', 'all'])

    def test_prompt_code_injects_params(self):
        code = acc.build_prompt_code(cfg('4bit', 3))
        params = ast.literal_eval(code.splitlines()[0].split('=', 1)[1].strip())
        self.assertEqual(params['concurrency'], 3)
        self.assertFalse(params['adapter'])
        self.assertTrue(acc.build_prompt_code(cfg('bf16', 8)).startswith("PARAMS = {'model_path': '/workspace/qwen-bf16/model'"))


class CorpusTests(unittest.TestCase):
    def test_4bit_mixed_has_vision_and_no_adapter(self):
        for n in (1, 2, 4, 8):
            c = cfg('4bit', n)
            acc.check_corpus(c, corpus(c))

    def test_bf16_mixed_keeps_adapter_split(self):
        c = cfg('bf16', 8)
        rows = [r for r in corpus(c) if r['wave'] == 'mixed']
        self.assertEqual(sum(r['useAdapter'] for r in rows), 2)
        self.assertEqual(sum(r['vision'] for r in rows), 2)
        acc.check_corpus(c, corpus(c))

    def test_adapter_rows_rejected_for_4bit(self):
        c = cfg('4bit', 4)
        rows = corpus(c)
        next(r for r in rows if r['wave'] == 'mixed')['useAdapter'] = True
        with self.assertRaises(AssertionError):
            acc.check_corpus(c, rows)


class StatusTests(unittest.TestCase):
    def test_valid_rows_pass_for_both_profiles(self):
        for profile, n in (('4bit', 4), ('bf16', 8)):
            c = cfg(profile, n)
            acc.validate_status(c, status_row(c))

    def test_4bit_requires_fp8_kv(self):
        c = cfg('4bit', 4)
        row = status_row(c)
        args = row['processes'][0]['args']
        args[args.index('--kv-cache-dtype') + 1] = 'auto'
        with self.assertRaises(AssertionError):
            acc.validate_status(c, row)

    def test_4bit_rejects_lora(self):
        c = cfg('4bit', 4)
        row = status_row(c)
        row['processes'][0]['args'].append('--enable-lora')
        with self.assertRaises(AssertionError):
            acc.validate_status(c, row)

    def test_concurrency_mismatch_fails(self):
        c = cfg('4bit', 4)
        row = status_row(c)
        row['gateway_health']['admission']['maxConcurrentRequests'] = 8
        with self.assertRaises(AssertionError):
            acc.validate_status(c, row)

    def test_busy_or_short_on_ram_fails(self):
        c = cfg('4bit', 4)
        busy = status_row(c)
        busy['gateway_health']['admission']['activeRequests'] = 1
        low = status_row(c)
        low['memory']['memory.current'] = low['memory']['memory.max'] - 4 * GIB
        for row in (busy, low):
            with self.assertRaises(AssertionError):
                acc.validate_status(c, row)

    def test_oom_growth_against_baseline_fails(self):
        c = cfg('4bit', 4)
        baseline = status_row(c)
        row = copy.deepcopy(baseline)
        row['memory']['memory.events']['oom_kill'] = 1
        with self.assertRaises(AssertionError):
            acc.validate_status(c, row, baseline)


class ResponseTests(unittest.TestCase):
    def headers(self, model, adapter=None, tokens=512, vision=False):
        h = [('X-Context-Trimmed', '0'), ('Cache-Control', 'no-store'),
             ('X-Chat-Diagnostics', json.dumps({'model': model, 'gatewayTrimmedMessages': 0, 'inputTokensAfter': tokens}))]
        if adapter is not None:
            h.append(('X-Chat-Adapter-Enabled', adapter))
        if vision:
            h.append(('X-Chat-Image-Accepted', 'true'))
        return h

    def row(self, adapter=False, vision=False):
        return {'useAdapter': adapter, 'vision': vision, 'target_input_tokens': 512, 'max_tokens': 128}

    def test_4bit_base_headers(self):
        c = cfg('4bit', 4)
        acc.check_headers(c, self.row(), self.headers('qwen-4bit'))
        acc.check_headers(c, self.row(), self.headers('qwen-4bit', adapter='false'))
        acc.check_headers(c, self.row(vision=True), self.headers('qwen-4bit', tokens=600, vision=True))
        with self.assertRaises(AssertionError):
            acc.check_headers(c, self.row(), self.headers('qwen-bf16-base'))

    def test_4bit_requests_omit_use_adapter_which_its_gateway_rejects(self):
        # The gateway answers 400 to any useAdapter field on a Pod without an adapter.
        c = cfg('4bit', 4)
        for row in (self.row(), self.row(vision=True)):
            row['content'] = 'What is 2+2?'
            self.assertNotIn('useAdapter', acc.payload(c, row))
        self.assertEqual(acc.adapter_choice(c, False), {})

    def test_vision_requests_ask_only_about_the_image(self):
        # The 4-bit model keeps an earlier "reply only 4" over a later override, so the vision
        # row checks vision alone instead of instruction overriding.
        for profile in ('4bit', 'bf16'):
            row = {**self.row(vision=True), 'content': 'Reply only with the integer 4.'}
            body = acc.payload(cfg(profile, 4), row)
            self.assertEqual(body['messages'], [{'role': 'user', 'content': acc.VISION_QUESTION}])
            self.assertTrue(body['image'].startswith('data:image/png;base64,'))

    def test_bf16_requests_keep_the_adapter_choice(self):
        c = cfg('bf16', 8)
        row = {**self.row(adapter=True), 'content': 'What is 2+2?'}
        self.assertIs(acc.payload(c, row)['useAdapter'], True)
        self.assertEqual(acc.adapter_choice(c, False), {'useAdapter': False})

    def test_bf16_adapter_headers(self):
        c = cfg('bf16', 8)
        acc.check_headers(c, self.row(adapter=True), self.headers('qwen-bf16', adapter='true'))
        with self.assertRaises(AssertionError):
            acc.check_headers(c, self.row(adapter=True), self.headers('qwen-bf16', adapter='false'))

    def test_oracles(self):
        self.assertTrue(acc.arithmetic_correct('4\n4\n4'))
        self.assertFalse(acc.arithmetic_correct('4 and 5'))
        self.assertFalse(acc.arithmetic_correct('four'))
        self.assertTrue(acc.vision_correct('Red.'))
        self.assertFalse(acc.vision_correct('red and blue'))

    def test_fixture_is_png(self):
        self.assertTrue(acc.png_uri().startswith('data:image/png;base64,iVBOR'))


if __name__ == '__main__':
    unittest.main()
