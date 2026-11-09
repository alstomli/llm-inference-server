"""Offline benchmark checks, including real concurrent HTTP streams (no GPU)."""
from contextlib import redirect_stdout, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('benchmark', Path(__file__).with_name('benchmark-chat.py'))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def frame(item):
    return ('data: ' + json.dumps(item) + '\n\n').encode()


def stream():
    return (b': keepalive\r\n\r\n'
            + frame({'choices': [{'delta': {'role': 'assistant'}}]})
            + frame({'choices': [{'delta': {'content': 'many tokens in this chunk'}}]})
            + frame({'choices': [{'delta': {'content': ' more'}}]})
            + frame({'choices': [{'delta': {}, 'finish_reason': 'length'}]})
            + frame({'choices': [], 'usage': {'completion_tokens': 20}})
            + b'event: chat-timing\ndata: {"timings":{"contextFitMs":50}}\n\n'
            + b'data: [DONE]\n\n')


class MetricTests(unittest.TestCase):
    def test_usage_is_not_chunk_count_and_role_is_not_ttft(self):
        clock = iter([1, 2, 3, 4, 5, 6, 7]).__next__
        result = benchmark.stream_metrics(BytesIO(stream()), 0, clock)
        self.assertEqual(result['ttft_seconds'], 2)
        self.assertEqual(result['completion_tokens'], 20)
        self.assertEqual(result['post_ttft_tokens_per_second'], 20 / 5)
        self.assertEqual(result['output_tokens_per_total_second'], 20 / 7)
        self.assertEqual(result['gateway_timings_ms']['contextFitMs'], 50)

    def test_crlf_and_multiline_sse(self):
        data = b'event: chat-timing\r\ndata: {"timings":\r\ndata: {}}\r\n\r\n'
        name, body = next(benchmark.events(BytesIO(data)))
        self.assertEqual(name, 'chat-timing')
        self.assertEqual(json.loads(body), {'timings': {}})

    def test_incomplete_or_error_streams_cannot_be_successes(self):
        for data in (
            stream().replace(b'data: [DONE]\n\n', b''),
            b'data: {"error":{"message":"private data"}}\n\n',
            stream().replace(frame({'choices': [], 'usage': {'completion_tokens': 20}}), b''),
            stream().replace(frame({'choices': [{'delta': {}, 'finish_reason': 'length'}]}), b''),
            b'data: {"choices":[]}',
        ):
            with self.subTest(data=data[:40]), self.assertRaises(benchmark.BenchmarkError):
                benchmark.stream_metrics(BytesIO(data), time.perf_counter())


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.jobs = []
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.fail = False
        self.redirect = False
        self.diagnostics = True
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if owner.redirect:
                    self.send_response(302)
                    self.send_header('Location', '/redirect-target')
                    self.end_headers()
                    return
                body = {'ready': True, 'model': 'qwen-bf16', 'diagnosticsVersion': int(owner.diagnostics),
                        'imageInputVersion': 1, 'imageInput': {'maxBytes': 262144, 'mimeTypes': ['image/png']},
                        'adapterToggleAvailable': True, 'admission': {'maxConcurrentRequests': 2},
                        'limits': {'outputTokens': 10240, 'contextTokens': 40960}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with owner.lock:
                    owner.jobs.append((self.headers['X-Chat-User-Id'], payload))
                    owner.active += 1
                    owner.peak = max(owner.peak, owner.active)
                try:
                    if owner.fail:
                        self.send_response(429)
                        self.end_headers()
                        self.wfile.write(b'private response body')
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('X-Chat-Image-Accepted', 'true')
                    self.send_header('X-Chat-Adapter-Enabled', 'false')
                    self.send_header('X-Chat-Diagnostics', json.dumps({'model': 'qwen-bf16-base', 'secret': 'do-not-save'}))
                    self.end_headers()
                    time.sleep(0.08)
                    self.wfile.write(stream())
                    self.wfile.flush()
                finally:
                    with owner.lock:
                        owner.active -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.stop_server)
        self.image = Path(self.temp.name) / 'sample.png'
        self.image.write_bytes(b'fixture image; decoding belongs to gateway')
        self.output = Path(self.temp.name) / 'report.json'

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def invoke(self, *extra):
        argv = ['benchmark-chat.py', '--image', str(self.image), '--output', str(self.output), *extra]
        environment = {'RUNPOD_BF16_CHAT_URL': self.base, 'RUNPOD_BF16_CHAT_TOKEN': 'private-token'}
        with patch('sys.argv', argv), patch.dict(os.environ, environment, clear=True), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return benchmark.main()

    def test_complete_matrix_two_users_base_model_and_private_report(self):
        self.assertEqual(self.invoke(), 0)
        report = json.loads(self.output.read_text())
        self.assertTrue(report['ok'])
        self.assertEqual(len(report['runs']), 5)  # Two warmups, three scenarios.
        self.assertEqual(len(self.jobs), 7)
        self.assertEqual(self.peak, 2)
        for run in report['runs'][-2:]:
            self.assertGreater(run['client_overlap_seconds'], 0)
            self.assertEqual(len(run['requests']), 2)
        self.assertNotEqual(self.jobs[-2][0], self.jobs[-1][0])
        for _, payload in self.jobs:
            self.assertFalse(payload['useAdapter'])
            self.assertFalse(payload['settings']['reuseCachedPrefixes'])
        saved = self.output.read_text()
        for private in ('private-token', 'do-not-save', self.base, 'data:image', benchmark.PROMPT):
            self.assertNotIn(private, saved)

    def test_failed_request_is_recorded_and_stops_run(self):
        self.fail = True
        self.assertEqual(self.invoke('--skip-warmup'), 1)
        report = json.loads(self.output.read_text())
        self.assertFalse(report['ok'])
        self.assertEqual(len(self.jobs), 1)
        self.assertEqual(report['runs'][0]['requests'][0]['error'], 'HTTP 429')
        self.assertNotIn('private response body', self.output.read_text())

    def test_existing_report_is_not_overwritten(self):
        self.output.write_text('keep')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(self.output.read_text(), 'keep')
        self.assertFalse(self.jobs)

    def test_same_user_and_missing_diagnostics_fail_before_load(self):
        with self.assertRaises(SystemExit):
            self.invoke('--user-b', '00000000-0000-4000-8000-000000000101')
        self.diagnostics = False
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.jobs)

    def test_redirect_is_not_followed(self):
        self.redirect = True
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertFalse(self.jobs)


if __name__ == '__main__':
    unittest.main()
