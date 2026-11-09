"""Offline checks for the strict deployment-checker identity contract."""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
from io import BytesIO, StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('checker', Path(__file__).with_name('check-chat-deployment.py'))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class Response(BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream'}


class CheckerTests(unittest.TestCase):
    def run_checker(self, health, arguments=(), environment=None):
        requests = []

        def urlopen(request, timeout):
            requests.append(request)
            if request.full_url.endswith('/health'):
                return Response(json.dumps(health).encode())
            return Response(b'data: {"choices":[{"delta":{"content":"ready"}}]}\n\ndata: [DONE]\n\n')

        with patch.object(checker, 'ROOT', Path(__file__).parent / '__quota_checker_test__' / 'server'), \
                patch.dict(checker.os.environ, {'RUNPOD_CHAT_URL': 'https://fixture.invalid', 'RUNPOD_CHAT_TOKEN': 'x' * 48, **(environment or {})}, clear=True), \
                patch('sys.argv', ['check-chat-deployment.py', *arguments]), patch.object(checker, 'urlopen', urlopen), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            checker.main()
        return requests

    def test_strict_health_then_stable_bounded_service_principal(self):
        health = {'ready': True, 'model': 'qwen', 'accountQuotaVersion': 1, 'accountQuotasEnforced': True}
        requests = self.run_checker(health)
        self.assertEqual(len(requests), 2)
        self.assertIsNone(requests[0].get_header('X-chat-user-id'))
        self.assertEqual(requests[1].get_header('X-chat-user-id'), '00000000-0000-4000-8000-000000000001')
        requests = self.run_checker(health, environment={'CHAT_CHECK_USER_ID': 'AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA'})
        self.assertEqual(requests[1].get_header('X-chat-user-id'), 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')

    def test_old_or_compatibility_health_fails_before_inference(self):
        for fields in [{}, {'accountQuotaVersion': 1, 'accountQuotasEnforced': False}, {'accountQuotaVersion': 2, 'accountQuotasEnforced': True}]:
            with self.subTest(fields=fields), self.assertRaises(AssertionError):
                self.run_checker({'ready': True, 'model': 'qwen', **fields})

    def test_invalid_identity_fails_before_network(self):
        with self.assertRaises(SystemExit):
            self.run_checker({}, arguments=['--user-id', 'not-a-uuid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
