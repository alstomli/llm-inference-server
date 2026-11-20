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
                patch.dict(checker.os.environ, {'RUNPOD_BF16_CHAT_URL': 'https://fixture.invalid', 'RUNPOD_BF16_CHAT_TOKEN': 'x' * 48, **(environment or {})}, clear=True), \
                patch('sys.argv', ['check-chat-deployment.py', *arguments]), patch.object(checker, 'urlopen', urlopen), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            checker.main()
        return requests

    def test_strict_health_then_stable_bounded_service_principal(self):
        health = {'ready': True, 'model': 'qwen-bf16', 'accountQuotaVersion': 1, 'accountQuotasEnforced': True}
        requests = self.run_checker(health)
        self.assertEqual(len(requests), 2)
        self.assertIsNone(requests[0].get_header('X-chat-user-id'))
        self.assertEqual(requests[1].get_header('X-chat-user-id'), '00000000-0000-4000-8000-000000000001')
        requests = self.run_checker(health, environment={'CHAT_CHECK_USER_ID': 'AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA'})
        self.assertEqual(requests[1].get_header('X-chat-user-id'), 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')

    def test_old_or_compatibility_health_fails_before_inference(self):
        for fields in [{}, {'accountQuotaVersion': 1, 'accountQuotasEnforced': False}, {'accountQuotaVersion': 2, 'accountQuotasEnforced': True}]:
            with self.subTest(fields=fields), self.assertRaises(AssertionError):
                self.run_checker({'ready': True, 'model': 'qwen-bf16', **fields})

    def test_removed_connection_cannot_supply_a_missing_bf16_target_or_token(self):
        for environment in ({'RUNPOD_CHAT_URL': 'https://retired.invalid', 'RUNPOD_CHAT_TOKEN': 'old'},
                            {'RUNPOD_BF16_CHAT_URL': 'https://fixture.invalid', 'RUNPOD_CHAT_TOKEN': 'old'},
                            {'RUNPOD_CHAT_URL': 'https://retired.invalid', 'RUNPOD_BF16_CHAT_TOKEN': 'new'}):
            with self.subTest(environment=environment), \
                    patch.object(checker, 'ROOT', Path(__file__).parent / '__quota_checker_test__' / 'server'), \
                    patch.dict(checker.os.environ, environment, clear=True), \
                    patch('sys.argv', ['check-chat-deployment.py']), patch.object(checker, 'urlopen') as network, \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                checker.main()
            network.assert_not_called()

    def test_4bit_profile_checks_its_own_gateway_for_the_4bit_alias(self):
        health = {'ready': True, 'model': 'qwen-4bit', 'accountQuotaVersion': 1, 'accountQuotasEnforced': True}
        environment = {'RUNPOD_BF16_CHAT_URL': 'https://bf16.invalid', 'RUNPOD_BF16_CHAT_TOKEN': 'b' * 48,
                       'RUNPOD_4BIT_CHAT_URL': 'https://int4.invalid/', 'RUNPOD_4BIT_CHAT_TOKEN': 'f' * 48}
        requests = self.run_checker(health, ['--profile', '4bit'], environment)
        self.assertEqual([r.full_url for r in requests], ['https://int4.invalid/health', 'https://int4.invalid/v1/chat/completions'])
        self.assertTrue(all(r.get_header('Authorization') == 'Bearer ' + 'f' * 48 for r in requests))
        with self.assertRaises(AssertionError):
            self.run_checker({**health, 'model': 'qwen-bf16'}, ['--profile', '4bit'], environment)

    def test_web_search_model_count_follows_the_profile_when_the_gateway_reports_it(self):
        connections = {'RUNPOD_4BIT_CHAT_URL': 'https://int4.invalid', 'RUNPOD_4BIT_CHAT_TOKEN': 'f' * 48}
        quota = {'ready': True, 'accountQuotaVersion': 1, 'accountQuotasEnforced': True}
        model = {'available': False, 'reason': 'disabled'}
        for profile, alias, count in (('bf16', 'qwen-bf16', 2), ('4bit', 'qwen-4bit', 1)):
            health = {**quota, 'model': alias}
            with self.subTest(profile=profile):
                # A gateway without versioned web search (the Python fallback) is not asked for a count.
                self.assertEqual(len(self.run_checker(health, ['--profile', profile], connections)), 2)
                right = {**health, 'webSearchVersion': 1, 'webSearch': {'models': [model] * count}}
                self.assertEqual(len(self.run_checker(right, ['--profile', profile], connections)), 2)
                for wrong in (0, 3 - count, 3):
                    with self.subTest(models=wrong), self.assertRaises(AssertionError):
                        self.run_checker({**right, 'webSearch': {'models': [model] * wrong}}, ['--profile', profile], connections)

    def test_4bit_profile_never_borrows_the_bf16_connection(self):
        for environment in ({'RUNPOD_4BIT_CHAT_URL': 'https://int4.invalid'},
                            {'RUNPOD_4BIT_CHAT_TOKEN': 'f' * 48}):
            with self.subTest(environment=environment), \
                    patch.object(checker, 'ROOT', Path(__file__).parent / '__quota_checker_test__' / 'server'), \
                    patch.dict(checker.os.environ, {'RUNPOD_BF16_CHAT_URL': 'https://bf16.invalid',
                                                    'RUNPOD_BF16_CHAT_TOKEN': 'b' * 48, **environment}, clear=True), \
                    patch('sys.argv', ['check-chat-deployment.py', '--profile', '4bit']), \
                    patch.object(checker, 'urlopen') as network, redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                checker.main()
            network.assert_not_called()

    def test_unknown_profile_fails_before_network(self):
        for value in ('4BIT', 'awq', ''):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.run_checker({}, arguments=['--profile', value])

    def test_expected_concurrency_verifies_active_and_three_times_waiting_slots(self):
        health = {'ready': True, 'model': 'qwen-bf16', 'accountQuotaVersion': 1, 'accountQuotasEnforced': True}
        for active, waiting in [(2, 6), (4, 12), (5, 15), (8, 24)]:
            with self.subTest(active=active):
                ok = {**health, 'admission': {'maxConcurrentRequests': active, 'maxWaitingRequests': waiting}}
                self.assertEqual(len(self.run_checker(ok, ['--expected-concurrency', str(active)])), 2)
        for admission in [{}, {'maxConcurrentRequests': 4, 'maxWaitingRequests': 6},
                          {'maxConcurrentRequests': 2, 'maxWaitingRequests': 12}]:
            with self.subTest(admission=admission), self.assertRaises(AssertionError):
                self.run_checker({**health, 'admission': admission}, ['--expected-concurrency', '4'])

    def test_expected_concurrency_outside_two_to_eight_fails_before_network(self):
        for value in ('1', '9', 'x'):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.run_checker({}, arguments=['--expected-concurrency', value])

    def test_invalid_identity_fails_before_network(self):
        with self.assertRaises(SystemExit):
            self.run_checker({}, arguments=['--user-id', 'not-a-uuid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
