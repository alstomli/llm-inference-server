"""Offline tests for the deployment safety decisions."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('deploy', Path(__file__).with_name('deploy-runpod.py'))
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
IMAGE = 'zihan000/qwen-inference@sha256:' + 'a' * 64
PREVIOUS = 'zihan000/qwen-inference@sha256:' + 'b' * 64
ENV = {'RUNPOD_BF16_POD_ID': 'bf16pod', 'DEPLOY_IMAGE': IMAGE,
       'RUNPOD_API_KEY': 'test-api', 'RUNPOD_BF16_CHAT_TOKEN': 'test-chat'}
BF16_ENV = {**ENV, 'DEPLOY_MODEL_PROFILE': 'bf16', 'RUNPOD_BF16_POD_ID': 'bf16pod',
            'RUNPOD_BF16_CHAT_TOKEN': 'test-bf16', 'TAVILY_API_KEY': 'fixture-provider-key'}
INT4_ENV = {'RUNPOD_API_KEY': 'test-api', 'DEPLOY_IMAGE': IMAGE, 'DEPLOY_MODEL_PROFILE': '4bit',
            'RUNPOD_4BIT_POD_ID': 'int4pod', 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-token'}
BOTH_ENV = {'RUNPOD_API_KEY': 'test-api', 'DEPLOY_IMAGE': IMAGE,
            'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-token',
            'RUNPOD_4BIT_POD_ID': 'int4pod', 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-token'}
POD_ENV = {'MODEL_PROFILE': 'bf16', 'SERVED_MODEL_NAME': 'qwen-bf16',
           'VLLM_BASE_MODEL_NAME': 'qwen-bf16-base', 'MAX_NUM_SEQS': '2',
           'MAX_MODEL_LEN': '40960', 'CPU_OFFLOAD_GB': '0', 'JUPYTER_PASSWORD': 'fixture-existing-secret'}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        # Any test without an explicit HTTP stub must remain strictly offline.
        network = patch.object(deploy, 'urlopen', side_effect=AssertionError('Unmocked network request'))
        network.start()
        self.addCleanup(network.stop)

    def test_pod_api_uses_runpod_rest_api_v2(self):
        # REST API v1 (rest.runpod.io) was retired; v2 names the image `image` and the state `status`.
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"image": "x"}'
        with patch.dict(os.environ, {'RUNPOD_API_KEY': 'test-api'}), \
                patch.object(deploy, 'urlopen', return_value=response) as opened:
            self.assertEqual(deploy.api('GET', '/pods/testpod'), {'image': 'x'})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.runpod.io/v2/pods/testpod')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-api')

    def test_web_staging_preserves_all_unrelated_settings_and_never_verifies(self):
        result = deploy.web_environment(POD_ENV, 'stage', '', 'fixture-provider-key')
        for key, value in POD_ENV.items():
            self.assertEqual(result[key], value)
        self.assertEqual(result['CHAT_WEB_SEARCH_TOOL_PARSER'], 'qwen3_xml')
        self.assertEqual(result['CHAT_WEB_SEARCH_ENABLED'], 'false')
        self.assertEqual(result['CHAT_WEB_SEARCH_MODEL_VERIFIED'], 'false')
        self.assertEqual(result['CHAT_WEB_SEARCH_VERIFIED_MODEL'], '')
        self.assertEqual(result['TAVILY_API_KEY'], 'fixture-provider-key')
        self.assertNotIn('TAVILY_API_KEY', POD_ENV)

    def test_web_enable_requires_staging_and_exact_alias_assertion(self):
        staged = deploy.web_environment(POD_ENV, 'stage', '', 'fixture-provider-key')
        for action, alias in [('enable-base', 'qwen-bf16-base'), ('enable-adapter', 'qwen-bf16')]:
            enabled = deploy.web_environment(staged, action, alias, 'fixture-provider-key')
            self.assertEqual(enabled['CHAT_WEB_SEARCH_VERIFIED_MODEL'], alias)
            self.assertEqual(enabled['CHAT_WEB_SEARCH_ENABLED'], 'true')
            self.assertEqual(enabled['CHAT_WEB_SEARCH_MODEL_VERIFIED'], 'true')
            self.assertEqual(enabled['JUPYTER_PASSWORD'], POD_ENV['JUPYTER_PASSWORD'])
            with self.assertRaises(ValueError):
                deploy.web_environment(staged, action, 'unverified', 'fixture-provider-key')
            with self.assertRaises(ValueError):
                deploy.web_environment(POD_ENV, action, alias, 'fixture-provider-key')

    def test_web_stage_rejects_wrong_profiles_aliases_missing_key_and_verification(self):
        for current, alias, key in [({**POD_ENV, 'MODEL_PROFILE': '4bit'}, '', 'fixture-key'),
                                    ({**POD_ENV, 'SERVED_MODEL_NAME': 'other'}, '', 'fixture-key'),
                                    ({**POD_ENV, 'CHAT_GATEWAY_IMPLEMENTATION': 'python'}, '', 'fixture-key'),
                                    (POD_ENV, '', ''), (POD_ENV, 'qwen-bf16-base', 'fixture-key')]:
            with self.assertRaises(ValueError):
                deploy.web_environment(current, 'stage', alias, key)

    def test_web_disable_preserves_key_parser_and_ordinary_serving_settings(self):
        staged = deploy.web_environment(POD_ENV, 'stage', '', 'fixture-provider-key')
        enabled = deploy.web_environment(staged, 'enable-base', 'qwen-bf16-base', 'fixture-provider-key')
        disabled = deploy.web_environment(enabled, 'disable', '', '')
        self.assertEqual(disabled['CHAT_WEB_SEARCH_ENABLED'], 'false')
        self.assertEqual(disabled['CHAT_WEB_SEARCH_MODEL_VERIFIED'], 'false')
        self.assertEqual(disabled['CHAT_WEB_SEARCH_VERIFIED_MODEL'], '')
        self.assertEqual(disabled['TAVILY_API_KEY'], enabled['TAVILY_API_KEY'])
        self.assertEqual(disabled['CHAT_WEB_SEARCH_TOOL_PARSER'], enabled['CHAT_WEB_SEARCH_TOOL_PARSER'])

    @patch.dict(os.environ, {**BF16_ENV, 'DEPLOY_WEB_SEARCH_ACTION': 'stage'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'check_web_capability', side_effect=RuntimeError('capability failed'))
    @patch.object(deploy, 'api', return_value={'image': IMAGE, 'status': 'RUNNING', 'env': POD_ENV})
    def test_web_capability_failure_rolls_back_full_environment_and_image_without_logging_key(self, api, check, apply, summary):
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual(apply.call_count, 2)
        target = apply.call_args_list[0].kwargs['env']
        self.assertEqual(target['CHAT_WEB_SEARCH_ENABLED'], 'false')
        self.assertEqual(target['TAVILY_API_KEY'], 'fixture-provider-key')
        self.assertEqual(apply.call_args_list[1].kwargs['env'], POD_ENV)
        self.assertEqual(apply.call_args_list[1].args[1], IMAGE)
        self.assertNotIn('fixture-provider-key', str(summary.call_args_list))
        self.assertNotIn('fixture-existing-secret', str(summary.call_args_list))

    def deploy_4bit_web_action(self, action, alias, pod_env, models):
        """Deploy a 4-bit web action against a fake Runpod API whose gateway health lists `models`."""
        environment = {**INT4_ENV, 'DEPLOY_WEB_SEARCH_ACTION': action, 'DEPLOY_WEB_SEARCH_VERIFIED_ALIAS': alias,
                       'TAVILY_API_KEY': 'fixture-provider-key'}
        health = {'webSearchVersion': 1, 'webSearch': {'reason': 'disabled' if action in ('stage', 'disable') else 'ready', 'models': models}}

        class Health:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, *args):
                return json.dumps(health).encode()

        patched = []

        def api(method, path, body=None):
            if method == 'PATCH':
                patched.append(body)
                return {}
            return {'image': PREVIOUS, 'status': 'RUNNING', 'env': pod_env, **(patched[-1] if patched else {})}

        with patch.dict(os.environ, environment, clear=True), patch.object(deploy, 'api', side_effect=api), \
                patch.object(deploy, 'check_transition'), patch.object(deploy, 'summary'), \
                patch.object(deploy.subprocess, 'run'), patch.object(deploy, 'urlopen', return_value=Health()):
            deploy.main()
        return patched

    def test_4bit_web_search_uses_its_single_alias_and_one_model(self):
        pod_env = {'MODEL_PROFILE': '4bit', 'SERVED_MODEL_NAME': 'qwen-4bit', 'MAX_NUM_SEQS': '4'}
        one = [{'model': 'qwen-4bit', 'useAdapter': False, 'available': False, 'reason': 'disabled'}]
        staged = self.deploy_4bit_web_action('stage', '', pod_env, one)[0]['env']
        self.assertEqual(staged['CHAT_WEB_SEARCH_TOOL_PARSER'], 'qwen3_xml')
        self.assertEqual(staged['CHAT_WEB_SEARCH_ENABLED'], 'false')
        ready = [{'model': 'qwen-4bit', 'useAdapter': False, 'available': True, 'reason': 'ready'}]
        enabled = self.deploy_4bit_web_action('enable-base', 'qwen-4bit', staged, ready)[0]['env']
        self.assertEqual(enabled['CHAT_WEB_SEARCH_VERIFIED_MODEL'], 'qwen-4bit')
        self.assertEqual(enabled['CHAT_WEB_SEARCH_ENABLED'], 'true')
        self.assertEqual(enabled['MAX_NUM_SEQS'], '4')
        disabled = self.deploy_4bit_web_action('disable', '', enabled, one)[0]['env']
        self.assertEqual(disabled['CHAT_WEB_SEARCH_ENABLED'], 'false')

    def test_4bit_web_check_fails_and_rolls_back_unless_exactly_one_model_is_advertised(self):
        pod_env = {'MODEL_PROFILE': '4bit', 'SERVED_MODEL_NAME': 'qwen-4bit'}
        two = [{'model': 'qwen-4bit', 'available': False, 'reason': 'disabled'},
               {'model': 'qwen-4bit-base', 'available': False, 'reason': 'disabled'}]
        with self.assertRaises(RuntimeError):
            self.deploy_4bit_web_action('stage', '', pod_env, two)

    @patch.object(deploy, 'api')
    def test_4bit_web_actions_refuse_the_adapter_and_bf16_aliases_before_any_pod_api(self, api):
        environment = {**INT4_ENV, 'TAVILY_API_KEY': 'fixture-provider-key'}
        for action, alias in [('enable-adapter', 'qwen-4bit'), ('enable-adapter', 'qwen-bf16'),
                              ('enable-base', 'qwen-bf16-base')]:
            with self.subTest(action=action, alias=alias), self.assertRaises(ValueError), patch.dict(os.environ, {
                    **environment, 'DEPLOY_WEB_SEARCH_ACTION': action, 'DEPLOY_WEB_SEARCH_VERIFIED_ALIAS': alias}, clear=True):
                deploy.main()
        api.assert_not_called()

    def test_4bit_web_settings_require_the_4bit_alias_without_an_adapter(self):
        good = {'MODEL_PROFILE': '4bit', 'SERVED_MODEL_NAME': 'qwen-4bit'}
        self.assertEqual(deploy.web_environment({**good, 'VLLM_BASE_MODEL_NAME': 'qwen-4bit'}, 'stage', '', 'key', '4bit')['CHAT_WEB_SEARCH_ENABLED'], 'false')
        for current in ({**good, 'SERVED_MODEL_NAME': 'qwen-bf16'}, {'MODEL_PROFILE': '4bit'},
                        {**good, 'VLLM_BASE_MODEL_NAME': 'qwen-4bit-base'}, {**good, 'MODEL_PROFILE': 'bf16'},
                        {**good, 'CHAT_GATEWAY_IMPLEMENTATION': 'python'}):
            with self.subTest(current=current), self.assertRaises(ValueError):
                deploy.web_environment(current, 'stage', '', 'key', '4bit')

    def deploy_with_live_check(self, environment, pod_profile='bf16'):
        """Run the deploy command against a fake Runpod API; return the PATCH body and checker argv."""
        result = self.deploy_against_fake_runpod(environment, pod_profile)
        return result['patched'], result['checker']

    def deploy_against_fake_runpod(self, environment, pod_profile='bf16'):
        """Run the deploy command against a fake Runpod API and record what it touched."""
        current = {'image': PREVIOUS, 'status': 'RUNNING', 'env': {'MODEL_PROFILE': pod_profile, 'MAX_NUM_SEQS': '1'}}
        patched, paths, checker_env = {}, [], {}

        def api(method, path, body=None):
            paths.append(path)
            if method == 'PATCH':
                patched.update(body)
                return {}
            return {**current, **patched}

        def run(argv, **kwargs):
            checker_env.update(os.environ)

        with patch.dict(os.environ, environment, clear=True), patch.object(deploy, 'api', side_effect=api), \
                patch.object(deploy, 'check_transition'), patch.object(deploy, 'summary'), \
                patch.object(deploy.subprocess, 'run', side_effect=run) as runner:
            deploy.main()
        return {'patched': patched, 'paths': set(paths), 'checker': runner.call_args.args[0], 'checker_env': checker_env}

    def test_each_profile_deploys_only_its_own_pod_and_checks_its_alias(self):
        both = BOTH_ENV
        for profile, pod, alias, prefix, token in [('bf16', 'bf16pod', 'qwen-bf16', 'RUNPOD_BF16', 'bf16-token'),
                                                   ('4bit', 'int4pod', 'qwen-4bit', 'RUNPOD_4BIT', 'int4-token')]:
            with self.subTest(profile=profile):
                result = self.deploy_against_fake_runpod({**both, 'DEPLOY_MODEL_PROFILE': profile}, profile)
                self.assertEqual(result['paths'], {'/pods/' + pod})
                checker = result['checker']
                self.assertEqual(checker[checker.index('--profile') + 1], profile)
                self.assertEqual(checker[checker.index('--expected-model') + 1], alias)
                self.assertEqual(result['checker_env'][prefix + '_CHAT_URL'], f'https://{pod}-8001.proxy.runpod.net')
                self.assertEqual(result['checker_env'][prefix + '_CHAT_TOKEN'], token)

    @patch.object(deploy, 'api')
    def test_equal_bf16_and_4bit_pod_ids_are_refused_before_any_pod_api(self, api):
        same = {'RUNPOD_API_KEY': 'test-api', 'DEPLOY_IMAGE': IMAGE,
                'RUNPOD_BF16_POD_ID': 'samepod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-token',
                'RUNPOD_4BIT_POD_ID': 'samepod', 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-token'}
        for profile in ('bf16', '4bit'):
            with self.subTest(profile=profile), \
                    patch.dict(os.environ, {**same, 'DEPLOY_MODEL_PROFILE': profile}, clear=True), \
                    self.assertRaisesRegex(ValueError, 'different Pods'):
                deploy.main()
        api.assert_not_called()

    @patch.object(deploy, 'deploy_image')
    def test_pod_whose_profile_setting_differs_from_the_chosen_profile_is_not_updated(self, apply):
        both = BOTH_ENV
        for chosen, configured in [('4bit', 'bf16'), ('bf16', '4bit'), ('4bit', None)]:
            pod_env = {} if configured is None else {'MODEL_PROFILE': configured}
            with self.subTest(chosen=chosen, configured=configured), \
                    patch.dict(os.environ, {**both, 'DEPLOY_MODEL_PROFILE': chosen}, clear=True), \
                    patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING', 'env': pod_env}), \
                    self.assertRaisesRegex(RuntimeError, 'MODEL_PROFILE=' + chosen):
                deploy.main()
        apply.assert_not_called()

    def test_any_concurrency_from_two_to_eight_is_applied_and_verified_by_the_live_check(self):
        for profile, environment in (('bf16', ENV), ('4bit', INT4_ENV)):
            for sequences in ('2', '4', '5', '8'):
                with self.subTest(profile=profile, sequences=sequences):
                    patched, checker = self.deploy_with_live_check({**environment, 'DEPLOY_MAX_NUM_SEQS': sequences}, profile)
                    self.assertEqual(patched['env'], {'MODEL_PROFILE': profile, 'MAX_NUM_SEQS': sequences})
                    self.assertEqual(checker[checker.index('--expected-concurrency') + 1], sequences)

    def test_image_only_deploy_keeps_concurrency_and_skips_its_live_check(self):
        patched, checker = self.deploy_with_live_check(ENV)
        self.assertNotIn('env', patched)
        self.assertNotIn('--expected-concurrency', checker)

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '4'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING', 'env': {
        'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '8', 'CHAT_MAX_CONCURRENT': '8',
        'CHAT_MAX_WAITING': '24', 'CHAT_RATE_LIMIT_PER_MINUTE': '80'}})
    def test_new_concurrency_drops_old_gateway_slot_overrides_but_keeps_rate_limit(self, api, apply, summary):
        deploy.main()
        self.assertEqual(apply.call_args.kwargs['env'], {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '4',
                                                         'CHAT_RATE_LIMIT_PER_MINUTE': '80'})

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}})
    def test_concurrency_failure_restores_image_and_full_environment(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        old = {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}
        self.assertEqual(apply.call_args_list[0].kwargs['env'], {**old, 'MAX_NUM_SEQS': '2'})
        self.assertEqual(apply.call_args_list[1].kwargs['env'], old)
        self.assertEqual([c.args[1] for c in apply.call_args_list], [IMAGE, PREVIOUS])
        self.assertNotIn('test-secret', str(summary.call_args_list))

    @patch.object(deploy, 'api')
    def test_concurrency_outside_two_to_eight_is_rejected_before_api(self, api):
        for sequences in ('1', '9', '0', '08', ' 4', 'x'):
            with self.subTest(sequences=sequences), \
                    patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': sequences}, clear=True), \
                    self.assertRaises(ValueError):
                deploy.main()
        api.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING'})
    def test_missing_environment_is_rejected_without_mutation(self, api, apply):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_not_called()

    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api')
    def test_image_and_environment_are_sent_in_one_update(self, api, transition, run):
        environment = {'MAX_NUM_SEQS': '2', 'CHAT_ENABLED': 'true'}
        api.side_effect = [{}, {'image': IMAGE, 'env': environment}]
        deploy.deploy_image('/pods/testpod', IMAGE, 'https://test.invalid', env=environment)
        self.assertEqual(api.call_args_list[0].args, ('PATCH', '/pods/testpod', {'image': IMAGE, 'env': environment}))
        transition.assert_called_once()
        run.assert_called_once()

    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api')
    def test_runpod_omitting_empty_settings_still_matches_the_requested_environment(self, api, transition, run):
        # Web staging clears CHAT_WEB_SEARCH_VERIFIED_MODEL to ''; Runpod v2 reports env without it.
        environment = {'MAX_NUM_SEQS': '2', 'CHAT_WEB_SEARCH_VERIFIED_MODEL': ''}
        api.side_effect = [{}, {'image': IMAGE, 'env': {'MAX_NUM_SEQS': '2'}}]
        deploy.deploy_image('/pods/testpod', IMAGE, 'https://test.invalid', env=environment)
        run.assert_called_once()

    @patch.object(deploy.time, 'sleep')
    @patch.object(deploy.time, 'monotonic', side_effect=[0, 1, 500])
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api')
    def test_environment_mismatch_names_the_differing_settings_without_values(self, api, transition, clock, sleep):
        environment = {'MAX_NUM_SEQS': '2', 'TAVILY_API_KEY': 'fixture-secret'}
        api.side_effect = [{}, {'image': IMAGE, 'env': {'MAX_NUM_SEQS': '3'}}]
        with self.assertRaises(RuntimeError) as raised:
            deploy.deploy_image('/pods/testpod', IMAGE, 'https://test.invalid', env=environment)
        message = str(raised.exception)
        self.assertIn('MAX_NUM_SEQS', message)
        self.assertIn('TAVILY_API_KEY', message)
        self.assertNotIn('fixture-secret', message)

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_failed_deploy_restores_previous_image(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual([call.args[1] for call in apply.call_args_list], [IMAGE, PREVIOUS])
        self.assertIn('Rollback passed', summary.call_args.args[0])

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image', side_effect=RuntimeError('failed'))
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_failed_rollback_is_reported(self, api, apply, summary):
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertTrue(any('ROLLBACK FAILED' in c.args[0] for c in summary.call_args_list))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_IMAGE': 'zihan000/qwen-inference:latest'}, clear=True)
    @patch.object(deploy, 'api')
    def test_mutable_requested_tag_is_rejected_before_api_call(self, api):
        with self.assertRaises(ValueError):
            deploy.main()
        api.assert_not_called()

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': PREVIOUS, 'status': 'EXITED'})
    def test_stopped_pod_is_not_updated(self, api, apply):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_not_called()


    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api')
    def test_runpod_managed_ssh_key_is_left_out_of_the_comparison(self, api, transition, run):
        # Runpod rewrites PUBLIC_KEY itself, so a requested value need not come back.
        api.side_effect = [{}, {'image': IMAGE, 'env': {'MAX_NUM_SEQS': '2', 'PUBLIC_KEY': 'ssh-ed25519 other'}}]
        deploy.deploy_image('/pods/testpod', IMAGE, 'https://test.invalid',
                            env={'MAX_NUM_SEQS': '2', 'PUBLIC_KEY': 'ssh-ed25519 sent'})
        run.assert_called_once()


POD_A = {'image': PREVIOUS, 'status': 'RUNNING', 'env': {
    'MODEL_PROFILE': '4bit', 'MAX_NUM_SEQS': '6', 'TAVILY_API_KEY': 'fixture-provider-key',
    'JUPYTER_PASSWORD': 'source-secret', 'PUBLIC_KEY': 'ssh-ed25519 source'}}


@patch.dict(os.environ, {'RUNPOD_API_KEY': 'test-api'}, clear=True)
@patch.object(deploy, 'summary')
class PodCommandTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(deploy, 'urlopen', side_effect=AssertionError('Unmocked network request'))
        network.start()
        self.addCleanup(network.stop)

    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api')
    def test_set_env_adds_settings_keeps_the_rest_and_checks_the_profile_concurrency(self, api, apply, summary):
        api.return_value = POD_A
        deploy.cli(['set-env', '--profile', '4bit', 'podb', 'METRICS_ENABLED=true', 'METRICS_PORT=8889'])
        args, kwargs = apply.call_args
        self.assertEqual(args[:3], ('/pods/podb', PREVIOUS, 'https://podb-8001.proxy.runpod.net'))
        self.assertEqual(kwargs['env'], {'MODEL_PROFILE': '4bit', 'MAX_NUM_SEQS': '6',
                                         'TAVILY_API_KEY': 'fixture-provider-key',
                                         'JUPYTER_PASSWORD': 'source-secret',
                                         'METRICS_ENABLED': 'true', 'METRICS_PORT': '8889'})
        self.assertEqual((kwargs['profile'], kwargs['concurrency']), ('4bit', 6))
        self.assertNotIn('fixture-provider-key', str(summary.call_args_list))

    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value=POD_A)
    def test_set_env_refuses_bad_assignments_runpod_keys_and_the_wrong_profile(self, api, apply, summary):
        for argv in (['--profile', '4bit', 'podb', 'lower=x'], ['--profile', '4bit', 'podb', 'NOEQUALS'],
                     ['--profile', '4bit', 'podb', 'PUBLIC_KEY=x'], ['--profile', 'bf16', 'podb', 'A=1']):
            with self.subTest(argv=argv), self.assertRaises((ValueError, RuntimeError)):
                deploy.cli(['set-env', *argv])
        apply.assert_not_called()

    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api')
    def test_set_env_failure_restores_the_previous_settings(self, api, apply, summary):
        api.return_value = POD_A
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.cli(['set-env', '--profile', '4bit', 'podb', 'METRICS_ENABLED=true'])
        restored = apply.call_args_list[1]
        self.assertEqual(restored.args[1], PREVIOUS)
        self.assertNotIn('PUBLIC_KEY', restored.kwargs['env'])
        self.assertEqual(restored.kwargs['env']['TAVILY_API_KEY'], 'fixture-provider-key')

    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api')
    def test_copy_env_copies_image_and_settings_but_keeps_the_targets_own(self, api, apply, summary):
        target = {'image': IMAGE, 'status': 'RUNNING',
                  'env': {'JUPYTER_PASSWORD': 'target-secret', 'PUBLIC_KEY': 'ssh-ed25519 target'}}
        api.side_effect = lambda method, path, body=None: POD_A if path == '/pods/poda' else target
        deploy.cli(['copy-env', '--profile', '4bit', 'poda', 'podb'])
        args, kwargs = apply.call_args
        self.assertEqual(args[:3], ('/pods/podb', PREVIOUS, 'https://podb-8001.proxy.runpod.net'))
        self.assertEqual(kwargs['env'], {'MODEL_PROFILE': '4bit', 'MAX_NUM_SEQS': '6',
                                         'TAVILY_API_KEY': 'fixture-provider-key',
                                         'JUPYTER_PASSWORD': 'target-secret'})
        self.assertNotIn('fixture-provider-key', str(summary.call_args_list))
        self.assertNotIn('target-secret', str(summary.call_args_list))

    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value=POD_A)
    def test_copy_env_refuses_one_pod_and_a_source_of_another_profile(self, api, apply, summary):
        for argv in (['--profile', '4bit', 'poda', 'poda'], ['--profile', 'bf16', 'poda', 'podb']):
            with self.subTest(argv=argv), self.assertRaises((ValueError, RuntimeError)):
                deploy.cli(['copy-env', *argv])
        apply.assert_not_called()

    @patch.object(deploy.time, 'sleep')
    @patch.object(deploy, 'api')
    def test_stop_requests_the_stop_action_and_waits_for_exited(self, api, sleep, summary):
        api.side_effect = [{'status': 'RUNNING'}, {}, {'status': 'RUNNING'}, {'status': 'EXITED'}]
        deploy.cli(['stop', 'poda'])
        self.assertEqual(api.call_args_list[1].args, ('POST', '/pods/poda/action', {'action': 'stop'}))
        self.assertEqual(api.call_count, 4)

    @patch.object(deploy, 'api', return_value={'status': 'EXITED'})
    def test_stop_leaves_a_stopped_pod_alone(self, api, summary):
        deploy.cli(['stop', 'poda'])
        api.assert_called_once_with('GET', '/pods/poda')

    @patch.object(deploy, 'api')
    def test_invalid_pod_ids_fail_before_any_pod_api(self, api, summary):
        for argv in (['stop', '../x'], ['set-env', '--profile', '4bit', 'a/b', 'A=1'],
                     ['copy-env', '--profile', '4bit', 'ok', 'bad id']):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                deploy.cli(argv)
        api.assert_not_called()


class LegacyEntryTests(unittest.TestCase):
    @patch.object(deploy, 'main')
    def test_no_arguments_still_runs_the_workflow_deploy(self, main):
        deploy.cli([])
        main.assert_called_once_with()


class DeploymentTargetTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(deploy, 'urlopen', side_effect=AssertionError('Unmocked network request'))
        network.start()
        self.addCleanup(network.stop)

    def test_bf16_target_never_falls_back_to_generic_connection(self):
        legacy = {'RUNPOD_POD_ID': 'retiredpod', 'RUNPOD_CHAT_TOKEN': 'retired-token'}
        for values in (legacy, {**legacy, 'RUNPOD_BF16_POD_ID': 'bf16pod'},
                       {**legacy, 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}):
            with self.assertRaises(ValueError):
                deploy.deployment_target(values)
        values = {**legacy, 'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}
        self.assertEqual(deploy.deployment_target(values), ('bf16pod', 'bf16-only'))
        self.assertEqual(deploy.deployment_target(values, 'bf16'), ('bf16pod', 'bf16-only'))
        for profile in ('unknown', ''):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                deploy.deployment_target(values, profile)

    def test_4bit_target_never_borrows_the_bf16_or_generic_connection(self):
        others = {'RUNPOD_POD_ID': 'retiredpod', 'RUNPOD_CHAT_TOKEN': 'retired-token',
                  'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}
        for values in (others, {**others, 'RUNPOD_4BIT_POD_ID': 'int4pod'},
                       {**others, 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-only'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                deploy.deployment_target(values, '4bit')
        values = {**others, 'RUNPOD_4BIT_POD_ID': 'int4pod', 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-only'}
        self.assertEqual(deploy.deployment_target(values, '4bit'), ('int4pod', 'int4-only'))

    @patch.object(deploy, 'api')
    @patch.object(deploy, 'deploy_image')
    def test_unknown_profiles_fail_before_any_pod_api_or_deployment(self, apply, api):
        configured = {**ENV, 'RUNPOD_4BIT_POD_ID': 'int4pod', 'RUNPOD_4BIT_CHAT_TOKEN': 'int4-token'}
        for profile in ('4BIT', 'awq', 'unknown', ''):
            with self.subTest(profile=profile), patch.dict(os.environ, {**configured, 'DEPLOY_MODEL_PROFILE': profile}, clear=True):
                with self.assertRaises(ValueError):
                    deploy.main()
        api.assert_not_called(); apply.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': 'bf16',
                            'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only',
                            'DEPLOY_FIRST_OFFICIAL_RELEASE': 'true'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': IMAGE, 'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_first_bf16_release_can_retry_the_same_configured_digest(self, api, apply, summary):
        deploy.main()
        apply.assert_called_once()
        self.assertEqual(apply.call_args.args[0], '/pods/bf16pod')
        self.assertIn('First official BF16 release passed', str(summary.call_args))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': 'bf16',
                            'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only',
                            'DEPLOY_FIRST_OFFICIAL_RELEASE': 'true'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image', side_effect=RuntimeError('failure'))
    @patch.object(deploy, 'api', return_value={'image': 'runpod/pytorch:old', 'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_first_bf16_failure_does_not_claim_an_official_rollback(self, api, apply, summary):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_called_once()
        self.assertIn('no previous official BF16 release exists', str(summary.call_args))

    @patch.dict(os.environ, {'DEPLOY_MODEL_PROFILE': 'bf16'}, clear=True)
    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api', side_effect=[{}, {'image': IMAGE}])
    def test_bf16_verification_uses_its_alias_and_long_startup_budget(self, api, transition, run):
        deploy.deploy_image('/pods/bf16pod', IMAGE, 'https://bf16.invalid')
        args = run.call_args.args[0]
        self.assertEqual(args[args.index('--expected-model') + 1], 'qwen-bf16')
        self.assertGreaterEqual(int(args[args.index('--wait') + 1]), 3600)
        self.assertGreater(int(args[args.index('--read-timeout') + 1]), 74)


    @patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': 'bf16',
                            'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'image': 'zihan000/qwen-inference@sha256:' + 'b' * 64,
                                            'status': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_later_bf16_failure_uses_the_same_official_image_rollback(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual(apply.call_count, 2)
        self.assertEqual(apply.call_args_list[1].args[1], 'zihan000/qwen-inference@sha256:' + 'b' * 64)
        self.assertIn('Rollback passed', str(summary.call_args_list))


if __name__ == '__main__':
    unittest.main()
