"""Offline tests for the deployment safety decisions."""
import importlib.util
from io import BytesIO
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
POD_ENV = {'MODEL_PROFILE': 'bf16', 'SERVED_MODEL_NAME': 'qwen-bf16',
           'VLLM_BASE_MODEL_NAME': 'qwen-bf16-base', 'MAX_NUM_SEQS': '2',
           'MAX_MODEL_LEN': '40960', 'CPU_OFFLOAD_GB': '0', 'JUPYTER_PASSWORD': 'fixture-existing-secret'}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        # Any test without an explicit HTTP stub must remain strictly offline.
        network = patch.object(deploy, 'urlopen', side_effect=AssertionError('Unmocked network request'))
        network.start()
        self.addCleanup(network.stop)

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
    @patch.object(deploy, 'api', return_value={'imageName': IMAGE, 'desiredStatus': 'RUNNING', 'env': POD_ENV})
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

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': '4bit', 'DEPLOY_WEB_SEARCH_ACTION': 'stage'}, clear=True)
    @patch.object(deploy, 'api')
    def test_web_actions_cannot_target_the_4bit_pod(self, api):
        with self.assertRaises(ValueError):
            deploy.main()
        api.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'urlopen', return_value=BytesIO(b'{"admission":{"maxConcurrentRequests":2,"maxWaitingRequests":6}}'))
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1'}})
    def test_success_requires_advertised_gateway_capacity(self, api, apply, summary, urlopen):
        deploy.main()
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(apply.call_args.kwargs['env'], {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '2'})
        self.assertEqual(urlopen.call_args.args[0].get_header('User-agent'), 'qwen-chat-deployment-check/1.0')
        self.assertIn('2 active requests', str(summary.call_args_list))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'urlopen', return_value=BytesIO(b'{"admission":{"maxConcurrentRequests":1,"maxWaitingRequests":6}}'))
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1'}})
    def test_wrong_live_capacity_triggers_full_rollback(self, api, apply, summary, urlopen):
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual(apply.call_count, 2)
        self.assertEqual(apply.call_args_list[1].args[1], PREVIOUS)
        self.assertEqual(apply.call_args_list[1].kwargs['env'], {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1'})

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}})
    def test_concurrency_failure_restores_image_and_full_environment(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        old = {'MODEL_PROFILE': 'bf16', 'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}
        self.assertEqual(apply.call_args_list[0].kwargs['env'], {**old, 'MAX_NUM_SEQS': '2'})
        self.assertEqual(apply.call_args_list[1].kwargs['env'], old)
        self.assertEqual([c.args[1] for c in apply.call_args_list], [IMAGE, PREVIOUS])
        self.assertNotIn('test-secret', str(summary.call_args_list))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '3'}, clear=True)
    @patch.object(deploy, 'api')
    def test_invalid_concurrency_is_rejected_before_api(self, api):
        with self.assertRaises(ValueError):
            deploy.main()
        api.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING'})
    def test_missing_environment_is_rejected_without_mutation(self, api, apply):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_not_called()

    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api')
    def test_image_and_environment_are_sent_in_one_update(self, api, transition, run):
        environment = {'MAX_NUM_SEQS': '2', 'CHAT_ENABLED': 'true'}
        api.side_effect = [{}, {'imageName': IMAGE, 'env': environment}]
        deploy.deploy_image('/pods/testpod', IMAGE, 'https://test.invalid', env=environment)
        self.assertEqual(api.call_args_list[0].args, ('PATCH', '/pods/testpod', {'imageName': IMAGE, 'env': environment}))
        transition.assert_called_once()
        run.assert_called_once()

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_failed_deploy_restores_previous_image(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual([call.args[1] for call in apply.call_args_list], [IMAGE, PREVIOUS])
        self.assertIn('Rollback passed', summary.call_args.args[0])

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image', side_effect=RuntimeError('failed'))
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
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
    @patch.object(deploy, 'api', return_value={'imageName': PREVIOUS, 'desiredStatus': 'EXITED'})
    def test_stopped_pod_is_not_updated(self, api, apply):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_not_called()


    def test_bf16_target_never_falls_back_to_generic_connection(self):
        legacy = {'RUNPOD_POD_ID': 'retiredpod', 'RUNPOD_CHAT_TOKEN': 'retired-token'}
        for values in (legacy, {**legacy, 'RUNPOD_BF16_POD_ID': 'bf16pod'},
                       {**legacy, 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}):
            with self.assertRaises(ValueError):
                deploy.deployment_target(values)
        values = {**legacy, 'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only'}
        self.assertEqual(deploy.deployment_target(values), ('bf16pod', 'bf16-only'))
        self.assertEqual(deploy.deployment_target(values, 'bf16'), ('bf16pod', 'bf16-only'))
        for profile in ('4bit', 'unknown', ''):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                deploy.deployment_target(values, profile)

    @patch.object(deploy, 'api')
    @patch.object(deploy, 'deploy_image')
    def test_removed_profiles_fail_before_any_pod_api_or_deployment(self, apply, api):
        for profile in ('4bit', 'unknown', ''):
            with self.subTest(profile=profile), patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': profile}, clear=True):
                with self.assertRaises(ValueError):
                    deploy.main()
        api.assert_not_called(); apply.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MODEL_PROFILE': 'bf16',
                            'RUNPOD_BF16_POD_ID': 'bf16pod', 'RUNPOD_BF16_CHAT_TOKEN': 'bf16-only',
                            'DEPLOY_FIRST_OFFICIAL_RELEASE': 'true'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': IMAGE, 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
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
    @patch.object(deploy, 'api', return_value={'imageName': 'runpod/pytorch:old', 'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_first_bf16_failure_does_not_claim_an_official_rollback(self, api, apply, summary):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_called_once()
        self.assertIn('no previous official BF16 release exists', str(summary.call_args))

    @patch.dict(os.environ, {'DEPLOY_MODEL_PROFILE': 'bf16'}, clear=True)
    @patch.object(deploy.subprocess, 'run')
    @patch.object(deploy, 'check_transition')
    @patch.object(deploy, 'api', side_effect=[{}, {'imageName': IMAGE}])
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
    @patch.object(deploy, 'api', return_value={'imageName': 'zihan000/qwen-inference@sha256:' + 'b' * 64,
                                            'desiredStatus': 'RUNNING', 'env': {'MODEL_PROFILE': 'bf16'}})
    def test_later_bf16_failure_uses_the_same_official_image_rollback(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual(apply.call_count, 2)
        self.assertEqual(apply.call_args_list[1].args[1], 'zihan000/qwen-inference@sha256:' + 'b' * 64)
        self.assertIn('Rollback passed', str(summary.call_args_list))


if __name__ == '__main__':
    unittest.main()
