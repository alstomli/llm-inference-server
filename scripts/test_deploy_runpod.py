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
ENV = {'RUNPOD_POD_ID': 'testpod', 'DEPLOY_IMAGE': IMAGE,
       'RUNPOD_API_KEY': 'test-api', 'RUNPOD_CHAT_TOKEN': 'test-chat'}


class DeploymentTests(unittest.TestCase):
    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'urlopen', return_value=BytesIO(b'{"admission":{"maxConcurrentRequests":2,"maxWaitingRequests":6}}'))
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING', 'env': {'MAX_NUM_SEQS': '1'}})
    def test_success_requires_advertised_gateway_capacity(self, api, apply, summary, urlopen):
        deploy.main()
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(apply.call_args.kwargs['env'], {'MAX_NUM_SEQS': '2'})
        self.assertIn('2 active requests', str(summary.call_args_list))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'urlopen', return_value=BytesIO(b'{"admission":{"maxConcurrentRequests":1,"maxWaitingRequests":6}}'))
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING', 'env': {'MAX_NUM_SEQS': '1'}})
    def test_wrong_live_capacity_triggers_full_rollback(self, api, apply, summary, urlopen):
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual(apply.call_count, 2)
        self.assertEqual(apply.call_args_list[1].args[1], 'old:1')
        self.assertEqual(apply.call_args_list[1].kwargs['env'], {'MAX_NUM_SEQS': '1'})

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING', 'env': {'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}})
    def test_concurrency_failure_restores_image_and_full_environment(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        old = {'MAX_NUM_SEQS': '1', 'CHAT_ENABLED': 'true', 'PRIVATE_VALUE': 'test-secret'}
        self.assertEqual(apply.call_args_list[0].kwargs['env'], {**old, 'MAX_NUM_SEQS': '2'})
        self.assertEqual(apply.call_args_list[1].kwargs['env'], old)
        self.assertEqual([c.args[1] for c in apply.call_args_list], [IMAGE, 'old:1'])
        self.assertNotIn('test-secret', str(summary.call_args_list))

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '3'}, clear=True)
    @patch.object(deploy, 'api')
    def test_invalid_concurrency_is_rejected_before_api(self, api):
        with self.assertRaises(ValueError):
            deploy.main()
        api.assert_not_called()

    @patch.dict(os.environ, {**ENV, 'DEPLOY_MAX_NUM_SEQS': '2'}, clear=True)
    @patch.object(deploy, 'deploy_image')
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING'})
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
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING'})
    def test_failed_deploy_restores_previous_image(self, api, apply, summary):
        apply.side_effect = [RuntimeError('failed'), None]
        with self.assertRaises(RuntimeError):
            deploy.main()
        self.assertEqual([call.args[1] for call in apply.call_args_list], [IMAGE, 'old:1'])
        self.assertIn('Rollback passed', summary.call_args.args[0])

    @patch.dict(os.environ, ENV, clear=True)
    @patch.object(deploy, 'summary')
    @patch.object(deploy, 'deploy_image', side_effect=RuntimeError('failed'))
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'RUNNING'})
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
    @patch.object(deploy, 'api', return_value={'imageName': 'old:1', 'desiredStatus': 'EXITED'})
    def test_stopped_pod_is_not_updated(self, api, apply):
        with self.assertRaises(RuntimeError):
            deploy.main()
        apply.assert_not_called()


if __name__ == '__main__':
    unittest.main()
