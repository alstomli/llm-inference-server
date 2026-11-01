"""Offline tests for the deployment safety decisions."""
import importlib.util
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
