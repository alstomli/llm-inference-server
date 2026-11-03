"""Run in the image: preserve authenticated administrative access without logging tokens."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT = Path('/usr/local/lib/start-jupyter.py')
spec = importlib.util.spec_from_file_location('jupyter_startup', SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class JupyterTests(unittest.TestCase):
    def test_authentication_cannot_be_disabled_by_missing_token(self):
        with self.assertRaises(ValueError):
            startup.configuration({})

    def test_template_token_opens_workspace_and_is_not_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            token = 'image-test-only-' + 'x' * 40
            env = {**os.environ, 'JUPYTER_PASSWORD': token,
                   'JUPYTER_PORT': str(port), 'JUPYTER_SERVER_ROOT': str(root),
                   'JUPYTER_RUNTIME_DIR': str(root / 'runtime'),
                   'JUPYTER_CONFIG_DIR': str(root / 'config')}
            config = startup.configuration(env)
            self.assertEqual(config['IdentityProvider']['token'], token)
            self.assertEqual(config['ServerApp']['root_dir'], str(root))
            self.assertFalse(config['FileContentsManager']['delete_to_trash'])
            process = subprocess.Popen(['python3', str(SCRIPT)], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                url = f'http://127.0.0.1:{port}/api/status'
                deadline = time.monotonic() + 45
                while True:
                    try:
                        with urlopen(Request(url, headers={'Authorization': 'token ' + token}), timeout=2) as response:
                            self.assertIn('started', json.load(response))
                            break
                    except (HTTPError, URLError, OSError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail('Authenticated Jupyter did not become ready')
                        time.sleep(0.25)
                with self.assertRaises(HTTPError) as error:
                    urlopen(url, timeout=2)
                self.assertEqual(error.exception.code, 403)
            finally:
                process.terminate()
                output = process.communicate(timeout=15)[0]
                self.assertNotIn(token, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
