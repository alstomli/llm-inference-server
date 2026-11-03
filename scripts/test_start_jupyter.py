"""Run in the image: preserve authenticated administrative access without logging tokens."""
import importlib.util
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor
from urllib.parse import quote
import websocket

SCRIPT = Path('/usr/local/lib/start-jupyter.py')
spec = importlib.util.spec_from_file_location('jupyter_startup', SCRIPT)
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class JupyterTests(unittest.TestCase):
    def test_authentication_cannot_be_disabled_by_missing_token(self):
        with self.assertRaises(ValueError):
            startup.configuration({})

    def test_proxy_origin_must_be_exact_and_cannot_be_a_wildcard(self):
        with tempfile.TemporaryDirectory() as root:
            env = {'JUPYTER_PASSWORD': 'test-only-token', 'JUPYTER_SERVER_ROOT': root,
                   'JUPYTER_PORT': '8888', 'RUNPOD_POD_ID': 'testpod'}
            expected = 'https://testpod-8888.proxy.runpod.net'
            self.assertEqual(startup.configuration(env)['ServerApp']['allow_origin'], expected)
            self.assertEqual(startup.configuration({**env, 'JUPYTER_ALLOW_ORIGIN': expected})['ServerApp']['allow_origin'], expected)
            for origin in ['*', 'https://wrong.example', 'https://otherpod-8888.proxy.runpod.net',
                           'http://testpod-8888.proxy.runpod.net',
                           'https://testpod-8889.proxy.runpod.net', expected + '?token=bad']:
                with self.subTest(origin=origin), self.assertRaises(ValueError):
                    startup.configuration({**env, 'JUPYTER_ALLOW_ORIGIN': origin})

    def test_template_token_opens_workspace_and_is_not_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            (root / 'volume-probe.txt').write_text('configured workspace root')
            token = 'image-test-only-' + 'x' * 40
            env = {**os.environ, 'JUPYTER_PASSWORD': token,
                   'JUPYTER_PORT': str(port), 'JUPYTER_SERVER_ROOT': str(root),
                   'JUPYTER_RUNTIME_DIR': str(root / 'runtime'),
                   'JUPYTER_CONFIG_DIR': str(root / 'config'), 'RUNPOD_POD_ID': 'testpod',
                   'JUPYTER_ALLOW_ORIGIN': ''}
            origin = f'https://testpod-{port}.proxy.runpod.net'
            config = startup.configuration(env)
            self.assertEqual(config['IdentityProvider']['token'], token)
            self.assertEqual(config['ServerApp']['root_dir'], str(root))
            self.assertFalse(config['FileContentsManager']['delete_to_trash'])
            self.assertEqual(config['ServerApp']['allow_origin'], origin)
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
                            process.terminate()
                            output = process.communicate(timeout=15)[0]
                            self.fail('Authenticated Jupyter did not become ready: ' + output.replace(token, '[redacted test token]'))
                        time.sleep(0.25)
                contents = f'http://127.0.0.1:{port}/api/contents/volume-probe.txt'
                with urlopen(Request(contents, headers={'Authorization': 'token ' + token}), timeout=2) as response:
                    self.assertEqual(json.load(response)['content'], 'configured workspace root')
                # A normal browser authenticates with cookies, not a token header.
                # Token-authenticated requests bypass Origin checks and cannot test the proxy.
                cookies = CookieJar()
                browser = build_opener(HTTPCookieProcessor(cookies))
                base = f'http://127.0.0.1:{port}'
                browser.open(base + '/login', timeout=3).close()
                browser.open(base + '/login?token=' + quote(token), timeout=3).close()
                self.assertTrue(any(cookie.name.startswith('username-') for cookie in cookies))
                with browser.open(Request(url, headers={'Origin': origin}), timeout=3) as response:
                    self.assertIn('started', json.load(response))
                with self.assertRaises(HTTPError) as wrong_origin:
                    browser.open(Request(url, headers={'Origin': 'https://wrong.example'}), timeout=3)
                # Jupyter APIHandler deliberately conceals rejected Origins with 404.
                self.assertEqual(wrong_origin.exception.code, 404)

                # Exact Origin permission must not disable the normal XSRF protection.
                terminals = base + '/api/terminals'
                with self.assertRaises(HTTPError) as missing_xsrf:
                    browser.open(Request(terminals, data=b'{}', headers={
                        'Origin': origin, 'Content-Type': 'application/json'}), timeout=3)
                self.assertEqual(missing_xsrf.exception.code, 403)
                xsrf = next(cookie.value for cookie in cookies if cookie.name == '_xsrf')
                headers = {'Origin': origin, 'X-XSRFToken': xsrf, 'Content-Type': 'application/json'}
                with browser.open(Request(terminals, data=b'{}', headers=headers), timeout=3) as response:
                    terminal = json.load(response)['name']
                cookie_header = '; '.join(cookie.name + '=' + cookie.value for cookie in cookies)
                ws_url = f'ws://127.0.0.1:{port}/terminals/websocket/{terminal}'
                with self.assertRaises(websocket.WebSocketBadStatusException) as wrong_ws:
                    websocket.create_connection(ws_url, cookie=cookie_header, origin='https://wrong.example', timeout=3)
                self.assertEqual(wrong_ws.exception.status_code, 403)
                ws = websocket.create_connection(ws_url, cookie=cookie_header, origin=origin, timeout=3)
                try:
                    ws.send(json.dumps(['stdin', "printf 'proxy-%s\\n' 'terminal-ready'\r"]))
                    output = ''
                    deadline = time.monotonic() + 10
                    while 'proxy-terminal-ready' not in output and time.monotonic() < deadline:
                        message = json.loads(ws.recv())
                        if message[0] == 'stdout':
                            output += message[1]
                    self.assertIn('proxy-terminal-ready', output)
                finally:
                    ws.close()
                    browser.open(Request(terminals + '/' + terminal, method='DELETE', headers=headers), timeout=3).close()
                with self.assertRaises(HTTPError) as error:
                    urlopen(url, timeout=2)
                self.assertEqual(error.exception.code, 403)
            finally:
                process.terminate()
                output = process.communicate(timeout=15)[0]
                self.assertNotIn(token, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
