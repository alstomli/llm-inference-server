import importlib.util
from pathlib import Path
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location('relay', Path(__file__).with_name('metrics-relay.py'))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


class RelayTests(unittest.TestCase):
    def test_authorization_and_route_boundaries(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), relay.handler_for('x' * 48))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = 'http://127.0.0.1:' + str(server.server_port)
        try:
            for headers in ({}, {'Authorization': 'Bearer wrong'}):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + '/metrics', headers=headers), timeout=3)
                self.assertEqual(error.exception.code, 401)
            headers = {'Authorization': 'Bearer ' + 'x' * 48}
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/v1/chat/completions', headers=headers), timeout=3)
            self.assertEqual(error.exception.code, 404)
            with patch.object(relay, 'collect_metrics', return_value=b'vllm:num_requests_running 0\n'):
                with urlopen(Request(base + '/metrics', headers=headers), timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'vllm:num_requests_running', response.read())
            with patch.object(relay, 'collect_metrics', side_effect=OSError('offline')):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + '/metrics', headers=headers), timeout=3)
                self.assertEqual(error.exception.code, 503)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

    def test_gpu_units_and_unsupported_values(self):
        class Result:
            stdout = '0, 12, 1024, 32768, 35, 55.5\n1, N/A, 512, 32768, N/A, N/A\n'
        with patch.object(relay.subprocess, 'run', return_value=Result()):
            text = relay.gpu_metrics()
        self.assertIn('gpu_memory_used_bytes{gpu="0"} 1073741824.0', text)
        self.assertIn('gpu_utilization_percent{gpu="0"} 12.0', text)
        self.assertNotIn('gpu_utilization_percent{gpu="1"}', text)


if __name__ == '__main__':
    unittest.main()
