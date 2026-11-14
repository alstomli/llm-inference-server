"""CPU-only tests for the bounded proof bridge and runtime resource guard."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace
import kv_runtime as runtime


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'reset-control.json'
        self.path.write_text(json.dumps({'expires': 1100, 'token': 'x' * 64}))
        self.path.chmod(0o600)

    def tearDown(self):
        self.temp.cleanup()

    def test_one_use(self):
        self.assertEqual(runtime.take_control(self.path, now=1000)['expires'], 1100)
        with self.assertRaises(FileNotFoundError):
            runtime.take_control(self.path, now=1000)

    @unittest.skipIf(os.name == 'nt', 'POSIX permissions and O_NOFOLLOW')
    def test_rejects_symlink_and_broad_permissions(self):
        link = self.root / 'link'
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            runtime.take_control(link, now=1000)
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):
            runtime.take_control(self.path, now=1000)

    def test_rejects_expired_and_unbounded_lifetime(self):
        for now in (1100, 499):
            with self.assertRaises(ValueError):
                runtime.take_control(self.path, now=now)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(runtime, 'ROOT', self.root)
        self.root_patch.start()
        self.middleware = runtime.ProofMiddleware(AsyncMock())
        self.middleware.engine = AsyncMock()
        self.middleware.engine.reset_prefix_cache.return_value = True
        self.middleware.server = AsyncMock()
        self.middleware.server.close = lambda: None
        self.middleware.require_idle = AsyncMock()
        (self.root / 'maintenance').touch()
        control = self.root / 'reset-control.json'
        control.write_text(json.dumps({'expires': time.time() + 300, 'token': 'x' * 64}))
        control.chmod(0o600)

    async def asyncTearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    async def call(self, token='x' * 64):
        reader = AsyncMock()
        reader.readline.return_value = json.dumps({'token': token}).encode() + b'\n'
        writer = AsyncMock()
        writer.write = lambda value: self.output.append(value)
        writer.close = lambda: None
        self.output = []
        await self.middleware.reset(reader, writer)
        return json.loads(b''.join(self.output))

    async def test_reset_preserves_cpu_and_maintenance(self):
        self.assertTrue((await self.call())['reset'])
        self.middleware.engine.reset_prefix_cache.assert_awaited_once_with(reset_running_requests=False, reset_connector=False)
        self.assertTrue((self.root / 'maintenance').exists())
        self.assertFalse((await self.call())['reset'])

    async def test_invalid_capability_never_resets(self):
        self.assertFalse((await self.call('wrong'))['reset'])
        self.middleware.engine.reset_prefix_cache.assert_not_awaited()

    async def test_busy_or_refused_engine_never_reports_success(self):
        self.middleware.require_idle.side_effect = RuntimeError('busy')
        self.assertFalse((await self.call())['reset'])
        self.middleware.engine.reset_prefix_cache.assert_not_awaited()

    async def test_native_false_is_failure(self):
        self.middleware.engine.reset_prefix_cache.return_value = False
        self.assertFalse((await self.call())['reset'])
        self.assertTrue((self.root / 'maintenance').exists())

    async def test_real_unix_socket_starts_resets_and_closes(self):
        self.middleware.app = AsyncMock()
        app = SimpleNamespace(state=SimpleNamespace(engine_client=self.middleware.engine))
        await self.middleware({'type': 'http', 'app': app, 'path': '/metrics', 'headers': []}, AsyncMock(), AsyncMock())
        socket_path = self.root / 'reset.sock'
        self.assertEqual(socket_path.stat().st_mode & 0o777, 0o600)
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(json.dumps({'token': 'x' * 64}).encode() + b'\n')
        await writer.drain()
        result = json.loads(await asyncio.wait_for(reader.readline(), 3))
        self.assertTrue(result['reset'])
        writer.close()
        await writer.wait_closed()
        self.assertFalse(socket_path.exists())
        self.assertTrue((self.root / 'maintenance').exists())

    async def test_real_socket_expiry_keeps_maintenance(self):
        self.middleware.socket_lifetime = 0.01
        app = SimpleNamespace(state=SimpleNamespace(engine_client=self.middleware.engine))
        await self.middleware({'type': 'http', 'app': app, 'path': '/metrics', 'headers': []}, AsyncMock(), AsyncMock())
        await asyncio.sleep(0.05)
        self.assertFalse((self.root / 'reset.sock').exists())
        self.assertTrue((self.root / 'maintenance').exists())

    async def test_required_metrics_missing_or_busy_fail_closed(self):
        self.middleware.engine.get_num_unfinished_requests = lambda: 0
        names = ['vllm:num_requests_running', 'vllm:num_requests_waiting',
                 'vllm:kv_offload_cpu_cache_usage_perc', 'vllm:kv_offload_cpu_cache_write_usage_perc',
                 'vllm:kv_offload_cpu_cache_read_usage_perc']
        client = AsyncMock()
        client.__aenter__.return_value = client
        with patch('httpx.AsyncClient', return_value=client):
            for values in ([0] * 5, [0] * 4 + [1], [0] * 4):
                client.get.return_value = SimpleNamespace(text='\n'.join(f'{name} {value}' for name, value in zip(names, values)), raise_for_status=lambda: None)
                if len(values) == 5 and not any(values):
                    await runtime.ProofMiddleware.require_idle(self.middleware)
                else:
                    with self.assertRaises(RuntimeError):
                        await runtime.ProofMiddleware.require_idle(self.middleware)

    async def test_ingress_remains_in_maintenance(self):
        self.middleware.started = True
        send = AsyncMock()
        await self.middleware({'type': 'http', 'path': '/v1/chat/completions', 'headers': []}, AsyncMock(), send)
        self.assertEqual(send.await_args_list[0].args[0]['status'], 503)
        self.middleware.app.assert_not_awaited()


class MemoryTests(unittest.TestCase):
    def test_proof_directory_and_secret_creation_are_strict(self):
        spec = importlib.util.spec_from_file_location('config', Path(__file__).with_name('kv-runtime-config.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'proof'
            module.prepare_proof(root)
            self.assertEqual(root.stat().st_mode & 0o777, 0o700)
            self.assertEqual((root / 'inference-token').stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError): module.prepare_proof(root)
            link = Path(directory) / 'link'
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError): module.prepare_proof(link)
            root.chmod(0o755)
            with self.assertRaises(ValueError): module.prepare_proof(root)
    def test_reserve_guards(self):
        spec = importlib.util.spec_from_file_location('config', Path(__file__).with_name('kv-runtime-config.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'memory.max').write_text(str(188_000_000_000))
            (root / 'memory.current').write_text(str(62_000_000_000))
            class Stats:
                f_bavail = 94_000_000_000
                f_frsize = 1
            with patch.object(module.os, 'statvfs', return_value=Stats()):
                module.check_memory(root)
                (root / 'memory.current').write_text(str(100_000_000_000))
                with self.assertRaises(RuntimeError): module.check_memory(root)
                (root / 'memory.current').write_text(str(62_000_000_000))
                Stats.f_bavail = 70_000_000_000
                with self.assertRaises(RuntimeError): module.check_memory(root)


if __name__ == '__main__':
    unittest.main()
