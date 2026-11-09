"""Boundary checks for the local source browser; no simulation dependencies."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

SPEC = importlib.util.spec_from_file_location('explorer_server', Path(__file__).resolve().parents[1] / 'tools/code_explorer/server.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SourceServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'hello.py').write_text('def hello():\n    return "world"\n')
        (self.root / '.env').write_text('PRIVATE_TOKEN=do-not-return')
        (self.root / 'mesh.bin').write_bytes(b'abc\0def')
        (self.root / 'outside').symlink_to('/etc/passwd')
        (self.root / 'secret_alias.py').symlink_to(self.root / '.env')
        self.state = module.ExplorerState(self.root)
        self.state.index = {'files': []}
        self.state.paths = {p: {'language': 'python'} for p in ('hello.py', '.env', 'mesh.bin', 'outside', 'secret_alias.py')}
        self.server = module.ExplorerServer(('127.0.0.1', 0), self.state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def get(self, path):
        with urlopen(self.url + path) as response:
            return json.load(response)

    def test_read_source_and_binary_metadata(self):
        self.assertIn('def hello', self.get('/api/source?path=hello.py')['content'])
        self.assertTrue(self.get('/api/source?path=mesh.bin')['binary'])

    def test_cannot_read_secrets_or_outside_repository(self):
        for path, code in [('.env', 403), ('outside', 403), ('secret_alias.py', 403), ('../etc/passwd', 400), ('unknown.py', 404)]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as ctx:
                self.get('/api/source?path=' + path)
            self.assertEqual(ctx.exception.code, code)

    def test_cross_origin_requests_rejected(self):
        request = Request(self.url + '/api/source?path=hello.py', headers={'Origin': 'https://unrelated.example'})
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request)
        self.assertEqual(ctx.exception.code, 403)

    def test_large_source_marked_truncated(self):
        (self.root / 'hello.py').write_text('x' * (module.SOURCE_LIMIT + 100))
        result = self.get('/api/source?path=hello.py')
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['content']), module.SOURCE_LIMIT)

    def test_unexpected_host_rejected(self):
        request = Request(self.url + '/api/source?path=hello.py', headers={'Host': 'unrelated.example'})
        with self.assertRaises(HTTPError) as ctx:
            urlopen(request)
        self.assertEqual(ctx.exception.code, 403)


if __name__ == '__main__':
    unittest.main()
