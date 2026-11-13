"""Small local HTTP surface for the real-arm console; request threads never touch hardware."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from urllib.parse import parse_qs, urlsplit

from core.paths import repo_dir


class TestingHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session, host='127.0.0.1', port=8766):
        self.session = session
        self.web_dir = repo_dir('gui') / 'real'
        super().__init__((host, port), self._handler())

    def _handler(self):
        session, web = self.session, self.web_dir
        static = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/styles.css': ('styles.css', 'text/css; charset=utf-8'),
                  '/favicon.svg': ('favicon.svg', 'image/svg+xml')}

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(5.)

            def reply(self, status, body=b'', content_type='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
                self.end_headers()
                if body:
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

            def json(self, status, value):
                self.reply(status, json.dumps(value).encode())

            def valid_host(self):
                try:
                    host = urlsplit('http://' + self.headers.get('Host', '')).hostname
                except ValueError:
                    return False
                if host == 'localhost':
                    return True
                try:
                    return ipaddress.ip_address(host).is_loopback or self.server.server_address[0] in ('0.0.0.0', '::') or host == self.server.server_address[0]
                except ValueError:
                    return False

            def do_GET(self):
                if not self.valid_host():
                    return self.json(403, {'error': 'Use the local console address.'})
                path = urlsplit(self.path)
                if path.path in static:
                    filename, kind = static[path.path]
                    return self.reply(200, (web / filename).read_bytes(), kind)
                if path.path == '/api/config':
                    return self.json(200, session.catalog())
                if path.path == '/api/state':
                    return self.json(200, session.snapshot())
                if path.path == '/api/camera':
                    name = parse_qs(path.query).get('name', [''])[0]
                    data = session.frame(name)
                    return self.reply(200, data, 'image/jpeg') if data else self.reply(204)
                if path.path.startswith('/api/report/'):
                    body = session.report(path.path.removeprefix('/api/report/'))
                    return self.reply(200, body) if body else self.json(404, {'error': 'Run report not found.'})
                return self.json(404, {'error': 'Not found.'})

            def do_POST(self):
                if not self.valid_host() or self.headers.get('Origin', 'http://' + self.headers.get('Host', '')) != 'http://' + self.headers.get('Host', ''):
                    return self.json(403, {'error': 'Commands must come from this console.'})
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self.json(415, {'error': 'Expected application/json.'})
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 16384:
                        raise ValueError('Request is too large or empty.')
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise ValueError('Expected a JSON object.')
                except (ValueError, UnicodeDecodeError) as error:
                    return self.json(400, {'error': str(error)})
                path = urlsplit(self.path).path
                try:
                    if path == '/api/connect':
                        session.request_connect(client_id=body.get('client_id'))
                    elif path == '/api/run':
                        session.request_run(body)
                    elif path == '/api/stop':
                        session.request_stop()
                    elif path == '/api/disconnect':
                        session.request_stop(disconnect=True)
                    elif path == '/api/heartbeat':
                        session.heartbeat(body.get('client_id'))
                    elif path == '/api/spawn' and hasattr(session, 'request_spawn'):
                        session.request_spawn()
                    elif path == '/api/end-trial' and hasattr(session, 'request_end_trial'):
                        session.request_end_trial(body)
                    else:
                        return self.json(404, {'error': 'Not found.'})
                except (ValueError, TypeError) as error:
                    return self.json(409, {'error': str(error)})
                return self.json(202, {'ok': True})

            def log_message(self, *args):
                pass

        return Handler
