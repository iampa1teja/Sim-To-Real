# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser recording GUI: live camera feeds plus Start / Stop / Discard / Spawn.

Serves the React page in ``source/gui/``, streams camera frames and recording
state over one Server-Sent Events connection, and accepts button clicks as
``POST /command``. Clicks only set request flags; the control loop consumes
them once per tick, so recorder transitions stay on its thread.
"""

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# source/gui/, found through the editable install (pip install -e) the container uses.
WEB_DIR = Path(__file__).resolve().parents[2] / "gui"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.jsx": ("app.jsx", "text/javascript; charset=utf-8"),
}
COMMANDS = ("start", "stop", "discard", "spawn")


class RecordingWebGui:
    def __init__(
        self,
        camera_names: list[str],
        host: str = "127.0.0.1",
        port: int = 8765,
        max_fps: float = 15.0,
        pane_height: int = 240,
        jpeg_quality: int = 80,
    ):
        self._camera_names = list(camera_names)
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._pane_height = pane_height
        self._jpeg_quality = jpeg_quality

        self._cond = threading.Condition()
        self._closed = False
        self._requests = dict.fromkeys(COMMANDS, False)
        self._state = {"active": False, "can_start": False, "status": ""}
        self._frames: dict[str, str | None] = dict.fromkeys(self._camera_names)
        self._payload = ""
        self._seq = 0
        self._last_publish = 0.0
        self.flush()  # so a browser that connects before the first frame gets valid state

        self._server =ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"

    # ── Control-loop API ─────────────────────────────────────────────────────

    def consume_requests(self) -> dict[str, bool]:
        with self._cond:
            requests = self._requests
            self._requests = dict.fromkeys(COMMANDS, False)
        return requests

    def set_state(self, active: bool, can_start: bool, status: str) -> None:
        """``active``: an episode is running (countdown or recording).
        ``can_start``: Start is allowed (e.g. a cube has been spawned)."""
        with self._cond:
            self._state = {"active": active, "can_start": can_start, "status": status}

    def update_images(self, frames: dict[str, np.ndarray | torch.Tensor | None]) -> None:
        """Publish HxWx3 uint8 RGB frames, at most ``max_fps`` times a second."""
        now = time.monotonic()
        if now - self._last_publish < self._min_interval:
            return
        self._last_publish = now
        encoded = {name: self._encode(image) for name, image in frames.items() if name in self._frames}
        with self._cond:
            self._frames.update(encoded)
        self.flush()

    def flush(self) -> None:
        """Push the current state (and last frames) to connected browsers now."""
        with self._cond:
            self._payload = json.dumps({**self._state, "cameras": self._camera_names, "frames": self._frames})
            self._seq += 1
            self._cond.notify_all()

    def destroy(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()

    # ── Internals ────────────────────────────────────────────────────────────

    def _encode(self, image: np.ndarray | torch.Tensor | None) -> str | None:
        if image is None:
            return None
        # Downsample before any GPU->CPU copy.
        step = max(1, image.shape[0] // self._pane_height)
        image = image[::step, ::step]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(image[..., :3], dtype=np.uint8)).save(
            buffer, format="JPEG", quality=self._jpeg_quality
        )
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _next_payload(self, last_seq: int) -> tuple[int, str | None]:
        """Block until there is a newer payload; (seq, None) means send a keepalive."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._seq != last_seq, timeout=5.0)
            if self._closed:
                raise ConnectionAbortedError
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._payload

    def _command(self, name: str) -> bool:
        if name not in COMMANDS:
            return False
        with self._cond:
            self._requests[name] = True
        return True

    def _make_handler(self):
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    self._stream_events()
                elif self.path in STATIC_FILES:
                    filename, content_type = STATIC_FILES[self.path]
                    body = (WEB_DIR / filename).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/command":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length < 256:
                        raise ValueError("bad request size")
                    command = json.loads(self.rfile.read(length)).get("cmd")
                except (ValueError, AttributeError):
                    self.send_error(400, "Expected JSON body {\"cmd\": ...}")
                    return
                if not gui._command(command):
                    self.send_error(400, f"Unknown command; expected one of {COMMANDS}")
                    return
                self.send_response(204)
                self.end_headers()

            def _stream_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        last_seq, payload = gui._next_payload(last_seq)
                        chunk = f"data: {payload}\n\n" if payload is not None else ": keepalive\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError, OSError):
                    return

            def log_message(self, *args):
                pass

        return Handler
