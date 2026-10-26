# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Browser page for paired sim/real evaluation: camera panes, trial controls and the gap summary.

Clicks are queued as requests; the control loop pops one per tick, so the simulator and the arm
are only driven from its thread. Stop is a flag the real rollout polls.
"""

import base64
import io
import json
import math
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .sim_real_eval import STAGE_ORDER

# source/gui/, found through the editable install (pip install -e) the container uses.
WEB_DIR = Path(__file__).resolve().parents[2] / "gui"
STATIC_FILES = {
    "/": ("sim_real.html", "text/html; charset=utf-8"),
    "/sim_real.html": ("sim_real.html", "text/html; charset=utf-8"),
    "/sim_real.jsx": ("sim_real.jsx", "text/javascript; charset=utf-8"),
}
CAMERAS = ("sim_room", "sim_wrist", "real_room", "real_wrist")
COMMANDS = ("spawn", "placed", "run_sim", "run_real", "stop", "label", "save", "discard", "exit")
MAX_NOTE = 500
MAX_BODY = 4096


def parse_command(body) -> dict:
    """Validate a POST /command body and return a normalized dict; raises ValueError."""
    if not isinstance(body, dict) or not isinstance(body.get("cmd"), str) or body["cmd"] not in COMMANDS:
        raise ValueError(f"Expected JSON object with \"cmd\" in {COMMANDS}")
    cmd = body["cmd"]
    if cmd == "spawn":
        if "start_id" in body:
            start_id = body["start_id"]
            if not isinstance(start_id, str) or not 0 < len(start_id) <= 100:
                raise ValueError("start_id must be a non-empty string of at most 100 characters")
            return {"cmd": cmd, "start_id": start_id}
        pose = {}
        for key, limit in (("x", 2.0), ("y", 2.0), ("yaw", 7.0)):
            value = body.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or abs(value) > limit):
                raise ValueError(f"spawn needs start_id or a finite {key} with |{key}| <= {limit:g}")
            pose[key] = float(value)
        return {"cmd": cmd, **pose}
    if cmd == "label":
        stage = body.get("stage")
        if not isinstance(stage, str) or stage not in STAGE_ORDER:
            raise ValueError(f"label stage must be one of {STAGE_ORDER}")
        note = body.get("note", "")
        if not isinstance(note, str):
            raise ValueError("note must be a string")
        note = note.strip()
        if len(note) > MAX_NOTE:
            raise ValueError(f"note must be at most {MAX_NOTE} characters")
        return {"cmd": cmd, "stage": stage, "note": note}
    return {"cmd": cmd}


class SimRealWebGui:
    def __init__(
        self,
        info: dict,
        host: str = "127.0.0.1",
        port: int = 8766,
        max_fps: float = 10.0,
        pane_height: int = 360,
        jpeg_quality: int = 80,
    ):
        self._info = info
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._pane_height = pane_height
        self._jpeg_quality = jpeg_quality

        self._cond = threading.Condition()
        self._closed = False
        self._session: dict | None = None
        self._trials: list[dict] = []
        self._summary: dict | None = None
        self._status = ""
        self._busy = False
        self._stop = False
        self._frames: dict[str, str | None] = dict.fromkeys(CAMERAS)
        self._requests: deque = deque(maxlen=1)
        self._payload = ""
        self._seq = 0
        self._last_publish = 0.0
        self.flush()  # so a browser that connects before the first frame gets valid state

        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self._server.server_port}/"

    # ── Control-loop API ─────────────────────────────────────────────────────

    def pop_request(self) -> dict | None:
        with self._cond:
            request = self._requests.popleft() if self._requests else None
        self.flush()  # clears "pending" in the browser
        return request

    def stop_requested(self) -> bool:
        with self._cond:
            return self._stop

    def clear_stop(self) -> None:
        with self._cond:
            self._stop = False
        self.flush()

    def set_session(self, snapshot: dict | None, trials: list[dict], summary: dict | None) -> None:
        with self._cond:
            self._session = snapshot
            self._trials = trials
            self._summary = summary
        self.flush()

    def set_status(self, text: str, busy: bool = False) -> None:
        with self._cond:
            self._status = text
            self._busy = busy
        self.flush()

    def update_images(self, frames: dict[str, np.ndarray | torch.Tensor | None]) -> None:
        """Publish HxWx3 uint8 RGB frames, at most ``max_fps`` times a second."""
        now = time.monotonic()
        if now - self._last_publish < self._min_interval:
            return
        self._last_publish = now
        encoded = {name: self._encode(image) for name, image in frames.items() if name in CAMERAS}
        with self._cond:
            self._frames.update(encoded)
        self.flush()

    def flush(self) -> None:
        """Push the current state (and last frames) to connected browsers now."""
        with self._cond:
            payload = {
                "info": self._info,
                "session": self._session,
                "trials": self._trials,
                "summary": self._summary,
                "status": self._status,
                "busy": self._busy,
                "stop_requested": self._stop,
                "frames": self._frames,
                "closed": self._closed,
                "pending": bool(self._requests),
            }
            self._payload = json.dumps(payload)
            self._seq += 1
            self._cond.notify_all()

    def destroy(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._busy = True
            self._status = "Session closed."
            self._requests.clear()
        self.flush()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)

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
            if self._closed and self._seq == last_seq:
                raise ConnectionAbortedError
            if self._seq == last_seq:
                return last_seq, None
            return self._seq, self._payload

    def _command(self, parsed: dict) -> bool:
        with self._cond:
            if self._closed:
                return False
            if parsed["cmd"] == "stop":
                self._stop = True
            elif self._busy or self._requests:
                return False
            else:
                self._requests.append(parsed)
        self.flush()
        return True

    def _make_handler(self):
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(10.0)

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
                    if not 0 < length <= MAX_BODY:
                        raise ValueError("bad request size")
                    parsed = parse_command(json.loads(self.rfile.read(length)))
                except (ValueError, AttributeError, TypeError) as error:
                    self.send_error(400, str(error))
                    return
                if not gui._command(parsed):
                    self.send_error(409, "Busy: another request is pending or a run is active")
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
