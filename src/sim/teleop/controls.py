# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One command queue for GUI requests and typed terminal commands."""
from queue import Empty, Queue
import shlex
from threading import Event, Thread


class CommandQueue:
    HELP = "[controls] Type start, save, discard, reset, quit or help and press Enter."
    ALIASES = {"s": "toggle", "start": "start", "save": "end", "end": "end", "right": "end",
               "c": "rerecord", "discard": "rerecord", "left": "rerecord", "rerecord": "rerecord",
               "r": "reset", "reset": "reset", "q": "stop", "quit": "stop", "exit": "stop", "escape": "stop"}
    GUI_COMMANDS = {"exit": "stop", "discard": "rerecord", "stop": "end", "save": "end",
                    "reset": "reset", "spawn": "spawn", "start": "start"}

    def __init__(self, enable_spawn=False):
        self.recording = False
        self._commands, self._closed = Queue(), Event()
        self._thread = None
        self._aliases = dict(self.ALIASES)
        self.help = self.HELP
        if enable_spawn:
            self._aliases.update(spawn="spawn", s="spawn", right="toggle")
            self.help += " spawn <object> places the named object; right starts/saves."

    def submit(self, command, **payload):
        if command not in {"start", "end", "rerecord", "stop", "reset", "spawn", "toggle", "prepare_rerecord"}:
            raise ValueError(f"Unknown control command: {command!r}")
        if not self._closed.is_set():
            self._commands.put((command, payload))

    def submit_gui(self, requests):
        for button, command in self.GUI_COMMANDS.items():
            if requests.get(button):
                payload = {"object": requests[button]} if button == "spawn" and isinstance(requests[button], str) else {}
                self.submit(command, **payload)
        if requests.get("rerecord"):
            self.submit("prepare_rerecord", rerecord_episode=requests.get("rerecord_episode"))

    def submit_text(self, text):
        text = text.strip()
        if not text:
            return False
        if text.lower() in ("help", "?"):
            print(self.help)
            return False
        try:
            words = shlex.split(text)
        except ValueError:
            words = []
        command = self._aliases.get(words[0].lower()) if words else None
        if command == "spawn" and len(words) == 2:
            self.submit(command, object=words[1])
            return True
        if len(words) != 1:
            command = None
        if command is None:
            print(f"[controls] Unknown command: {text!r}. Type help.")
            return False
        self.submit(command)
        return True

    def start_terminal(self, *, stop_on_eof=False):
        """Read without blocking physics; EOF leaves an active GUI/eval running."""
        if self._thread is not None:
            return
        print(self.help)

        def read():
            while not self._closed.is_set():
                try:
                    text = input("control> ")
                except (EOFError, OSError):
                    if stop_on_eof:
                        self.submit("stop")
                    return
                if self._closed.is_set():
                    return
                self.submit_text(text)
                if self._aliases.get(text.strip().lower()) == "stop":
                    return

        self._thread = Thread(target=read, daemon=True)
        self._thread.start()

    def consume_requests(self):
        # One transition per tick: a queued start/save pair cannot collapse.
        try:
            command, payload = self._commands.get_nowait()
        except Empty:
            return {}
        if command == "toggle":
            command = "end" if self.recording else "start"
        requests = {command: True, **payload}
        if command == "reset" and self.recording:
            requests["end"] = True
        return requests

    def set_recording(self, recording):
        self.recording = bool(recording)

    def cleanup(self):
        # The daemon input thread may be waiting for Enter.
        self._closed.set()
