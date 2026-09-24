"""Adapt LeRobot's standard keyboard events to the teleop episode loop."""

from queue import Empty, SimpleQueue


class LeRobotKeyboardControl:
    HELP = "[controls] S: spawn/respawn cube  R: reset  Right: start/save  Left: discard/re-record  Esc: quit"

    def __init__(self):
        from lerobot.utils.control_utils import init_keyboard_listener

        self.recording = False
        self.spawn_listener = None
        self._spawn_requests = SimpleQueue()
        self._custom_keys_down = set()
        self.listener, self.events = init_keyboard_listener()
        if self.listener is None:
            raise RuntimeError("LeRobot keyboard listener is unavailable without a display.")
        # pynput can start a thread successfully but fail to connect its X11
        # RECORD stream. Detect that instead of advertising dead controls.
        self.listener.join(timeout=0.1)
        if not self.listener.is_alive():
            self.cleanup()
            raise RuntimeError("LeRobot keyboard event thread exited; use terminal controls.")
        # Keep LeRobot's standard arrow/Escape listener. Add only the custom
        # spawn/reset keys, using the same backend and without overlapping keys.
        from pynput import keyboard
        try:
            self.spawn_listener = keyboard.Listener(
                on_press=self._on_spawn_press, on_release=self._on_spawn_release
            )
            self.spawn_listener.start()
            self.spawn_listener.join(timeout=0.1)
            if not self.spawn_listener.is_alive():
                raise RuntimeError("Spawn-key listener could not start.")
        except Exception:
            self.cleanup()
            raise
        print(self.HELP)

    def _on_spawn_press(self, key):
        char = (getattr(key, "char", None) or "").lower()
        if char in ("s", "r") and char not in self._custom_keys_down:
            self._custom_keys_down.add(char)
            if char == "r":
                self._spawn_requests.put("reset")
            elif not self.recording:
                self._spawn_requests.put("spawn")

    def _on_spawn_release(self, key):
        self._custom_keys_down.discard((getattr(key, "char", None) or "").lower())

    def set_recording(self, recording):
        # This means an active episode, including its preparation countdown;
        # it does not require a dataset recorder.
        self.recording = recording

    def consume_requests(self):
        if not self.listener.is_alive() or not self.spawn_listener.is_alive():
            raise RuntimeError("Keyboard listener stopped. Restart with --control_input terminal.")
        requests = dict.fromkeys(("start", "end", "rerecord", "stop", "reset", "spawn"), False)
        try:
            command = self._spawn_requests.get_nowait()
            requests[command] = command == "reset" or not self.recording
        except Empty:
            pass
        stop = self.events["stop_recording"]
        discard = self.events["rerecord_episode"]
        advance = self.events["exit_early"]
        for key in self.events:
            self.events[key] = False
        # LeRobot sets exit_early together with both discard and stop.
        if stop:
            requests["stop"] = True
        elif discard:
            requests["rerecord"] = self.recording
        elif advance:
            requests["end" if self.recording else "start"] = True
        return requests

    def cleanup(self):
        for listener in (self.spawn_listener, self.listener):
            if listener is None:
                continue
            try:
                if listener.is_alive():
                    listener.stop()
                listener.join(timeout=1.0)
            except Exception as exc:
                # A closed X connection must never skip robot disconnection.
                print(f"[controls] Keyboard listener cleanup: {exc}")
