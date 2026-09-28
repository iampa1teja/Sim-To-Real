"""Queue LeRobot-style keyboard commands for the teleoperation control loop."""

from queue import Empty, SimpleQueue


class LeRobotKeyboardControl:
    HELP = "[controls] S: spawn/respawn cube  R: reset  Right: start/save  Left: discard/re-record  Esc: quit"

    def __init__(self):
        from pynput import keyboard

        self.recording = False
        self._requests = SimpleQueue()
        self._keys_down = set()
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        try:
            self.listener.start()
            self.listener.join(timeout=0.1)
            if not self.listener.is_alive():
                raise RuntimeError("Keyboard event thread exited; use terminal controls.")
        except BaseException:
            self.cleanup()
            raise
        print(self.HELP)

    def _on_press(self, key):
        if key in self._keys_down:
            return
        self._keys_down.add(key)
        name = getattr(key, "name", None) or (getattr(key, "char", None) or "").lower()
        command = {"right": "advance", "left": "rerecord", "esc": "stop", "s": "spawn", "r": "reset"}.get(name)
        if command == "advance":
            command = "end" if self.recording else "start"
        if command is not None:
            self._requests.put(command)

    def _on_release(self, key):
        self._keys_down.discard(key)

    def set_recording(self, recording):
        """Set the control-loop episode state, including a preparation countdown."""
        self.recording = recording

    def consume_requests(self):
        if not self.listener.is_alive():
            raise RuntimeError("Keyboard listener stopped. Restart with --control_input terminal.")
        requests = dict.fromkeys(("start", "end", "rerecord", "stop", "reset", "spawn"), False)
        while True:
            try:
                command = self._requests.get_nowait()
            except Empty:
                break
            requests[command] = True
        return requests

    def cleanup(self):
        try:
            if self.listener.is_alive():
                self.listener.stop()
            self.listener.join(timeout=1.0)
        except Exception as exc:
            print(f"[controls] Keyboard listener cleanup: {exc}")
