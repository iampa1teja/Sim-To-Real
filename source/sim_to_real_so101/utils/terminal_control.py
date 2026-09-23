"""Read Python input() commands without blocking the simulation/control loop."""

from queue import Empty, Queue
from threading import Event, Thread


class TerminalInputControl:
    HELP = (
        '[controls] Type a command and press Enter: '
        's = start/save, save = save episode, c = discard, '
        'r = save/reset, q = quit, help = show commands.'
    )
    ALIASES = {
        's': 'toggle', 'start': 'start', 'save': 'end', 'end': 'end',
        'right': 'end', 'c': 'rerecord', 'discard': 'rerecord',
        'left': 'rerecord', 'rerecord': 'rerecord',
        'r': 'reset', 'reset': 'reset', 'q': 'stop', 'quit': 'stop',
        'exit': 'stop', 'escape': 'stop',
    }

    def __init__(self):
        self.recording = False
        self._commands = Queue()
        self._closed = Event()
        print(self.HELP)
        self._thread = Thread(target=self._read_input, daemon=True)
        self._thread.start()

    def _read_input(self):
        while not self._closed.is_set():
            try:
                command = input('control> ').strip().lower()
            except (EOFError, OSError):
                if not self._closed.is_set():
                    print('[controls] Terminal input closed; requesting clean shutdown.')
                    self._commands.put('stop')
                return
            if self._closed.is_set():
                return
            if not command:
                continue
            if command in ('help', '?'):
                print(self.HELP)
                continue
            action = self.ALIASES.get(command)
            if action is None:
                print(f'[controls] Unknown command: {command!r}. Type help.')
                continue
            self._commands.put(action)
            if action == 'stop':
                return

    def consume_requests(self):
        requests = dict.fromkeys(('start', 'end', 'rerecord', 'stop', 'reset'), False)
        # Resolve one command per tick using the master loop's current state;
        # queued "start, save" must not collapse into a single transition.
        try:
            command = self._commands.get_nowait()
        except Empty:
            return requests
        if command == 'toggle':
            command = 'end' if self.recording else 'start'
        if command in ('end', 'rerecord') and not self.recording:
            print('[controls] No active recording.')
            return requests
        requests[command] = True
        if command == 'reset' and self.recording:
            requests['end'] = True
        return requests

    def set_recording(self, recording):
        self.recording = recording

    def cleanup(self):
        # input() may be waiting for Enter; a daemon thread must not delay exit.
        self._closed.set()
