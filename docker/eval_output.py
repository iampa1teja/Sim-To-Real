#!/usr/bin/env python3
"""Keep complete evaluator output on disk and concise, live output in the terminal.

Only routine banners and known initialization details are omitted. Warnings,
errors and otherwise unrecognized output always remain visible.
"""
from __future__ import annotations

import argparse
import codecs
from collections import deque
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DIAGNOSTIC = re.compile(r"warn|error|exception|traceback|fatal", re.I)
TABLE_BORDER = re.compile(r"^[|+][-=+|]+[|+]$")
ROUTINE = re.compile(
    r"^(?:"
    r"=+$"
    r"|\[INFO\]\[(?:AppLauncher|IsaacLab)\]:"
    r"|\[Info\] \[carb\] Logging to file:"
    r"|\[INFO\] (?:Command|Event|Recorder|Action|Observation|Termination|Reward|Curriculum) Manager:"
    r"|\[INFO\]: (?:Parsing configuration from:|Base environment:|Scene manager:|"
    r"Time taken for |Gym observation space:|Gym action space:|Click 'R'|"
    r"Initializing device with Rerun visualization:|Initialized LeRobot device)"
    r"|(?:Environment device|Environment seed|Physics step-size|Rendering step-size|"
    r"Environment step-size|Number of environments|Environment spacing|Source prim name|"
    r"Global prim paths|Replicate physics)\s*:"
    r"|\[Pick-place eval\] Reset event order:"
    r"|(?:Audited|Resolved) \d+ packages? in "
    r"|Requirement already satisfied:"
    r")"
)


class LiveOutput:
    def __init__(self, stream=sys.stdout, *, verbose=False, clock=time.monotonic):
        self.stream, self.verbose, self.clock = stream, verbose, clock
        self.last_progress = -float("inf")
        self.episode = None
        self.progress_visible = False
        self.in_table = False

    def emit(self, record: str) -> None:
        line = ANSI.sub("", record).strip("\r\n")
        stripped = line.strip()
        border = bool(TABLE_BORDER.fullmatch(stripped))
        table_row = self.in_table and stripped.startswith("|")
        self.in_table = border or table_row
        # tqdm can share a line with a log printed during inference.
        if line.startswith("Rollout") and not DIAGNOSTIC.search(line):
            split = re.search(r"\[(?:INFO)\]", line)
            if split:
                self.emit(line[:split.start()] + "\r")
                self.emit(line[split.start():] + "\n")
                return
            now = self.clock()
            episode = line.partition(":")[0]
            terminal = self.stream.isatty()
            if not terminal and episode == self.episode and now-self.last_progress < 5 and "100%" not in line:
                return
            self.last_progress, self.episode = now, episode
            self.stream.write(("\r" if terminal else "") + line + ("" if terminal else "\n"))
            self.stream.flush()
            self.progress_visible = terminal
            return
        if not self.verbose and (not stripped or (not DIAGNOSTIC.search(line) and (border or table_row or ROUTINE.search(stripped)))):
            return
        if self.progress_visible:
            self.stream.write("\n")
            self.progress_visible = False
        self.stream.write(line + "\n")
        self.stream.flush()

    def finish(self):
        if self.progress_visible:
            self.stream.write("\n")
            self.stream.flush()


def show_tail(path: Path, stream=sys.stderr) -> None:
    """Bound failure output while retaining both the actual error and final context."""
    tail, errors = deque(maxlen=20), deque(maxlen=12)
    try:
        with path.open(errors="replace") as handle:
            for line in handle:
                for record in line.split("\r"):
                    clean = ANSI.sub("", record).strip()
                    if not clean:
                        continue
                    tail.append(clean)
                    if re.search(r"error|exception|traceback|fatal", clean, re.I):
                        errors.append(clean)
    except OSError as exc:
        print(f"Cannot read diagnostic log {path}: {exc}", file=stream)
        return
    print(f"Diagnostic log: {path}", file=stream)
    earlier_errors = [line for line in errors if line not in tail]
    if earlier_errors:
        print("Errors:", file=stream)
        for line in earlier_errors:
            print(line, file=stream)
    print("Last output:", file=stream)
    print("\n".join(tail), file=stream)


def run(command: list[str], logfile: Path, *, verbose=False) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    output = LiveOutput(verbose=verbose)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    with logfile.open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        interrupted = None
        def forward(signum, _frame):
            nonlocal interrupted
            interrupted = signum
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass
        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        while chunk := os.read(process.stdout.fileno(), 8192):
            log.write(chunk)
            log.flush()
            pending += decoder.decode(chunk)
            records = re.split(r"(?<=[\r\n])", pending)
            pending = records.pop()
            for record in records:
                output.emit(record)
        pending += decoder.decode(b"", final=True)
        if pending:
            output.emit(pending)
        status = process.wait()
    output.finish()
    if status and not interrupted:
        print(f"Command failed (exit {status}).", file=sys.stderr)
        show_tail(logfile)
    if interrupted is not None:
        return 128+interrupted
    return status if status >= 0 else 128-status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--tail", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.tail:
        show_tail(args.tail)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.log or not command:
        parser.error("--log PATH and a command after -- are required")
    try:
        return run(command, args.log, verbose=args.verbose or os.environ.get("EVAL_VERBOSE") == "1")
    except OSError as exc:
        print(f"ERROR: {exc}; diagnostic log: {args.log}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
