#!/usr/bin/env python3
"""Stop only container processes tagged as belonging to one evaluation invocation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time


def find_owned(run_id: str):
    field = b"SO101_EVAL_RUN_ID=" + run_id.encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        descriptor = None
        try:
            if field not in (entry / "environ").read_bytes().split(b"\0"):
                continue
            pid = int(entry.name)
            # Hold the actual process identity across inspection and signaling.
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                descriptor = os.pidfd_open(pid)
            if field not in (entry / "environ").read_bytes().split(b"\0"):
                if descriptor is not None:
                    os.close(descriptor)
                continue
            yield pid, descriptor
        except (OSError, ValueError):
            if descriptor is not None:
                os.close(descriptor)


def signal_owned(run_id: str, signum: int, *, already=None) -> set[int]:
    found = set()
    for pid, descriptor in find_owned(run_id):
        found.add(pid)
        try:
            if already is None or pid not in already:
                if descriptor is not None:
                    signal.pidfd_send_signal(descriptor, signum)
                else:
                    os.kill(pid, signum)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return found


def cleanup(run_id: str, grace: float = 8, kill_wait: float = 2, quiet_period: float = .5) -> int:
    if not run_id or "\0" in run_id:
        raise ValueError("A nonempty, valid evaluation run ID is required")
    deadline = time.monotonic() + grace
    signaled = set()
    empty_since = None
    while True:
        found = signal_owned(run_id, signal.SIGTERM, already=signaled)
        signaled.update(found)
        now = time.monotonic()
        if not found:
            if empty_since is None:
                empty_since = now
            if now-empty_since >= quiet_period:
                if signaled:
                    print(f"Stopped {len(signaled)} owned evaluation process(es).", flush=True)
                return 0
        else:
            empty_since = None
        if now >= deadline and found:
            break
        time.sleep(0.1)
    deadline = time.monotonic() + kill_wait
    empty_since = None
    while True:
        found = signal_owned(run_id, signal.SIGKILL)
        signaled.update(found)
        now = time.monotonic()
        if not found:
            if empty_since is None:
                empty_since = now
            if now-empty_since >= quiet_period:
                print(f"Stopped {len(signaled)} owned evaluation process(es); forced shutdown was required.", flush=True)
                return 0
        else:
            empty_since = None
        if now >= deadline and found:
            print(f"ERROR: Owned evaluation processes remain after shutdown: {sorted(found)}", file=sys.stderr)
            return 1
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--grace", type=float, default=8)
    parser.add_argument("--kill-wait", type=float, default=2)
    parser.add_argument("--quiet-period", type=float, default=.5)
    args = parser.parse_args()
    if args.grace < 0 or args.kill_wait < 0 or args.quiet_period < 0:
        parser.error("Shutdown timeouts must be nonnegative")
    return cleanup(args.run_id, args.grace, args.kill_wait, args.quiet_period)


if __name__ == "__main__":
    sys.exit(main())
