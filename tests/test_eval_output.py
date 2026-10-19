"""Logging keeps the original diagnostics while presenting useful live output."""
import importlib.util
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest


HELPER = Path(__file__).resolve().parents[1] / "docker/eval_output.py"
spec = importlib.util.spec_from_file_location("eval_output", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_filters_only_known_routine_output_and_preserves_diagnostics():
    stream = io.StringIO()
    output = module.LiveOutput(stream)
    for line in (
        "[INFO] Event Manager: <EventManager> contains 2 active terms.",
        "+--------+------------------+", "| Index | Name |", "| 0 | spawn_cube |",
        "\x1b[36m===================\x1b[0m", "Audited 39 packages in 5ms",
        "2026-10-19T14:28:43Z [627ms] [Warning] [new.plugin] unexpected warning",
        "| WARNING: a surprising warning formatted as a table |",
        "UserWarning: camera calibration changed", "[Error] CUDA out of memory",
        "[INFO]: Robot USD: /assets/ARM_new.usd", "New diagnostic with unknown format",
        "Success rate: 0.00% (0/1)", "Results JSON: /outputs/result.json",
    ):
        output.emit(line)
    text = stream.getvalue()
    assert "Event Manager" not in text and "spawn_cube" not in text and "Audited" not in text
    for required in ("unexpected warning", "surprising warning", "UserWarning", "CUDA out of memory",
                     "ARM_new.usd", "unknown format", "Success rate", "Results JSON"):
        assert required in text


def test_noninteractive_progress_is_throttled_without_losing_completion_or_logs():
    stream = io.StringIO()
    ticks = iter([0, 1, 2, 7, 8])
    output = module.LiveOutput(stream, clock=lambda: next(ticks))
    output.emit("Rollout (ep 1/2): 0%| | 0/900\r")
    output.emit("Rollout (ep 1/2): 1%| | 8/900\r")
    output.emit("Rollout (ep 1/2): 2%| | 16/900[INFO]: Model state validated\n")
    output.emit("Rollout (ep 1/2): 10%| | 90/900\r")
    output.emit("Rollout (ep 1/2): 100%| | 900/900\r")
    text = stream.getvalue()
    assert text.count("Rollout") == 3
    assert "100%" in text and "Model state validated" in text


def test_interactive_progress_overwrites_line_and_separates_diagnostics():
    class Terminal(io.StringIO):
        def isatty(self):
            return True
    stream = Terminal()
    output = module.LiveOutput(stream)
    output.emit("Rollout (ep 1/1): 0%| | 0/10\r")
    output.emit("WARNING: keep visible\n")
    assert stream.getvalue() == "\rRollout (ep 1/1): 0%| | 0/10\nWARNING: keep visible\n"


def test_progress_never_throttles_a_warning_appended_without_newline():
    stream = io.StringIO()
    output = module.LiveOutput(stream, clock=lambda: 0)
    output.emit("Rollout (ep 1/1): 0%| | 0/10\r")
    output.emit("Rollout (ep 1/1): 1%| | 1/10[Warning] unknown GPU warning\n")
    assert "unknown GPU warning" in stream.getvalue()


def test_command_failure_retains_exact_bytes_exit_status_and_diagnostic_context(tmp_path):
    logfile = tmp_path / "full.log"
    original = b"[INFO] Event Manager: hidden detail\n\rRollout (ep 1/1): 0%| | 0/5\r\nWarning: keep me\nERROR: inference failed\nlast context\n"
    code = f"import sys; sys.stdout.buffer.write({original!r}); sys.stdout.flush(); sys.exit(7)"
    result = subprocess.run([sys.executable, str(HELPER), "--log", str(logfile), "--", sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 7
    assert logfile.read_bytes() == original
    assert "hidden detail" not in result.stdout
    assert "Warning: keep me" in result.stdout and "ERROR: inference failed" in result.stderr
    assert "last context" in result.stderr and str(logfile) in result.stderr


def test_failure_tail_retains_earlier_error_when_last_lines_are_shutdown_noise(tmp_path):
    logfile = tmp_path / "full.log"
    logfile.write_text("ERROR: real failure\n" + "shutdown info\n"*30)
    stream = io.StringIO()
    module.show_tail(logfile, stream)
    assert "real failure" in stream.getvalue()
    assert stream.getvalue().count("shutdown info") == 20


def test_verbose_mode_keeps_routine_output(tmp_path):
    logfile = tmp_path / "full.log"
    result = subprocess.run([sys.executable, str(HELPER), "--verbose", "--log", str(logfile), "--",
                             sys.executable, "-c", "print('[INFO] Event Manager: full detail')"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "full detail" in result.stdout


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_forwarded_signal_retains_signal_exit_status_when_child_exits_zero(tmp_path, signum):
    logfile = tmp_path / "signal.log"
    code = """import signal, sys, time
def stop(_signum, _frame):
    print('child saved and exited cleanly', flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
time.sleep(60)
"""
    process = subprocess.Popen([sys.executable, str(HELPER), "--log", str(logfile), "--",
                                sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 128+signum
        assert "child saved" in stdout and "child saved" in logfile.read_text()
        assert "Command failed" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_evaluator_cleanup_only_stops_exact_run_tag(tmp_path, ignore_term):
    cleanup = HELPER.with_name("eval_cleanup.py")
    run_id = "pytest-" + uuid.uuid4().hex
    receipt = tmp_path / "graceful-shutdown.txt"
    code = """import pathlib, signal, sys, time
def stop(_signum, _frame):
    pathlib.Path(sys.argv[1]).write_text('saved before shutdown')
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[2] == 'ignore' else stop)
print('ready', flush=True)
time.sleep(60)
"""
    target = subprocess.Popen([sys.executable, "-c", code, str(receipt), "ignore" if ignore_term else "save"],
                              env=os.environ | {"SO101_EVAL_RUN_ID": run_id}, stdout=subprocess.PIPE, text=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"],
                                 env=os.environ | {"SO101_EVAL_RUN_ID": run_id + "-other"}, stdout=subprocess.PIPE,
                                 text=True)
    try:
        assert target.stdout.readline().strip() == "ready"
        assert unrelated.stdout.readline().strip() == "ready"
        result = subprocess.run([sys.executable, str(cleanup), run_id, "--grace", "0.2", "--kill-wait", "1"],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, result.stderr
        target.wait(timeout=2)
        assert unrelated.poll() is None
        if ignore_term:
            assert "forced shutdown" in result.stdout
            assert not receipt.exists()
        else:
            assert receipt.read_text() == "saved before shutdown"
    finally:
        for child in (target, unrelated):
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)


def test_evaluator_cleanup_observes_delayed_tagged_start_during_quiet_interval(monkeypatch):
    cleanup_spec = importlib.util.spec_from_file_location("eval_cleanup", HELPER.with_name("eval_cleanup.py"))
    cleanup = importlib.util.module_from_spec(cleanup_spec)
    cleanup_spec.loader.exec_module(cleanup)
    run_id = "pytest-delayed-" + uuid.uuid4().hex
    first_scan = threading.Event()
    target = []
    real_signal_owned = cleanup.signal_owned
    def observe(*args, **kwargs):
        found = real_signal_owned(*args, **kwargs)
        if not first_scan.is_set():
            assert not found
            first_scan.set()
        return found
    monkeypatch.setattr(cleanup, "signal_owned", observe)
    def start_after_first_empty_scan():
        first_scan.wait(timeout=2)
        time.sleep(.05)
        target.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                       env=os.environ | {"SO101_EVAL_RUN_ID": run_id}))
    thread = threading.Thread(target=start_after_first_empty_scan)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 env=os.environ | {"SO101_EVAL_RUN_ID": run_id + "-other"})
    thread.start()
    try:
        assert cleanup.cleanup(run_id, grace=1, kill_wait=1, quiet_period=.3) == 0
        thread.join(timeout=2)
        assert first_scan.is_set() and len(target) == 1
        target[0].wait(timeout=2)
        assert unrelated.poll() is None
    finally:
        thread.join(timeout=2)
        for child in [*target, unrelated]:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
