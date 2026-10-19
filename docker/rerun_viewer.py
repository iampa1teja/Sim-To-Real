#!/usr/bin/env python3
"""Run the container SDK's matching Rerun viewer natively on the Linux host."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time


SDK_QUERY = """
import hashlib, json, os
from pathlib import Path
import rerun, rerun_cli
binary = Path(rerun_cli.__file__).with_name('rerun')
with binary.open('rb') as source:
    digest = hashlib.file_digest(source, 'sha256').hexdigest()
print(json.dumps({'version': rerun.__version__, 'binary': str(binary),
                  'sha256': digest, 'display': os.environ.get('DISPLAY', '')}))
"""


def sdk_info(container: str) -> dict:
    result = subprocess.run(
        ["docker", "exec", container, "/isaac-sim/python.sh", "-c", SDK_QUERY],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", info["version"]):
        raise RuntimeError("Container reported an invalid Rerun version")
    if not re.fullmatch(r"[a-f0-9]{64}", info["sha256"]):
        raise RuntimeError("Container reported an invalid Rerun binary digest")
    return info


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def matching_binary(container: str, info: dict, cache_root: Path | None = None) -> Path:
    """Reuse only the exact executable shipped alongside the active SDK."""
    if cache_root is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "so101" / "rerun"
    directory = cache_root / info["version"] / info["sha256"]
    binary = directory / "rerun"
    if binary.is_file() and os.access(binary, os.X_OK) and _digest(binary) == info["sha256"]:
        return binary
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=directory) as temporary:
        copied = Path(temporary) / "rerun"
        subprocess.run(["docker", "cp", f"{container}:{info['binary']}", str(copied)], check=True)
        copied.chmod(0o755)
        if _digest(copied) != info["sha256"]:
            raise RuntimeError("Copied Rerun viewer does not match the container SDK binary")
        version = subprocess.run([str(copied), "--version"], check=True, capture_output=True, text=True).stdout
        if info["version"] not in version.split():
            raise RuntimeError(f"Rerun executable version does not match SDK {info['version']}: {version.strip()}")
        os.replace(copied, binary)
    return binary


def ensure_port_available(port: int) -> None:
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise RuntimeError(f"Rerun port {port} is already in use; close its owner or choose RERUN_PORT") from error


def check_display(env: dict[str, str]) -> None:
    if not env.get("DISPLAY"):
        raise RuntimeError("Rerun needs an X11 display; set DISPLAY and XAUTHORITY on the host")
    x11 = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    display = x11.XOpenDisplay(env["DISPLAY"].encode())
    if not display:
        raise RuntimeError(f"Cannot open host X11 display {env['DISPLAY']}; check DISPLAY and XAUTHORITY")
    x11.XCloseDisplay(display)


def native_viewer_environment(env: dict[str, str], version: str, cache_root: Path | None = None) -> dict[str, str]:
    """Isolate SDK-version state and avoid probing unrelated Mesa devices."""
    env = env.copy()
    if cache_root is None:
        cache_root = Path(env.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "so101" / "rerun"
    data_home = cache_root / version / "viewer-data"
    data_home.mkdir(parents=True, exist_ok=True)
    env["XDG_DATA_HOME"] = str(data_home)
    env.setdefault("WGPU_BACKEND", "vulkan")
    if env.get("DISPLAY") and "WINIT_X11_SCALE_FACTOR" not in env:
        try:
            result = subprocess.run(["xrandr", "--current"], env=env, capture_output=True,
                                    text=True, timeout=2, check=True)
            outputs = re.findall(r"^(\S+) connected[^\n]* (\d+)mm x (\d+)mm\s*$",
                                 result.stdout, re.MULTILINE)
            # VNC has no physical monitor dimensions. Match its existing 1x scale
            # explicitly instead of asking winit to derive DPI from zero millimeters.
            if outputs and all(name.startswith(("VNC", "Virtual")) and width == height == "0"
                               for name, width, height in outputs):
                env["WINIT_X11_SCALE_FACTOR"] = "1"
        except (OSError, subprocess.SubprocessError):
            pass
    if "VK_DRIVER_FILES" not in env and "VK_ICD_FILENAMES" not in env:
        for candidate in (Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
                          Path("/etc/vulkan/icd.d/nvidia_icd.json")):
            if candidate.is_file():
                env["VK_DRIVER_FILES"] = str(candidate)
                break
    return env


def run_viewer(binary: Path, port: int, env: dict[str, str], startup_timeout: float = 30) -> int:
    command = [str(binary), "--bind", "127.0.0.1", "--port", str(port),
               "--memory-limit", "10%", "--window-size", "1280x720", "--expect-data-soon"]
    process = subprocess.Popen(command, env=env)
    previous_handlers = {}
    shutdown_deadline = None

    def forward(signum, _frame):
        nonlocal shutdown_deadline
        if process.poll() is None:
            if shutdown_deadline is None:
                shutdown_deadline = time.monotonic() + 5
            process.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward)
    try:
        deadline = time.monotonic() + startup_timeout
        while process.poll() is None:
            if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                process.kill()
                process.wait()
                raise RuntimeError("Rerun viewer required forced shutdown during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Rerun viewer did not start listening on port {port}")
                time.sleep(0.05)
        else:
            raise RuntimeError(f"Rerun viewer exited during startup (exit {process.returncode})")
        print(f"[RERUN] Native viewer ready: rerun+http://127.0.0.1:{port}/proxy", flush=True)
        while process.poll() is None:
            try:
                process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                    process.kill()
        returncode = process.wait()
        return 128 - returncode if returncode < 0 else returncode
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="teleop")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    runtime = None
    try:
        ensure_port_available(args.port)
        info = sdk_info(args.container)
        env = os.environ.copy()
        if not env.get("DISPLAY"):
            env["DISPLAY"] = info["display"]
        check_display(env)
        existing = Path(env["XDG_RUNTIME_DIR"]) if env.get("XDG_RUNTIME_DIR") else None
        if (existing is None or not existing.is_dir() or existing.stat().st_uid != os.getuid()
                or stat.S_IMODE(existing.stat().st_mode) != 0o700):
            runtime = tempfile.TemporaryDirectory(prefix="so101-rerun-runtime-")
            env["XDG_RUNTIME_DIR"] = runtime.name
        binary = matching_binary(args.container, info)
        env = native_viewer_environment(env, info["version"])
        print(f"[RERUN] Using SDK-matched native viewer {info['version']}", flush=True)
        return run_viewer(binary, args.port, env)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"[RERUN ERROR] {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if runtime is not None:
            runtime.cleanup()


if __name__ == "__main__":
    sys.exit(main())
