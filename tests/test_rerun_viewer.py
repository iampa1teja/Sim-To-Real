"""Native viewer compatibility and failures, without launching a GUI."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rerun_viewer", ROOT / "docker/rerun_viewer.py")
viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer)


def test_viewer_cache_matches_exact_sdk_binary_and_repairs_wrong_copy(tmp_path, monkeypatch):
    payload = b"SDK 0.26.2 viewer binary fixture"
    info = {"version": "0.26.2", "binary": "/sdk/rerun_cli/rerun",
            "sha256": hashlib.sha256(payload).hexdigest()}
    copies = []

    def run(command, **kwargs):
        if command[:2] == ["docker", "cp"]:
            copies.append(command[2])
            Path(command[3]).write_bytes(payload)
            return SimpleNamespace(returncode=0)
        assert command[-1] == "--version"
        return SimpleNamespace(returncode=0, stdout="rerun-cli 0.26.2\n")

    monkeypatch.setattr(viewer.subprocess, "run", run)
    binary = viewer.matching_binary("teleop", info, tmp_path)
    assert binary.read_bytes() == payload and os.access(binary, os.X_OK)
    assert viewer.matching_binary("teleop", info, tmp_path) == binary
    assert len(copies) == 1
    binary.write_bytes(b"a mismatched host viewer")
    assert viewer.matching_binary("teleop", info, tmp_path).read_bytes() == payload
    assert len(copies) == 2


def test_wrong_version_is_never_installed(tmp_path, monkeypatch):
    payload = b"wrong executable"
    info = {"version": "0.26.2", "binary": "/sdk/rerun", "sha256": hashlib.sha256(payload).hexdigest()}

    def run(command, **kwargs):
        if command[:2] == ["docker", "cp"]:
            Path(command[3]).write_bytes(payload)
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0, stdout="rerun-cli 0.33.0\n")

    monkeypatch.setattr(viewer.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="version does not match"):
        viewer.matching_binary("teleop", info, tmp_path)
    assert not list(tmp_path.rglob("rerun"))


def test_busy_port_and_early_viewer_exit_fail_before_evaluation(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with pytest.raises(RuntimeError, match="already in use"):
            viewer.ensure_port_available(port)
    binary = tmp_path / "rerun"
    binary.write_text(f"#!{sys.executable}\nraise SystemExit(7)\n")
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match="exit 7"):
        viewer.run_viewer(binary, port, os.environ.copy(), startup_timeout=1)


def test_native_viewer_isolates_versions_and_preserves_graphics_overrides(tmp_path):
    original = {"WGPU_BACKEND": "gl", "VK_ICD_FILENAMES": "/custom/driver.json"}
    environment = viewer.native_viewer_environment(original, "0.26.2", tmp_path)
    assert environment["WGPU_BACKEND"] == "gl"
    assert environment["VK_ICD_FILENAMES"] == "/custom/driver.json"
    assert "VK_DRIVER_FILES" not in environment
    assert environment["XDG_DATA_HOME"] != viewer.native_viewer_environment({}, "0.33.0", tmp_path)["XDG_DATA_HOME"]
    assert viewer.native_viewer_environment({}, "0.26.2", tmp_path)["WGPU_BACKEND"] == "vulkan"
    assert original == {"WGPU_BACKEND": "gl", "VK_ICD_FILENAMES": "/custom/driver.json"}


def test_virtual_display_scale_uses_one_only_for_missing_vnc_dimensions(tmp_path, monkeypatch):
    output = ["VNC-0 connected 1536x841+0+0 0mm x 0mm\n"]
    monkeypatch.setattr(viewer.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=output[0]))
    environment = {"DISPLAY": ":1"}
    assert viewer.native_viewer_environment(environment, "0.26.2", tmp_path)["WINIT_X11_SCALE_FACTOR"] == "1"
    environment["WINIT_X11_SCALE_FACTOR"] = "1.5"
    assert viewer.native_viewer_environment(environment, "0.26.2", tmp_path)["WINIT_X11_SCALE_FACTOR"] == "1.5"
    output[0] = "HDMI-0 connected 1920x1080+0+0 520mm x 290mm\n"
    assert "WINIT_X11_SCALE_FACTOR" not in viewer.native_viewer_environment({"DISPLAY": ":0"}, "0.26.2", tmp_path)


@pytest.mark.parametrize("endpoint", [None, "rerun+http://127.0.0.1:19876/proxy"])
def test_sim_visualization_connects_to_requested_viewer_without_spawning(monkeypatch, endpoint):
    # Exercise the production initializer without importing Isaac or hardware.
    tree = ast.parse((ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LeRobotSO101Interface")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_device")
    calls = []
    monkeypatch.setitem(sys.modules, "rerun", SimpleNamespace(
        init=lambda name: calls.append(("init", name)),
        connect_grpc=lambda url: calls.append(("connect", url))))
    if endpoint:
        monkeypatch.setenv("RERUN_CONNECT_URL", endpoint)
    else:
        monkeypatch.delenv("RERUN_CONNECT_URL", raising=False)
    context = {"os": os, "uuid": uuid, "make_robot_from_config": lambda cfg: "robot",
               "init_rerun": lambda **kw: calls.append(("spawn", kw["session_name"]))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT), "exec"), context)
    iface = SimpleNamespace(make_cfg=lambda: "config", port=None, id="simulation")
    context["init_device"](iface, visualize=True)
    assert iface.robot == "robot"
    if endpoint:
        assert [call[0] for call in calls] == ["init", "connect"]
        assert calls[1][1] == endpoint
    else:
        assert [call[0] for call in calls] == ["spawn"]
