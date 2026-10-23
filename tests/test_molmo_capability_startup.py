"""Research-server startup failures must retain evidence, not become scores."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capability_sweep", ROOT / "scripts/molmo_capability_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def investigation(tmp_path):
    study = object.__new__(sweep.Investigation)
    study.snapshot = tmp_path
    study.server = study.server_key = None
    study.viewer = Mock()
    study.viewer.poll.return_value = None
    return study


def test_immediate_crash_reports_exit_state_and_logs_before_cleanup(monkeypatch, tmp_path):
    launch = Mock()
    monkeypatch.setattr(sweep, "run", launch)
    monkeypatch.setattr(sweep, "port_free", lambda port: True)
    inspect = SimpleNamespace(returncode=0, stdout=json.dumps(
        dict(Running=False, ExitCode=139, OOMKilled=False)), stderr="")
    logs = SimpleNamespace(returncode=0, stdout="starting runtime\n", stderr="Segmentation fault\n")
    subprocess_run = Mock(side_effect=[inspect, logs])
    monkeypatch.setattr(sweep.subprocess, "run", subprocess_run)
    study = investigation(tmp_path)
    with pytest.raises(RuntimeError, match="server exited before becoming ready") as error:
        study.start_server({"num_steps": 10})
    assert '"ExitCode": 139' in str(error.value)
    assert '"OOMKilled": false' in str(error.value)
    assert "Segmentation fault" in str(error.value)
    assert "--rm" not in launch.call_args.args[0]
    assert study.server is not None  # Retained for the caller's finally cleanup.
    assert study.server_key is None  # Never marked ready.


def test_missing_container_reports_inspection_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "run", Mock())
    monkeypatch.setattr(sweep, "port_free", lambda port: True)
    missing = SimpleNamespace(returncode=1, stdout="", stderr="No such container")
    monkeypatch.setattr(sweep.subprocess, "run", Mock(return_value=missing))
    with pytest.raises(RuntimeError, match="No such container"):
        investigation(tmp_path).start_server({})


def test_ready_server_requires_a_live_owned_container(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "run", Mock())
    monkeypatch.setattr(sweep, "port_free", Mock(side_effect=[True, False]))
    live = SimpleNamespace(returncode=0, stdout=json.dumps(dict(Running=True, ExitCode=0)), stderr="")
    monkeypatch.setattr(sweep.subprocess, "run", Mock(return_value=live))
    study = investigation(tmp_path)
    study.start_server({"num_steps": 10})
    assert json.loads(study.server_key) == {"num_steps": 10}
