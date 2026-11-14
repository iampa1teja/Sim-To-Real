"""Factory recordings still validate and retain their arm alignment marker."""

import pytest

from utils.arms.calibration import resolve_calibration
from utils.datasets.alignment import RecordingAlignmentError
from sim.teleop import agent
from sim.teleop.config import AgentConfig, DatasetConfig


def test_factory_recording_prepares_alignment_even_without_device_calibration_paths(synthetic_arm, monkeypatch, tmp_path):
    selection = resolve_calibration(synthetic_arm, environ={})
    assert selection.follower_path is None and selection.leader_path is None
    monkeypatch.setattr("utils.arms.calibration.resolve_calibration", lambda **_: selection)
    ports = []
    monkeypatch.setattr(agent, "validate_serial_ports", lambda values: ports.append(values))
    root = tmp_path / "recording"
    cfg = AgentConfig(arm=synthetic_arm.name,
                      dataset=DatasetConfig(repo_id="local/factory", root=str(root), task_name="test"))

    chosen, contract, roots = agent._prepare(cfg)

    assert chosen is selection
    assert roots == [str(root)]
    assert contract["arm"] == synthetic_arm.name
    assert contract["unit"] is None
    assert ports == [{"leader": cfg.leader.port}]
    assert not root.exists()


def test_factory_alignment_mismatch_is_rejected_before_device_port_validation(synthetic_arm, monkeypatch, tmp_path):
    selection = resolve_calibration(synthetic_arm, environ={})
    monkeypatch.setattr("utils.arms.calibration.resolve_calibration", lambda **_: selection)
    root = tmp_path / "recording"
    (root / "meta").mkdir(parents=True)
    marker = root / "meta/recording_alignment.json"
    marker.write_text('{"arm":"different"}\n')
    monkeypatch.setattr(agent, "validate_serial_ports", lambda _: pytest.fail("Devices must not be validated yet"))
    cfg = AgentConfig(arm=synthetic_arm.name,
                      dataset=DatasetConfig(repo_id="local/factory", root=str(root), task_name="test"))

    with pytest.raises(RecordingAlignmentError):
        agent._prepare(cfg)
    assert marker.read_text() == '{"arm":"different"}\n'
