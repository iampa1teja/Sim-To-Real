"""Locked sim mapping, independent hardware IDs and historical recording bytes."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.joint_calibration import (
    SETUP_PATH, file_sha256, load_device_calibration, resolve_calibration,
)
from sim_to_real_so101.utils.recording_alignment import recording_contract

FIXTURE = ROOT / "tests/fixtures/rpd_robot.json"


def hardware_file(root, identifier="cli", role="follower"):
    group = "robots" if role == "follower" else "teleoperators"
    kind = "so101_follower" if role == "follower" else "so101_leader"
    path = root / group / kind / f"{identifier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(FIXTURE.read_bytes())
    return path


def test_missing_environment_ids_do_not_change_mapping_or_break_sim_import(tmp_path):
    env = dict(os.environ, ROBOT_ID="absent", TELEOP_ID="also-absent", HF_LEROBOT_CALIBRATION=str(tmp_path),
               SO101_LEADER_CALIBRATION=str(tmp_path / "missing-leader.json"))
    env.pop("SO101_FOLLOWER_CALIBRATION", None)
    selection = resolve_calibration(environ=env)
    assert selection.mapping == json.loads(SETUP_PATH.read_text())["sim_joint_mapping"]
    assert selection.follower_path is None
    result = subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = ['source', 'tests']; "
                             "import types; p = types.ModuleType('pxr'); "
                             "[setattr(p, n, types.ModuleType(n)) for n in ['Gf', 'Sdf', 'Usd', 'UsdGeom', 'UsdLux', 'UsdShade']]; "
                             "sys.modules['pxr'] = p; "
                             "import sim_to_real_so101.assets.real_setup"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_ids_select_hardware_files_without_selecting_a_sim_mapping(tmp_path):
    path = hardware_file(tmp_path)
    env = {"ROBOT_ID": "ignored", "TELEOP_ID": "missing", "HF_LEROBOT_CALIBRATION": str(tmp_path)}
    selection = resolve_calibration(environ=env, follower_id="cli")
    assert selection.follower_path == path and selection.leader_path is None
    assert not selection.provenance["explicit_selection"]
    selection.verify_loaded(SimpleNamespace(calibration_fpath=path), "follower")
    with pytest.raises(ValueError, match="Cannot read"):
        resolve_calibration(environ=env, leader_id="missing")


def test_explicit_mapping_and_hardware_id_remain_independent_with_difference_table(tmp_path):
    path = hardware_file(tmp_path)
    changed = tmp_path / "different.json"
    data = json.loads(path.read_text())
    data["elbow_flex"]["range_min"] += 20
    changed.write_text(json.dumps(data))
    selection = resolve_calibration(environ={"HF_LEROBOT_CALIBRATION": str(tmp_path),
                                             "SO101_FOLLOWER_CALIBRATION": str(changed)}, follower_id="cli")
    assert selection.follower_path == path
    with pytest.raises(ValueError) as error:
        selection.verify_loaded(SimpleNamespace(calibration_fpath=path), "follower")
    text = str(error.value)
    assert file_sha256(changed) in text and file_sha256(path) in text
    assert all(joint in text for joint in selection.mapping)
    assert "delta min/max" in text and "refusing to connect" in text


def test_direct_client_path_wins_without_changing_id_and_only_loads_memory(tmp_path):
    path = hardware_file(tmp_path)
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": "/missing"},
                                    follower_id="logical", follower_path=path)
    robot = SimpleNamespace(id="logical", config=SimpleNamespace(id="logical"), bus=SimpleNamespace())

    def load():
        robot.calibration = json.loads(robot.calibration_fpath.read_text())

    robot._load_calibration = load
    load_device_calibration(robot, path)
    selection.verify_loaded(robot, "follower")
    assert robot.id == robot.config.id == "logical"
    assert robot.bus.calibration is robot.calibration


def test_no_selection_contract_matches_fixture_from_before_0a32271(tmp_path):
    expected = json.loads((ROOT / "tests/fixtures/recording_contract_pre_0a32271.json").read_text())["contract"]
    path = hardware_file(tmp_path, "rpd_robot")
    env = {"ROBOT_ID": "absent", "TELEOP_ID": "absent", "HF_LEROBOT_CALIBRATION": str(tmp_path)}
    for selection in (resolve_calibration(environ=env), resolve_calibration(environ=env, follower_id="rpd_robot")):
        assert recording_contract(follower_id="rpd_robot", leader_id="rpd_teleop", calibration_selection=selection) == expected
    selected = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(path)})
    contract = recording_contract(follower_id="rpd_robot", leader_id="rpd_teleop", calibration_selection=selected)
    assert contract["calibration_sha256"] != expected["calibration_sha256"]
    assert contract["selected_calibrations"]["follower"] == file_sha256(path)
