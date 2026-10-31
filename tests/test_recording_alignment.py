"""Exercise the real recorder control tick without connecting hardware or starting Isaac."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import textwrap

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.recording_alignment import (
    MARKER, mark_recording_root, recording_contract, synced_start_pose, validate_recording_root,
)
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS

ORDER = [name + ".pos" for name in FOLLOWER_JOINTS]
SETUP = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())


def run_control_tick(*, leader_roll, follower_roll):
    leader = dict(zip(ORDER, [15., 25., 35., 45., leader_roll, 9.]))
    follower = dict(zip(ORDER, [10., 20., 30., 40., follower_roll, 8.]))
    original = dict(leader)
    initial = torch.tensor([[.1, .2, .3, .4, -.03, .06]])
    timeline, captured = [], []

    def write(q, velocity):
        timeline.append("write")
        robot.data.joint_pos = q.clone()
        assert torch.count_nonzero(velocity) == 0

    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=initial.clone()), write_joint_state_to_sim=write,
                            set_joint_position_target=lambda q: timeline.append("target"))

    def compute():
        timeline.append("compute")
        return {"policy": {"joint_pos_obs": robot.data.joint_pos.clone()}, "visual": {"frame": 0}}

    sensor = SimpleNamespace(cfg=SimpleNamespace(data_types=["rgb"]),
                             update=lambda *args, **kwargs: timeline.append("sensor"))
    scene = type("Scene", (dict,), {})({"robot": robot})
    scene.sensors = {"room": sensor, "wrist": sensor}
    raw = SimpleNamespace(scene=scene, num_envs=1, sim=SimpleNamespace(render=lambda: timeline.append("render")),
                          observation_manager=SimpleNamespace(compute=compute))

    # Execute production mapping and inverse mapping without importing hardware drivers.
    path = ROOT / "source/sim_to_real_so101/utils/lerobot_interface.py"
    tree = ast.parse(path.read_text())
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LeRobotSO101Interface")
    wanted = {"get_raw_actions_tensor", "get_mapped_actions_vectorized", "get_raw_actions_from_radians",
              "real_to_sim_obs_processor"}
    methods = [n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    cls = ast.ClassDef(name="Interface", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(path), "exec"), scope)
    iface = scope["Interface"]()
    iface.device, iface.SO101_JOINT_ORDER = "cpu", ORDER
    for attribute, source_key, bound in [("joint_mins", "runtime_joint_limits", "joint_min"),
                                         ("joint_maxs", "runtime_joint_limits", "joint_max"),
                                         ("mapping_joint_mins", "sim_joint_mapping", "joint_min"),
                                         ("mapping_joint_maxs", "sim_joint_mapping", "joint_max")]:
        setattr(iface, attribute, torch.tensor([SETUP[source_key][n][bound] for n in FOLLOWER_JOINTS]))
    iface.robot = SimpleNamespace(get_action=lambda: leader)
    iface.sim_to_real_dataset_processor = lambda state, visual: (state, visual, {}, {})

    def sim_frame(action, state, images):
        timeline.append("frame")
        return {"action": action.tolist(), "state": state.tolist()}

    real_sent, sim_sent = [], []
    follower_iface = SimpleNamespace(
        robot=SimpleNamespace(get_observation=lambda: (timeline.append("measure") or follower),
                              send_action=lambda action: real_sent.append(dict(action))),
        make_real_dataset_frame=lambda action, obs, task: {"action": [action[n] for n in ORDER]},
    )
    recorder = SimpleNamespace(make_sim_frame=sim_frame, make_auxiliary_frame=lambda *args: {})
    group = SimpleNamespace(recorders={"sim": recorder},
                            add_frames=lambda frames, auxiliary: captured.append(frames))
    namespace = dict(torch=torch, leader_iface=iface, follower_iface=follower_iface, env=SimpleNamespace(
        unwrapped=raw, step=lambda action: (sim_sent.append(action.clone()) or ({}, None, None, None, None))),
        sync_episode_start=True, obs=None, actions=torch.zeros((1, 6)),
        episode_cube=SimpleNamespace(park=lambda: None), recording=True, recorder_group=group, cube_trajectory=[], base_body_id=0,
        args_cli=SimpleNamespace(task_name="task", rerun=False),
        synced_start_pose=synced_start_pose,
        _cube_pose_in_base=lambda *args: torch.zeros(7))
    source = (ROOT / "source/sim_to_real_so101/scripts/pick_place_agent.py").read_text()
    start = source.index("            # ── Control tick")
    end = source.index("            gui.set_state(", start)
    exec(compile(textwrap.dedent(source[start:end]), "recorder_control_tick", "exec"), namespace)
    expected = iface.get_mapped_actions_vectorized(iface.get_raw_actions_tensor(follower))
    torch.testing.assert_close(robot.data.joint_pos[0], expected)
    assert timeline == ["measure", "write", "target", "render", "render", "sensor", "sensor", "compute", "frame"]
    assert captured[0]["sim"]["action"] == captured[0]["real"]["action"] == list(leader.values())
    assert leader == original
    drive_start = source.index("            if follower_iface is not None:",
                               source.index("            gui.update_images", end))
    drive_end = source.index("            remaining =", drive_start)
    exec(compile(textwrap.dedent(source[drive_start:drive_end]), "recorder_drive_tick", "exec"), namespace)
    assert real_sent == [leader]
    expected_target = iface.get_mapped_actions_vectorized(iface.get_raw_actions_tensor(leader))
    torch.testing.assert_close(sim_sent[0][0], expected_target)
    executed_raw = iface.get_raw_actions_from_radians(sim_sent[0][0])
    return executed_raw, captured[0]["sim"]["action"]


@pytest.mark.parametrize("leader_roll,follower_roll", [(22., 12.), (-35., -37.)])
def test_recorder_syncs_all_six_joints_and_executes_stored_leader_action(leader_roll, follower_roll):
    executed, stored = run_control_tick(leader_roll=leader_roll, follower_roll=follower_roll)
    torch.testing.assert_close(executed, torch.tensor(stored), atol=3e-5, rtol=0)


def test_different_episode_leader_starts_cannot_create_roll_action_gap():
    gaps = []
    for leader_roll, follower_roll in [(22., 12.), (-35., -37.)]:
        executed, stored = run_control_tick(leader_roll=leader_roll, follower_roll=follower_roll)
        gaps.append(float(executed[4]) - stored[4])
    np.testing.assert_allclose(gaps, [0., 0.], atol=1e-5, rtol=0)


def test_legacy_or_changed_calibration_root_cannot_be_resumed(tmp_path):
    contract = recording_contract(follower_id="follower", leader_id="leader")
    root = tmp_path / "new"
    validate_recording_root(root, contract)
    root.mkdir()
    (root / "data").mkdir()
    with pytest.raises(ValueError, match="NEW dataset root"):
        validate_recording_root(root, contract)
    mark_recording_root(root, contract)
    validate_recording_root(root, contract)
    for version in [1, 2]:
        mark_recording_root(root, dict(contract, version=version))
        with pytest.raises(ValueError, match="legacy and v2 roots"):
            validate_recording_root(root, contract)
    mark_recording_root(root, contract)
    changed = dict(contract, roll_resolved=not contract["roll_resolved"])
    with pytest.raises(ValueError, match="NEW dataset root"):
        validate_recording_root(root, changed)
    assert json.loads((root / MARKER).read_text()) == contract


def test_recording_contract_changes_when_mapping_or_fit_status_changes():
    base = recording_contract(follower_id="f", leader_id="l", setup=SETUP)
    assert base["version"] == 3
    assert base["wrist_rezero"] is False and base["roll_resolved"] is False
    assert base["wrist_roll_zero_offset_degrees"] == 0
    assert base["roll_zero_measurement"] == "unmeasured constant"
    assert base["synced_joints"] == list(FOLLOWER_JOINTS)
    setup = json.loads(json.dumps(SETUP))
    setup["sim_joint_mapping_source"]["wrist_roll_fit_status"] = "verified"
    verified = recording_contract(follower_id="f", leader_id="l", setup=setup)
    assert verified["roll_resolved"] and not verified["wrist_rezero"]
    assert verified["calibration_sha256"] != base["calibration_sha256"]


def test_invalid_measured_start_refuses_before_pose_write():
    with pytest.raises(ValueError, match="finite"):
        synced_start_pose(np.array([0., 0., 0., 0., np.nan, 0.]))


def test_legacy_root_guard_runs_before_serial_preflight():
    source = (ROOT / "source/sim_to_real_so101/scripts/pick_place_agent.py").read_text()
    assert source.index("validate_recording_root(root, alignment_contract)") < source.index("validate_serial_ports(args_cli, parser)")
