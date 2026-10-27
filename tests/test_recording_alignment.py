"""Exercise the real recorder control tick without connecting hardware or starting Isaac."""
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
    MARKER, mark_recording_root, recording_contract, synced_start_pose, validate_recording_root, wrist_target,
)
from sim_to_real_so101.utils.calibration_dataset import units_to_degrees
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS

ORDER = [name + ".pos" for name in FOLLOWER_JOINTS]
SETUP = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())


@pytest.mark.parametrize("resolved", [False, True])
def test_recorder_frame_zero_sync_and_paired_actions_with_real_control_tick(resolved):
    leader = dict(zip(ORDER, [15., 25., 35., 45., 22., 9.]))
    follower = dict(zip(ORDER, [10., 20., 30., 40., 12., 8.]))
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

    def tensor(values):
        return torch.tensor([values[n] for n in ORDER], dtype=torch.float32)

    def mapped(values):
        return torch.deg2rad(torch.tensor(units_to_degrees(values.numpy(), SETUP["sim_joint_mapping"]),
                                         dtype=torch.float32))

    def sim_frame(action, state, images):
        timeline.append("frame")
        return {"action": action.tolist(), "state": state.tolist()}

    iface = SimpleNamespace(
        robot=SimpleNamespace(get_action=lambda: leader), get_raw_actions_tensor=tensor,
        get_mapped_actions_vectorized=mapped,
        real_to_sim_obs_processor=lambda action: (None, mapped(tensor(action))),
        sim_to_real_dataset_processor=lambda state, visual: (state, visual, {}, {}),
        joint_mins=torch.tensor([-180.] * 6), joint_maxs=torch.tensor([180.] * 6),
    )
    follower_iface = SimpleNamespace(
        robot=SimpleNamespace(get_observation=lambda: (timeline.append("measure") or follower)),
        make_real_dataset_frame=lambda action, obs, task: {"action": [action[n] for n in ORDER]},
    )
    recorder = SimpleNamespace(make_sim_frame=sim_frame, make_auxiliary_frame=lambda *args: {})
    group = SimpleNamespace(recorders={"sim": recorder},
                            add_frames=lambda frames, auxiliary: captured.append(frames))
    namespace = dict(torch=torch, leader_iface=iface, follower_iface=follower_iface, env=SimpleNamespace(unwrapped=raw),
                     roll_resolved=resolved, sync_episode_start=True, obs=None, leader_wrist_start=None,
                     wrist_roll_index=4, recording=True, recorder_group=group, cube_trajectory=[], base_body_id=0,
                     args_cli=SimpleNamespace(task_name="task", rerun=False),
                     synced_start_pose=synced_start_pose, wrist_target=wrist_target,
                     _cube_pose_in_base=lambda *args: torch.zeros(7))
    source = (ROOT / "source/sim_to_real_so101/scripts/pick_place_agent.py").read_text()
    start = source.index("            # ── Control tick")
    end = source.index("            gui.set_state(", start)
    exec(compile(textwrap.dedent(source[start:end]), "recorder_control_tick", "exec"), namespace)
    expected = mapped(tensor(follower))
    if not resolved:
        expected[4] = initial[0, 4]
    torch.testing.assert_close(robot.data.joint_pos[0], expected)
    assert timeline == ["measure", "write", "target", "render", "render", "sensor", "sensor", "compute", "frame"]
    assert captured[0]["sim"]["action"] == captured[0]["real"]["action"] == list(leader.values())
    assert leader == original
    target = namespace["mapped_action"]
    if resolved:
        torch.testing.assert_close(target, mapped(tensor(leader)))
    else:
        assert target[4] == pytest.approx(initial[0, 4].item())
        assert target[4] != pytest.approx(mapped(tensor(leader))[4].item())


def test_wrist_fallback_preserves_leader_delta_only_when_unresolved():
    assert wrist_target(25., -2., 20., roll_resolved=False) == 3.
    assert wrist_target(25., -2., 20., roll_resolved=True) == 25.


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
    changed = dict(contract, roll_resolved=not contract["roll_resolved"])
    with pytest.raises(ValueError, match="NEW dataset root"):
        validate_recording_root(root, changed)
    assert json.loads((root / MARKER).read_text()) == contract


def test_recording_contract_changes_when_mapping_or_fit_status_changes():
    base = recording_contract(follower_id="f", leader_id="l", setup=SETUP)
    setup = json.loads(json.dumps(SETUP))
    setup["sim_joint_mapping_source"]["wrist_roll_fit_status"] = "verified"
    verified = recording_contract(follower_id="f", leader_id="l", setup=setup)
    assert verified["roll_resolved"] and not verified["wrist_rezero"]
    assert verified["calibration_sha256"] != base["calibration_sha256"]


def test_invalid_measured_start_refuses_before_pose_write():
    with pytest.raises(ValueError, match="finite"):
        synced_start_pose(np.array([0., 0., 0., 0., np.nan, 0.]), np.zeros(6), roll_resolved=False)
