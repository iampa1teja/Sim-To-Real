"""Role-aware rollout ranges and optional wrist handling without hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.rollout.real_rollout import (
    RolloutSettings, clamp_action, joint_ranges, run_episode, sim_start_roll_units,
)


def test_interleaved_parallel_gripper_uses_selected_codec(synthetic_arm):
    mapping = synthetic_arm.runtime_joint_limits
    assert joint_ranges(synthetic_arm, mapping) == ((-100, 100), (0, 100), (-100, 100), (0, 100))
    mapping["pan"].update(raw_min=110, raw_max=-70)
    assert joint_ranges(synthetic_arm, mapping)[0] == (-70, 110)
    assert sim_start_roll_units({}, arm=synthetic_arm) is None


def test_episode_without_wrist_roll_clamps_each_declared_range(synthetic_arm):
    keys = ["pan.pos", "left.pos", "elbow.pos", "right.pos"]
    sent = []
    robot = SimpleNamespace(get_observation=lambda: dict(zip(keys, (0, 50, 0, 50))),
                            send_action=sent.append)
    policy = SimpleNamespace(get_raw_action=lambda *_: dict(zip(keys, (200, -5, -200, 150))))
    settings = RolloutSettings(fps=1, episode_seconds=1, enable_motion=True, max_start_error=200,
                               raw_ranges=joint_ranges(synthetic_arm, synthetic_arm.runtime_joint_limits))
    result = run_episode(robot, policy, keys, {}, settings, clock=lambda: 0, sleep=lambda _: None)
    assert result.status == "completed"
    assert sent == [dict(zip(keys, (100, 0, -100, 100)))]


def test_bad_action_cannot_reach_robot():
    keys = ["axis.pos"]
    sent = []
    robot = SimpleNamespace(get_observation=lambda: {keys[0]: 0}, send_action=sent.append)
    policy = SimpleNamespace(get_raw_action=lambda *_: {keys[0]: np.nan})
    settings = RolloutSettings(fps=1, episode_seconds=1, enable_motion=True, raw_ranges=((-100, 100),))
    result = run_episode(robot, policy, keys, {}, settings, clock=lambda: 0, sleep=lambda _: None)
    assert result.status == "policy_error"
    assert not sent
    with pytest.raises(ValueError, match="One raw range"):
        clamp_action({keys[0]: 0}, keys, ())


def test_degree_mode_wrist_start_is_encoded_with_calibrated_zero(monkeypatch):
    import math
    from dataclasses import replace
    from core.arms.spec import load_arm_spec
    from core.arms.calibration import resolve_calibration
    from core.arms.mapping import units_to_physical
    monkeypatch.delenv("S2R_ARMS_DIR", raising=False)
    arm, _ = load_arm_spec("so101")
    arm = replace(arm, lerobot=replace(arm.lerobot, use_degrees=True))
    selection = resolve_calibration(arm, unit="rpd_robot", environ={})
    raw = sim_start_roll_units(selection.mapping, mount=SimpleNamespace(start_joint_positions={"wrist_roll": 0.0}), arm=arm)
    # SDK zero is encoder-range midpoint (2047.5), physical zero is encoder2048.
    assert raw != 0.0
    joint = arm.wrist_roll.joint
    assert units_to_physical([raw], {joint: selection.mapping[joint]}, [joint])[0] == pytest.approx(0, abs=1e-12)


def test_rollout_yaml_selectors_and_explicit_flags(tmp_path):
    from core.rollout.real_rollout import parse_args

    path = tmp_path / 'rollout.yaml'
    path.write_text('''arm: so101
scene: blank_table
task: stack_two_cubes
variant: teleop
policy: example
fps: 10
enable_motion: false
rename_map:
  external: room
''')
    args = parse_args(['--config_path', str(path), '--fps', '15'])
    assert (args.arm, args.scene, args.task, args.variant) == ('so101', 'blank_table', 'stack_two_cubes', 'teleop')
    assert args.policy == 'example' and args.fps == 15
    assert args.enable_motion is False and args.action_horizon is None
    assert args.lang_instruction == 'Stack the blue cube on top of the red cube'
    import json
    assert json.loads(args.rename_map) == {'external': 'room'}


@pytest.mark.parametrize('option', ['enable_motion', 'release_on_exit'])
def test_rollout_yaml_refuses_non_boolean_motion_options(tmp_path, option):
    from core.rollout.real_rollout import parse_args

    path = tmp_path / 'rollout.yaml'
    path.write_text(f'policy: example\n{option}: "false"\n')
    with pytest.raises(SystemExit):
        parse_args(['--config_path', str(path)])
