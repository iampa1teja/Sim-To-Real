"""Collect task-defined milestones before the environment automatically resets."""
from __future__ import annotations

import numbers

from core.tasks.base import Pose, StepState
from core.tasks.loader import load_task
from core.tasks.stages import StageTrace


def _value(value, index):
    if isinstance(value, (numbers.Number, str, type(None))):
        return value
    selected = value[index]
    if hasattr(selected, "numel") and selected.numel() == 1:
        return selected.item()
    if hasattr(selected, "size") and not callable(selected.size) and selected.size == 1:
        return selected.item()
    return selected


def reset_stage_traces(env, env_ids=None):
    from sim.envs.mdp import env_indices

    for index in env_indices(env, env_ids).tolist():
        env._stage_traces[index] = StageTrace(env._stage_task)
    env._benchmark_trace = env._stage_traces[0]


def install_stage_tracking(env, task=None):
    """Run each success term once, then inspect that exact pre-reset frame."""
    from core.arms.spec import load_arm_spec
    from sim.envs.mdp import as_torch

    task = load_task(task) if isinstance(task, (str, type(None))) else task
    arm = getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]
    robot = env.scene["robot"]
    ee_ids, _ = robot.find_bodies(arm.links.ee)
    if len(ee_ids) != 1:
        raise ValueError(f"Expected exactly one robot body named {arm.links.ee!r}")
    gripper_ids = robot.find_joints([joint.usd for joint in arm.joints if joint.role == "gripper"],
                                  preserve_order=True)[0] if arm.gripper_joint_names else []
    env._stage_task = task
    env._stage_traces = [StageTrace(task) for _ in range(env.num_envs)]
    env._benchmark_trace = env._stage_traces[0]
    previous_lengths = [-1] * env.num_envs
    previous_steps = [-1] * env.num_envs
    cfg = env.termination_manager.get_term_cfg("success")
    original = cfg.func

    def observe(raw, **params):
        result = original(raw, **params)
        signals = task.signals(raw)
        body_positions = as_torch(robot.data.body_link_pos_w)
        body_rotations = as_torch(robot.data.body_link_quat_w)
        joints = as_torch(robot.data.joint_pos)
        objects = {name: (as_torch(raw.scene[name].data.root_pos_w),
                          as_torch(raw.scene[name].data.root_quat_w)) for name in task.spec.objects}
        for index, trace in enumerate(raw._stage_traces):
            length = int(raw.episode_length_buf[index].item())
            step = int(raw.common_step_counter)
            if previous_steps[index] != step and previous_lengths[index] >= length:
                trace = raw._stage_traces[index] = StageTrace(task)
            previous_lengths[index] = length
            previous_steps[index] = step
            flags = {name: _value(value, index) for name, value in signals.items()}
            flags["success"] = bool(_value(result, index))
            state = StepState(
                objects={name: Pose.from_lists(position[index].tolist(), rotation[index].tolist())
                         for name, (position, rotation) in objects.items()},
                ee=Pose.from_lists(body_positions[index, ee_ids[0]].tolist(), body_rotations[index, ee_ids[0]].tolist()),
                gripper=float(joints[index, gripper_ids].mean().item()) if len(gripper_ids) else 0.0,
                signals=flags,
            )
            trace.observe(state)
        raw._benchmark_trace = raw._stage_traces[0]
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg("success", cfg)
    return env._stage_traces
