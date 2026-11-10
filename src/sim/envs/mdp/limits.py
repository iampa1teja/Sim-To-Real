"""Simulator limits events."""
import torch
from ._common import _arm_spec, env_indices


def apply_runtime_joint_limits(env, env_ids):
    """Apply per-joint arm limits: revolute degrees become radians; prismatic limits stay metres."""
    arm = _arm_spec(env)
    robot = env.scene["robot"]
    missing = [joint.usd for joint in arm.joints if joint.usd not in robot.joint_names]
    if missing:
        raise ValueError(f"arm.toml joint names {missing} not in the USD articulation {robot.joint_names}")
    joint_ids = [robot.joint_names.index(joint.usd) for joint in arm.joints]
    values = torch.tensor([joint.limits for joint in arm.joints], device=env.device, dtype=torch.float32)
    revolute = torch.tensor([joint.type == "revolute" for joint in arm.joints], device=env.device)
    values[revolute] = torch.deg2rad(values[revolute])
    env_ids = env_indices(env, env_ids)
    limits = values.unsqueeze(0).repeat(len(env_ids), 1, 1)
    robot.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=joint_ids, env_ids=env_ids)
    print(f"[INFO]: Runtime joint limits applied from arm {arm.name!r}: {arm.runtime_joint_limits}")
