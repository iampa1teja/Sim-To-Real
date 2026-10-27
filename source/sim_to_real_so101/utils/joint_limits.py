"""JSON-sourced control and PhysX limits; never author limits into the USD asset."""
import json
import math

from .joint_calibration import FOLLOWER_JOINTS, SETUP_PATH

USD_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")


def runtime_joint_limits(setup=None):
    if setup is None:
        setup = json.loads(SETUP_PATH.read_text())
    bounds = setup["runtime_joint_limits"]
    if set(bounds) != set(FOLLOWER_JOINTS):
        raise ValueError("runtime_joint_limits must specify all six follower joints")
    for name, pair in bounds.items():
        values = [pair.get("joint_min"), pair.get("joint_max")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Runtime limits for {name} must be finite")
        if values[0] >= values[1]:
            raise ValueError(f"Runtime limits for {name} must increase")
    return {name: dict(bounds[name]) for name in FOLLOWER_JOINTS}


def apply_runtime_joint_limits(env, env_ids=None):
    """Isaac Lab 2.x API, confirmed in the installed articulation implementation."""
    import torch

    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in USD_JOINTS]
    bounds = runtime_joint_limits()
    values = torch.tensor([[bounds[n]["joint_min"], bounds[n]["joint_max"]]
                           for n in FOLLOWER_JOINTS], device=env.device, dtype=torch.float32)
    count = env.num_envs if env_ids is None else len(env_ids)
    limits = torch.deg2rad(values).unsqueeze(0).repeat(count, 1, 1)
    robot.write_joint_position_limit_to_sim(limits, joint_ids=joint_ids, env_ids=env_ids)
    print(f"[INFO]: Runtime joint limits applied from real_setup.json: {bounds}")
