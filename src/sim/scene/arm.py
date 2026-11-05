from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import quat_from_euler_xyz

# The toml reading lives in core so hardware code can use it without Isaac.
from core.arms.spec import arms_dir, load_arm_spec  # noqa: F401  (re-exported)


def _quat_from_euler_degrees(angles: list[float]) -> tuple[float, ...]:
    """Use Isaac Lab's own helper so the order always matches what ArticulationCfg expects."""
    roll, pitch, yaw = (torch.tensor([math.radians(a)]) for a in angles)
    return tuple(quat_from_euler_xyz(roll, pitch, yaw)[0].tolist())


def load_arm_cfg(name: str, variant: str | None = None) -> ArticulationCfg:
    spec, arm_dir = load_arm_spec(name, variant)
    spawn = spec["spawn"]
    state = spec["initial_state"]

    usd_path = (arm_dir / spec["usd_path"]).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD for arm '{name}' not found: {usd_path}")

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            activate_contact_sensors=spawn.get("activate_contact_sensors", False),
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(**spawn.get("rigid_props", {})),
            articulation_props=sim_utils.NewtonArticulationRootPropertiesCfg(
                **spawn.get("articulation_props", {})
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(state["position"]),
            rot=_quat_from_euler_degrees(state["rotation_euler_degrees"]),
            joint_pos=dict(state["joint_positions"]),
        ),
        actuators={
            key: ImplicitActuatorCfg(**params) for key, params in spec["actuators"].items()
        },
    )
