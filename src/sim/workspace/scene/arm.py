from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from sim.workspace.scene.contact_materials import (
    GRASP_FRICTION, GRASP_ROLLING_FRICTION, bind_contact_material, enable_collision_meshes,
)

# The toml reading lives in core so hardware code can use it without Isaac.
from utils.arms.spec import ArmSpec, arms_dir, load_arm_spec  # noqa: F401  (re-exported)


@configclass
class ArmUsdFileCfg(sim_utils.UsdFileCfg):
    """Keep arm selection with the spawn config, including when another arm is selected later."""

    arm_spec: ArmSpec | None = None


def _quat_from_euler_degrees(angles: list[float]) -> tuple[float, ...]:
    """Use Isaac Lab's own helper so the order always matches what ArticulationCfg expects."""
    roll, pitch, yaw = (torch.tensor([math.radians(a)]) for a in angles)
    return tuple(quat_from_euler_xyz(roll, pitch, yaw)[0].tolist())


def _spawn_arm(prim_path, cfg, translation=None, orientation=None, **kwargs):
    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    arm = cfg.arm_spec
    if arm is None:
        raise ValueError("ArmUsdFileCfg needs arm_spec")
    enable_collision_meshes(root, arm)
    if arm.gripper is not None:
        bodies = tuple(root.GetPath().AppendChild(name) for name in arm.gripper.friction_bodies)
        bind_contact_material(root, lambda prim: any(prim.GetPath().HasPrefix(path) for path in bodies),
                              friction=GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION)
    return root


def load_arm_cfg(name: str | ArmSpec | None = None, variant: str | None = None) -> ArticulationCfg:
    if isinstance(name, ArmSpec):
        if variant is not None:
            raise ValueError("A loaded ArmSpec already includes its variant")
        spec = name
    else:
        spec, _ = load_arm_spec(name, variant)
    spawn = spec.spawn
    state = spec.initial_state

    usd_path = spec.usd_path
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD for arm '{spec.name}' not found: {usd_path}")

    return ArticulationCfg(
        spawn=ArmUsdFileCfg(
            func=_spawn_arm,
            arm_spec=spec,
            usd_path=str(usd_path),
            activate_contact_sensors=spawn.get("activate_contact_sensors", False),
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(**spawn.get("rigid_props", {})),
            articulation_props=sim_utils.NewtonArticulationRootPropertiesCfg(
                **spawn.get("articulation_props", {})
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(state.position),
            rot=_quat_from_euler_degrees(state.rotation_euler_degrees),
            joint_pos=dict(spec.initial_joint_positions),
        ),
        actuators={
            key: ImplicitActuatorCfg(**params) for key, params in spec.actuators.items()
        },
    )
