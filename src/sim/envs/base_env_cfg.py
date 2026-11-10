"""Base env config for any arm in arms/<name>/arm.toml, on Newton (MJWarp)."""

from dataclasses import fields, is_dataclass
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from sim.envs.mdp import (
    JointPositionActionCfg,
    apply_runtime_joint_limits,
    ee_frame_state,
    joint_pos,
    joint_pos_rel,
    randomize_robot_color,
    reset_joints_by_offset,
)
from sim.scene.arm import load_arm_cfg, load_arm_spec
from core.arms.spec import ArmSpec


def _config_to_dict(cfg):
    """Include immutable manifest snapshots in Isaac Lab's config export."""
    from isaaclab.utils.dict import class_to_dict

    def export(value):
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: export(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: export(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(export(item) for item in value)
        return value

    return export(class_to_dict(cfg))


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Scene with only the arm."""

    env_spacing = 4.0
    num_envs = 1
    robot: ArticulationCfg = None


@configclass
class ActionsCfg:
    """Absolute joint position targets for every arm joint."""

    joint_positions: JointPositionActionCfg = None


@configclass
class BaseObservationsCfg:
    """Observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (order preserved)."""

        joint_pos_obs = ObsTerm(func=joint_pos)
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        ee_frame_state: ObsTerm = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseEventsCfg:
    """Reset events."""

    reset_robot_position: EventTerm = None
    runtime_joint_limits = EventTerm(func=apply_runtime_joint_limits, mode="startup")
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": ["beige"]},
    )


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Teleop base env: arm only, Newton physics."""

    arm_spec: ArmSpec | None = None
    scene: BaseSceneCfg = BaseSceneCfg()
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        from sim.solver_numerics import install_mjwarp_precision_guard

        install_mjwarp_precision_guard()
        arm = self.arm_spec or load_arm_spec()[0]
        self.arm_spec = arm
        self.scene.robot = load_arm_cfg(arm).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_positions = JointPositionActionCfg(
            asset_name="robot", joint_names=list(arm.usd_joint_names),
            scale=1.0, use_default_offset=False, preserve_order=True,
        )
        for observation in (self.observations.policy.joint_pos_obs, self.observations.policy.joint_pos_rel):
            observation.params = {"asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(arm.usd_joint_names), preserve_order=True,
            )}
        self.observations.policy.ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={"robot_cfg": SceneEntityCfg(
                "robot", body_names=[arm.links.base, arm.links.ee], preserve_order=True,
            )},
        )
        self.events.reset_robot_position = EventTerm(
            func=reset_joints_by_offset, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(arm.usd_joint_names), preserve_order=True),
                "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
            },
        )
        self.decimation = 2
        self.episode_length_s = 5.0

        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(-0.25, -0.4, 0.22), lookat=(0.15, 0.0, 0.12),
        )
        # Defaults alone do not create a visualizer in Isaac Lab 3.0.
        self.sim.visualizer_cfgs = [self.sim.default_visualizer_cfg]

        # Integrate the native PD drives implicitly: Euler is unstable for the
        # SO101's light links at these gains and the 120 Hz physics timestep.
        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(integrator="implicitfast"))
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# configclass installs its own exporter after the class body has been processed.
BaseEnvCfg.to_dict = _config_to_dict
