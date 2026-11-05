"""Base env config for any arm in arms/<name>/arm.toml, on Newton (MJWarp)."""

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from sim.envs.mdp import (
    JointPositionActionCfg,
    ee_frame_state,
    joint_pos,
    joint_pos_rel,
    randomize_robot_color,
    reset_joints_by_offset,
)
from sim.scene.arm import load_arm_cfg, load_arm_spec
from sim.scene.real_setup import ARM_NAME

ARM_SPEC, _ = load_arm_spec(ARM_NAME)
JOINT_NAMES = list(ARM_SPEC["joints"]["names"])
BASE_LINK = ARM_SPEC["links"]["base"]
EE_LINK = ARM_SPEC["links"]["ee"]


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Scene with only the arm."""

    env_spacing = 4.0
    num_envs = 1
    robot: ArticulationCfg = load_arm_cfg(ARM_NAME).replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Absolute joint position targets for every arm joint."""

    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class BaseObservationsCfg:
    """Observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations (order preserved)."""

        joint_pos_obs = ObsTerm(func=joint_pos)
        joint_pos_rel = ObsTerm(func=joint_pos_rel)
        ee_frame_state = ObsTerm(
            func=ee_frame_state,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK, EE_LINK], preserve_order=True),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseEventsCfg:
    """Reset events."""

    reset_robot_position = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_set_robot_visual_material = EventTerm(
        func=randomize_robot_color,
        mode="reset",
        params={"color_names": ["beige"]},
    )


@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Teleop base env: arm only, Newton physics."""

    scene: BaseSceneCfg = BaseSceneCfg()
    observations: BaseObservationsCfg = BaseObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0

        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)

        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
