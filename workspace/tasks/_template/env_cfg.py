"""Compose stacking objects with any compatible arm and declarative scene."""
import re
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from core.scenes import Pose, transform_pose
from sim.envs.mdp import park_object, time_out
from sim.envs.mdp.starts import assert_reset_event_order, reset_from_starts
from sim.envs.scene_builder import drop_unused_camera_outputs
from sim.scene.cameras import euler_xyz_degrees_to_xyzw
from sim.scene.contact_materials import GRASP_FRICTION, GRASP_ROLLING_FRICTION, contact_material, contact_properties
from .mdp import reset_target, reset_tracking, success


@configclass
class Terminations:
    time_out = DoneTerm(func=time_out, time_out=True)
    success = DoneTerm(func=success)


def apply_teleop(cfg, task):
    arm, scene = cfg.arm_spec, cfg.scene_spec
    if arm.gripper is None or not arm.gripper.contact_bodies:
        raise ValueError(f"Task {task.name!r} requires gripper contact_bodies")
    state = cfg.scene.robot.init_state
    for name, spec in task.spec.objects.items():
        pose = transform_pose(Pose(spec.position, euler_xyz_degrees_to_xyzw(spec.rotation_euler_degrees)),
                              spec.params.get("frame", "world"), scene, Pose(state.pos, state.rot))
        setattr(cfg.scene, name, RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            spawn=sim_utils.CuboidCfg(
                size=spec.size, rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=spec.params["mass"]),
                collision_props=contact_properties(), physics_material_path="physicsMaterial",
                physics_material=contact_material(GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.params["color"]), roughness=.5),
            ), init_state=RigidObjectCfg.InitialStateCfg(pos=pose.position, rot=pose.orientation_xyzw),
        ))
    names = arm.gripper.contact_bodies
    expression = re.escape(names[0]) if len(names) == 1 else "(" + "|".join(map(re.escape, names)) + ")"
    cfg.scene.contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + expression, update_period=0.0,
        history_length=1, debug_vis=False, filter_prim_paths_expr=["{ENV_REGEX_NS}/cube_a"],
    )
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.events.reset_target = EventTerm(func=reset_target, mode="reset")
    cfg.events.reset_from_starts = EventTerm(func=park_object, mode="reset",
                                            params={"asset_cfg": SceneEntityCfg("cube_a")})
    cfg.events.reset_tracking = EventTerm(func=reset_tracking, mode="reset")
    cfg.decimation = task.spec.raw["episode"]["decimation"]
    cfg.sim.render_interval = cfg.decimation


def apply_eval(cfg, task):
    apply_teleop(cfg, task)
    cfg.events.reset_target = None
    cfg.events.check_eval_order = EventTerm(func=assert_reset_event_order, mode="startup")
    cfg.events.reset_from_starts = EventTerm(func=reset_from_starts, mode="reset",
        params={"task": task.name, "starts_dir": None, "mode": "cycle", "random_fraction": 0.0,
                "seed": None, "reset_robot": True})
    cfg.terminations = Terminations()
    cfg.episode_length_s = task.spec.raw["episode"]["eval_steps"] * cfg.decimation * cfg.sim.dt
    drop_unused_camera_outputs(cfg)
