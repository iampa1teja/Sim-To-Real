"""Compose all three hollow rigid bowls before Newton imports the topology."""
import re
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import spawn_from_usd
from core.scenes import Pose, transform_pose
from sim.envs.mdp import time_out
from sim.envs.mdp.starts import assert_reset_event_order, reset_from_starts
from sim.envs.scene_builder import drop_unused_camera_outputs
from sim.scene.cameras import euler_xyz_degrees_to_xyzw
from sim.scene.contact_materials import bind_contact_material, GRASP_FRICTION, GRASP_ROLLING_FRICTION
from .mdp import reset_tracking, success


def spawn_bowl(prim_path, cfg, translation=None, orientation=None, **kwargs):
    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    bind_contact_material(root, lambda prim: True, friction=GRASP_FRICTION,
                          rolling_friction=GRASP_ROLLING_FRICTION)
    return root


@configclass
class BowlCfg(sim_utils.UsdFileCfg):
    func = spawn_bowl


@configclass
class Terminations:
    time_out = DoneTerm(func=time_out, time_out=True)
    success = DoneTerm(func=success)


def apply_teleop(cfg, task):
    arm, scene = cfg.arm_spec, cfg.scene_spec
    if arm.gripper is None or not arm.gripper.contact_bodies:
        raise ValueError('Stacking bowls requires gripper contact bodies')
    state = cfg.scene.robot.init_state
    for name, spec in task.spec.objects.items():
        pose = transform_pose(Pose(spec.position, euler_xyz_degrees_to_xyzw(spec.rotation_euler_degrees)),
                              spec.params.get('frame', 'world'), scene, Pose(state.pos, state.rot))
        setattr(cfg.scene, name, RigidObjectCfg(prim_path='{ENV_REGEX_NS}/'+name,
            spawn=BowlCfg(usd_path=str(spec.usd),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.params['color']), roughness=.4)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pose.position, rot=pose.orientation_xyzw)))
    names = arm.gripper.friction_bodies or arm.gripper.contact_bodies
    pattern = '('+'|'.join(map(re.escape, names))+')'
    cfg.scene.contact_grasp = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+pattern,
        update_period=0., history_length=1, debug_vis=False,
        filter_prim_paths_expr=['{ENV_REGEX_NS}/'+name for name in task.spec.tracked_objects])
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.events.check_eval_order = EventTerm(func=assert_reset_event_order, mode='startup')
    cfg.events.reset_from_starts = EventTerm(func=reset_from_starts, mode='reset',
        params=dict(task=task.name, mode='random', reset_robot=False))
    cfg.events.reset_tracking = EventTerm(func=reset_tracking, mode='reset')
    # Three compound hollow bowls produce substantially more contact rows.
    # Capacity only: contact materials, solver objective and timing stay unchanged.
    cfg.sim.physics.solver_cfg.njmax = max(cfg.sim.physics.solver_cfg.njmax or 0, 131072)
    cfg.sim.physics.solver_cfg.nconmax = max(cfg.sim.physics.solver_cfg.nconmax or 0, 32768)
    cfg.sim.physics.solver_cfg.iterations = max(cfg.sim.physics.solver_cfg.iterations, 100)
    cfg.decimation = task.spec.raw['episode']['decimation']
    cfg.sim.render_interval = cfg.decimation


def apply_eval(cfg, task):
    apply_teleop(cfg, task)
    cfg.terminations = Terminations()
    cfg.episode_length_s = task.spec.raw['episode']['eval_steps']*cfg.decimation*cfg.sim.dt
    drop_unused_camera_outputs(cfg)
