from core.arms.spec import load_arm_spec
from core.scenes import Pose, load_scene_spec, transform_pose

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm, 
    ObservationGroupCfg as ObsGroup, 
    ObservationTermCfg as ObsTerm, 
    SceneEntityCfg, 
    TerminationTermCfg as DoneTerm
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from sim.envs.mdp import ROBOT_COLORS, park_object, randomize_robot_color, time_out
from .mdp import (
    check_pick_place_eval_event_order,
    object_grasped,
    object_placed_in_container,
    object_placed_in_container_termination,
    randomize_cube_color,
    reset_cube_from_recorded_starts,
    reset_pick_place_state,
)
from sim.envs.scene_builder import drop_unused_camera_outputs
from sim.scene.cameras import euler_xyz_degrees_to_xyzw
from sim.scene.appearance import randomize_scene_light_exposure
from sim.scene.contact_materials import (
    GRASP_FRICTION, GRASP_ROLLING_FRICTION, contact_material, contact_properties,
)


CUBE_PRIM = "{ENV_REGEX_NS}/CUBE"
BOX_PRIM = "{ENV_REGEX_NS}/WhiteBox"

def _material(name, task, scene):
    values = scene.appearance.get("object_materials", {}).get(name, {})
    params = task.spec.objects[name].params
    fallback = params.get("material", {})
    color = values.get("color", fallback.get("color", params.get("color", (0, 0, 1) if name == "blue_cube" else (1, 1, 1))))
    roughness = values.get("roughness", fallback.get("roughness", params.get("roughness", 0.5)))
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(value) for value in color),
                                       roughness=float(roughness), metallic=0.0)


def _box_pose(box, scene, arm_pose=None):
    pose = Pose(box.position, euler_xyz_degrees_to_xyzw(box.rotation_euler_degrees))
    return transform_pose(pose, box.params.get("frame", "world"), scene, arm_pose)


def _place_params(task):
    box = task.spec.objects["white_box"]
    success = task.spec.raw.get("success", {})
    return {
        "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
        "object_name": "blue_cube",
        "container_name": "white_box",
        "container_size": tuple(box.params["size_outer"]),
        "container_floor_z": box.params["floor_z"],
        "container_inner_half_size": tuple(box.params["inner_half_size"]),
        "min_lift": success.get("min_lift", 0.01),
        "warmup_steps": success.get("warmup_steps", 30),
        "force_threshold": success.get("force_threshold", 2.0),
    }

@configclass
class SubtaskCfg(ObsGroup):
    cube_grasped: ObsTerm = None
    cube_placed: ObsTerm = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False

@configclass
class PickPlaceTerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=time_out, time_out=True)
    success: DoneTerm = None

def apply_teleop(cfg, task):
    box = task.spec.objects["white_box"]
    cube = task.spec.objects["blue_cube"]
    episode = task.spec.raw.get("episode", {})
    scene = getattr(cfg, "scene_spec", None) or load_scene_spec()
    state = cfg.scene.robot.init_state
    box_pose = _box_pose(box, scene, Pose(state.pos, state.rot))

    arm = getattr(cfg, "arm_spec", None) or load_arm_spec()[0]
    if arm.gripper is None or not arm.gripper.contact_bodies:
        raise ValueError(f"Task {task.name!r} needs arm gripper contact_bodies")
    contact_bodies = arm.gripper.contact_bodies
    body_pattern = contact_bodies[0] if len(contact_bodies) == 1 else "(" + "|".join(contact_bodies) + ")"
    cfg.scene.contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body_pattern,
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[CUBE_PRIM],
    )
    cfg.scene.white_box = RigidObjectCfg(
        prim_path=BOX_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=str(box.usd), visual_material=_material("white_box", task, scene)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=box_pose.position, rot=box_pose.orientation_xyzw),
    )
    cfg.scene.blue_cube = RigidObjectCfg(
        prim_path=CUBE_PRIM,
        spawn=sim_utils.CuboidCfg(
            size=cube.size,
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cube.params["mass"]),
            collision_props=contact_properties(),
            physics_material_path="physicsMaterial",
            physics_material=contact_material(GRASP_FRICTION, rolling_friction=GRASP_ROLLING_FRICTION),
            visual_material=_material("blue_cube", task, scene),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cube.position, rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # Teleop: the cube stays parked out of the scene until spawn.py asks for it at the jaw
    # (sim.generate.object_spawner). Eval replaces this event with recorded starts.
    cfg.events.spawn_cube = EventTerm(
        func=park_object, mode="reset",
        params={"asset_cfg": SceneEntityCfg("blue_cube")},
    )
    cfg.events.reset_tracking = EventTerm(func=reset_pick_place_state, mode="reset")

    params = _place_params(task)
    cfg.observations.subtask_terms = SubtaskCfg(
        cube_grasped=ObsTerm(
            func=object_grasped,
            params={k: params[k] for k in
                    ("contact_sensor_cfg", "object_name", "min_lift", "warmup_steps", "force_threshold")},
        ),
        cube_placed=ObsTerm(func=object_placed_in_container, params=dict(params)),
    )

    cfg.decimation = episode.get("decimation", 4)
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.robot.spawn.activate_contact_sensors = True

def apply_eval(cfg, task):
    apply_teleop(cfg, task)
    episode = task.spec.raw.get("episode", {})
    success = task.spec.raw.get("success", {})

    cfg.events.check_eval_order = EventTerm(func=check_pick_place_eval_event_order, mode="startup")
    cfg.events.spawn_cube = EventTerm(
        func = reset_cube_from_recorded_starts, mode = "reset", 
        params = {
            "asset_cfg": SceneEntityCfg("blue_cube"), "starts_dir": None, 
            "mode": "cycle", "random_fraction": 0.0, "seed": None,
            "reset_robot": True,
        },
    )
    cfg.terminations = PickPlaceTerminationsCfg(
        success=DoneTerm(
            func=object_placed_in_container_termination, time_out=False,
            params={**_place_params(task), "confirm_steps": success.get("confirm_steps", 25)},
        ),
    )
    cfg.episode_length_s = episode.get("eval_steps", 450) * cfg.decimation * cfg.sim.dt
    drop_unused_camera_outputs(cfg)


def apply_eval_dr(cfg, task):
    apply_eval(cfg, task)
    dr = task.spec.raw.get("dr", {})
    scene = getattr(cfg, "scene_spec", None) or load_scene_spec()
    base_color = scene.appearance.get("object_materials", {}).get("blue_cube", {}).get("color", (0, 0, 1))
    cube_colors = {
        name: tuple(base_color) if value == "from_setup" else tuple(value)
        for name, value in dr.get("cube_colors", {"blue": "from_setup"}).items()
    }

    cfg.events.eval_room_light = EventTerm(
        func=randomize_scene_light_exposure, mode="reset",
        params={"exposure_range": tuple(dr.get("room_light_exposure_range", (-3.0, 1.0)))},
    )
    cfg.events.eval_robot_color = EventTerm(
        func=randomize_robot_color, mode="reset",
        params={"color_names": list(ROBOT_COLORS)},
    )
    cfg.events.eval_cube_color = EventTerm(
        func=randomize_cube_color, mode="reset",
        params={"colors": cube_colors},
    )
