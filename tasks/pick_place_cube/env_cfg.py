import numpy as np 

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
    randomize_pick_place_room_light,
    reset_cube_from_recorded_starts,
    reset_pick_place_state,
)
from sim.envs.rpd_meeting_room_env_cfg import drop_unused_camera_outputs
from sim.scene.real_setup import SETUP, robot_rotation
from sim.scene.contact_materials import (
    GRASP_FRICTION, GRASP_ROLLING_FRICTION, contact_material, contact_properties,
)


CUBE_PRIM = "{ENV_REGEX_NS}/CUBE"
BOX_PRIM = "{ENV_REGEX_NS}/WhiteBox"

_TABLE_RIGHT = np.array([0.996801706, 0.079914694, 0.00263459])
_TABLE_FRONT_EDGE = -0.6953624951871441
_TABLE_POINT = (0.65, -0.40, 0.797413)

def _material(name):
    values = SETUP["object_materials"][name]
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(values["color"]), roughness=values["roughness"], metallic=0.0)

def _box_position(box):
    normal = np.array(SETUP["robot"]["table_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    right = _TABLE_RIGHT / np.linalg.norm(_TABLE_RIGHT)
    inward = np.cross(normal, right)
    inward /= np.linalg.norm(inward)
    half_depth = box.params["size_outer"][1] / 2
    return tuple(float(v) for v in np.linalg.solve(
        np.stack([right, inward, normal]),
        [
            np.dot(SETUP["robot"]["position"], right) + box.params["right_offset"],
            _TABLE_FRONT_EDGE + box.params["front_edge_clearance"] + half_depth,
            np.dot(_TABLE_POINT, normal) + 0.001,
        ],
    ))

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

    cfg.scene.contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[CUBE_PRIM],
    )
    cfg.scene.white_box = RigidObjectCfg(
        prim_path=BOX_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=str(box.usd), visual_material=_material("white_box")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_box_position(box), rot=tuple(robot_rotation())),
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
            visual_material=_material("blue_cube"),
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
    cube_colors = {
        name: tuple(SETUP["object_materials"]["blue_cube"]["color"]) if value == "from_setup" else tuple(value)
        for name, value in dr.get("cube_colors", {"blue": "from_setup"}).items()
    }

    cfg.events.eval_room_light = EventTerm(
        func=randomize_pick_place_room_light, mode="reset",
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
