import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from sim_to_real_so101 import assets as _assets_pkg
from sim_to_real_so101.assets.real_setup import SETUP, robot_rotation

from .my_room_env_cfg import (
    MyRoomSceneCfg,
    MyRoomEventsCfg,
    MyRoomEnvCfg,
    MyRoomObservationsCfg,
)

from sim_to_real_so101.mdp import (
    reset_object_pose,
    reset_cube_from_recorded_starts,
    randomize_cube_color,
    randomize_pick_place_room_light,
    check_pick_place_eval_event_order,
    randomize_robot_color,
    ROBOT_COLORS,
    reset_pick_place_state,
    object_grasped,
    object_placed_in_container,
    object_placed_in_container_termination,
    time_out,
)

assets_path = os.path.dirname(os.path.abspath(_assets_pkg.__file__))

WHITE_BOX_USD = f"{assets_path}/usd/Pick-Place/White-Box-Physics.usda"
# Put the long side flush with the front edge, with the box inside the table
# boundary and to the robot's right. Clearance is from the face, not centre.
WHITE_BOX_FRONT_EDGE_CLEARANCE = 0.0
# With the measured base extent and real_setup.json robot position, this gives
# 14.5 cm base-to-box and 30 cm box-to-right-table-edge clearance.
WHITE_BOX_RIGHT_OFFSET = 0.247172813204101
_BOX_HALF_DEPTH = 0.0425  # White-Box.usd is 0.135 x 0.085 x 0.045 m.
_TABLE_NORMAL = np.array(SETUP["robot"]["table_normal"], dtype=float)
_TABLE_NORMAL /= np.linalg.norm(_TABLE_NORMAL)
_TABLE_RIGHT = np.array([0.996801706, 0.079914694, 0.00263459])
_TABLE_RIGHT /= np.linalg.norm(_TABLE_RIGHT)
_TABLE_INWARD = np.cross(_TABLE_NORMAL, _TABLE_RIGHT)
_TABLE_INWARD /= np.linalg.norm(_TABLE_INWARD)
# Minimum tabletop vertex projection on _TABLE_INWARD, in the packaged room.
_TABLE_FRONT_EDGE = -0.6953624951871441
# Measured from C_Tabletop_14: the robot's base origin is below the tabletop
# collider and must not be used as its surface height.
_TABLE_POINT = (0.65, -0.40, 0.797413)
WHITE_BOX_INITIAL_POSITION = tuple(np.linalg.solve(
    np.stack([_TABLE_RIGHT, _TABLE_INWARD, _TABLE_NORMAL]),
    [
        np.dot(SETUP["robot"]["position"], _TABLE_RIGHT) + WHITE_BOX_RIGHT_OFFSET,
        _TABLE_FRONT_EDGE + WHITE_BOX_FRONT_EDGE_CLEARANCE + _BOX_HALF_DEPTH,
        np.dot(_TABLE_POINT, _TABLE_NORMAL) + 0.001,
    ],
))
WHITE_BOX_INITIAL_ORIENTATION = tuple(robot_rotation())  # wxyz, aligned to table

# Cube pose is overwritten every reset by reset_object_pose; this is only the
# spawn-time default before the first reset event fires.
BLUE_CUBE_INITIAL_POSITION = (0.0, 0.0, 0.05)

CUBE_COLOR = tuple(SETUP["object_materials"]["blue_cube"]["color"])
CUBE_ROUGHNESS = SETUP["object_materials"]["blue_cube"]["roughness"]
CUBE_SIZE = 0.02
CUBE_MASS = 0.008

# White-Box.usd geometry, in its own frame: origin at the centre of the outer
# bottom face, +Z up; X spans +/-6.75 cm, Y +/-4.25 cm, Z 0 to 4.5 cm. The box's
# pose is read live from the scene, so these stay valid wherever it is placed.
WHITE_BOX_SIZE = (0.135, 0.085, 0.045)  # length (X), width (Y), height (Z), metres
WHITE_BOX_INNER_HALF_SIZE = (0.065625, 0.04064896)  # measured USD inner walls, metres
WHITE_BOX_FLOOR_Z = 0.0018  # inside floor height above the origin, metres



@configclass
class PickPlaceSceneCfg(MyRoomSceneCfg):
    """ 
    Extends the real room scene with a cube, container, and grasp contact sensor.

    Inherits:
        room, camera_realsense_rgb, realsense_depth, camera_wrist_cam, robot, ee_frame
    Adds:
        white_box, blue_cube, contact_grasp
    """
    contact_grasp = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/jaw",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/CUBE"],
    )

    white_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WhiteBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WHITE_BOX_USD,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(SETUP["object_materials"]["white_box"]["color"]),
                roughness=SETUP["object_materials"]["white_box"]["roughness"],
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=WHITE_BOX_INITIAL_POSITION,
            rot=WHITE_BOX_INITIAL_ORIENTATION,
        ),
    )

    blue_cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CUBE",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=CUBE_COLOR,
                roughness=CUBE_ROUGHNESS,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=BLUE_CUBE_INITIAL_POSITION,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

@configclass
class PickPlaceEventsCfg(MyRoomEventsCfg):
    """
    Extends MyRoomEventsCfg with cube placement and per-episode tracking resets.

    Inherits:
        all events from configure_scene_updates (mode = "startup") 
        set_robot_visual_material (mode = "startup")
        reset_set_robot_visual_material (mode = "reset")
        reset_robot_position (mode = "reset", from EventCfg Via MyRoomEventsCfg)
    """
    spawn_cube: EventTerm = EventTerm(
        func=reset_object_pose,
        mode = "reset",
        params={
            "asset_cfg": SceneEntityCfg("blue_cube"),
            "pose_range": {},
        },
    )

    reset_tracking = EventTerm(func=reset_pick_place_state, mode="reset")


@configclass
class PickPlaceObservationsCfg(MyRoomObservationsCfg):
    """Adds grasp/placement subtask labels on top of MyRoom's visual observations."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

        cube_grasped = ObsTerm(
            func=object_grasped,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "object_name": "blue_cube",
                "min_lift": 0.01,  # m above resting height
                "warmup_steps": 30,
                "force_threshold": 2,  # N
            },
        )

        cube_placed = ObsTerm(
            func=object_placed_in_container,
            params={
                "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
                "object_name": "blue_cube",
                "container_name": "white_box",
                "container_size": WHITE_BOX_SIZE,
                "container_floor_z": WHITE_BOX_FLOOR_Z,
                "container_inner_half_size": WHITE_BOX_INNER_HALF_SIZE,
                "min_lift": 0.01,
                "warmup_steps": 30,
                "force_threshold": 2,  # N
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class PickPlaceTerminationsCfg:
    """Termination terms for the pick-place evaluation task."""

    time_out = DoneTerm(
        func=time_out,
        time_out=True,
    )

    success = DoneTerm(
        func=object_placed_in_container_termination,
        time_out=False,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_grasp"),
            "object_name": "blue_cube",
            "container_name": "white_box",
            "container_size": WHITE_BOX_SIZE,
            "container_floor_z": WHITE_BOX_FLOOR_Z,
            "container_inner_half_size": WHITE_BOX_INNER_HALF_SIZE,
            "min_lift": 0.01,
            "warmup_steps": 30,
            "force_threshold": 2,  # N
            "confirm_steps": 25,
        },
    )


@configclass
class PickPlaceEnvCfg(MyRoomEnvCfg):
    """
    Base config — teleop/data-collection. subtask_terms give grasp/placement
    observations (not included in the dataset schema), but no auto-termination (matches how
    SO101TeleopEnvCfg leaves terminations=None for teleop).
    """
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg()
    events: PickPlaceEventsCfg = PickPlaceEventsCfg()
    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4  # 120 Hz physics, 30 Hz pick-place control.
        self.sim.render_interval = self.decimation
        self.scene.robot.spawn.activate_contact_sensors = True
        # Single-arm teleop does not need multi-million-contact training buffers.
        self.sim.physx.gpu_max_rigid_contact_count = 2**18
        self.sim.physx.gpu_max_rigid_patch_count = 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**18
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**18
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**18


@configclass
class PickPlaceEvalEventsCfg(PickPlaceEventsCfg):
    check_eval_order = EventTerm(func=check_pick_place_eval_event_order, mode="startup")
    # Replacing the inherited field preserves its manager insertion order:
    # reset_robot_position -> spawn_cube -> reset_tracking.
    spawn_cube = EventTerm(
        func=reset_cube_from_recorded_starts, mode="reset",
        params={"asset_cfg": SceneEntityCfg("blue_cube"), "starts_dir": None,
                "mode": "cycle", "random_fraction": 0.0, "seed": None,
                # Start the arm at each episode's recorded first-frame state (demo start pose).
                "reset_robot": True},
    )


@configclass
class PickPlaceEvalEnvCfg(PickPlaceEnvCfg):
    """Evaluate grasp-and-place success, with bounded episodes."""

    events: PickPlaceEvalEventsCfg = PickPlaceEvalEventsCfg()
    terminations: PickPlaceTerminationsCfg = PickPlaceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Match the existing vial evaluation task's 450 control-step horizon.
        self.episode_length_s = 450 * self.decimation * self.sim.dt


@configclass
class PickPlaceEvalDREnvCfg(PickPlaceEvalEnvCfg):
    """Recorded-start evaluation with room-light, robot-colour and cube-colour DR (cameras fixed)."""

    def __post_init__(self):
        super().__post_init__()
        # MyRoom has no LightStudio/lightbox_light, sky_light or mat assets.
        # Skip lightbox exposure, HDRI and mat rotation rather than spawn new
        # scene geometry/lights. The measured room tube light is handled below.
        from .task_env_cfg import TaskEventCfg
        reference_events = TaskEventCfg()
        # These events append AFTER MyRoom's reset_set_robot_visual_material.
        self.events.eval_room_light = EventTerm(
            func=randomize_pick_place_room_light, mode="reset",
            params={"exposure_range": reference_events.reset_lightbox_light_exposure.params["exposure_range"]},
        )
        self.events.eval_robot_color = EventTerm(
            func=randomize_robot_color, mode="reset",
            params={"color_names": list(ROBOT_COLORS)},
        )
        self.events.eval_cube_color = EventTerm(
            func=randomize_cube_color, mode="reset",
            params={"colors": {"blue": CUBE_COLOR, "red": (0.8, 0.05, 0.05)}},
        )
        # Cameras are deliberately NOT randomized: the sim cameras are calibrated to the real
        # ones (real_setup.json), so the view the policy sees stays identical to the robot's.
