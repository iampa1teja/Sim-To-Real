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
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101 import assets as _assets_pkg
from sim_to_real_so101.assets.real_setup import SETUP

from .my_room_env_cfg import (
    MyRoomSceneCfg,
    MyRoomEventsCfg,
    MyRoomEnvCfg,
    MyRoomObservationsCfg,
    _camera,
)

from sim_to_real_so101.mdp import (
    reset_object_pose,
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
WHITE_BOX_RIGHT_OFFSET = 0.25
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

CUBE_COLOR = (0.0, 0.0, 1.0)
CUBE_ROUGHNESS = 0.75
CUBE_SIZE = 0.02
CUBE_MASS = 0.008

# Pick-Place framing: slightly wider FOV / re-aimed vs. the stock MyRoom
# realsense values so the box + cube both stay in frame.
_PICK_PLACE_REALSENSE_VALUES = {**SETUP["realsense"], "focal_length": 18.0}
_PICK_PLACE_REALSENSE_ROTATION = euler_angles_to_quat(
    np.array([40.0, 6.0, -165.0]), degrees=True
)

# White-Box.usd geometry, in its own frame: origin at the centre of the outer
# bottom face, +Z up; X spans +/-6.75 cm, Y +/-4.25 cm, Z 0 to 4.5 cm. The box's
# pose is read live from the scene, so these stay valid wherever it is placed.
WHITE_BOX_SIZE = (0.135, 0.085, 0.045)  # length (X), width (Y), height (Z), metres
WHITE_BOX_FLOOR_Z = 0.0018  # inside floor height above the origin, metres



@configclass
class PickPlaceSceneCfg(MyRoomSceneCfg):
    """ 
    Extends the real room scene with a single rigid body pick target. 

    Inherits:
        room, camera_realsense_rgb, camera_realsense_depth, camera_wrist_cam, robot, ee_frame
    Adds:
        white_box: RigidObjectCfg for the pick target
    """
    camera_realsense_rgb = _camera(
        "{ENV_REGEX_NS}/realsense_camera_rgb",
        _PICK_PLACE_REALSENSE_VALUES,
        _PICK_PLACE_REALSENSE_ROTATION,
        ["rgb", "instance_id_segmentation_fast"],
    )

    white_box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WhiteBox",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WHITE_BOX_USD,
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
    Extends MyRoomEventsCfg with a pre-reset box pose randomization term. 

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
    labels in the recorded dataset, but no auto-termination (matches how
    SO101TeleopEnvCfg leaves terminations=None for teleop).
    """
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg()
    events: PickPlaceEventsCfg = PickPlaceEventsCfg()
    observations: PickPlaceObservationsCfg = PickPlaceObservationsCfg()

    def __post_init__(self):
        super().__post_init__()