import os

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
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

from sim_to_real_so101.mdp import reset_object_pose

assets_path = os.path.dirname(os.path.abspath(_assets_pkg.__file__))

WHITE_BOX_USD = f"{assets_path}/usd/Pick-Place/White-Box.usd"
WHITE_BOX_INITIAL_POSITION = (0.0, 0.0, 0.0)  # TODO: set to actual initial position
WHITE_BOX_INITIAL_ORIENTATION = (0.0, 0.0, 0.0, 1.0)  # TODO: set to actual initial orientation

# Cube pose is overwritten every reset by reset_object_pose; this is only the
# spawn-time default before the first reset event fires. Kept distinct from
# WHITE_BOX_INITIAL_POSITION so the two objects never co-locate at (0,0,0).
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
class PickPlaceEnvCfg(MyRoomEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg()
    events: PickPlaceEventsCfg = PickPlaceEventsCfg() 
    observations: MyRoomObservationsCfg = MyRoomObservationsCfg()

    def __post_init__(self):
        super().__post_init__()