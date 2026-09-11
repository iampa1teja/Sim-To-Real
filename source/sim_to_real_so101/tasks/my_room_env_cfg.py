import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from .so101_env_cfg import (
    SO101TeleopEnvCfg,
    LerobotSo101BaseSceneCfg,
    EventCfg,
    ObservationsCfg,
)

camera_object = TiledCameraCfg(
    prim_path="",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100,
        focal_length=13.5,
        focus_distance=0.05,
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=euler_angles_to_quat(
            np.array([0, 0, 0]), degrees=True
        ),
        convention="opengl",
    ),
)

@configclass
class MyRoomSceneCfg(LerobotSo101BaseSceneCfg):

    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/usdz/room.usdz",
        ),
    )

    camera_ego = camera_object.replace()
    camera_ego.prim_path = "{ENV_REGEX_NS}/Robot/gripper/gripper_cam"
    camera_ego.offset.pos = (-0.005, 0.06, -0.062)
    camera_ego.offset.rot = euler_angles_to_quat(
        np.array([-45, 0, 0]), degrees=True
    )

@configclass
class MyRoomEnvCfg(SO101TeleopEnvCfg):
    scene: MyRoomSceneCfg = MyRoomSceneCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
