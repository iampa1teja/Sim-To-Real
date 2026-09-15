import os
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
            usd_path=os.environ.get("ROOM_USD_PATH", "/workspace/usdz/digital_twin.usdz"),
        ),
        # The reconstructed digital twin already uses metres and Z-up.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
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

    def __post_init__(self) -> None:
        super().__post_init__()

        # Rest the complete base on the measured tabletop, inside its near edge.
        # The robot mesh bottom is 0.03008145 m above its articulation origin;
        # subtract that offset and align the base with the table's slight slope.
        self.scene.robot.init_state.pos = (0.05, 0.38, 0.781066857)
        self.scene.robot.init_state.rot = (
            0.999957023, 0.009252938, -0.000579532, 0.0
        )

        # Frame the room-scale scene and the complete tabletop.
        self.viewer.eye = (2.2, 2.5, 1.8)
        self.viewer.lookat = (0.05, 0.0, 0.80)
