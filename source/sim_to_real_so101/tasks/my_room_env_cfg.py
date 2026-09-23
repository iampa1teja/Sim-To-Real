"""The user's cleared real workbench, shared by teleop, zero and eval agents."""
import os

import numpy as np
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.sensors import spawn_camera
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat

from sim_to_real_so101.assets.real_setup import (
    DEFAULT_SETUP_USDZ, SETUP, camera_rotation, robot_rotation, apply_robot_appearance,
)
from sim_to_real_so101.mdp import image, image_raw
from .so101_env_cfg import (
    EventCfg, LerobotSo101BaseSceneCfg, ObservationsCfg, SO101TeleopEnvCfg,
)

# Share the fitted plastic color with the packaged preview robot.
MY_ROOM_ARM_COLOR = tuple(SETUP["appearance"]["robot_beige"])


def _resolve_room_usd_path() -> str:
    """Use the portable photo-matched setup unless explicitly overridden."""
    override = os.environ.get("ROOM_USD_PATH")
    if override:
        requested = os.path.abspath(os.path.expanduser(override))
        # Retain compatibility with the old editable wrapper's broken articulation.
        if requested == "/workspace/usdz/digital_twin.usd" and os.path.isfile(requested + "z"):
            return requested + "z"
        return requested
    return DEFAULT_SETUP_USDZ


def _spawn_room_without_embedded_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep the room; Isaac Lab owns the controlled robot and three sensors."""
    if cfg.usd_path == DEFAULT_SETUP_USDZ:
        # Filter at composition time; removing the embedded articulation after
        # spawning can invalidate Isaac Lab's already registered robot callbacks.
        runtime_cfg = cfg.replace(usd_path=DEFAULT_SETUP_USDZ.replace('.usdz', '_room.usda'))
        return spawn_from_usd(prim_path, runtime_cfg, translation=translation, orientation=orientation, **kwargs)
    room = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for prim in reversed(list(Usd.PrimRange(room))):
        if prim.GetName() == "Robot" or prim.IsA(UsdGeom.Camera):
            prim.SetActive(False)
    return room


def _spawn_named_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_camera(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    # USD identifiers use underscores; the viewport menu shows the requested labels.
    prim.SetDisplayName({
        "realsense_camera_rgb": "realsense_camera-rgb",
        "realsense_camera_depth": "realsense_camera-depth",
        "wrist_cam": "wrist_cam",
    }[prim.GetName()])
    return prim


def _camera(path, values, rotation, data_types):
    return TiledCameraCfg(
        prim_path=path,
        update_period=0.0,
        height=values["height"], width=values["width"],
        data_types=data_types,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            func=_spawn_named_camera,
            focal_length=values["focal_length"],
            horizontal_aperture=values["horizontal_aperture"],
            vertical_aperture=values["vertical_aperture"],
            clipping_range=(0.01, 100.0), f_stop=0.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(values["position"]), rot=rotation, convention="opengl",
        ),
    )


@configclass
class MyRoomSceneCfg(LerobotSo101BaseSceneCfg):
    room = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Room",
        spawn=sim_utils.UsdFileCfg(func=_spawn_room_without_embedded_robot, usd_path=_resolve_room_usd_path()),
    )
    camera_realsense_rgb = _camera(
        "{ENV_REGEX_NS}/realsense_camera_rgb", SETUP["realsense"], camera_rotation(),
        ["rgb", "instance_id_segmentation_fast"],
    )
    # Separate depth sensor, aligned to RGB. It is deliberately not a camera_*
    # RGB recording entry: metric depth must not be encoded as a colour video.
    realsense_depth = _camera(
        "{ENV_REGEX_NS}/realsense_camera_depth", SETUP["realsense"], camera_rotation(), ["depth"],
    )
    camera_wrist_cam = _camera(
        "{ENV_REGEX_NS}/Robot/gripper/wrist_cam", SETUP["wrist"],
        euler_angles_to_quat(np.array(SETUP["wrist"]["euler_degrees"]), degrees=True),
        ["rgb", "depth", "instance_id_segmentation_fast"],
    )


def _visual_term(sensor, kind):
    return ObsTerm(
        func=image if kind == "rgb" else image_raw,
        params={"sensor_cfg": SceneEntityCfg(sensor), "data_type": kind,
                **({"normalize": False} if kind == "rgb" else {})},
    )


@configclass
class MyRoomObservationsCfg(ObservationsCfg):
    @configclass
    class VisualCfg(ObsGroup):
        rgb_realsense_rgb = _visual_term("camera_realsense_rgb", "rgb")
        depth_realsense_rgb = _visual_term("realsense_depth", "depth")
        instance_id_seg_realsense_rgb = _visual_term("camera_realsense_rgb", "instance_id_segmentation_fast")
        rgb_wrist_cam = _visual_term("camera_wrist_cam", "rgb")
        depth_wrist_cam = _visual_term("camera_wrist_cam", "depth")
        instance_id_seg_wrist_cam = _visual_term("camera_wrist_cam", "instance_id_segmentation_fast")

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    visual: VisualCfg = VisualCfg()


def _configure_scene_updates(env, env_ids):
    for path in ("/OmniverseKit_Front", "/OmniverseKit_Top", "/OmniverseKit_Right"):
        prim = env.sim.stage.GetPrimAtPath(path)
        if prim.IsValid(): prim.SetActive(False)


def _set_real_robot_materials(env, env_ids):
    apply_robot_appearance(
        env.sim.stage,
        '/World/envs/env_0/Robot',
        arm_color=MY_ROOM_ARM_COLOR,
    )


@configclass
class MyRoomEventsCfg(EventCfg):
    configure_scene_updates = EventTerm(func=_configure_scene_updates, mode="startup")
    set_robot_visual_material = EventTerm(func=_set_real_robot_materials, mode='startup')
    reset_set_robot_visual_material = EventTerm(func=_set_real_robot_materials, mode='reset')


@configclass
class MyRoomEnvCfg(SO101TeleopEnvCfg):
    scene: MyRoomSceneCfg = MyRoomSceneCfg()
    events: MyRoomEventsCfg = MyRoomEventsCfg()
    observations: MyRoomObservationsCfg = MyRoomObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = tuple(SETUP["robot"]["position"])
        self.scene.robot.init_state.rot = robot_rotation()
        self.scene.robot.init_state.joint_pos = SETUP["robot"]["joint_positions"].copy()
        self.sim.render.rendering_mode = "balanced"
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.carb_settings = {
            "/rtx/sceneDb/ambientLightIntensity": SETUP["appearance"]["ambient_intensity"],
            "/rtx/post/histogram/enabled": False,
            "/rtx-transient/dlssg/enabled": False,
        }
        self.viewer.eye = tuple(SETUP["viewer"]["eye"])
        self.viewer.lookat = tuple(SETUP["viewer"]["lookat"])
