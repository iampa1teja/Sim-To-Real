"""Photo-matched room scene: the measured table, calibrated room and wrist cameras, and fitted lighting.

The scene data comes from ``scenes/$S2R_SCENE`` through ``sim.scene.real_setup``.

Isaac Lab 3.0 changes from v1's my_room_env_cfg:
- ``self.sim.render.*`` no longer exists. The RTX quality settings moved to each camera's
  ``renderer_cfg`` (``IsaacRtxRendererCfg.global_settings``). They are process-global, so every camera carries the same values.
- Cameras request the Isaac RTX renderer explicitly. Under Newton the default renderer can be the
  Newton warp renderer, which ignores the OpenCV lens model and the room's materials.
- ``mdp.image`` was replaced by the ``image_rgb`` class; RGB uses ``image_rgb_raw`` (no normalisation, plus the fitted colour response).
- Offsets are XYZW quaternions.
"""
import os

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.sensors import spawn_camera
from isaaclab.utils import configclass
from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

from sim.envs.base_env_cfg import BaseEnvCfg, BaseEventsCfg, BaseObservationsCfg, BaseSceneCfg
from sim.envs.mdp import apply_runtime_joint_limits, image_raw, image_rgb_raw
from sim.scene.calibrated_camera import CalibratedPinholeCameraCfg, CalibratedTiledCamera, apply_opencv_intrinsics
from sim.scene.calibrated_viewport_cfg import CalibratedKitVisualizerCfg
from sim.scene.contact_materials import bind_contact_material
from sim.scene.gripper_geometry import cache_scene_geometry
from sim.scene.real_setup import (
    DEFAULT_SETUP_USDZ, SETUP, apply_robot_appearance, apply_room_appearance, apply_room_lighting,
    camera_rotation, quaternion_tuple, robot_rotation,
)

# Share the fitted plastic color with the packaged preview robot.
ROOM_ARM_COLOR = tuple(SETUP["appearance"]["robot_beige"])


def _rtx_renderer() -> IsaacRtxRendererCfg:
    """v1's MyRoom render settings, now carried by the camera's RTX renderer config."""
    return IsaacRtxRendererCfg(
        class_type="sim.scene.calibrated_renderer:CalibratedRtxRenderer",
        global_settings=IsaacRtxRendererGlobalSettingsCfg(
            antialiasing_mode="DLAA",
            enable_ambient_occlusion=True,
            enable_global_illumination=True,
            enable_dlssg=False,
            ambient_light_intensity=SETUP["appearance"]["ambient_intensity"],
            carb_settings={"/rtx/post/histogram/enabled": False},
        ),
    )


def _euler_xyz_degrees_to_xyzw(angles) -> tuple[float, float, float, float]:
    """Extrinsic XYZ degrees -> XYZW, composed the same way as real_setup.camera_rotation."""
    x, y, z = angles
    rotation = (Gf.Rotation(Gf.Vec3d(1, 0, 0), x)
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), y)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), z))
    return quaternion_tuple(rotation)


def _resolve_room_usd_path() -> str:
    """Use the scene's photo-matched room unless ROOM_USD_PATH overrides it."""
    override = os.environ.get("ROOM_USD_PATH")
    return os.path.abspath(os.path.expanduser(override)) if override else DEFAULT_SETUP_USDZ


def _spawn_room_without_embedded_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Keep the room; Isaac Lab owns the controlled robot and the cameras."""
    if cfg.usd_path == DEFAULT_SETUP_USDZ:
        # Filter at composition time; removing the embedded articulation after
        # spawning can invalidate Isaac Lab's already registered robot callbacks.
        runtime_cfg = cfg.replace(usd_path=DEFAULT_SETUP_USDZ.replace(".usdz", "_room.usda"))
        room = spawn_from_usd(prim_path, runtime_cfg, translation=translation, orientation=orientation, **kwargs)
    else:
        room = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        for prim in reversed(list(Usd.PrimRange(room))):
            if prim.GetName() == "Robot" or prim.IsA(UsdGeom.Camera):
                prim.SetActive(False)
    bind_contact_material(room, lambda prim: "tabletop" in str(prim.GetPath()).lower())
    return room


def _spawn_named_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_camera(
        prim_path, cfg.replace(intrinsics=None, response=None),
        translation=translation, orientation=orientation, **kwargs,
    )
    if cfg.intrinsics is not None:
        apply_opencv_intrinsics(prim, cfg.intrinsics)
    # USD identifiers use underscores; the viewport menu shows the requested labels.
    prim.SetDisplayName({
        "realsense_camera_rgb": "realsense_camera-rgb",
        "realsense_camera_depth": "realsense_camera-depth",
        "wrist_cam": "wrist_cam",
    }[prim.GetName()])
    return prim


def _camera(path, values, rotation, data_types):
    return TiledCameraCfg(
        class_type=CalibratedTiledCamera,
        prim_path=path,
        update_period=0.0,
        height=values["height"],
        width=values["width"],
        data_types=data_types,
        renderer_cfg=_rtx_renderer(),
        spawn=CalibratedPinholeCameraCfg(
            func=_spawn_named_camera,
            intrinsics=({**values["calibration"], "width": values["width"], "height": values["height"]}
                        if "calibration" in values else None),
            response=values.get("response"),
            focal_length=values["focal_length"],
            horizontal_aperture=values["horizontal_aperture"],
            vertical_aperture=values["vertical_aperture"],
            clipping_range=(0.01, 100.0),
            f_stop=0.0,
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(values["position"]), rot=rotation, convention="opengl"),
    )


@configclass
class RpdMeetingRoomSceneCfg(BaseSceneCfg):
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
        _euler_xyz_degrees_to_xyzw(SETUP["wrist"]["euler_degrees"]),
        ["rgb", "depth", "instance_id_segmentation_fast"],
    )


def _visual_term(sensor, kind):
    return ObsTerm(
        func=image_rgb_raw if kind == "rgb" else image_raw,
        params={"sensor_cfg": SceneEntityCfg(sensor), "data_type": kind},
    )


@configclass
class RpdMeetingRoomObservationsCfg(BaseObservationsCfg):
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
        if prim.IsValid():
            prim.SetActive(False)
    # Apply the same material settings as the packaged USDZ. Custom room overrides can
    # have unrelated materials and lighting.
    if env.cfg.scene.room.spawn.usd_path == DEFAULT_SETUP_USDZ:
        for root in env.scene.env_prim_paths:
            apply_room_appearance(env.sim.stage, root + "/Room/Room")
            apply_room_lighting(env.sim.stage, root + "/Room/Room")


def _set_real_robot_materials(env, env_ids):
    apply_robot_appearance(env.sim.stage, "/World/envs/env_0/Robot", arm_color=ROOM_ARM_COLOR)


@configclass
class RpdMeetingRoomEventsCfg(BaseEventsCfg):
    cache_static_geometry = EventTerm(func=cache_scene_geometry, mode="startup")
    runtime_joint_limits = EventTerm(func=apply_runtime_joint_limits, mode="startup")
    configure_scene_updates = EventTerm(func=_configure_scene_updates, mode="startup")
    set_robot_visual_material = EventTerm(func=_set_real_robot_materials, mode="startup")
    # Replaces the base env's beige colour event with the fitted real-arm materials.
    reset_set_robot_visual_material = EventTerm(func=_set_real_robot_materials, mode="reset")


@configclass
class RpdMeetingRoomEnvCfg(BaseEnvCfg):
    sim_joint_mapping: dict = SETUP.get("sim_joint_mapping", {})
    scene: RpdMeetingRoomSceneCfg = RpdMeetingRoomSceneCfg()
    events: RpdMeetingRoomEventsCfg = RpdMeetingRoomEventsCfg()
    observations: RpdMeetingRoomObservationsCfg = RpdMeetingRoomObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Newton supplies stable contact manifolds for the triangulated tabletop;
        # MJWarp still solves the dynamics. Native MJWarp contacts let the cube
        # rock and creep here, even with nonzero friction and multi-CCD enabled.
        self.sim.physics.solver_cfg.use_mujoco_contacts = False
        self.sim.physics.solver_cfg.cone = "elliptic"
        self.sim.physics.solver_cfg.impratio = 1000.0
        self.sim.physics.solver_cfg.tolerance = 1e-8
        # Refresh moving-jaw contacts every 2.08 ms within the existing 120 Hz
        # physics tick; control and camera rendering keep their 30 Hz cadence.
        self.sim.physics.num_substeps = 4
        self.sim.physics.collision_decimation = 1
        # Keep the scene's surveyed reference separate from the contact-fitted
        # mount correction so the table and box retain their recorded world poses.
        self.scene.robot.init_state.pos = tuple(
            p + delta for p, delta in zip(
                SETUP["robot"]["position"],
                SETUP["robot"].get("mount_translation_correction_m", (0.0, 0.0, 0.0)),
                strict=True,
            )
        )
        self.scene.robot.init_state.rot = robot_rotation()
        self.scene.robot.init_state.joint_pos = SETUP["robot"]["joint_positions"].copy()
        self.sim.default_visualizer_cfg = CalibratedKitVisualizerCfg(
            camera_prim_path="/World/envs/env_0/realsense_camera_rgb",
            window_width=SETUP["realsense"]["width"],
            window_height=SETUP["realsense"]["height"],
            eye=tuple(SETUP["viewer"]["eye"]),
            lookat=tuple(SETUP["viewer"]["lookat"]),
        )
        self.sim.visualizer_cfgs = [self.sim.default_visualizer_cfg]


def drop_unused_camera_outputs(cfg, depth=False, instance_id_seg=False):
    """Stop rendering depth / instance-id segmentation nobody consumes.

    Policy eval uses only RGB; recorders read depth and segmentation only when enabled.
    Pass the flags the caller will actually use. Lists are rebuilt, not mutated in place.
    """
    if not depth:
        cfg.scene.realsense_depth = None
        cfg.scene.camera_wrist_cam.data_types = [t for t in cfg.scene.camera_wrist_cam.data_types if t != "depth"]
        cfg.observations.visual.depth_realsense_rgb = None
        cfg.observations.visual.depth_wrist_cam = None
    if not instance_id_seg:
        seg = "instance_id_segmentation_fast"
        cfg.scene.camera_realsense_rgb.data_types = [t for t in cfg.scene.camera_realsense_rgb.data_types if t != seg]
        cfg.scene.camera_wrist_cam.data_types = [t for t in cfg.scene.camera_wrist_cam.data_types if t != seg]
        cfg.observations.visual.instance_id_seg_realsense_rgb = None
        cfg.observations.visual.instance_id_seg_wrist_cam = None
