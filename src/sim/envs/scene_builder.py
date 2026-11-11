"""Build an arm and declarative scene without scene-specific source branches."""
from __future__ import annotations

import hashlib
import importlib.util
import re
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

from core.arms.spec import ArmSpec
from core.scenes import SceneSpec, load_mount_spec, load_scene_spec, mount_joint_positions, mount_pose
from sim.envs.base_env_cfg import BaseEnvCfg, _config_to_dict
from sim.envs.mdp import env_indices, image_raw, image_rgb_raw
from sim.scene.appearance import apply_robot_appearance, apply_scene_lighting
from sim.scene.cameras import build_camera_cfg
from sim.scene.contact_materials import bind_contact_material
from sim.scene.geometry import anchor_prim, cache_scene_geometry
from sim.scene.renderer import build_renderer_cfg
from sim.scene.viewport import CalibratedKitVisualizerCfg


@configclass
class SceneRoomUsdCfg(sim_utils.UsdFileCfg):
    scene_spec: SceneSpec | None = None


def spawn_scene_room(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Bind declared static contacts before Newton imports the room topology."""
    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    scene = cfg.scene_spec
    if scene is None:
        raise ValueError("SceneRoomUsdCfg needs scene_spec")
    for anchor in scene.anchors.values():
        if not anchor.contact_material:
            continue
        target = anchor_prim(root.GetStage(), root, anchor)
        if target is None:
            raise ValueError(f"Contact anchor prim {anchor.prim_name!r} is missing in {scene.name!r}")
        if anchor.contact_prim_pattern is not None:
            pattern = re.compile(anchor.contact_prim_pattern)
            bind_contact_material(root, lambda prim: pattern.fullmatch(prim.GetName()) is not None)
        else:
            path = target.GetPath()
            bind_contact_material(root, lambda prim: prim.GetPath().HasPrefix(path))
    return root


def hide_default_viewports(stage):
    """Disable unused SDK orthographic cameras, preserving calibrated scene cameras."""
    for path in ("/OmniverseKit_Front", "/OmniverseKit_Top", "/OmniverseKit_Right"):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)


def _content_path(env, index, scene):
    path = env.scene.env_prim_paths[index] + "/Room"
    return path + "/" + scene.room_root.strip("/") if scene.room_root else path


def configure_scene_updates(env, env_ids, scene_spec):
    """Apply generic visual-only lighting after the physics model has been imported."""
    hide_default_viewports(env.sim.stage)
    for index in env_indices(env, env_ids).tolist():
        apply_scene_lighting(env.sim.stage, _content_path(env, index, scene_spec), scene_spec.lighting)


def set_robot_materials(env, env_ids, scene_spec):
    arm = env.cfg.arm_spec
    for index in env_indices(env, env_ids).tolist():
        apply_robot_appearance(env.sim.stage, env.scene.env_prim_paths[index] + "/Robot", arm, scene_spec.appearance)


def _visual_term(sensor, kind):
    return ObsTerm(func=image_rgb_raw if kind == "rgb" else image_raw,
                   params={"sensor_cfg": SceneEntityCfg(sensor), "data_type": kind})


@configclass
class VisualObservationsCfg(ObsGroup):
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


def _add_camera(cfg, name, camera, renderer, used_names, *, parent="", response=None):
    sensor_name = "camera_" + name
    if sensor_name in used_names:
        raise ValueError(f"Camera name {name!r} collides with an existing scene entity")
    used_names.add(sensor_name)
    prim_name = getattr(camera, "prim_name", None) or name
    path = "{ENV_REGEX_NS}/" + (parent.strip("/") + "/" if parent else "") + prim_name
    sensor = build_camera_cfg(path, camera, renderer, response=response)
    setattr(cfg.scene, sensor_name, sensor)
    outputs = {kind: sensor_name for kind in camera.data_types}
    if getattr(camera, "depth_sensor", False):
        depth_name = camera.depth_sensor_name or name + "_depth"
        if depth_name in used_names:
            raise ValueError(f"Depth sensor name {depth_name!r} collides with an existing scene entity")
        used_names.add(depth_name)
        depth_path = "{ENV_REGEX_NS}/" + (camera.depth_prim_name or name + "_depth")
        depth = build_camera_cfg(depth_path, camera, renderer, data_types=["depth"], response=response,
                                 display_name=camera.metadata.get("depth_display_name"))
        setattr(cfg.scene, depth_name, depth)
        outputs["depth"] = depth_name
    # Keep policy RGB, depth and segmentation order stable across camera declarations.
    priority = ("rgb", "depth", "instance_id_segmentation_fast")
    ordered = [kind for kind in priority if kind in outputs] + [kind for kind in outputs if kind not in priority]
    for kind in ordered:
        prefix = "instance_id_seg" if kind == "instance_id_segmentation_fast" else kind
        setattr(cfg.observations.visual, prefix + "_" + name, _visual_term(outputs[kind], kind))
    return sensor


def _apply_physics(cfg, values):
    for key, value in values.items():
        if key == "solver":
            for name, setting in value.items():
                if not hasattr(cfg.sim.physics.solver_cfg, name):
                    raise ValueError(f"Unsupported solver setting {name!r}")
                setattr(cfg.sim.physics.solver_cfg, name, setting)
        elif key in ("dt", "render_interval"):
            setattr(cfg.sim, key, value)
        elif hasattr(cfg.sim.physics, key):
            setattr(cfg.sim.physics, key, value)
        elif hasattr(cfg.sim.physics.solver_cfg, key):
            setattr(cfg.sim.physics.solver_cfg, key, value)
        else:
            raise ValueError(f"Unsupported scene physics setting {key!r}")


def _customize(cfg, scene):
    if not scene.hook:
        return
    filename, attribute = scene.hook.split(":", 1)
    path = (scene.directory / filename).resolve()
    module_name = "_s2r_scene_hook_" + hashlib.sha256(str(path).encode()).hexdigest()
    if module_name not in sys.modules:
        definition = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(definition)
        sys.modules[module_name] = module
        try:
            definition.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    function = getattr(sys.modules[module_name], attribute)
    if not callable(function):
        raise TypeError(f"Scene hook {scene.hook!r} is not callable")
    function(cfg, scene)


@configclass
class SceneEnvCfg(BaseEnvCfg):
    scene_spec: SceneSpec | None = None

    def __post_init__(self):
        super().__post_init__()
        scene = self.scene_spec or load_scene_spec()
        self.scene_spec = scene
        arm = self.arm_spec
        mount = load_mount_spec(scene, arm)
        pose = mount_pose(scene, mount, arm)
        self.scene.robot.init_state.pos = pose.position
        self.scene.robot.init_state.rot = pose.orientation_xyzw
        self.scene.robot.init_state.joint_pos = mount_joint_positions(arm, mount)
        self.scene.room = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Room",
            spawn=SceneRoomUsdCfg(func=spawn_scene_room, usd_path=str(scene.room_usd), scene_spec=scene),
        )
        self.observations.visual = VisualObservationsCfg()
        renderer = build_renderer_cfg(scene.renderer)
        cameras = {}
        sensor_names = set()
        for name, camera in scene.cameras.items():
            response = scene.appearance.get("camera_response", {}).get(name)
            cameras[name] = _add_camera(self, name, camera, renderer, sensor_names, response=response)
        for name, camera in arm.cameras.items():
            if name in cameras:
                raise ValueError(f"Scene and arm both declare camera {name!r}")
            cameras[name] = _add_camera(self, name, camera, renderer, sensor_names, parent="Robot/" + camera.parent_link)
        _apply_physics(self, scene.physics)
        self.events.cache_static_geometry = EventTerm(func=cache_scene_geometry, mode="startup")
        self.events.configure_scene_updates = EventTerm(
            func=configure_scene_updates, mode="startup", params={"scene_spec": scene},
        )
        self.events.set_robot_visual_material = EventTerm(
            func=set_robot_materials, mode="startup", params={"scene_spec": scene},
        )
        self.events.reset_set_robot_visual_material = EventTerm(
            func=set_robot_materials, mode="reset", params={"scene_spec": scene},
        )
        viewer = scene.viewer
        if viewer.camera is None:
            visualizer = KitVisualizerCfg(eye=viewer.eye, lookat=viewer.lookat)
        else:
            if viewer.camera not in cameras:
                raise ValueError(f"Viewer camera {viewer.camera!r} is not declared by scene or arm")
            camera = cameras[viewer.camera]
            visualizer = CalibratedKitVisualizerCfg(
                camera_prim_path=camera.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0"),
                window_width=camera.width, window_height=camera.height,
                eye=viewer.eye, lookat=viewer.lookat,
            )
        self.sim.default_visualizer_cfg = visualizer
        self.sim.visualizer_cfgs = [visualizer]
        _customize(self, scene)


# configclass installs its own exporter after the class body has been processed.
SceneEnvCfg.to_dict = _config_to_dict


def build_scene_cfg(scene: str | SceneSpec | None = None, arm: ArmSpec | None = None):
    spec = scene if isinstance(scene, SceneSpec) else load_scene_spec(scene)
    return SceneEnvCfg(scene_spec=spec, arm_spec=arm)


def drop_unused_camera_outputs(cfg, depth=False, instance_id_seg=False):
    """Disable unconsumed outputs on any declared fixed or arm camera."""
    def keep(kind):
        return (depth or kind != "depth") and (instance_id_seg or kind != "instance_id_segmentation_fast")

    for name, sensor in list(vars(cfg.scene).items()):
        if sensor is not None and hasattr(sensor, "data_types"):
            sensor.data_types = [kind for kind in sensor.data_types if keep(kind)]
            if not sensor.data_types:
                setattr(cfg.scene, name, None)
    visual = getattr(cfg.observations, "visual", None)
    if visual is not None:
        for name, term in list(vars(visual).items()):
            if isinstance(term, ObsTerm) and not keep(term.params.get("data_type")):
                setattr(visual, name, None)
