import os
import importlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import rerun as rr
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot.processor import make_default_processors
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.feature_utils import combine_feature_dicts,  build_dataset_frame
from lerobot.utils.constants import ACTION, OBS_STR

from core.arms.calibration import load_device_calibration, resolve_calibration
from core.arms.spec import load_arm_spec

def make_camera_configs(cameras: dict, fps: int,) -> dict:
    result = {}
    for name, camera in cameras.items():
        index_or_path = camera.get("index_or_path", Path("null"),)
        if isinstance(index_or_path, str):
            try:
                index_or_path = int(index_or_path)
            except ValueError:
                index_or_path = Path(index_or_path)

        result[name] = OpenCVCameraConfig(index_or_path=index_or_path, fps=fps, width=camera["width"], height=camera["height"], fourcc=camera.get("fourcc"),)
    return result


def make_device_config(kind: str, port: str, id: str, cameras: dict, fps: int, selection=None, options=None,):
    """Create a LeRobot 0.6.1 leader or follower config.

    The config class is resolved dynamically from the arm specification.
    No SO101-specific LeRobot config classes are imported here.
    ``options`` adds follower-config fields (e.g. max_relative_target, disable_torque_on_disconnect).
    """
    spec = getattr(selection, "arm", None) or load_arm_spec()[0]
    lerobot = spec.lerobot
    for module in lerobot.modules:
        importlib.import_module(module)

    common = {
        "port": port,
        "id": id,
        "use_degrees": lerobot.use_degrees,
    }

    path = getattr(selection, f"{kind}_path", None) if selection is not None else None
    if path is not None and lerobot.calibration == "per_unit":
        common["calibration_dir"] = path.parent

    if kind == "follower":
        config_class = RobotConfig.get_choice_class(lerobot.follower_type)
        supported = {item.name for item in fields(config_class)}
        common = {name: value for name, value in common.items() if name in supported}
        if "cameras" in supported:
            common["cameras"] = make_camera_configs(cameras, fps)
        elif cameras:
            raise ValueError(f"Follower {lerobot.follower_type!r} does not support cameras")
        unsupported = set(options or {}) - supported
        if unsupported:
            raise ValueError(f"Follower {lerobot.follower_type!r} does not support options {sorted(unsupported)}")
        return config_class(**common, **(options or {}))

    if kind == "leader":
        config_class = TeleoperatorConfig.get_choice_class(lerobot.leader_type)
        supported = {item.name for item in fields(config_class)}
        common = {name: value for name, value in common.items() if name in supported}
        return config_class(**common)

    raise ValueError(f"Unsupported device kind: {kind!r}")


class Device:
    def __init__(self, kind: str, cfg, selection=None,):
        if kind not in ("leader", "follower"):
            raise ValueError(f"Unsupported device kind: {kind!r}")
        self.kind = kind
        self.cfg = cfg
        self.selection = selection
        self.device = None


    def connect(self, calibrate: bool = False):
        if self.is_connected:
            return
        
        if self.kind == "leader":
            self.device = make_teleoperator_from_config(self.cfg)
        else:
            self.device = make_robot_from_config(self.cfg)

        if self.selection is None:
            self.selection = resolve_calibration(**{f"{self.kind}_id": self.cfg.id})

        path = getattr(self.selection, f"{self.kind}_path")
        if path is not None and self.selection.arm.lerobot.calibration == "per_unit":
            load_device_calibration(self.device, path)
        self.selection.verify_loaded(self.device, self.kind)

        if not calibrate:
            def refuse(*_args, **_kwargs):
                raise RuntimeError("Calibration is disabled for this session")
            self.device.calibrate = refuse
            if hasattr(self.device, "_save_calibration"):
                self.device._save_calibration = refuse
            bus = getattr(self.device, "bus", None)
            if bus is not None and hasattr(bus, "write_calibration"):
                bus.write_calibration = refuse
        self.device.connect(calibrate=calibrate)

        if not calibrate and not self.device.is_calibrated:
            raise RuntimeError(f"{self.kind} motor calibration does not match the selected file")


    @property
    def is_connected(self) -> bool:
        return bool(getattr(self.device, "is_connected", False,))


    def get_action(self, *args, **kwargs):
        if self.kind != "leader":
            raise RuntimeError("get_action() is only available on a leader.")

        if not self.is_connected:
            raise RuntimeError("Leader is not connected.")

        return self.device.get_action(*args, **kwargs,)


    def get_observation(self, *args, **kwargs):
        if self.kind != "follower":
            raise RuntimeError("get_observation() is only available on a follower.")

        if not self.is_connected:
            raise RuntimeError("Follower is not connected.")

        return self.device.get_observation(*args, **kwargs,)


    def send_action(self, action, *args, **kwargs):
        if self.kind != "follower":
            raise RuntimeError("send_action() is only available on a follower.")

        if not self.is_connected:
            raise RuntimeError("Follower is not connected.")

        return self.device.send_action(action,*args,**kwargs,)


    def disconnect(self) -> None:
        robot = self.device
        if robot is None:
            return

        if self.selection is not None and self.selection.arm.lerobot.calibration == "factory":
            if self.is_connected:
                robot.disconnect()
            return

        errors = []
        bus = getattr(robot, "bus", None)

        try:
            if bus is not None and bus.is_connected:
                if self.kind == "follower":
                    bus.disconnect(robot.config.disable_torque_on_disconnect)
                else:
                    bus.disconnect()
        except BaseException as exc:
            errors.append(exc)

        for camera in getattr(robot, "cameras", {},).values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"{self.kind.capitalize()} disconnection failed: {errors}") from errors[0]


def connect_device(kind: str, port: str, id: str, cameras: dict, fps: int, selection=None, calibrate: bool = False,
                   options=None,) -> Device:
    cfg = make_device_config(kind=kind, port=port, id=id, cameras=cameras, fps=fps, selection=selection, options=options,)
    device = Device(kind=kind, cfg=cfg, selection=selection,)

    try:
        device.connect(calibrate=calibrate)
    except BaseException:
        try:
            device.disconnect()
        except BaseException as cleanup_error:
            print(
                f"[ERROR]: Partial {kind} disconnect failed: "
                f"{type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )
        raise

    print(f"[INFO]: Physical {kind} connected: port={port}, id={id}")
    return device

@dataclass
class RealRecordingPipeline:
    teleop_action_processor: Any
    robot_action_processor: Any
    robot_observation_processor: Any
    features: dict


def real_recording_features(follower: Device, use_videos: bool = True,) -> RealRecordingPipeline:
    if follower.kind != "follower":
        raise RuntimeError("Real recording features require a follower.")

    if not follower.is_connected:
        raise RuntimeError("Follower must be connected before creating recording features.")

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = follower.device

    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features,
            ),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features,
            ),
            use_videos=use_videos,
        ),
    )

    return RealRecordingPipeline(
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        features=features,
    )


def real_dataset_frame(recording: RealRecordingPipeline, leader_action: dict, follower_observation: dict, task_name: str,) -> dict:
    processed_observation = ( recording.robot_observation_processor(follower_observation))
    processed_action = (recording.teleop_action_processor((leader_action, follower_observation)))
    observation_frame = build_dataset_frame(recording.features, processed_observation, prefix=OBS_STR,)
    action_frame = build_dataset_frame(recording.features, processed_action, prefix=ACTION,)

    return {
        **observation_frame,
        **action_frame,
        "task": task_name,
    }

def init_rerun(session_name: str):
    viewer_url = os.environ.get("RERUN_CONNECT_URL")
    rr.init(session_name)
    if viewer_url:
        rr.connect_grpc(viewer_url)
    return rr
