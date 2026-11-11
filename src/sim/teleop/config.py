"""
Arm, scene and task are chosen here and exported as S2R_ARM / S2R_SCENE / S2R_TASK by
``apply_selection``, which must run before anything imports ``core`` or ``sim`` modules (they read
those variables at import time). The env opened is S2R-<task>-<variant> (sim.envs registers it);
an empty task opens the bare scene, S2R-<scene>.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Literal

from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from sim.app import add_launch_arguments


def _env(name: str, default: str | None = None):
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass
class LeaderConfig:
    port: str = _env("TELEOP_PORT", "/dev/ttyACM0")
    id: str = _env("TELEOP_ID", "leader_arm_1")


@dataclass
class FollowerConfig:
    # False keeps sim-only teleop; the physical follower and its cameras stay closed.
    enabled: bool = False
    port: str = _env("ROBOT_PORT", "/dev/ttyACM1")
    id: str = _env("ROBOT_ID", "follower_arm_1")
    unit: str | None = _env("S2R_UNIT")
    # name -> OpenCV index or /dev/video path. Opened only when recording a real dataset.
    cameras: dict[str, str] = field(default_factory=lambda: {
        name: value for name, value in (("gripper", os.getenv("CAMERA_GRIPPER")),
                                        ("external", os.getenv("CAMERA_EXTERNAL"))) if value
    })
    camera_width: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "640")))
    camera_height: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "480")))
    camera_fourcc: str | None = _env("CAMERA_FOURCC")


@dataclass
class DatasetConfig:
    # Recording is on when repo_id, root and task_name are all set.
    repo_id: str | None = None
    root: str | None = None
    task_name: str | None = None
    # Real-follower dataset; defaults to "<repo_id>-real" / "<root>-real".
    real_repo_id: str | None = None
    real_root: str | None = None
    on_alignment_mismatch: Literal["error", "new"] = "error"
    save_mp4: bool = False
    depth: bool = False
    instance_id_seg: bool = False
    image_writer_threads_per_camera: int = 4

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id and self.root and self.task_name)


@dataclass
class TelemetryConfig:
    # New JSONL path for passive follower / mapped / sim joint-angle telemetry; needs the follower.
    joint_angle_log: str | None = None


@dataclass
class GuiConfig:
    # v1 browser recorder at http://<host>:<port>/ (sim and real camera panes, start/save/discard/spawn,
    # re-record a saved episode). Kept until the new GUI replaces it.
    enabled: bool = False
    # 0.0.0.0 to reach it from another machine.
    host: str = "127.0.0.1"
    port: int = 8765
    # Preview refresh limit, not the dataset fps; 0 means every tick.
    fps: float = 15.0


@dataclass
class AgentConfig:
    arm: str = _env("S2R_ARM", DEFAULT_ARM)
    scene: str = _env("S2R_SCENE", DEFAULT_SCENE)
    task: str = _env("S2R_TASK", DEFAULT_TASK)
    variant: str | None = None
    num_envs: int | None = None
    use_fabric: bool = True
    seed: int = 101
    fps: int = field(default_factory=lambda: int(os.getenv("CONTROL_FPS", "30")))
    # False connects both arms with calibration refused (v1 --no_calibration).
    calibrate: bool = True
    control_input: Literal["terminal", "keyboard"] = "terminal"
    # Local spawn.py listener; 0 disables it.
    spawn_port: int = 6001
    spawn_authkey: str = "so101-spawn"
    # Existing scene entity placed at the simulated gripper when an episode starts.
    sim_reset_object_name: str | None = None
    sim_reset_object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    leader: LeaderConfig = field(default_factory=LeaderConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)

    def __post_init__(self):
        """Cross-field checks (v1 lerobot_agent 142-157)."""
        ds = self.dataset
        if any((ds.repo_id, ds.root, ds.task_name)) and not ds.enabled:
            raise ValueError("dataset.repo_id, dataset.root and dataset.task_name must be set together.")
        if (ds.real_repo_id or ds.real_root) and not ds.enabled:
            raise ValueError("dataset.real_repo_id / dataset.real_root require the base dataset settings.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if ds.image_writer_threads_per_camera < 0:
            raise ValueError("dataset.image_writer_threads_per_camera cannot be negative.")
        if self.follower.enabled and os.path.realpath(self.leader.port) == os.path.realpath(self.follower.port):
            raise ValueError("Leader and follower serial ports must be different.")
        if self.telemetry.joint_angle_log and not self.follower.enabled:
            raise ValueError("telemetry.joint_angle_log requires follower.enabled.")
        if self.gui.enabled and not 0 < self.gui.port < 65536:
            raise ValueError("gui.port must be between 1 and 65535.")
        if self.gui.fps < 0:
            raise ValueError("gui.fps cannot be negative.")
        if self.num_envs not in (None, 1):
            raise ValueError("The teleop agents drive one environment; num_envs must be 1.")

    @property
    def recording(self) -> bool:
        return self.dataset.enabled


def parse_config(argv: list[str] | None = None) -> tuple[AgentConfig, argparse.Namespace]:
    """Parse ``(AgentConfig, isaac_args)``.

    AppLauncher flags are split off first with ``parse_known_args``; everything else goes to
    draccus, which also handles ``--config_path``.
    """
    import draccus
    parser = argparse.ArgumentParser(add_help=False)
    add_launch_arguments(parser)
    isaac_args, rest = parser.parse_known_args(argv)
    config_path = None
    if "--config_path" in rest:
        index = rest.index("--config_path")
        config_path = rest[index + 1]
        rest = rest[:index] + rest[index + 2:]
    else:
        for item in rest:
            if item.startswith("--config_path="):
                config_path = item.split("=", 1)[1]
                rest = [arg for arg in rest if arg is not item]
                break
    cfg = draccus.parse(config_class=AgentConfig, config_path=config_path, args=rest)
    # Every teleop agent renders the cameras it records or streams.
    isaac_args.enable_cameras = True
    return cfg, isaac_args
