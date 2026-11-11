"""Shared setup for the recording agents: signals, serial ports, real cameras, recorders, cleanup.

Takes plain values so the same code serves CLI and
YAML configs. Device connection lives in ``sim.teleop.devices``. LeRobot imports are
delayed until a recorder is built (after the Isaac app has launched).
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
from pathlib import Path


def exit_cleanly_on_termination() -> None:
    """Route Docker stop and terminal hangup through the main loop's finalizers."""

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, interrupt)


def validate_serial_ports(ports: dict[str, str]) -> None:
    """Check device paths without opening ports or commanding either arm.

    ``ports`` maps role ("leader", "follower") to its serial path. Raises ValueError.
    """
    resolved = {}
    for role, port in ports.items():
        path = Path(port)
        if not path.exists():
            available = sorted(
                str(p) for pattern in ("ttyACM*", "ttyUSB*", "serial/by-id/*")
                for p in Path("/dev").glob(pattern) if p.exists()
            )
            raise ValueError(
                f"Physical {role} serial port {port!r} does not exist. "
                f"Available serial paths: {', '.join(available) or 'none'}. "
                f"Reconnect the {role} USB cable and set its port to the verified device path "
                "(prefer /dev/serial/by-id/...). In Docker, make sure the device is visible inside "
                "the container. Ports are not reassigned between leader and follower."
            )
        if not stat.S_ISCHR(path.stat().st_mode):
            raise ValueError(f"Physical {role} port {port!r} is not a serial device.")
        if not os.access(path, os.R_OK | os.W_OK):
            raise ValueError(f"Physical {role} port {port!r} requires read/write permission.")
        resolved[role] = os.path.realpath(port)
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"Leader and follower serial ports must be different: {ports}")


def derived_real_repo_id(sim_repo_id: str) -> str:
    """``user/name`` -> ``user/name-real``."""
    if "/" in sim_repo_id:
        namespace, name = sim_repo_id.rsplit("/", 1)
        return f"{namespace}/{name}-real"
    return f"{sim_repo_id}-real"


def derived_real_root(sim_root: str) -> str:
    """``.../name`` -> ``.../name-real``."""
    path = Path(sim_root)
    return os.fspath(path.with_name(f"{path.name}-real"))


def prepare_recording_roots(roots: list[str], contract: dict, *, on_mismatch: str = "error"):
    """Recover and validate a dataset pair, selecting a calibration suffix when requested.

    Selection never creates directories or rewrites alignment markers. The recorder owns
    initialization, and a repeated launch with the same contract selects the same pair.
    """
    from core.datasets.alignment import RecordingAlignmentError, validate_recording_root
    from .rerecording import recover_replacement

    if on_mismatch not in ("error", "new"):
        raise ValueError(f"Unknown recording alignment policy: {on_mismatch!r}")
    if not roots or len({Path(root).resolve() for root in roots}) != len(roots):
        raise ValueError("Recording roots must be nonempty and distinct.")

    def validate(candidates):
        # Recover both sides of an interrupted paired publication before inspecting either.
        recover_replacement(candidates[0])
        for root in candidates:
            validate_recording_root(root, contract)

    try:
        validate(roots)
        return roots, ""
    except RecordingAlignmentError:
        if on_mismatch == "error":
            raise

    suffix = "-cal-" + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    candidates = [os.fspath(Path(root).with_name(Path(root).name + suffix)) for root in roots]
    # A conflicting candidate is an error: never relabel or overwrite existing recordings.
    validate(candidates)
    return candidates, suffix


def real_camera_specs(cameras: dict[str, str | int], *, fps: int, width: int, height: int,
                      fourcc: str | None = None) -> dict:
    """Physical camera specs for the follower, one entry per ``name: index_or_path``.

    v1 had two fixed names (``gripper``, ``external``); names now come from the config.
    The result feeds ``sim.teleop.devices.make_camera_configs``.
    """
    return {
        name: {"fps": fps, "width": width, "height": height, "fourcc": fourcc, "index_or_path": index_or_path}
        for name, index_or_path in cameras.items()
        if index_or_path not in (None, "")
    }


def cleanup(label: str, close) -> None:
    """Attempt every cleanup operation, including after an interrupted save."""
    try:
        close()
    except BaseException as exc:
        print(f"[ERROR]: {label}: {type(exc).__name__}: {exc}")


def disconnect(device, label: str) -> None:
    """Disconnect a ``sim.teleop.devices.Device`` (None is a no-op)."""
    if device is None:
        return
    try:
        device.disconnect()
        print(f"[INFO]: {label} disconnected.")
    except Exception as exc:
        print(f"[ERROR]: Failed to disconnect {label}: {exc}")


def make_recorders(
    *,
    task_name: str,
    repo_id: str,
    repo_root: str,
    fps: int,
    device: str,
    sim_cameras: dict,
    follower=None,
    real_repo_id: str | None = None,
    real_repo_root: str | None = None,
    real_cameras: dict | None = None,
    save_mp4: bool = False,
    depth: bool = False,
    instance_id_seg: bool = False,
    image_writer_threads_per_camera: int = 4,
    arm=None,
):
    """Build the synchronized recorders; the caller calls ``init_datasets()``.

    Returns ``(recorders, real_pipeline)``. ``real_pipeline`` is the
    ``RealRecordingPipeline`` for ``sim.teleop.devices.real_dataset_frame`` and is
    None when there is no physical follower.
    """
    from .lerobot_recorder import LeRobotRecorder, SynchronizedLeRobotRecorders
    from core.arms.spec import load_arm_spec

    arm = arm or load_arm_spec()[0]

    common = {
        "task_name": task_name,
        "fps": fps,
        "device": device,
        "arm": arm,
        "image_writer_threads_per_camera": image_writer_threads_per_camera,
    }
    recorders = {"sim": LeRobotRecorder(
        repo_id=repo_id,
        dataset_root=repo_root,
        cameras=sim_cameras,
        save_mp4=save_mp4,
        depth=depth,
        instance_id_seg=instance_id_seg,
        **common,
    )}

    real_pipeline = None
    if follower is not None:
        from sim.teleop.devices import real_recording_features

        real_root = real_repo_root or derived_real_root(repo_root)
        if Path(real_root).resolve() == Path(repo_root).resolve():
            raise ValueError("Real and simulation dataset roots must differ.")
        real_pipeline = real_recording_features(follower, use_videos=True)
        recorders["real"] = LeRobotRecorder(
            repo_id=real_repo_id or derived_real_repo_id(repo_id),
            dataset_root=real_root,
            cameras=real_cameras or {},
            features=real_pipeline.features,
            # LeRobot 0.6.1 names every SO follower "so_follower"; keep the arm.toml type (v1: so101_follower).
            robot_type=arm.lerobot.follower_type,
            **common,
        )
    return SynchronizedLeRobotRecorders(recorders), real_pipeline
