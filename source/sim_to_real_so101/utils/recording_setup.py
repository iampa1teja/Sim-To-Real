"""Shared setup for manual SO-101 recorder scripts.

Imports that depend on simulation or LeRobot are delayed until after app launch.
"""

import os
from pathlib import Path
import signal
import stat


def exit_cleanly_on_termination():
    """Route Docker stop and terminal hangup through the main loop's finalizers."""
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, interrupt)


def validate_serial_ports(args, parser):
    """Check device paths without opening ports or commanding either arm."""
    ports = [("leader", "--port / TELEOP_PORT", args.port)]
    if args.enable_real_follower:
        ports.append(("follower", "--follower_port / ROBOT_PORT", args.follower_port))
    for role, option, port in ports:
        path = Path(port)
        if not path.exists():
            available = sorted(
                str(p) for pattern in ("ttyACM*", "ttyUSB*", "serial/by-id/*")
                for p in Path("/dev").glob(pattern) if p.exists()
            )
            parser.error(
                f"Physical {role} serial port {port!r} does not exist. "
                f"Available serial paths: {', '.join(available) or 'none'}. "
                f"Reconnect the {role} USB cable and set {option} to its verified "
                "device path (prefer /dev/serial/by-id/...). In Docker, ensure "
                "the device is visible inside the container. Ports are not "
                "automatically reassigned between leader and follower."
            )
        if not stat.S_ISCHR(path.stat().st_mode):
            parser.error(f"Physical {role} port {port!r} is not a serial device.")
        if not os.access(path, os.R_OK | os.W_OK):
            parser.error(f"Physical {role} port {port!r} requires read/write permission.")


def _derived_real_repo_id(sim_repo_id: str) -> str:
    if "/" in sim_repo_id:
        namespace, name = sim_repo_id.rsplit("/", 1)
        return f"{namespace}/{name}-real"
    return f"{sim_repo_id}-real"


def _derived_real_root(sim_root: str) -> str:
    path = Path(sim_root)
    return os.fspath(path.with_name(f"{path.name}-real"))


def real_camera_specs(args) -> dict:
    cameras = {}
    common = {
        "fps": args.fps,
        "width": args.real_camera_width,
        "height": args.real_camera_height,
        "fourcc": args.real_camera_fourcc,
    }
    if args.real_gripper_camera:
        cameras["gripper"] = {
            **common,
            "index_or_path": args.real_gripper_camera,
        }
    if args.real_external_camera:
        cameras["external"] = {
            **common,
            "index_or_path": args.real_external_camera,
        }
    return cameras


def cleanup(label, close):
    """Attempt every cleanup operation, including after an interrupted save."""
    try:
        close()
    except BaseException as exc:
        print(f"[ERROR]: {label}: {type(exc).__name__}: {exc}")


def disconnect(interface, label: str) -> None:
    if interface is None:
        return
    try:
        interface.disconnect()
        print(f"[INFO]: {label} disconnected.")
    except Exception as exc:
        print(f"[ERROR]: Failed to disconnect {label}: {exc}")


def connect_interface(args, device, cameras, kind, visualize=False, joint_mapping=None):
    """Connect a leader/follower, cleaning partial resources on failure."""
    from .lerobot_interface import LeRobotSO101Interface

    follower = kind == "follower"
    interface = LeRobotSO101Interface(
        device=device,
        port=args.follower_port if follower else args.port,
        id=args.follower_id if follower else args.robot_id,
        cameras=cameras,
        fps=args.fps,
        kind=kind,
        joint_mapping=joint_mapping,
    )
    try:
        interface.init_device(visualize=visualize)
        interface.connect()
    except BaseException:
        cleanup(f"Partial {kind} disconnect", interface.disconnect)
        raise
    print(f"[INFO]: Physical {kind} connected: port={interface.port}, id={interface.id}")
    return interface


def make_recorders(args, device, sim_cameras, follower, real_cameras, batch_encoding_size=1):
    """Build synchronized recorder owners; the caller initializes their datasets."""
    from .lerobot_recorder import LeRobotRecorder, SynchronizedLeRobotRecorders

    common = {
        "task_name": args.task_name,
        "fps": args.fps,
        "device": device,
        "image_writer_threads_per_camera": args.image_writer_threads_per_camera,
        "batch_encoding_size": batch_encoding_size,
    }
    recorders = {"sim": LeRobotRecorder(
        repo_id=args.repo_id,
        dataset_root=args.repo_root,
        cameras=sim_cameras,
        save_mp4=args.save_mp4,
        depth=args.depth,
        instance_id_seg=args.instance_id_seg,
        **common,
    )}
    if follower is not None:
        real_root = args.real_repo_root or _derived_real_root(args.repo_root)
        if Path(real_root).resolve() == Path(args.repo_root).resolve():
            raise ValueError("Real and simulation dataset roots must differ.")
        recorders["real"] = LeRobotRecorder(
            repo_id=args.real_repo_id or _derived_real_repo_id(args.repo_id),
            dataset_root=real_root,
            cameras=real_cameras,
            features=follower.make_recording_features(use_videos=True),
            robot_type=follower.robot.name,
            **common,
        )
    return SynchronizedLeRobotRecorders(recorders)
