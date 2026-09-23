import argparse
import json
import os
import stat
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 pick-and-place agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Pick-Place")

# Leader
parser.add_argument("--port", type=str, default=os.getenv("TELEOP_PORT", "/dev/ttyACM0"))
parser.add_argument("--robot_id", type=str, default=os.getenv("TELEOP_ID", "leader_arm_1"))

# Follower
parser.add_argument("--enable_real_follower", action="store_true", default=False)
parser.add_argument("--follower_port", type=str, default=os.getenv("ROBOT_PORT", "/dev/ttyACM1"))
parser.add_argument("--follower_id", type=str, default=os.getenv("ROBOT_ID", "follower_arm_1"))

# Dataset
parser.add_argument("--repo_id", type=str, default=None)
parser.add_argument("--repo_root", type=str, default=None)
parser.add_argument("--task_name", type=str, default=None)
parser.add_argument("--real_repo_id", type=str, default=None)
parser.add_argument("--real_repo_root", type=str, default=None)
parser.add_argument("--save_mp4", action="store_true", default=False)
parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--instance_id_seg", action="store_true", default=False)
parser.add_argument("--image_writer_threads_per_camera", type=int, default=4)

# Real cameras
parser.add_argument("--real_gripper_camera", type=str, default=os.getenv("CAMERA_GRIPPER"))
parser.add_argument("--real_external_camera", type=str, default=os.getenv("CAMERA_EXTERNAL"))
parser.add_argument("--real_camera_width", type=int, default=int(os.getenv("CAMERA_WIDTH", "640")))
parser.add_argument("--real_camera_height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "480")))
parser.add_argument("--real_camera_fourcc", type=str, default=os.getenv("CAMERA_FOURCC") or None)

parser.add_argument("--fps", type=int, default=int(os.getenv("CONTROL_FPS", "30")))
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")

# Countdown duration (seconds) between env.reset() and start of recording
parser.add_argument("--reset_wait_s", type=int, default=10,
                    help="Seconds to wait after env.reset() before recording begins.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

dataset_args = (args_cli.repo_id, args_cli.repo_root, args_cli.task_name)
if any(dataset_args) and not all(dataset_args):
    parser.error("--repo_id, --repo_root, and --task_name must be supplied together.")
if (args_cli.real_repo_id or args_cli.real_repo_root) and not all(dataset_args):
    parser.error("Real dataset overrides require the base dataset arguments.")
if args_cli.fps <= 0:
    parser.error("--fps must be positive.")
if args_cli.enable_real_follower and (
    os.path.realpath(args_cli.port) == os.path.realpath(args_cli.follower_port)
):
    parser.error("Leader and follower serial ports must be different.")


# ---------------------------------------------------------------------------
# Serial port pre-flight
# ---------------------------------------------------------------------------
def _validate_serial_ports():
    ports = [("leader", "--port / TELEOP_PORT", args_cli.port)]
    if args_cli.enable_real_follower:
        ports.append(("follower", "--follower_port / ROBOT_PORT", args_cli.follower_port))
    for role, option, port in ports:
        path = Path(port)
        if not path.exists():
            available = sorted(
                str(p) for pattern in ("ttyACM*", "ttyUSB*", "serial/by-id/*")
                for p in Path("/dev").glob(pattern) if p.exists()
            )
            parser.error(
                f"Physical {role} serial port {port!r} does not exist. "
                f"Available: {', '.join(available) or 'none'}. Set {option}."
            )
        if not stat.S_ISCHR(path.stat().st_mode):
            parser.error(f"Physical {role} port {port!r} is not a serial device.")
        if not os.access(path, os.R_OK | os.W_OK):
            parser.error(f"Physical {role} port {port!r} requires read/write permission.")


_validate_serial_ports()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import parse_env_cfg
from isaacsim.core.prims import XFormPrim

import sim_to_real_so101.tasks  # noqa: F401
from sim_to_real_so101.utils.keyboard import KeyboardControl
from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
from sim_to_real_so101.utils.lerobot_recorder import (
    LeRobotRecorder,
    SynchronizedLeRobotRecorders,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disconnect(interface: LeRobotSO101Interface | None, label: str) -> None:
    if interface is None:
        return
    try:
        interface.disconnect()
        print(f"[INFO]: {label} disconnected.")
    except Exception as exc:
        print(f"[ERROR]: Failed to disconnect {label}: {exc}")


def _cube_pose_relative_to_base(env) -> tuple[list, list]:
    """Return blue cube pose expressed in robot-base frame."""
    scene = env.unwrapped.scene
    cube = scene["blue_cube"]
    robot = scene["robot"]

    cube_pos_w  = cube.data.root_pos_w    # (1, 3)
    cube_quat_w = cube.data.root_quat_w   # (1, 4) wxyz

    robot_pos_w  = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    pos_base, quat_base = math_utils.subtract_frame_transforms(
        robot_pos_w, robot_quat_w, cube_pos_w, cube_quat_w
    )
    return pos_base[0].tolist(), quat_base[0].tolist()


def _save_episode_object_pose(
    repo_root: str,
    episode_index: int,
    position: list,
    orientation: list,
) -> None:
    """Write cube pose JSON sidecar at <repo_root>/pick_place_meta/episode_N.json."""
    meta_dir = Path(repo_root) / "pick_place_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sidecar = meta_dir / f"episode_{episode_index:06d}.json"
    sidecar.write_text(
        json.dumps({
            "episode_index": episode_index,
            "cube_position_robot_base": position,
            "cube_orientation_robot_base_wxyz": orientation,
        }, indent=2)
    )
    print(f"[INFO]: Cube pose saved → {sidecar}")


def _log_rerun_frame(visual_obs: dict, real_observation: dict | None, action: dict) -> None:
    """Stream 4-camera frame to Rerun viewer."""
    from lerobot.utils.visualization_utils import log_rerun_data
    observation = {
        "sim_rgb":   visual_obs.get("rgb_realsense_rgb"),
        "sim_wrist": visual_obs.get("rgb_wrist_cam"),
    }
    if real_observation is not None:
        # Raw follower observation dict is keyed by camera name directly
        # (see _real_camera_specs()), not by the "observation.images.*"
        # prefix used only on the processed LeRobot dataset frame.
        observation["real_rgb"]   = real_observation.get("external")
        observation["real_wrist"] = real_observation.get("gripper")
    log_rerun_data(observation=observation, action=action)


def _derived_real_repo_id(sim_repo_id: str) -> str:
    if "/" in sim_repo_id:
        namespace, name = sim_repo_id.rsplit("/", 1)
        return f"{namespace}/{name}-real"
    return f"{sim_repo_id}-real"


def _derived_real_root(sim_root: str) -> str:
    path = Path(sim_root)
    return os.fspath(path.with_name(f"{path.name}-real"))


def _real_camera_specs() -> dict:
    cameras = {}
    common = {
        "fps": args_cli.fps,
        "width": args_cli.real_camera_width,
        "height": args_cli.real_camera_height,
        "fourcc": args_cli.real_camera_fourcc,
    }
    if args_cli.real_gripper_camera:
        cameras["gripper"] = {**common, "index_or_path": args_cli.real_gripper_camera}
    if args_cli.real_external_camera:
        cameras["external"] = {**common, "index_or_path": args_cli.real_external_camera}
    return cameras


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    keyboard_control = None
    env = None
    leader_iface = None
    follower_iface = None
    recorder_group = None
    recording = False

    try:
        # ── Env ───────────────────────────────────────────────────────────
        use_fabric = not args_cli.disable_fabric
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=use_fabric,
        )
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        # ── Sim camera discovery ──────────────────────────────────────────
        sim_cameras = {}
        for obj in env.unwrapped.scene.keys():
            if obj.startswith("camera_"):
                camera_cfg = getattr(env.unwrapped.scene.cfg, obj)
                sim_cameras[obj.replace("camera_", "")] = {
                    "height": camera_cfg.height,
                    "width": camera_cfg.width,
                }
                print(f"[INFO]: Sim camera: {obj.replace('camera_', '')}")

        # ── Leader ────────────────────────────────────────────────────────
        leader_iface = LeRobotSO101Interface(
            device=env.unwrapped.device,
            port=args_cli.port,
            id=args_cli.robot_id,
            cameras=sim_cameras,
            fps=args_cli.fps,
            kind="leader",
        )
        leader_iface.init_device(visualize=args_cli.rerun)
        leader_iface.connect()
        print(f"[INFO]: Leader connected: port={args_cli.port}, id={args_cli.robot_id}")

        # ── Follower (optional) ───────────────────────────────────────────
        recording_mode = bool(all(dataset_args))
        real_cameras = _real_camera_specs() if recording_mode else {}

        if args_cli.enable_real_follower:
            follower_iface = LeRobotSO101Interface(
                device=env.unwrapped.device,
                port=args_cli.follower_port,
                id=args_cli.follower_id,
                cameras=real_cameras,
                fps=args_cli.fps,
                kind="follower",
            )
            follower_iface.init_device()
            try:
                follower_iface.connect()
            except Exception as exc:
                _disconnect(follower_iface, "partially connected follower")
                raise RuntimeError(f"Follower connection failed: {exc}") from exc
            print("[INFO]: Follower connected.")

        # ── Recorders ────────────────────────────────────────────────────
        if recording_mode:
            sim_recorder = LeRobotRecorder(
                task_name=args_cli.task_name,
                repo_id=args_cli.repo_id,
                dataset_root=args_cli.repo_root,
                fps=args_cli.fps,
                device=env.unwrapped.device,
                cameras=sim_cameras,
                save_mp4=args_cli.save_mp4,
                depth=args_cli.depth,
                instance_id_seg=args_cli.instance_id_seg,
                image_writer_threads_per_camera=args_cli.image_writer_threads_per_camera,
            )
            recorders = {"sim": sim_recorder}

            if follower_iface is not None:
                real_features  = follower_iface.make_recording_features(use_videos=True)
                real_repo_id   = args_cli.real_repo_id   or _derived_real_repo_id(args_cli.repo_id)
                real_repo_root = args_cli.real_repo_root or _derived_real_root(args_cli.repo_root)
                if Path(real_repo_root).resolve() == Path(args_cli.repo_root).resolve():
                    raise ValueError("Real and sim dataset roots must differ.")
                recorders["real"] = LeRobotRecorder(
                    task_name=args_cli.task_name,
                    repo_id=real_repo_id,
                    dataset_root=real_repo_root,
                    fps=args_cli.fps,
                    device=env.unwrapped.device,
                    cameras=real_cameras,
                    features=real_features,
                    robot_type=follower_iface.robot.name,
                    image_writer_threads_per_camera=args_cli.image_writer_threads_per_camera,
                )

            recorder_group = SynchronizedLeRobotRecorders(recorders)
            recorder_group.init_datasets()
            print(f"[INFO]: Recording ready — episode {recorder_group.episode_index}")
        else:
            print("[INFO]: Recording disabled. Supply --repo_id, --repo_root, --task_name.")

        # ── Per-tick state ────────────────────────────────────────────────
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        wrist_roll_index = 4
        leader_wrist_start = None
        sim_wrist_start    = 0.0

        keyboard_control = KeyboardControl()
        print("[controls] S: start/save  →: save  ←/C: discard  R: reset  Esc: quit")

        # ── Episode outer loop ────────────────────────────────────────────
        phase = "setup"   # "setup" | "resetting" | "recording"
        reset_deadline = None

        print("\n⏳  Setup the cube position")

        while simulation_app.is_running():
            loop_start = time.perf_counter()
            requests = keyboard_control.consume_requests()

            # ── Quit ──────────────────────────────────────────────────────
            if requests["stop"]:
                if recording and recorder_group is not None:
                    recorder_group.save_episode()
                    recording = False
                break

            # ── S pressed in setup phase → trigger env.reset() countdown ─
            if requests["start"] and phase == "setup":
                env.reset()
                leader_wrist_start = None
                phase = "resetting"
                reset_deadline = time.perf_counter() + args_cli.reset_wait_s
                keyboard_control.set_recording(False)
                print(f"\n🔄  Reset the env [{args_cli.reset_wait_s} SEC]")

            # ── Countdown expired → start recording ───────────────────────
            if phase == "resetting" and time.perf_counter() >= reset_deadline:
                phase = "recording"
                if recording_mode and recorder_group is not None:
                    cube_pos, cube_quat = _cube_pose_relative_to_base(env)
                    _save_episode_object_pose(
                        args_cli.repo_root,
                        recorder_group.episode_index,
                        cube_pos,
                        cube_quat,
                    )
                    recording = True
                    keyboard_control.set_recording(True)
                print(f"\n🎬  Recording episode {recorder_group.episode_index if recorder_group else '—'}…")

            # ── Save episode ──────────────────────────────────────────────
            if requests["end"] and phase == "recording":
                if recorder_group is not None:
                    recorder_group.save_episode()
                recording = False
                phase = "setup"
                keyboard_control.set_recording(False)
                print("\n⏳  Setup the cube position")

            # ── Discard / re-record ───────────────────────────────────────
            if requests["rerecord"] and phase == "recording":
                if recorder_group is not None:
                    recorder_group.cancel_episode()
                recording = False
                phase = "setup"
                keyboard_control.set_recording(False)
                print("\n⏳  Setup the cube position")

            # ── Mid-episode reset (save + go back to setup) ────────────────
            if requests["reset"] and phase == "recording":
                if recorder_group is not None:
                    recorder_group.save_episode()
                recording = False
                phase = "setup"
                keyboard_control.set_recording(False)
                print("\n⏳  Setup the cube position")

            # ── Control tick (runs in all phases) ─────────────────────────
            with torch.inference_mode():
                try:
                    leader_action = leader_iface.robot.get_action()
                except Exception as exc:
                    raise RuntimeError(f"Failed to read leader: {exc}") from exc

                if follower_iface is not None:
                    try:
                        follower_iface.robot.send_action(leader_action)
                    except Exception as exc:
                        raise RuntimeError(f"Follower action write failed: {exc}") from exc

                raw_action, mapped_action = leader_iface.real_to_sim_obs_processor(leader_action)

                # Wrist-roll re-zero
                if leader_wrist_start is None:
                    leader_wrist_start = mapped_action[wrist_roll_index].clone()
                mapped_action[wrist_roll_index] = (
                    sim_wrist_start
                    + mapped_action[wrist_roll_index]
                    - leader_wrist_start
                )
                mapped_action[wrist_roll_index] = torch.clamp(
                    mapped_action[wrist_roll_index],
                    leader_iface.joint_mins[wrist_roll_index] * torch.pi / 180,
                    leader_iface.joint_maxs[wrist_roll_index] * torch.pi / 180,
                )

                actions[:] = mapped_action
                try:
                    obs, _, _, _, _ = env.step(actions)
                except Exception as exc:
                    raise RuntimeError(f"env.step() failed: {exc}") from exc

                if recording and recorder_group is not None:
                    visual_obs = obs.get("visual")
                    if visual_obs is None:
                        raise RuntimeError("No 'visual' observation group for recording.")

                    joint_pos_obs = obs["policy"]["joint_pos_obs"][0]
                    (
                        sim_observation,
                        visual_buffers,
                        depth_buffers,
                        instance_id_seg_buffers,
                    ) = leader_iface.sim_to_real_dataset_processor(joint_pos_obs, visual_obs)

                    sim_recorder = recorder_group.recorders["sim"]
                    frames = {
                        "sim": sim_recorder.make_sim_frame(raw_action, sim_observation, visual_buffers)
                    }
                    auxiliary = {
                        "sim": sim_recorder.make_auxiliary_frame(
                            visual_buffers, depth_buffers, instance_id_seg_buffers
                        )
                    }

                    follower_observation = None
                    if follower_iface is not None:
                        try:
                            follower_observation = follower_iface.robot.get_observation()
                        except Exception as exc:
                            raise RuntimeError(f"Follower observation failed: {exc}") from exc
                        frames["real"] = follower_iface.make_real_dataset_frame(
                            leader_action, follower_observation, args_cli.task_name
                        )

                    recorder_group.add_frames(frames, auxiliary)

                    if args_cli.rerun:
                        _log_rerun_frame(visual_obs, follower_observation, leader_action)

            remaining = (1.0 / args_cli.fps) - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("[INFO]: Interrupted; shutting down cleanly.")
    except Exception:
        if recorder_group is not None and recording:
            recorder_group.cancel_episode()
            recording = False
        raise
    finally:
        if recorder_group is not None:
            try:
                if recorder_group.frame_count:
                    recorder_group.cancel_episode()
            except Exception as exc:
                print(f"[ERROR]: {exc}")
            try:
                recorder_group.finalize()
            except Exception as exc:
                print(f"[ERROR]: {exc}")

        if keyboard_control is not None:
            keyboard_control.cleanup()
        _disconnect(follower_iface, "physical follower")
        _disconnect(leader_iface, "physical leader")
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()