import argparse
from importlib import import_module
import os
import signal
import sys
import traceback
from isaaclab.app import AppLauncher
from sim_to_real_so101.utils.recording_setup import (
    cleanup as _cleanup,
    connect_interface,
    exit_cleanly_on_termination,
    disconnect as _disconnect,
    make_recorders,
    real_camera_specs,
    validate_serial_ports,
)

parser = argparse.ArgumentParser(description="SO-101 pick-and-place agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Lerobot-So101-Teleop-Pick-Place")
parser.add_argument("--control_input", choices=("lerobot", "terminal", "isaac"), default="lerobot",
                    help="Episode controls; LeRobot arrows by default, terminal fallback if unavailable.")

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
parser.add_argument(
    "--encode_every", type=int, default=0,
    help="Start background encoding every N saved episodes. "
         "Default 0 encodes on request or exit, so saving returns quickly. "
         "Recovered PNG episodes are encoded automatically when the dataset is reopened.",
)

# Real cameras
parser.add_argument("--real_gripper_camera", type=str, default=os.getenv("CAMERA_GRIPPER"))
parser.add_argument("--real_external_camera", type=str, default=os.getenv("CAMERA_EXTERNAL"))
parser.add_argument("--real_camera_width", type=int, default=int(os.getenv("CAMERA_WIDTH", "640")))
parser.add_argument("--real_camera_height", type=int, default=int(os.getenv("CAMERA_HEIGHT", "480")))
parser.add_argument("--real_camera_fourcc", type=str, default=os.getenv("CAMERA_FOURCC") or None)

parser.add_argument("--fps", type=int, default=int(os.getenv("CONTROL_FPS", "30")))
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--rerun", action="store_true", default=False, help="Enable Rerun visualization")

# Preparation countdown before recording; starting does not reset the scene.
parser.add_argument("--reset_wait_s", type=int, default=10,
                    help="Preparation seconds before recording begins (does not reset the scene).")

# Browser GUI (camera feeds + Start/Stop/Discard/Spawn), alongside --control_input.
parser.add_argument(
    "--gui_host", type=str, default="127.0.0.1",
    help="Interface for the recorder web GUI. It can drive the real robot, so keep "
         "it on localhost unless you trust the network.",
)
parser.add_argument("--gui_port", type=int, default=8765)
parser.add_argument("--gui_fps", type=float, default=15.0, help="Max camera refresh rate in the GUI.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

dataset_args = (args_cli.repo_id, args_cli.repo_root, args_cli.task_name)
if any(dataset_args) and not all(dataset_args):
    parser.error("--repo_id, --repo_root, and --task_name must be supplied together.")
if (args_cli.real_repo_id or args_cli.real_repo_root) and not all(dataset_args):
    parser.error("Real dataset overrides require the base dataset arguments.")
if args_cli.fps <= 0:
    parser.error("--fps must be positive.")
if args_cli.image_writer_threads_per_camera < 0:
    parser.error("--image_writer_threads_per_camera cannot be negative.")
if args_cli.reset_wait_s < 0:
    parser.error("--reset_wait_s cannot be negative.")
if args_cli.encode_every < 0:
    parser.error("--encode_every cannot be negative.")
if args_cli.enable_real_follower and (
    os.path.realpath(args_cli.port) == os.path.realpath(args_cli.follower_port)
):
    parser.error("Leader and follower serial ports must be different.")


# ---------------------------------------------------------------------------
# Serial port pre-flight
# ---------------------------------------------------------------------------
validate_serial_ports(args_cli, parser)
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------
try:
    import time

    import gymnasium as gym
    import torch

    import_module("isaaclab_tasks")  # Registers Isaac Lab tasks.
    import isaaclab.utils.math as math_utils
    from isaaclab_tasks.utils import parse_env_cfg

    import_module("sim_to_real_so101.tasks")  # Registers workshop tasks.
    from sim_to_real_so101.utils.episode_cube import EpisodeCube
    from sim_to_real_so101.utils.episode_metadata import save_cube_trajectory, validate_cube_trajectories
    from sim_to_real_so101.utils.recording_web_gui import RecordingWebGui

    try:
        from lerobot.utils.utils import say as _lerobot_say
    except ImportError:
        _lerobot_say = None

except BaseException:
    simulation_app.close()
    raise

# GUI pane titles, in display order (two per row).
SIM_RGB, SIM_WRIST, REAL_RGB, REAL_WRIST = "Sim: realsense", "Sim: wrist", "Real: external", "Real: gripper"
# GUI button -> episode request key used by every --control_input. The GUI's
# "stop" means stop *recording* (save), which is "end"; "stop" here means quit.
GUI_REQUESTS = {"start": "start", "stop": "end", "save": "end", "encode": "encode",
                "exit": "stop", "discard": "rerecord", "spawn": "spawn"}



def say(text: str, blocking: bool = False) -> None:
    """Use LeRobot speech when available; audio must not abort an episode."""
    global _lerobot_say
    if _lerobot_say is None:
        return
    try:
        _lerobot_say(text, blocking=blocking)
    except OSError as exc:
        # LeRobot delegates to spd-say on Linux; minimal containers may not
        # include it. Warn once and retain the normal on-screen status messages.
        print(f"[WARNING]: LeRobot speech unavailable ({exc}); continuing without audio.")
        _lerobot_say = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cube_pose_in_base(env, base_body_id: int) -> torch.Tensor:
    """Cube pose in the robot's base link frame (env 0) as (x, y, z, qw, qx, qy, qz)."""
    scene = env.unwrapped.scene
    cube, robot = scene["blue_cube"], scene["robot"]
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, base_body_id],
        robot.data.body_quat_w[:, base_body_id],
        cube.data.root_pos_w,
        cube.data.root_quat_w,
    )
    # Kept on-device; converted once at save to avoid a GPU sync every tick.
    return torch.cat([pos, quat], dim=-1)[0].clone()


def _save_episode(recorder_group, trajectory, cube_size):
    """Validate the sidecar timeline before committing either dataset."""
    count = recorder_group.frame_count
    if len(trajectory) != count:
        raise ValueError(f"Cube trajectory has {len(trajectory)} frames; datasets have {count}.")
    episode_index = recorder_group.save_episode()
    if episode_index is not None:
        poses = torch.stack(trajectory).cpu().tolist()
        save_cube_trajectory(args_cli.repo_root, episode_index, poses, cube_size, args_cli.fps)
    return episode_index


def _gui_frames(visual_obs: dict | None, real_observation: dict | None) -> dict:
    visual_obs = visual_obs or {}
    real_observation = real_observation or {}

    def first_env(key):
        images = visual_obs.get(key)
        return None if images is None else images[0]

    return {
        SIM_RGB: first_env("rgb_realsense_rgb"),
        SIM_WRIST: first_env("rgb_wrist_cam"),
        REAL_RGB: real_observation.get("external"),
        REAL_WRIST: real_observation.get("gripper"),
    }


def _gui_status(phase: str, reset_deadline: float | None, recorder_group, cube_visible: bool, note: str) -> str:
    """Status line for the GUI; ``note`` reports the last save/discard between episodes."""
    if phase == "resetting":
        return f"● Recording starts in {max(0.0, reset_deadline - time.perf_counter()):.0f} s"
    if phase == "recording":
        if recorder_group is None:
            return "● Episode active (dataset recording disabled)"
        seconds = recorder_group.frame_count / args_cli.fps
        return f"● Recording episode {recorder_group.episode_index}: {seconds:.1f} s"
    hint = "Start recording when ready." if cube_visible else "Spawn the cube, then start recording."
    return f"{note} {hint}" if note else hint


def _log_rerun_frame(visual_obs: dict, real_observation: dict | None, action: dict) -> None:
    """Stream 4-camera frame to Rerun viewer."""
    from lerobot.utils.visualization_utils import log_rerun_data
    observation = {
        "sim_rgb":   visual_obs.get("rgb_realsense_rgb"),
        "sim_wrist": visual_obs.get("rgb_wrist_cam"),
    }
    if real_observation is not None:
        # Raw follower observation dict is keyed by camera name directly
        # (see real_camera_specs(args_cli)), not by the "observation.images.*"
        # prefix used only on the processed LeRobot dataset frame.
        observation["real_rgb"]   = real_observation.get("external")
        observation["real_wrist"] = real_observation.get("gripper")
    observation = {
        key: (value[0].detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
        for key, value in observation.items() if value is not None
    }
    log_rerun_data(observation=observation, action=action)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main():
    # Physics buffers updated during stepping may be inference tensors. Keep
    # keyboard-driven spawn/reset/save transitions in the same context.
    keyboard_control = None
    gui = None
    env = None
    leader_iface = None
    follower_iface = None
    recorder_group = None
    recording = False
    exit_encode = True

    try:
        exit_cleanly_on_termination()
        # ── Env ───────────────────────────────────────────────────────────
        use_fabric = not args_cli.disable_fabric
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=use_fabric,
        )
        env_cfg.seed = args_cli.seed
        if env_cfg.scene.num_envs != 1:
            raise ValueError("Manual recorder scripts require num_envs == 1.")
        if env_cfg.terminations is not None and any(
            value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
        ):
            raise ValueError("Manual recording requires a teleop task without automatic terminations.")
        physics_steps = 1.0 / (args_cli.fps * env_cfg.sim.dt)
        if abs(physics_steps - round(physics_steps)) > 1e-6 or physics_steps < 1:
            raise ValueError("--fps must divide the pick-place physics rate exactly.")
        env_cfg.decimation = round(physics_steps)
        env_cfg.sim.render_interval = env_cfg.decimation
        # The manual episode loop owns cube spawning, not env.reset().
        env_cfg.events.spawn_cube = None
        env_cfg.scene.blue_cube.init_state.pos = (0.0, 0.0, -10.0)
        env_cfg.scene.blue_cube.spawn.visible = False
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()
        episode_cube = EpisodeCube(env)
        episode_cube.hide()

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
        leader_iface = connect_interface(
            args_cli, env.unwrapped.device, sim_cameras, "leader", visualize=args_cli.rerun
        )

        recording_mode = bool(all(dataset_args))
        # Opened whenever configured: the GUI shows them even when not recording.
        real_cameras = real_camera_specs(args_cli)

        if args_cli.enable_real_follower:
            follower_iface = connect_interface(args_cli, env.unwrapped.device, real_cameras, "follower")

        if recording_mode:
            recorder_group = make_recorders(
                args_cli, env.unwrapped.device, sim_cameras, follower_iface, real_cameras,
                batch_encoding_size=args_cli.encode_every or sys.maxsize,
            )
            recorder_group.enable_background_encoding()
            recorder_group.init_datasets()
            validate_cube_trajectories(
                args_cli.repo_root, recorder_group.recorders["sim"].episode_lengths(), args_cli.fps
            )
            print(f"[INFO]: Recording ready — episode {recorder_group.episode_index}")
        else:
            print("[INFO]: Recording disabled. Supply --repo_id, --repo_root, --task_name.")

        # ── Per-tick state ────────────────────────────────────────────────
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        wrist_roll_index = 4
        leader_wrist_start = None

        if args_cli.control_input == "isaac":
            from sim_to_real_so101.utils.keyboard import KeyboardControl
            keyboard_control = KeyboardControl(enable_spawn=True)
            print("[controls] Focus the Isaac viewport: S spawn cube, Right start/save, "
                  "Left discard, R reset, Esc quit")
        elif args_cli.control_input == "lerobot":
            from sim_to_real_so101.utils.lerobot_keyboard import LeRobotKeyboardControl
            try:
                keyboard_control = LeRobotKeyboardControl()
            except Exception as exc:
                print(f"[controls] LeRobot listener unavailable ({exc}); using terminal commands.")
        if keyboard_control is None:
            from sim_to_real_so101.utils.terminal_control import TerminalInputControl
            keyboard_control = TerminalInputControl(enable_spawn=True)

        gui = RecordingWebGui(
            [SIM_RGB, SIM_WRIST, REAL_RGB, REAL_WRIST],
            host=args_cli.gui_host,
            port=args_cli.gui_port,
            max_fps=args_cli.gui_fps,
        )
        print(f"[INFO]: Recorder GUI running — open {gui.url} in a browser.")

        # ── Episode outer loop ────────────────────────────────────────────
        phase = "setup"   # "setup" | "resetting" | "recording"
        reset_deadline = None
        # Cube pose for each recorded frame; written next to the episode only if it is saved.
        cube_trajectory: list[torch.Tensor] = []
        base_body_id = env.unwrapped.scene["robot"].find_bodies("base")[0][0]
        cube_size = tuple(env.unwrapped.scene["blue_cube"].cfg.spawn.size)
        gui_note = ""

        print("\n[INFO]: Setup ready. Spawn a cube for practice, or start an episode.")

        while simulation_app.is_running():
            loop_start = time.perf_counter()
            requests = keyboard_control.consume_requests()
            gui_requests = gui.consume_requests()
            for button, request in GUI_REQUESTS.items():
                if gui_requests.get(button):
                    requests[request] = True
            if gui_requests.get("exit"):
                requests["exit_encode"] = gui_requests["exit_encode"]

            # ── Quit ──────────────────────────────────────────────────────
            if requests["stop"]:
                exit_encode = requests.get("exit_encode", True)
                print("[INFO]: Quit requested; closing the teleoperation loop.")
                gui.set_state(
                    active=False, can_start=False, busy=True, status="Closing recorder…",
                    pending_videos=recorder_group.pending_video_episodes if recorder_group else 0,
                    encoding=recorder_group.encoding if recorder_group else False,
                    encode_progress=recorder_group.encode_progress if recorder_group else "",
                )
                if recorder_group is not None:
                    recorder_group.begin_shutdown(exit_encode)
                if recording and recorder_group is not None:
                    if requests["rerecord"]:
                        recorder_group.cancel_episode()
                    else:
                        _save_episode(recorder_group, cube_trajectory, cube_size)
                    recording = False
                episode_cube.hide()
                break

            if requests.get("encode") and recorder_group is not None:
                recorder_group.start_encoding()

            # Practice/position the cube without resetting the arm, starting
            # a countdown, or touching dataset buffers.
            finishing = requests["end"] or requests["rerecord"] or requests["reset"]
            if requests.get("spawn") and not finishing:
                if phase == "setup":
                    episode_cube.show()
                    print("[INFO]: Cube spawned for practice; recording remains off.")
                else:
                    print("[INFO]: Finish the active episode before respawning the cube.")

            # Right starts from the current scene; only R resets and S spawns.
            if requests["start"] and phase == "setup" and not finishing:
                if not episode_cube.visible:
                    print("[INFO]: Spawn the cube with S before starting an episode.")
                else:
                    phase = "resetting"
                    reset_deadline = time.perf_counter() + args_cli.reset_wait_s
                    keyboard_control.set_recording(True)
                    print(f"\n[INFO]: Recording starts in {args_cli.reset_wait_s} seconds; "
                          "keeping the current cube pose.")
                    say("Ready to start recording", blocking=False)

            # Save/discard/reset during countdown cancels preparation. During
            # an episode it finishes exactly once, before recording another frame.
            if finishing:
                gui.set_state(
                    active=phase != "setup", can_start=False, busy=True, status="Finishing episode…",
                    pending_videos=recorder_group.pending_video_episodes if recorder_group else 0,
                    encoding=recorder_group.encoding if recorder_group else False,
                    encode_progress=recorder_group.encode_progress if recorder_group else "",
                )
                if recording and recorder_group is not None:
                    if requests["rerecord"]:
                        recorder_group.cancel_episode()
                        gui_note = "Episode discarded."
                        say("Episode discarded", blocking=False)
                    else:
                        episode_index = _save_episode(recorder_group, cube_trajectory, cube_size)
                        if episode_index is None:
                            gui_note = "Nothing recorded; episode not saved."
                        else:
                            gui_note = f"Saved episode {episode_index}."
                            say("Episode saved", blocking=False)
                cube_trajectory = []
                recording = False
                phase = "setup"
                reset_deadline = None
                keyboard_control.set_recording(False)
                if requests["reset"]:
                    env.reset()
                    leader_wrist_start = None
                episode_cube.hide()
                print("\n[INFO]: Setup ready; cube cleared. Start the next episode when ready.")

            # ── Countdown expired → start recording ───────────────────────
            if phase == "resetting" and time.perf_counter() >= reset_deadline:
                phase = "recording"
                gui_note = ""
                if recording_mode and recorder_group is not None:
                    cube_trajectory = []
                    recording = True
                    say("Recording", blocking=False)
                if recording:
                    print(f"\n[INFO]: Recording episode {recorder_group.episode_index}.")
                else:
                    print("\n[INFO]: Episode active (dataset recording disabled).")

            # ── Control tick (runs in all phases) ─────────────────────────
            try:
                leader_action = leader_iface.robot.get_action()
            except Exception as exc:
                raise RuntimeError(f"Failed to read leader: {exc}") from exc

            _, mapped_action = leader_iface.real_to_sim_obs_processor(leader_action)

            # Wrist-roll re-zero
            if leader_wrist_start is None:
                leader_wrist_start = mapped_action[wrist_roll_index].clone()
                sim_wrist_start = env.unwrapped.observation_manager.compute()["policy"]["joint_pos_obs"][
                    0, wrist_roll_index
                ].clone()
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

            obs = env.unwrapped.observation_manager.compute()
            visual_obs = obs.get("visual")

            # Read every tick: the GUI shows the real cameras even when not recording.
            follower_observation = None
            if follower_iface is not None:
                try:
                    follower_observation = follower_iface.robot.get_observation()
                except Exception as exc:
                    raise RuntimeError(f"Follower observation failed: {exc}") from exc

            if recording and recorder_group is not None:
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
                    "sim": sim_recorder.make_sim_frame(
                        leader_iface.get_raw_actions_from_radians(mapped_action), sim_observation, visual_buffers
                    )
                }
                auxiliary = {
                    "sim": sim_recorder.make_auxiliary_frame(
                        visual_buffers, depth_buffers, instance_id_seg_buffers
                    )
                }

                if follower_observation is not None:
                    frames["real"] = follower_iface.make_real_dataset_frame(
                        leader_action, follower_observation, args_cli.task_name
                    )

                recorder_group.add_frames(frames, auxiliary)
                # Same pre-command state as the images just recorded, so entry i
                # lines up with dataset frame_index i.
                cube_trajectory.append(_cube_pose_in_base(env, base_body_id))

                if args_cli.rerun:
                    _log_rerun_frame(visual_obs, follower_observation, leader_action)

            gui.set_state(
                active=phase != "setup",
                can_start=phase == "setup" and episode_cube.visible,
                status=_gui_status(phase, reset_deadline, recorder_group, episode_cube.visible, gui_note),
                recording=recording,
                pending_videos=recorder_group.pending_video_episodes if recorder_group else 0,
                encoding=recorder_group.encoding if recorder_group else False,
                encode_progress=recorder_group.encode_progress if recorder_group else "All saved episodes are encoded",
            )
            gui.update_images(_gui_frames(visual_obs, follower_observation))

            if follower_iface is not None:
                try:
                    follower_iface.robot.send_action(leader_action)
                except Exception as exc:
                    raise RuntimeError(f"Follower action write failed: {exc}") from exc

            actions[:] = mapped_action
            episode_cube.park()
            try:
                obs, _, _, _, _ = env.step(actions)
            except Exception as exc:
                raise RuntimeError(f"env.step() failed: {exc}") from exc
            remaining = (1.0 / args_cli.fps) - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        exit_encode = False
        print("[INFO]: Interrupted; shutting down cleanly.")
    except Exception:
        # Kit shutdown may exit before Python reports an uncaught exception.
        traceback.print_exc()
        raise
    finally:
        # Once cleanup starts, signals request cancellation instead of raising
        # between finalizers and skipping the remaining device/app shutdowns.
        def interrupt_cleanup(signum, frame):
            nonlocal exit_encode
            exit_encode = False
            if recorder_group is not None:
                recorder_group.begin_shutdown(False)

        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is not None:
                signal.signal(signum, interrupt_cleanup)
        if keyboard_control is not None:
            _cleanup("Keyboard cleanup", keyboard_control.cleanup)
        _cleanup("Follower disconnect", lambda: _disconnect(follower_iface, "physical follower"))
        _cleanup("Leader disconnect", lambda: _disconnect(leader_iface, "physical leader"))
        def exit_progress(pending):
            if gui is not None:
                gui.set_state(
                    active=False, can_start=False, busy=True,
                    status=(f"Encoding {pending} remaining episode(s) before exit…"
                            if exit_encode else "Closing recorder without encoding…"),
                    pending_videos=pending,
                    encoding=recorder_group.encoding if recorder_group else False,
                    encode_progress=recorder_group.encode_progress if recorder_group else "",
                )
        _cleanup("GUI shutdown status", lambda: exit_progress(
            recorder_group.pending_video_episodes if recorder_group else 0))
        if recorder_group is not None:
            _cleanup("Dataset finalization", lambda: recorder_group.finalize(
                encode=exit_encode, progress=exit_progress))
        if gui is not None:
            _cleanup("GUI shutdown", gui.destroy)
        if env is not None:
            _cleanup("Environment close", env.close)
        _cleanup("Simulation app close", simulation_app.close)


if __name__ == "__main__":
    main()
