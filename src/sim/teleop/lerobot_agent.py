import os 
import time 
import traceback
from functools import partial
from copy import deepcopy

import torch 

from sim.recording.recording_setup import (
    cleanup, 
    derived_real_repo_id,
    derived_real_root,
    disconnect,
    exit_cleanly_on_termination,
    make_recorders,
    prepare_recording_roots,
    real_camera_specs,
    validate_serial_ports
)
from sim.teleop.config import AgentConfig, apply_selection, parse_config

def _prepare(cfg: AgentConfig):
    from core.arms.joint_calibration import resolve_calibration
    from core.utils.recording_alignment import recording_contract

    selection = resolve_calibration(
        follower_id=cfg.follower.id if cfg.follower.enabled else None, 
        leader_id=cfg.leader.id,
        follower_path=cfg.follower.calibration_path,
    )
    selection.print_paths()
    contract = recording_contract(follower_id=cfg.follower.id, leader_id=cfg.leader.id, calibration_selection=selection)

    roots = []
    if cfg.recording and (selection.follower_path or selection.leader_path):
        roots.append(cfg.dataset.root)
        if cfg.follower.enabled:
            roots.append(cfg.dataset.real_root or derived_real_root(cfg.dataset.root))
        roots, suffix = prepare_recording_roots(
            roots, contract, on_mismatch=cfg.dataset.on_alignment_mismatch)
        if suffix:
            if cfg.follower.enabled:
                cfg.dataset.real_root = roots[1]
                cfg.dataset.real_repo_id = (cfg.dataset.real_repo_id or
                                            derived_real_repo_id(cfg.dataset.repo_id)) + suffix
            cfg.dataset.root = roots[0]
            cfg.dataset.repo_id += suffix
            print("[recording] Calibration differs from the configured dataset; "
                  "using separate folders and preserving existing recordings.", flush=True)
        for role, root in zip(("sim", "real"), roots):
            print(f"[recording] {role} dataset: {root}", flush=True)
    elif cfg.recording:
        from sim.recording.rerecording import recover_replacement

        recover_replacement(cfg.dataset.root)
    ports = {"leader": cfg.leader.port}
    if cfg.follower.enabled:
        ports["follower"] = cfg.follower.port
    validate_serial_ports(ports)
    return selection, contract, roots

def _env_id(cfg: AgentConfig):
    import sim.envs

    if sim.envs.TASK is None:
        return sim.envs.scene_gym_id()
    return sim.envs.TASK.gym_id(cfg.variant)

def _camera_dropper():
    from importlib import import_module 
    from sim.envs import scene_entry_point 

    module = import_module(scene_entry_point().split(":",1)[0])
    return getattr(module, "drop_unused_camera_outputs", None)

def _make_env_cfg(cfg: AgentConfig, env_id, device, selection):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=1, use_fabric=cfg.use_fabric)
    env_cfg.seed = cfg.seed

    if selection.provenance.get("mapping_source") == "follower_file":
        env_cfg.sim_joint_mapping = deepcopy(selection.mapping)
    if getattr(env_cfg, "sim_joint_mapping", None) not in (None, selection.mapping):
        raise ValueError("Environment mapping differs from the selected follower calibration.")
    if env_cfg.terminations is not None and any(
        value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
    ):
        raise ValueError(f"{env_id} has terminations; manual teleop needs a variant without them (e.g. teleop).")

    physics_steps = 1.0 / (cfg.fps * env_cfg.sim.dt)
    if abs(physics_steps - round(physics_steps)) > 1e-6 or physics_steps < 1:
        raise ValueError("fps must divide the physics rate exactly")
    env_cfg.decimation = round(physics_steps)
    env_cfg.sim.render_interval = env_cfg.decimation 

    dropper = _camera_dropper()
    if dropper is not None:
        dropper(env_cfg, depth=cfg.dataset.depth, instance_id_seg=cfg.dataset.instance_id_seg)
    return env_cfg

def _sim_cameras(env):
    cameras = {}
    for key in env.unwrapped.scene.keys():
        if key.startswith("camera_"):
            camera_cfg = getattr(env.unwrapped.scene.cfg, key)
            name = key[len("camera_"):]
            cameras[name] = {"height": camera_cfg.height, "width": camera_cfg.width}
            print(f"[INFO]: Found simulation camera: {name}")
    if not cameras:
        print("[WARNING]: No simulation cameras found.")
    return cameras

# At episode start, put the configured object beside the sim jaw (through the spawner, so it stops being held parked).
def _place_object_near_ee(spawner, cfg: AgentConfig):
    name = cfg.sim_reset_object_name
    if not name:
        return False
    spawner.spawn(name, offset=tuple(cfg.sim_reset_object_offset))
    return True


# Episode controls: terminal Enter-commands or the Isaac keyboard listener. With spawnable objects, S spawns
# (v1 layout: keyboard S = spawn, Right = start/save; terminal s/spawn = spawn, right = start/save).
def _make_controls(kind, enable_spawn=False):
    if kind == "keyboard":
        from sim.teleop.keyboard import KeyboardControl

        if enable_spawn:
            print("[controls] S: spawn object at the jaw; Right: start/save; Left/C: discard; R: reset; Escape: quit.")
        else:
            print("[controls] S: start/save; Right: save; Left/C: discard; R: reset; Escape: quit.")
        return KeyboardControl(enable_spawn=enable_spawn)
    from sim.teleop.terminal_control import TerminalInputControl

    return TerminalInputControl(enable_spawn=enable_spawn)

# Object pose in the robot base frame as (x, y, z, qw, qx, qy, qz): the sidecar format eval and DR replay read.
# Stays on device; converted once at save.
def _object_pose_in_base(env, name, base_id):
    import isaaclab.utils.math as math_utils
    from sim.envs.mdp.resets import as_torch

    scene = env.unwrapped.scene
    robot, obj = scene["robot"], scene[name]
    pos, quat = math_utils.subtract_frame_transforms(
        as_torch(robot.data.body_link_pos_w)[:, base_id], as_torch(robot.data.body_link_quat_w)[:, base_id],
        as_torch(obj.data.root_pos_w), as_torch(obj.data.root_quat_w),
    )
    return torch.cat([pos, quat[:, 3:4], quat[:, :3]], dim=-1)[0].clone()


def _write_sidecar(root, index, *, poses, size, fps):
    from sim.recording.episode_metadata import save_cube_trajectory

    save_cube_trajectory(root, index, poses, size, fps)


# Save the synchronized episode and its object trajectory sidecar (same frame count). During a GUI re-record
# the session splices it into the chosen slot and returns that index.
def _save_episode(recorders, trajectory, tracked, fps):
    if tracked is not None and len(trajectory) != recorders.frame_count:
        raise ValueError(f"Object trajectory has {len(trajectory)} frames; datasets have {recorders.frame_count}.")
    sidecar = None
    if tracked is not None:
        poses = torch.stack(trajectory).cpu().tolist() if trajectory else []
        sidecar = partial(_write_sidecar, poses=poses, size=tracked[1], fps=fps)
    index = recorders.save_episode(sidecar)
    trajectory.clear()
    return index


# Browser GUI (v1 recorder page). Buttons map onto the same request keys as the keyboard and terminal:
# Stop and Save end the episode, Discard is "rerecord" (throw the episode away), Exit quits.
# Encode does nothing here: LeRobot 0.6.1 encodes video while recording.
GUI_REQUESTS = {"start": "start", "stop": "end", "save": "end", "exit": "stop", "discard": "rerecord",
                "spawn": "spawn"}


# One pane per sim camera, then one per real camera.
def _gui_panes(sim_cameras, real_cameras):
    return [f"Sim: {name}" for name in sim_cameras] + [f"Real: {name}" for name in real_cameras]


def _gui_frames(visual_obs, sim_cameras, follower_observation, real_cameras):
    frames = {}
    for name in sim_cameras:
        images = (visual_obs or {}).get(f"rgb_{name}")
        frames[f"Sim: {name}"] = None if images is None else images[0]
    for name in real_cameras:
        frames[f"Real: {name}"] = (follower_observation or {}).get(name)
    return frames


# Status line under the GUI buttons; ``note`` reports the last save / discard / re-record step.
def _gui_status(recording, recorders, note, fps):
    if recording:
        return f"\u25cf Recording episode {recorders.episode_index}: {recorders.frame_count / fps:.1f} s"
    if recorders is None:
        return "Recording is disabled (set dataset.repo_id, dataset.root and dataset.task_name)."
    return f"{note} Start recording when ready." if note else "Start recording when ready."


def _reset_for_buffer(env, controls, spawner):
    env.reset()
    spawner.reset()
    controls.set_recording(False)
    print("[INFO]: RESET/BUFFER phase. Reposition the real scene and leader, then start the next episode.")


def run(cfg: AgentConfig, simulation_app, sim_device, selection, contract, roots):  # noqa: C901
    controls = None
    env = None
    leader = None
    follower = None
    recorders = None
    real_pipeline = None
    telemetry = None
    spawner = None
    gui = None
    recording = False
    exit_code = 0
    calibrate = cfg.calibrate and not cfg.telemetry.joint_angle_log

    try:
        # Isaac / LeRobot imports only after the app is up; a failure here still closes the app below.
        import gymnasium as gym

        import sim.envs
        from core.arms.joint_calibration import ROLL_JOINT
        from core.arms.mapping import JointMapping
        from core.utils.recording_alignment import mark_recording_root
        from sim.generate.object_spawner import ObjectSpawner
        from sim.recording.lerobot_recorder import split_visual_obs
        from sim.recording.rerecording import RerecordingSession
        from sim.teleop.devices import connect_device, real_dataset_frame

        exit_cleanly_on_termination()
        if cfg.telemetry.joint_angle_log:
            from core.calibration.joint_angle_check import JointAngleTelemetry

            telemetry = JointAngleTelemetry(cfg.telemetry.joint_angle_log, selection=selection,
                                            provenance={"agent": "lerobot_agent", "sim_wrist_start_rezero": True})

        env_id = _env_id(cfg)
        env_cfg = _make_env_cfg(cfg, env_id, sim_device, selection)
        env = gym.make(env_id, cfg=env_cfg)
        device = env.unwrapped.device
        print(f"[INFO]: {env_id}  obs: {env.observation_space}  action: {env.action_space}")
        env.reset()
        # Spawnable task objects start parked; `spawn --object <name>` puts one at the sim jaw (Newton-safe teleport).
        spawner = ObjectSpawner(env, task=sim.envs.TASK, port=cfg.spawn_port, authkey=cfg.spawn_authkey)

        sim_cameras = _sim_cameras(env)
        mapping = JointMapping.from_setup(selection, device=device)
        roll = mapping.names.index(ROLL_JOINT)

        leader = connect_device("leader", cfg.leader.port, cfg.leader.id, cameras={}, fps=cfg.fps,
                                selection=selection, calibrate=calibrate)

        # Physical cameras open only when a real dataset will consume them (follower-only teleop stays light).
        f = cfg.follower
        real_cameras = real_camera_specs(f.cameras, fps=cfg.fps, width=f.camera_width, height=f.camera_height,
                                         fourcc=f.camera_fourcc) if cfg.recording else {}
        if f.enabled:
            follower = connect_device("follower", f.port, f.id, cameras=real_cameras, fps=cfg.fps,
                                      selection=selection, calibrate=calibrate)

        ds = cfg.dataset
        if cfg.recording:
            recorders, real_pipeline = make_recorders(
                task_name=ds.task_name, repo_id=ds.repo_id, repo_root=ds.root, fps=cfg.fps, device=device,
                sim_cameras=sim_cameras, follower=follower, real_repo_id=ds.real_repo_id,
                real_repo_root=ds.real_root, real_cameras=real_cameras, save_mp4=ds.save_mp4, depth=ds.depth,
                instance_id_seg=ds.instance_id_seg, image_writer_threads_per_camera=ds.image_writer_threads_per_camera,
            )
            recorders.init_datasets()
            for root in roots:
                mark_recording_root(root, contract)
            # Same recorder API; adds GUI re-recording of a saved episode (originals untouched until Save).
            recorders = RerecordingSession(recorders)
            paths = {name: os.fspath(rec.dataset_root) for name, rec in recorders.recorders.items()}
            print(f"[INFO]: Recording ready at episode {recorders.episode_index}: {paths}")
        else:
            print("[INFO]: Recording disabled; set dataset.repo_id, dataset.root and dataset.task_name to enable it.")

        # The single spawnable task object is tracked per frame into <sim root>/pick_place_meta (v1 sidecars).
        tracked = None
        trajectory = []
        if len(spawner.offsets) == 1:
            from sim.envs.base_env_cfg import BASE_LINK
            from sim.generate.object_spawner import _half_extents

            name = next(iter(spawner.offsets))
            half = _half_extents(sim.envs.TASK, name)
            size = getattr(env.unwrapped.scene[name].cfg.spawn, "size", None) or (half and tuple(2 * v for v in half))
            if size is None:
                raise ValueError(f"{name}: set size in task.toml so its trajectory sidecar records the object size.")
            tracked = (name, tuple(float(v) for v in size))
            base_id = env.unwrapped.scene["robot"].find_bodies(BASE_LINK)[0][0]
        elif spawner.offsets:
            print(f"[WARNING]: {len(spawner.offsets)} spawnable objects; no object trajectory sidecar is written.")
        if recorders is not None and tracked is not None:
            from sim.recording.episode_metadata import validate_cube_trajectories

            validate_cube_trajectories(recorders.recorders["sim"].dataset_root,
                                       recorders.recorders["sim"].episode_lengths(), cfg.fps)

        if cfg.sim_reset_object_name and cfg.sim_reset_object_name not in env.unwrapped.scene.keys():
            raise ValueError(f"sim_reset_object_name={cfg.sim_reset_object_name!r} is not in {env_id}: "
                             f"{list(env.unwrapped.scene.keys())}")

        actions = torch.zeros(env.action_space.shape, device=device)
        roll_min, roll_max = torch.deg2rad(mapping.clamp_mins[roll]), torch.deg2rad(mapping.clamp_maxs[roll])
        leader_roll_start = None
        sim_roll_start = None
        previous_target = None
        previous_leader_action = None
        # Obs from the last env.step(); None forces compute() after scene edits outside env.step().
        obs = None

        # Controls start only after connection / calibration prompts are finished.
        controls = _make_controls(cfg.control_input, enable_spawn=bool(spawner.offsets))
        if cfg.gui.enabled:
            from sim.recording.web_gui import RecordingWebGui

            gui = RecordingWebGui(_gui_panes(sim_cameras, real_cameras), host=cfg.gui.host, port=cfg.gui.port,
                                  max_fps=cfg.gui.fps)
            print(f"[INFO]: Recorder GUI running; open {gui.url} in a browser.")
        gui_note = ""
        print("[INFO]: RESET/BUFFER phase. Start the episode when ready (keys above).")

        with torch.inference_mode():
            while simulation_app.is_running():
                loop_start = time.perf_counter()
                if telemetry is not None:
                    # Log the finished tick before any reset or the next command.
                    telemetry.observe(follower.get_observation(), mapping, env.unwrapped.scene["robot"],
                                      previous_target, previous_command=previous_leader_action)
                if spawner.poll():
                    obs = None
                requests = controls.consume_requests()
                if gui is not None:
                    gui_requests = gui.consume_requests()
                    for button, request in GUI_REQUESTS.items():
                        if gui_requests.get(button):
                            requests[request] = True
                    # Re-record: copy the datasets aside; the next episode replaces the chosen one on Save.
                    if gui_requests.get("rerecord") and recorders is not None and not recording:
                        gui.set_state(active=False, can_start=False, busy=True,
                                      status="Preparing a safe copy for re-recording...")
                        try:
                            recorders.begin(gui_requests["rerecord_episode"])
                            gui_note = f"Re-recording episode {recorders.target}. Discard keeps the original."
                        except (ValueError, OSError) as exc:
                            gui_note = f"Cannot re-record: {exc}"
                if requests.get("spawn"):
                    # Same object and offset as the episode-start placement; the first spawnable object by default.
                    try:
                        spawner.spawn(cfg.sim_reset_object_name, offset=tuple(cfg.sim_reset_object_offset))
                        obs = None
                    except ValueError as exc:
                        print(f"[WARNING]: Spawn failed: {exc}")

                if requests["stop"]:
                    if recording and requests["rerecord"]:
                        recorders.cancel_episode()
                        trajectory.clear()
                    elif recording:
                        _save_episode(recorders, trajectory, tracked, cfg.fps)
                    if recording:
                        recording = False
                        controls.set_recording(False)
                    break

                # rerecord discards the episode; end and reset save it. All three go back to the buffer phase.
                if requests["rerecord"] or requests["end"] or requests["reset"]:
                    if recording and requests["rerecord"]:
                        recorders.cancel_episode()
                        trajectory.clear()
                        gui_note = "Episode discarded."
                    elif recording:
                        replacing = recorders.target
                        index = _save_episode(recorders, trajectory, tracked, cfg.fps)
                        if index is None:
                            gui_note = "Nothing recorded; episode not saved."
                        elif replacing is not None:
                            gui_note = f"Replaced episode {index}; the original dataset was backed up."
                        else:
                            gui_note = f"Saved episode {index}."
                    elif recorders is not None and recorders.target is not None:
                        recorders.cancel_episode()
                        gui_note = "Re-record cancelled; original episode kept."
                    recording = False
                    _reset_for_buffer(env, controls, spawner)
                    leader_roll_start = None
                    obs = None

                if requests["start"] and not recording and not any(
                        requests[key] for key in ("end", "rerecord", "reset")):
                    if recorders is None:
                        print("[WARNING]: Recording is disabled; restart with the dataset settings before pressing S.")
                    else:
                        _place_object_near_ee(spawner, cfg)
                        obs = None
                        recording = True
                        gui_note = ""
                        controls.set_recording(True)
                        print(f"[INFO]: Recording episode {recorders.episode_index}.")

                # Keep this dict untouched: it goes to the follower and to the real dataset as-is.
                try:
                    leader_action = leader.get_action()
                except Exception as exc:
                    raise RuntimeError(f"Failed to read the physical leader: {exc}") from exc
                target = mapping.to_radians(mapping.raw_from_actions(leader_action))

                if obs is None:
                    obs = env.unwrapped.observation_manager.compute()
                # Wrist roll is relative: the sim keeps its own start angle and follows the leader's change.
                if leader_roll_start is None:
                    leader_roll_start = target[roll].clone()
                    sim_roll_start = obs["policy"]["joint_pos_obs"][0, roll].clone()
                target[roll] = torch.clamp(sim_roll_start + target[roll] - leader_roll_start, roll_min, roll_max)

                # The real follower is read every tick while recording, or for the GUI's real camera panes.
                follower_observation = None
                if follower is not None and (recording or gui is not None):
                    try:
                        follower_observation = follower.get_observation()
                    except Exception as exc:
                        raise RuntimeError(f"Follower observation failed; this episode is discarded: {exc}") from exc

                if recording:
                    visual_obs = obs.get("visual")
                    if visual_obs is None:
                        raise RuntimeError(f"{env_id} has no 'visual' observation group to record.")
                    rgb, depth, seg = split_visual_obs(visual_obs, sim_cameras)
                    sim_recorder = recorders.recorders["sim"]
                    frames = {"sim": sim_recorder.make_sim_frame(
                        mapping.to_raw(target), mapping.to_raw(obs["policy"]["joint_pos_obs"][0]), rgb)}
                    auxiliary = {"sim": sim_recorder.make_auxiliary_frame(rgb, depth, seg)}
                    if follower is not None:
                        frames["real"] = real_dataset_frame(real_pipeline, leader_action, follower_observation,
                                                            ds.task_name)
                    recorders.add_frames(frames, auxiliary)
                    if tracked is not None:
                        trajectory.append(_object_pose_in_base(env, tracked[0], base_id))

                if gui is not None:
                    gui.set_state(
                        active=recording,
                        can_start=recorders is not None and not recording,
                        status=_gui_status(recording, recorders, gui_note, cfg.fps),
                        recording=recording,
                        encode_progress="Videos are encoded while recording.",
                        saved_episodes=recorders.saved_episodes if recorders is not None else 0,
                        rerecord_episode=recorders.target if recorders is not None else None,
                    )
                    gui.update_images(_gui_frames(obs.get("visual"), sim_cameras, follower_observation, real_cameras))

                if follower is not None:
                    try:
                        # The original LeRobot action, never the mapped sim tensor.
                        follower.send_action(leader_action)
                    except Exception as exc:
                        raise RuntimeError(f"Follower action write failed; this episode is discarded: {exc}") from exc

                actions[:] = target
                spawner.hold()
                try:
                    obs, _, _, _, _ = env.step(actions)
                except Exception as exc:
                    raise RuntimeError(f"Isaac Sim action application failed: {exc}") from exc
                previous_target = target.clone()
                previous_leader_action = dict(leader_action)

                remaining = (1.0 / cfg.fps) - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)

    except KeyboardInterrupt:
        print("[INFO]: Interrupted; shutting down cleanly.")
    except Exception:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        if telemetry is not None:
            cleanup("Joint telemetry close", telemetry.close)
        if controls is not None:
            cleanup("Controls cleanup", controls.cleanup)
        if gui is not None:
            cleanup("Recorder GUI shutdown", gui.destroy)
        cleanup("Follower disconnect", lambda: disconnect(follower, "physical follower"))
        cleanup("Leader disconnect", lambda: disconnect(leader, "physical leader"))
        if spawner is not None:
            cleanup("Spawn listener shutdown", spawner.close)
        if recorders is not None:
            cleanup("Dataset finalization", recorders.finalize)
        if env is not None:
            cleanup("Environment close", env.close)
        cleanup("Simulation app close", partial(simulation_app.close, exit_code=exit_code))

def main(argv=None):
    cfg, isaac_args = parse_config(argv)
    apply_selection(cfg)
    selection, contract, roots = _prepare(cfg)

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(isaac_args).app
    run(cfg, simulation_app, isaac_args.device, selection, contract, roots)


if __name__ == "__main__":
    main()
