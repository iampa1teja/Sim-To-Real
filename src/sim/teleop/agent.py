"""Recording setup/teardown; the tick and request state machine live in teleop.loop."""
import os
from functools import partial
from copy import deepcopy

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
from sim.app import apply_selection, launch, close_app
from sim.teleop.config import AgentConfig, parse_config
from sim.teleop.loop import LoopResources, gui_panes, run_loop

def _prepare(cfg: AgentConfig):
    from utils.arms.calibration import resolve_calibration
    from utils.datasets.alignment import recording_contract

    selection = resolve_calibration(
        follower_id=cfg.follower.id if cfg.follower.enabled else None,
        leader_id=cfg.leader.id,
        unit=cfg.follower.unit,
    )
    selection.print_paths()
    contract = recording_contract(follower_id=cfg.follower.id, leader_id=cfg.leader.id, calibration_selection=selection)

    roots = []
    if cfg.recording:
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

def _make_env_cfg(cfg: AgentConfig, env_id, device, selection):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(env_id, device=device, num_envs=1, use_fabric=cfg.use_fabric)
    env_cfg.seed = cfg.seed

    env_cfg.sim_joint_mapping = deepcopy(selection.mapping)
    if env_cfg.terminations is not None and any(
        value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
    ):
        raise ValueError(f"{env_id} has terminations; manual teleop needs a variant without them (e.g. teleop).")

    physics_steps = 1.0 / (cfg.fps * env_cfg.sim.dt)
    if abs(physics_steps - round(physics_steps)) > 1e-6 or physics_steps < 1:
        raise ValueError("fps must divide the physics rate exactly")
    env_cfg.decimation = round(physics_steps)
    env_cfg.sim.render_interval = env_cfg.decimation

    from sim.envs.scene_builder import drop_unused_camera_outputs

    drop_unused_camera_outputs(env_cfg, depth=cfg.dataset.depth, instance_id_seg=cfg.dataset.instance_id_seg)
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

# Episode controls: terminal Enter-commands or the Isaac keyboard listener. With spawnable objects, S spawns
# (v1 layout: keyboard S = spawn, Right = start/save; terminal s/spawn = spawn, right = start/save).
def _make_controls(kind, enable_spawn=False):
    if kind == "keyboard":
        from sim.teleop.controls.keyboard import KeyboardControl

        if enable_spawn:
            print("[controls] S: spawn object at the jaw; Right: start/save; Left/C: discard; R: reset; Escape: quit.")
        else:
            print("[controls] S: start/save; Right: save; Left/C: discard; R: reset; Escape: quit.")
        return KeyboardControl(enable_spawn=enable_spawn)
    from sim.teleop.controls.terminal import TerminalInputControl

    return TerminalInputControl(enable_spawn=enable_spawn)

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
    exit_code = 0
    calibrate = cfg.calibrate and not cfg.telemetry.joint_angle_log

    try:
        # Isaac / LeRobot imports only after the app is up; a failure here still closes the app below.
        import gymnasium as gym

        import sim.envs
        from utils.arms.spec import load_arm_spec
        from utils.arms.mapping import JointMapping
        from utils.datasets.alignment import mark_recording_root
        from sim.datagen.spawner import ObjectSpawner
        from sim.recording.rerecording import RerecordingSession
        from sim.teleop.devices import connect_device

        exit_cleanly_on_termination()
        if cfg.telemetry.joint_angle_log:
            from utils.calibration.telemetry import JointAngleTelemetry

            telemetry = JointAngleTelemetry(cfg.telemetry.joint_angle_log, selection=selection,
                                            provenance={"agent": "sim.teleop.agent", "sim_wrist_start_rezero": True})

        env_id = _env_id(cfg)
        env_cfg = _make_env_cfg(cfg, env_id, sim_device, selection)
        env = gym.make(env_id, cfg=env_cfg)
        device = env.unwrapped.device
        print(f"[INFO]: {env_id}  obs: {env.observation_space}  action: {env.action_space}")
        env.reset()
        # Spawnable task objects start parked; `spawn --object <name>` puts one at the sim jaw (Newton-safe teleport).
        spawner = ObjectSpawner(env, task=sim.envs.TASK, port=cfg.spawn_port, authkey=cfg.spawn_authkey)

        sim_cameras = _sim_cameras(env)
        arm, _ = load_arm_spec(cfg.arm)
        mapping = JointMapping.from_arm(arm, selection, device=device)

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

        # Track every declared object, including fixed objects and tasks with multiple spawnables.
        tracked = None
        base_id = None
        task = sim.envs.TASK
        if task is not None:
            from sim.datagen.spawner import _half_extents

            tracked = {}
            for name in task.spec.tracked_objects:
                half = _half_extents(task, name)
                size = (getattr(env.unwrapped.scene[name].cfg.spawn, "size", None)
                        or (tuple(2 * value for value in half) if half is not None else None))
                if size is None:
                    raise ValueError(f"{name}: declare size or size_outer in task.toml for its trajectory sidecar.")
                tracked[name] = tuple(float(value) for value in size)
            base_id = env.unwrapped.scene["robot"].find_bodies(arm.links.base)[0][0]
        if recorders is not None and tracked is not None:
            from utils.datasets.task_meta import validate_task_trajectories

            validate_task_trajectories(recorders.recorders["sim"].dataset_root,
                                       recorders.recorders["sim"].episode_lengths(), cfg.fps,
                                       task=task, arm=arm, tracked_objects=tracked,
                                       reference_frame=f"robot base link ({arm.links.base})")

        if cfg.sim_reset_object_name and cfg.sim_reset_object_name not in env.unwrapped.scene.keys():
            raise ValueError(f"sim_reset_object_name={cfg.sim_reset_object_name!r} is not in {env_id}: "
                             f"{list(env.unwrapped.scene.keys())}")

        # Controls start only after connection / calibration prompts are finished.
        controls = _make_controls(cfg.control_input, enable_spawn=bool(spawner.offsets))
        if cfg.gui.enabled:
            from sim.recording.web_gui import RecordingWebGui

            gui = RecordingWebGui(gui_panes(sim_cameras, real_cameras), host=cfg.gui.host, port=cfg.gui.port,
                                  max_fps=cfg.gui.fps)
            print(f"[INFO]: Recorder GUI running; open {gui.url} in a browser.")
        print("[INFO]: RESET/BUFFER phase. Start the episode when ready (keys above).")
        resources = LoopResources(
            env=env, env_id=env_id, leader=leader, follower=follower, mapping=mapping, arm=arm, task=task,
            controls=controls, spawner=spawner, sim_cameras=sim_cameras, real_cameras=real_cameras,
            recorders=recorders, real_pipeline=real_pipeline, gui=gui, telemetry=telemetry,
            tracked=tracked, base_id=base_id,
        )
        run_loop(cfg, simulation_app, resources)

    except KeyboardInterrupt:
        print("[INFO]: Interrupted; shutting down cleanly.")
    except Exception:
        exit_code = 1
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
        cleanup("Simulation app close", partial(close_app, simulation_app, success=exit_code == 0))

def main(argv=None):
    cfg, isaac_args = parse_config(argv)
    apply_selection(cfg)
    selection, contract, roots = _prepare(cfg)

    simulation_app = launch(isaac_args, enable_cameras=True)
    run(cfg, simulation_app, isaac_args.device, selection, contract, roots)


if __name__ == "__main__":
    main()
