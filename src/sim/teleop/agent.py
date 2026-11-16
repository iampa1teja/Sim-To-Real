"""Recording setup/teardown; the tick and request state machine live in teleop.loop."""
import os
from functools import partial

from sim.recording import setup

from sim import apply_selection, launch, close_app
from sim.teleop.config import AgentConfig, parse_config
from sim.teleop.loop import LoopResources, gui_panes, run_loop

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

        import sim.workspace.envs
        from utils.arms.arms import load_arm_spec
        from utils.arms.mapping import JointMapping
        from utils.datasets.alignment import mark_recording_root
        from sim.teleop.spawner import ObjectSpawner
        from sim.recording.rerecording import RerecordingSession
        from sim.teleop.devices import connect_device

        setup.exit_cleanly_on_termination()
        if cfg.telemetry.joint_angle_log:
            from utils.calibration.telemetry import JointAngleTelemetry

            telemetry = JointAngleTelemetry(cfg.telemetry.joint_angle_log, selection=selection,
                                            provenance={"agent": "sim.teleop.agent", "sim_wrist_start_rezero": True})

        from core.tasks.task import load_task
        task = load_task(cfg.task) if cfg.task else None
        env_id = setup.environment_id(cfg)
        env_cfg = setup.make_env_cfg(cfg, env_id, sim_device, selection)
        env = gym.make(env_id, cfg=env_cfg)
        device = env.unwrapped.device
        print(f"[INFO]: {env_id}  obs: {env.observation_space}  action: {env.action_space}")
        env.reset()
        # Spawnable task objects start parked; `spawn --object <name>` puts one at the sim jaw (Newton-safe teleport).
        spawner = ObjectSpawner(env, task=task, port=cfg.spawn_port, authkey=cfg.spawn_authkey)

        sim_cameras = setup.simulation_cameras(env)
        arm, _ = load_arm_spec(cfg.arm)
        mapping = JointMapping.from_arm(arm, selection, device=device)

        leader = connect_device("leader", cfg.leader.port, cfg.leader.id, cameras={}, fps=cfg.fps,
                                selection=selection, calibrate=calibrate)

        # Physical cameras open only when a real dataset will consume them (follower-only teleop stays light).
        f = cfg.follower
        real_cameras = setup.real_camera_specs(f.cameras, fps=cfg.fps, width=f.camera_width, height=f.camera_height,
                                         fourcc=f.camera_fourcc) if cfg.recording else {}
        if f.enabled:
            follower = connect_device("follower", f.port, f.id, cameras=real_cameras, fps=cfg.fps,
                                      selection=selection, calibrate=calibrate)

        ds = cfg.dataset
        if cfg.recording:
            recorders, real_pipeline = setup.make_recorders(
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

        tracked, base_id = setup.prepare_task_tracking(env, task, arm, recorders, cfg.fps)

        if cfg.sim_reset_object_name and cfg.sim_reset_object_name not in env.unwrapped.scene.keys():
            raise ValueError(f"sim_reset_object_name={cfg.sim_reset_object_name!r} is not in {env_id}: "
                             f"{list(env.unwrapped.scene.keys())}")

        # Controls start only after connection / calibration prompts are finished.
        from sim.teleop.controls import CommandQueue

        controls = CommandQueue(enable_spawn=bool(spawner.offsets))
        if cfg.control_input == "terminal":
            controls.start_terminal(stop_on_eof=not cfg.gui.enabled)
        if cfg.gui.enabled:
            from sim.recording.web_gui import RecordingWebGui

            gui = RecordingWebGui(gui_panes(sim_cameras, real_cameras), host=cfg.gui.host, port=cfg.gui.port,
                                  max_fps=cfg.gui.fps)
            print(f"[INFO]: Recorder GUI running; open {gui.url} in a browser.")
        print("[INFO]: RESET/BUFFER phase. Start the episode when ready.")
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
            setup.cleanup("Joint telemetry close", telemetry.close)
        if controls is not None:
            setup.cleanup("Controls cleanup", controls.cleanup)
        if gui is not None:
            setup.cleanup("Recorder GUI shutdown", gui.destroy)
        setup.cleanup("Follower disconnect", lambda: setup.disconnect(follower, "physical follower"))
        setup.cleanup("Leader disconnect", lambda: setup.disconnect(leader, "physical leader"))
        if spawner is not None:
            setup.cleanup("Spawn listener shutdown", spawner.close)
        if recorders is not None:
            setup.cleanup("Dataset finalization", recorders.finalize)
        if env is not None:
            setup.cleanup("Environment close", env.close)
        setup.cleanup("Simulation app close", partial(close_app, simulation_app, success=exit_code == 0))

def main(argv=None):
    cfg, isaac_args = parse_config(argv)
    apply_selection(cfg)
    selection, contract, roots = setup.prepare_recording(cfg)

    simulation_app = launch(isaac_args, enable_cameras=True)
    run(cfg, simulation_app, isaac_args.device, selection, contract, roots)


if __name__ == "__main__":
    main()
