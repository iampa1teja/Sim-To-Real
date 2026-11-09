"""Hardware-free integration check, run with s2r-entrypoint python in s2r-sim.

Uses the full teleop scene, including all camera outputs and contacts. Pass
--gui to also check Kit's active camera. Never opens a hardware device or dataset.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(S2R_ARM="so101", S2R_SCENE="rpd_meeting_room", S2R_TASK="pick_place_cube")
    result = {"gui": args.gui, "steps": args.steps}
    app = env = None
    try:
        from isaaclab.app import AppLauncher
        from sim.app import rendering_kit_args

        app = AppLauncher(
            headless=not args.gui, visualizer="kit" if args.gui else None,
            enable_cameras=True, kit_args=rendering_kit_args(),
        ).app
        import gymnasium as gym
        import numpy as np
        import torch
        import isaaclab.utils.math as math_utils
        from isaaclab.sim import SimulationContext
        from isaaclab_tasks.utils import parse_env_cfg
        import sim.envs as envs
        from core.arms.mapping import JointMapping
        from sim.envs.mdp import as_torch
        from sim.envs.mdp.resets import _env_prim_path
        from sim.generate.object_spawner import ObjectSpawner
        from sim.recording.lerobot_recorder import LeRobotRecorder
        from sim.scene.calibrated_viewport import CalibratedKitVisualizer
        from sim.scene.gripper_geometry import table_plane

        camera_path = "/World/envs/env_0/realsense_camera_rgb"

        def camera_attributes():
            prim = SimulationContext.instance().stage.GetPrimAtPath(camera_path)
            # Include authored pose, apertures and the OpenCV lens model.
            return {a.GetName(): str(a.Get()) for a in prim.GetAttributes()}

        original_pose_hook = CalibratedKitVisualizer._apply_cfg_camera_pose_if_configured
        original_view_hook = CalibratedKitVisualizer._setup_initial_camera_view

        def pose_hook(self):
            result["camera_before_visualizer"] = camera_attributes()
            original_pose_hook(self)

        def view_hook(self):
            original_view_hook(self)
            result["camera_after_visualizer"] = camera_attributes()

        CalibratedKitVisualizer._apply_cfg_camera_pose_if_configured = pose_hook
        CalibratedKitVisualizer._setup_initial_camera_view = view_hook
        env_id = envs.TASK.gym_id("teleop")
        cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1, use_fabric=True)
        result["integrator"] = cfg.sim.physics.solver_cfg.integrator
        env = gym.make(env_id, cfg=cfg)
        e = env.unwrapped
        robot = e.scene["robot"]
        obs, _ = env.reset()
        # A 20 mm cube cannot be inserted between the closed fingers. Open the
        # jaw through the normal action path before testing jaw-relative spawn;
        # the later sinusoid keeps that jaw open while the arm moves.
        jaw = robot.find_joints("Jaw")[0][0]
        fixture_target = as_torch(robot.data.joint_pos).clone()
        fixture_target[:, jaw] = 0.6
        for _ in range(30):
            obs, *_ = env.step(fixture_target)
        result["spawn_jaw_position_rad"] = float(as_torch(robot.data.joint_pos)[0, jaw])
        assert result["spawn_jaw_position_rad"] > 0.5, result["spawn_jaw_position_rad"]
        mapping = JointMapping.from_setup(device=e.device)
        # Frame construction only: no dataset is opened or written.
        recorder = LeRobotRecorder("runtime check", "local/check", "/tmp/unused-s2r-check", 30, e.device)
        spawner = ObjectSpawner(e, task=envs.TASK, port=0)
        result["spawn"] = spawner.spawn("blue_cube")
        result["resolved_robot_path"] = _env_prim_path(e, robot, 0)
        assert e.sim.stage.GetPrimAtPath(result["resolved_robot_path"] + "/gripper").IsValid()

        cube = e.scene["blue_cube"]
        position = as_torch(cube.data.root_pos_w)[0]
        rotation = math_utils.matrix_from_quat(as_torch(cube.data.root_quat_w)[0])
        plane = position.new_tensor(table_plane(e.sim.stage, e.sim.stage.GetPrimAtPath(e.scene.env_prim_paths[0])))
        normal = torch.cat((-plane[:2], plane.new_ones(1)))
        support = (normal @ rotation).abs().sum() * 0.01  # task.toml: 2 cm cube
        gap = position[2] - (plane[:2] * position[:2]).sum() - plane[2] - support
        result["spawn_table_clearance_m"] = float(gap)
        assert float(gap) >= 0.00019, result["spawn"]

        initial = as_torch(robot.data.joint_pos).clone()
        limits = as_torch(robot.data.joint_pos_limits)
        max_error = max_relative_error = max_abs = max_raw = 0.0
        samples, durations = [], []
        early_images = {}
        phases = torch.arange(6, device=e.device) * 0.7
        viewport = None
        if args.gui:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            assert viewport is not None
        with torch.inference_mode():
            for step in range(args.steps):
                target = initial + 0.1 * torch.sin(phases + step * 0.04)
                target = torch.clamp(target, limits[..., 0] + 0.01, limits[..., 1] - 0.01)
                started = time.perf_counter()
                obs, *_ = env.step(target)
                q = as_torch(robot.data.joint_pos)
                default = as_torch(robot.data.default_joint_pos)
                measured = obs["policy"]["joint_pos_obs"]
                relative = obs["policy"]["joint_pos_rel"]
                torch.testing.assert_close(measured, q, rtol=0, atol=1e-6)
                torch.testing.assert_close(relative, q - default, rtol=0, atol=1e-6)
                assert torch.isfinite(q).all() and q.abs().max() < 4.0, q
                max_error = max(max_error, float((measured - q).abs().max()))
                max_relative_error = max(max_relative_error, float((relative - (q - default)).abs().max()))
                max_abs = max(max_abs, float(q.abs().max()))
                frame = recorder.make_sim_frame(mapping.to_raw(target[0]), mapping.to_raw(measured[0]), {})
                np.testing.assert_allclose(frame["observation.state"], mapping.to_raw(q[0]).cpu().numpy())
                assert np.isfinite(frame["observation.state"]).all()
                max_raw = max(max_raw, float(np.abs(frame["observation.state"]).max()))
                if viewport is not None:
                    assert str(viewport.get_active_camera()) == camera_path
                if step == 10:
                    for name in ("camera_realsense_rgb", "camera_wrist_cam"):
                        early_images[name] = as_torch(e.scene[name].data.output["rgb"]).clone()
                if step in (0, args.steps - 1):
                    samples.append({"step": step + 1, "observation": measured.cpu().tolist(), "joint_pos": q.cpu().tolist()})
                durations.append(time.perf_counter() - started)
        result.update(samples=samples, max_joint_error_rad=max_error, max_relative_error_rad=max_relative_error,
                      max_abs_joint_rad=max_abs, max_abs_recorded_state=max_raw,
                      median_step_with_checks_ms=float(np.median(durations[10:]) * 1000))
        result["images"] = {}
        for name in ("rgb_realsense_rgb", "rgb_wrist_cam"):
            image = obs["visual"][name]
            spatial_std = image.float().reshape(-1, 3).std(dim=0)
            assert tuple(image.shape) == (1, 480, 640, 3) and spatial_std.min() > 5, (name, spatial_std)
            result["images"][name] = {"shape": list(image.shape), "spatial_std": spatial_std.cpu().tolist()}
        # Compare raw renders so fitted color-response noise cannot masquerade as motion.
        for name, early in early_images.items():
            current = as_torch(e.scene[name].data.output["rgb"])
            changed = int(((current.to(torch.int16) - early.to(torch.int16)).abs() > 2).any(dim=-1).sum())
            result["images"][name + "_changed_pixels"] = changed
            assert changed > 100, (name, changed)
        if viewport is not None:
            result["active_camera"] = str(viewport.get_active_camera())
            result["viewport_resolution"] = list(viewport.get_texture_resolution())
            assert result["viewport_resolution"] == [640, 480]
            assert result["camera_before_visualizer"] == result["camera_after_visualizer"]
            assert result["camera_after_visualizer"] == camera_attributes()
        result["passed"] = True
    except BaseException:
        result["exception"] = traceback.format_exc()
        result["passed"] = False
    finally:
        args.output.write_text(json.dumps(result, indent=2))
        print("RUNTIME_CHECK " + json.dumps(result), flush=True)
        if env is not None:
            env.close()
        if app is not None:
            app.close(exit_code=0 if result.get("passed") else 1)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
