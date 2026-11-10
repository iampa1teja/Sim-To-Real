"""Headless contact and mounting check; opens no hardware or recording datasets.

Run with s2r-entrypoint python in the existing s2r-sim container. The production
teleop configuration and all cameras are retained; no solver settings are overridden.
"""

import argparse
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=736)
    parser.add_argument("--settle", type=int, default=60)
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()
    if args.frames < 2 or args.settle < 0:
        parser.error("--frames must be at least 2 and --settle must be nonnegative")
    os.environ.update(S2R_ARM="so101", S2R_SCENE="rpd_meeting_room", S2R_TASK="pick_place_cube")
    result = {"passed": False, "frames": args.frames, "settle_frames": args.settle,
              "physics_only": args.physics_only, "seed": 101}
    app = env = None
    samples = []
    try:
        from sim.app import launch, close_app

        app = launch(headless=True, enable_cameras=True)
        import gymnasium as gym
        import numpy as np
        import torch
        from pxr import Usd, UsdPhysics
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_newton.physics import NewtonManager
        import sim.envs as envs
        from sim.envs.mdp import as_torch
        from sim.scene.geometry import table_plane
        from sim.scene.contact_materials import CONTACT_FRICTION, GRASP_FRICTION
        from core.arms.spec import load_arm_spec
        from core.scenes import load_scene_spec, load_mount_spec
        from core.scenes.anchors import mount_pose

        env_id = envs.TASK.gym_id("teleop")
        cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1, use_fabric=True)
        cfg.seed = 101
        env = gym.make(env_id, cfg=cfg)
        e = env.unwrapped
        env.reset()
        robot, cube = e.scene["robot"], e.scene["blue_cube"]
        arm = load_arm_spec()[0]
        scene = load_scene_spec()
        mount = load_mount_spec(scene, arm)
        base = robot.find_bodies(arm.links.base)[0][0]
        base_pos = as_torch(robot.data.body_link_pos_w)[0, base].cpu().numpy().copy()
        origin = as_torch(e.scene.env_origins)[0].cpu().numpy()
        correction = np.asarray(mount.translation_correction_m)
        expected = np.asarray(mount_pose(scene, mount, arm).position) + origin
        nominal = expected - correction
        root = e.sim.stage.GetPrimAtPath(e.scene.env_prim_paths[0])
        plane = np.asarray(table_plane(e.sim.stage, root))
        height = float(plane[:2] @ base_pos[:2] + plane[2])
        result["mount"] = {
            "nominal_reference_world_m": nominal.tolist(),
            "inferred_mount_correction_m": correction.tolist(),
            "base_world_m": base_pos.tolist(), "expected_world_m": expected.tolist(),
            "base_quat_xyzw": as_torch(robot.data.body_link_quat_w)[0, base].cpu().tolist(),
            "env_origin_m": origin.tolist(), "error_mm": ((base_pos - expected) * 1000).tolist(),
            "tabletop_plane": plane.tolist(), "tabletop_z_at_base_m": height,
            "table_above_base_mm": float((height - base_pos[2]) * 1000),
        }
        print("MOUNT " + json.dumps(result["mount"]), flush=True)
        np.testing.assert_allclose(base_pos, expected, rtol=0, atol=1e-6)

        fixed = []
        for prim in Usd.PrimRange(e.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if joint.GetJointEnabledAttr().Get() and bool(body0) != bool(body1):
                    fixed.append({"path": str(prim.GetPath()), "body0": list(map(str, body0)),
                                  "body1": list(map(str, body1))})
        result["world_fixed_joints"] = fixed
        assert len(fixed) == 1, fixed

        model, solver = NewtonManager.get_model(), NewtonManager._solver
        mj = solver.mj_model
        geom_to_shape = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
        shape_mu = model.shape_material_mu.numpy()
        gpu_mu = solver.mjw_model.geom_friction.numpy().reshape(-1, mj.ngeom, 3)[0]
        labels = [model.shape_label[int(s)] if s >= 0 else mj.geom(g).name
                  for g, s in enumerate(geom_to_shape[:mj.ngeom])]
        result["robot_colliders"] = sum("/Robot/" in label for label in labels)
        assert result["robot_colliders"] == 30, result["robot_colliders"]
        result["geom_friction"] = []
        counts = {"cube": 0, "jaw": 0, "tabletop": 0}
        for geom, label in enumerate(labels):
            kind = ("cube" if "/CUBE" in label else "jaw" if any(
                part in label for part in ("/Robot/gripper/", "/Robot/jaw/")) else
                "tabletop" if "C_Tabletop_" in label else None)
            if kind is None:
                continue
            shape = int(geom_to_shape[geom])
            row = {"kind": kind, "label": label, "geom": geom,
                   "model_mu": float(shape_mu[shape]), "cpu_friction": mj.geom_friction[geom].tolist(),
                   "gpu_friction": gpu_mu[geom].tolist(), "condim": int(mj.geom_condim[geom])}
            result["geom_friction"].append(row)
            counts[kind] += 1
            print("FRICTION " + json.dumps(row), flush=True)
            np.testing.assert_allclose(
                [row["model_mu"], row["cpu_friction"][0], row["gpu_friction"][0]],
                CONTACT_FRICTION if kind == "tabletop" else GRASP_FRICTION,
                rtol=0, atol=1e-7)
            assert row["condim"] == 6, row
        result["target_geom_counts"] = counts
        assert counts == {"cube": 1, "jaw": 16, "tabletop": 49}, counts

        # Align the 2 cm cube with the measured table and place it 0.2 mm above contact.
        xy = np.asarray([0.65, -0.40]) + origin[:2]
        normal = np.asarray([-plane[0], -plane[1], 1.0])
        pos = np.r_[xy, plane[:2] @ xy + plane[2] + 0.01 * np.linalg.norm(normal) + 0.0002]
        pose = torch.tensor([list(pos) + list(mount_pose(scene, mount, arm).orientation_xyzw)], device=e.device, dtype=torch.float32)
        env_ids = torch.tensor([0], device=e.device)
        cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=e.device), env_ids=env_ids)
        target = as_torch(robot.data.joint_pos).clone()
        with torch.inference_mode():
            for _ in range(args.settle + args.frames):
                if args.physics_only:
                    robot.set_joint_position_target(target)
                    for _ in range(e.cfg.decimation):
                        e.scene.write_data_to_sim()
                        e.sim.step(render=False)
                        e.scene.update(e.physics_dt)
                else:
                    env.step(target)
                samples.append(as_torch(cube.data.root_pos_w)[0].cpu().numpy().copy())
        measured = np.asarray(samples[args.settle:])
        steps = np.linalg.norm(np.diff(measured, axis=0), axis=1)
        excursion = np.linalg.norm(measured - measured[0], axis=1)
        base_after = as_torch(robot.data.body_link_pos_w)[0, base].cpu().numpy()
        result["rest"] = {
            "dt_s": e.step_dt, "duration_s": (args.frames - 1) * e.step_dt,
            "cube_start_world_m": measured[0].tolist(), "cube_end_world_m": measured[-1].tolist(),
            "net_motion_mm": float(excursion[-1] * 1000), "path_mm": float(steps.sum() * 1000),
            "max_from_start_mm": float(excursion.max() * 1000),
            "max_step_mm": float(steps.max() * 1000), "z_range_mm": float(np.ptp(measured[:, 2]) * 1000),
            "base_after_world_m": base_after.tolist(),
        }
        assert np.isfinite(measured).all(), "Nonfinite cube pose"
        assert excursion.max() <= 1e-6, result["rest"]
        assert steps.max() <= 1e-6, result["rest"]
        np.testing.assert_allclose(base_after, base_pos, rtol=0, atol=1e-6)
        result["passed"] = True
    except BaseException:
        result["exception"] = traceback.format_exc()
    finally:
        try:
            if env is not None:
                try:
                    env.close()
                except BaseException:
                    result.update(passed=False, cleanup_exception=traceback.format_exc())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if samples:
                result["positions_file"] = str(args.output) + ".positions.npy"
                np.save(result["positions_file"], np.asarray(samples))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("CONTACT_MOUNT_CHECK " + json.dumps(
                {key: value for key, value in result.items() if key != "geom_friction"}), flush=True)
        finally:
            if app is not None:
                close_app(app, success=result["passed"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
