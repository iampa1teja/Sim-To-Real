"""Profile real two-camera AV1 recording without connecting hardware.

Runs the production scene and recording code. All output goes to the explicitly
provided new directory. GPU stage barriers add profiling overhead; fidelity,
camera resolution, physics rate and codec remain unchanged.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", default=None, help="Optional arm calibration unit for recording")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--kit-args", default="")
    parser.add_argument("--no-sync", action="store_true", help="Throughput check without attribution barriers")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new directory")
    args.output.mkdir(parents=True)
    os.environ.update(S2R_ARM="so101", S2R_SCENE="rpd_meeting_room", S2R_TASK="pick_place_cube")
    result = {"passed": False, "frames": args.frames, "warmup": args.warmup,
              "synchronized_stage_timing": not args.no_sync, "kit_args": args.kit_args,
              "utc_start_s": time.time(), "ticks": [], "first_raw_frames": []}
    app = env = recorder = profiler = None
    from tools.diagnostics.profiling import GpuSampler
    sampler = GpuSampler()
    sampler.start()
    try:
        result["compute_processes_before"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except Exception as error:
        result["process_query_error"] = str(error)
    try:
        from sim.app import launch, close_app
        print("PROFILE_STARTUP_BEGIN " + str(time.time()), flush=True)
        app = launch(headless=True, enable_cameras=True,
                          kit_args=args.kit_args)
        import gymnasium as gym
        import numpy as np
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import sim.envs as envs
        from core.arms.mapping import JointMapping
        from core.arms.spec import load_arm_spec
        from core.arms.calibration import resolve_calibration
        from sim.envs.mdp import as_torch
        from sim.envs.scene_builder import drop_unused_camera_outputs
        from sim.datagen.spawner import ObjectSpawner
        from sim.recording.lerobot_recorder import LeRobotRecorder, split_visual_obs
        from tools.diagnostics.profiling import TickProfiler

        profiler = TickProfiler((lambda: None) if args.no_sync else torch.cuda.synchronize)
        profiler.install()
        cfg = parse_env_cfg(envs.TASK.gym_id("teleop"), device="cuda:0", num_envs=1, use_fabric=True)
        cfg.seed = 101
        drop_unused_camera_outputs(cfg, depth=False, instance_id_seg=False)
        assert cfg.decimation == 4 and cfg.sim.render_interval == 4
        result["configuration"] = {"physics_dt_s": cfg.sim.dt, "decimation": cfg.decimation,
                                   "render_interval": cfg.sim.render_interval,
                                   "cameras": 2, "resolution": [640, 480], "fps": 30}
        env = gym.make(envs.TASK.gym_id("teleop"), cfg=cfg)
        e = env.unwrapped
        obs, _ = env.reset()
        result["runtime"] = {
            "sim_has_gui": e.sim.has_gui,
            "visualizers": [{"type": type(viz).__name__,
                             "runtime_headless": getattr(viz, "_runtime_headless", None),
                             "configured_headless": getattr(viz.cfg, "headless", None),
                             "pumps_app_update": viz.pumps_app_update()} for viz in e.sim.visualizers],
            "carb_settings": {key: e.sim.get_setting(key) for key in (
                "/persistent/UJITSO/geometry", "/UJITSO/geometry", "/rtx-transient/hydra/geometrystreaming/active",
                "/rtx/hydra/readTransformsFromFabricInRenderDelegate",
                "/rtx/hydra/readGPUTransformsFromFabricInRenderDelegate", "/app/window/enabled")},
        }
        from pxr import UsdRender
        from omni.kit.viewport.window import get_viewport_window_instances
        result["runtime"]["viewport_windows"] = [
            {"render_product": str(window.viewport_api.render_product_path),
             "updates_enabled": bool(window.viewport_api.updates_enabled)}
            for window in get_viewport_window_instances()]
        result["runtime"]["render_products"] = [
            {"path": str(prim.GetPath()), "resolution": list(UsdRender.Product(prim).GetResolutionAttr().Get()),
             "camera": list(map(str, UsdRender.Product(prim).GetCameraRel().GetTargets()))}
            for prim in e.sim.stage.Traverse() if prim.IsA(UsdRender.Product)]
        robot = e.scene["robot"]
        # Capture resolved physics as well as the outer control tick. Internal
        # substeps, contact changes and camera-response optimizations can change
        # between runs without changing decimation or render_interval.
        from isaaclab_newton.physics import NewtonManager
        model, solver = NewtonManager.get_model(), NewtonManager._solver
        mj = solver.mj_model
        shape_bodies = model.shape_body.numpy()
        body_masses = model.body_mass.numpy()
        cube_bodies = sorted({int(shape_bodies[index]) for index, label in enumerate(model.shape_label)
                              if "/CUBE/" in label and int(shape_bodies[index]) >= 0})
        result["configuration"].update({
            "physics_num_substeps": cfg.sim.physics.num_substeps,
            "solver_dt_s": float(NewtonManager.get_solver_dt()),
            "solver_steps_per_control_tick": cfg.decimation * cfg.sim.physics.num_substeps,
            "collision_decimation": cfg.sim.physics.collision_decimation,
            "solver": {name: getattr(cfg.sim.physics.solver_cfg, name) for name in
                       ("integrator", "use_mujoco_contacts", "cone", "impratio", "enable_multiccd")},
            "cube_mass_requested_kg": cfg.scene.blue_cube.spawn.mass_props.mass,
            "cube_mass_resolved_kg": {model.body_label[index]: float(body_masses[index]) for index in cube_bodies},
            "actuators": {name: {field: getattr(actuator, field) for field in
                                 ("joint_names_expr", "effort_limit_sim", "stiffness", "damping")}
                          for name, actuator in cfg.scene.robot.actuators.items()},
            "resolved_actuator_forceranges": [
                {"name": mj.actuator(index).name, "limited": bool(mj.actuator_forcelimited[index]),
                 "range": mj.actuator_forcerange[index].tolist()}
                for index in range(mj.nu)],
            "resolved_joint_actuator_force_ranges": [
                {"name": mj.joint(index).name, "limited": bool(mj.jnt_actfrclimited[index]),
                 "range": mj.jnt_actfrcrange[index].tolist()} for index in range(mj.njnt)],
            "robot_collision_geoms": 0,
            "resolved_contact_geoms": [],
            "camera_response_byte_lut_active": {
                name: getattr(getattr(e.scene[name], "_response", None), "_byte_lut", None) is not None
                for name in ("camera_realsense_rgb", "camera_wrist_cam")},
        })
        geom_to_shape = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
        shape_mu = model.shape_material_mu.numpy()
        gpu_friction = solver.mjw_model.geom_friction.numpy().reshape(-1, mj.ngeom, 3)[0]
        for geom in range(mj.ngeom):
            shape = int(geom_to_shape[geom])
            label = model.shape_label[shape] if shape >= 0 else mj.geom(geom).name
            if "/Robot/" in label:
                result["configuration"]["robot_collision_geoms"] += 1
            if any(part in label for part in ("/CUBE/", "/Robot/gripper/", "/Robot/jaw/", "C_Tabletop_")):
                result["configuration"]["resolved_contact_geoms"].append({
                    "path": label, "model_mu": float(shape_mu[shape]) if shape >= 0 else None,
                    "cpu_friction": mj.geom_friction[geom].tolist(), "gpu_friction": gpu_friction[geom].tolist(),
                    "condim": int(mj.geom_condim[geom]), "solimp": mj.geom_solimp[geom].tolist(),
                    "solref": mj.geom_solref[geom].tolist(),
                })
        # The 20 mm spawn fixture needs an open jaw; closed-jaw overlap is not
        # a valid way to profile normal recording physics.
        opening = as_torch(robot.data.joint_pos).clone()
        jaw = robot.find_joints("Jaw")[0][0]
        opening[:, jaw] = 0.6
        for _ in range(30):
            obs, *_ = env.step(opening)
        assert float(as_torch(robot.data.joint_pos)[0, jaw]) > 0.5
        spawner = ObjectSpawner(e, task=envs.TASK, port=0)
        spawner.spawn("blue_cube")
        cameras = {"realsense_rgb": {"height": 480, "width": 640},
                   "wrist_cam": {"height": 480, "width": 640}}
        recorder = LeRobotRecorder("Performance profile", "local/performance-profile",
                                   str(args.output / "dataset"), 30, e.device, cameras=cameras)
        recorder.init_dataset()
        arm, _ = load_arm_spec()
        selection = resolve_calibration(arm, unit=args.unit) if args.unit else None
        mapping = JointMapping.from_arm(arm, selection, device=e.device)
        if selection is not None:
            from core.datasets.alignment import recording_contract, mark_recording_root
            contract = recording_contract(follower_id=None, leader_id=None, calibration_selection=selection)
            mark_recording_root(args.output / "dataset", contract)
        initial = as_torch(robot.data.joint_pos).clone()
        limits = as_torch(robot.data.joint_pos_limits)
        phases = torch.arange(6, device=e.device) * 0.7
        for camera in cameras:
            raw = as_torch(e.scene["camera_" + camera].data.output["rgb"])[0].cpu().numpy()
            result["first_raw_frames"].append({
                "tick": -1, "camera": camera, "utc_s": time.time(),
                "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                "spatial_std": raw.reshape(-1, 3).std(axis=0).tolist(),
                "shape": list(raw.shape), "all_zero": bool(not raw.any()),
            })
        print("PROFILE_RECORDING_BEGIN " + str(time.time()), flush=True)
        sampler.phase = "recording"
        with torch.inference_mode():
            for tick in range(args.warmup + args.frames):
                started = time.perf_counter()
                profiler.begin_tick()
                target = initial + 0.08 * torch.sin(phases + tick * 0.04)
                target = torch.clamp(target, limits[..., 0] + 0.01, limits[..., 1] - 0.01)
                # Match the recorder loop: write the completed observation before advancing simulation.
                with profiler.measure("dataset_frame_write"):
                    rgb, _, _ = split_visual_obs(obs["visual"], cameras)
                    frame = recorder.make_sim_frame(mapping.to_raw(target[0]),
                                                    mapping.to_raw(obs["policy"]["joint_pos_obs"][0]), rgb)
                    recorder.add_frame(frame)
                with profiler.measure("object_parking"):
                    spawner.hold()
                obs, *_ = env.step(target)
                # Inspect raw RTX buffers, bypassing fitted response/noise, during startup only.
                if tick < 10:
                    for camera in cameras:
                        sensor = e.scene["camera_" + camera]
                        raw = as_torch(sensor.data.output["rgb"])[0].cpu().numpy()
                        result["first_raw_frames"].append({
                            "tick": tick, "camera": camera, "utc_s": time.time(),
                            "frame": np.asarray(as_torch(sensor.frame).cpu()).tolist() if hasattr(sensor.frame, "torch") else str(sensor.frame),
                            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                            "spatial_std": raw.reshape(-1, 3).std(axis=0).tolist(),
                            "shape": list(raw.shape), "all_zero": bool(not raw.any()),
                        })
                if not args.no_sync:
                    torch.cuda.synchronize()
                busy = time.perf_counter() - started
                sleep_requested = max(0.0, 1 / 30 - busy)
                sleep_started = time.perf_counter()
                if sleep_requested:
                    time.sleep(sleep_requested)
                elapsed = time.perf_counter() - started
                row = {"tick": tick, "utc_s": time.time(), "warmup": tick < args.warmup,
                       **{name + "_ms": value * 1000 for name, value in profiler.current.items()},
                       **{name + "_calls": value for name, value in profiler.calls.items()},
                       "busy_ms": busy * 1000, "sleep_ms": (time.perf_counter() - sleep_started) * 1000,
                       "total_ms": elapsed * 1000, "sleep_requested_ms": sleep_requested * 1000}
                row["unattributed_ms"] = max(0.0, row["busy_ms"] - sum(
                    value * 1000 for value in profiler.current.values()))
                result["ticks"].append(row)
                if tick in (0, 9, args.warmup - 1, args.warmup + args.frames - 1):
                    print("PROFILE_TICK " + json.dumps(row), flush=True)
        assert torch.isfinite(as_torch(robot.data.joint_pos)).all(), "Nonfinite recording joint state"
        profiler.current = None
        result["runtime"]["viewport_windows_after_recording"] = [
            {"render_product": str(window.viewport_api.render_product_path),
             "updates_enabled": bool(window.viewport_api.updates_enabled)}
            for window in get_viewport_window_instances()]
        # A stationary parked-object sequence isolates repeated park writes for visual comparison.
        sampler.phase = "parked_artifacts"
        from PIL import Image
        import sim.envs.mdp as mdp
        original_park = mdp.park_object
        park_writes = [0]
        def counted_park(*values, **options):
            park_writes[0] += 1
            return original_park(*values, **options)
        mdp.park_object = counted_park
        spawner.reset()
        try:
            for frame_index in range(121):
                spawner.hold()
                env.step(initial)
                if frame_index in (0, 10, 60, 120):
                    raw = as_torch(e.scene["camera_realsense_rgb"].data.output["rgb"])[0].cpu().numpy()
                    Image.fromarray(raw).save(args.output / f"parked-{frame_index:03d}.png")
        finally:
            mdp.park_object = original_park
        result["parked_pose_write_calls"] = park_writes[0]
        sampler.phase = "save"
        saved = time.perf_counter()
        recorder.save_episode()
        recorder.finalize()
        recorder = None
        result["save_finalize_ms"] = (time.perf_counter() - saved) * 1000
        # Decode every saved frame. A metadata count alone cannot establish camera fidelity.
        import av
        import pyarrow.parquet as pq
        dataset = args.output / "dataset"
        expected = args.warmup + args.frames
        result["parquet_frames"] = sum(pq.read_table(path).num_rows for path in (dataset / "data").rglob("*.parquet"))
        result["video_frames"] = {}
        result["decoded_first_frame_spatial_std"] = {}
        for path in (dataset / "videos").rglob("*.mp4"):
            with av.open(str(path)) as container:
                count = 0
                for frame in container.decode(video=0):
                    if count == 0:
                        pixels = frame.to_ndarray(format="rgb24")
                        result["decoded_first_frame_spatial_std"][str(path.relative_to(dataset))] = pixels.reshape(-1, 3).std(axis=0).tolist()
                    count += 1
                result["video_frames"][str(path.relative_to(dataset))] = count
        assert result["parquet_frames"] == expected, result["parquet_frames"]
        assert len(result["video_frames"]) == 2 and set(result["video_frames"].values()) == {expected}, result["video_frames"]
        assert not any(frame["all_zero"] for frame in result["first_raw_frames"])
        rows = [row for row in result["ticks"] if not row["warmup"]]
        fields = sorted(name for name in set().union(*(row.keys() for row in rows)) if name.endswith("_ms"))
        result["summary_ms"] = {name: {"mean": float(np.mean([row.get(name, 0) for row in rows])),
                                               "max": float(max(row.get(name, 0) for row in rows)),
                                               "p95": float(np.percentile([row.get(name, 0) for row in rows], 95))}
                                for name in fields}
        result["summary_calls"] = {
            name: {"mean": float(np.mean([row.get(name, 0) for row in rows])),
                   "max": max(row.get(name, 0) for row in rows)}
            for name in sorted(set().union(*(row.keys() for row in rows))) if name.endswith("_calls")}
        result["effective_fps"] = 1000 / result["summary_ms"]["total_ms"]["mean"]
        result["mean_headroom_ms"] = 1000 / 30 - result["summary_ms"]["busy_ms"]["mean"]
        result["recorded_rgb_bytes_per_tick"] = 2 * 640 * 480 * 3
        result["deadline_misses"] = sum(row["busy_ms"] > 1000 / 30 for row in rows)
        result["fidelity_passed"] = True
        result["holds_30hz_with_headroom"] = result["mean_headroom_ms"] > 1 and result["deadline_misses"] == 0
        result["passed"] = True
    except BaseException:
        result["exception"] = traceback.format_exc()
    finally:
        try:
            sampler.close()
            result["gpu_samples"] = sampler.samples
            result["gpu_sampler_error"] = sampler.error
            steady = [sample for sample in sampler.samples if sample["phase"] == "recording"]
            if steady:
                result["gpu_summary"] = {key: {"mean": sum(s[key] for s in steady) / len(steady),
                                               "max": max(s[key] for s in steady)}
                                         for key in ("gpu_util_percent", "memory_util_percent", "vram_used_mib")}
            if profiler:
                profiler.close()
            if recorder:
                recorder.finalize()
            if env:
                env.close()
            result["utc_end_s"] = time.time()
            (args.output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
            if result["ticks"]:
                fields = sorted(set().union(*(row.keys() for row in result["ticks"])))
                with (args.output / "ticks.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(result["ticks"])
            print("PROFILE_RESULT " + json.dumps({key: value for key, value in result.items()
                                                   if key not in ("ticks", "gpu_samples", "first_raw_frames")}), flush=True)
        finally:
            if app:
                close_app(app, success=result["passed"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
