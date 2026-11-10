"""Render measured free-physics replay states without stepping their physics again.

Run inside the existing s2r-sim container after the physics replay has finished:
  python tools/diagnostics/render_grasp_replay.py --trajectory outputs/grasp_replay/winner.npz \
    --output-dir outputs/grasp_replay/winner-render

The NPZ contains actual joint positions and cube world poses sampled before
each action at 30 Hz. This renderer never reads a source dataset or sidecar.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import numpy as np


JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
CAMERA_NAMES = ("realsense_rgb", "wrist_cam")
ARRAY_WIDTHS = {
    "joint_pos_rad": 6,
    "cube_pose_world_xyzw": 7,
    "robot_base_pose_world_xyzw": 7,
    "action_raw": 6,
}


def load_trajectory(path):
    """Validate the captured states before creating Kit or touching the GPU."""
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ARRAY_WIDTHS) | {"time_s"}
        missing -= set(archive.files)
        if missing:
            raise ValueError(f"Trajectory is missing {sorted(missing)}")
        data = {key: np.array(archive[key], copy=True) for key in (*ARRAY_WIDTHS, "time_s")}
        if "joint_names" in archive and tuple(archive["joint_names"].tolist()) != JOINT_NAMES:
            raise ValueError(f"joint_names must be ordered {JOINT_NAMES}")
    times = data["time_s"]
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("time_s must contain at least two sample times")
    count = len(times)
    for key, width in ARRAY_WIDTHS.items():
        if data[key].shape != (count, width):
            raise ValueError(f"{key} shape {data[key].shape}; expected {(count, width)}")
    for key, values in data.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN/Inf")
    intervals = np.diff(times)
    if not np.allclose(intervals, 1 / 30, rtol=0, atol=1e-5):
        raise ValueError("Captured states must be uniformly sampled at 30 Hz")
    for key in ("cube_pose_world_xyzw", "robot_base_pose_world_xyzw"):
        if not np.allclose(np.linalg.norm(data[key][:, 3:], axis=1), 1, rtol=0, atol=1e-3):
            raise ValueError(f"{key} contains non-unit XYZW quaternions")
    return data


def pose_error(actual, expected):
    """Metres and quaternion-sign-invariant orientation error in radians."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    position = np.linalg.norm(actual[:3] - expected[:3])
    cosine = abs(float(np.dot(actual[3:], expected[3:])))
    cosine /= np.linalg.norm(actual[3:]) * np.linalg.norm(expected[3:])
    angle = 2 * np.arccos(np.clip(cosine, -1, 1))
    return float(position), float(angle)


class VideoWriter:
    def __init__(self, path, shape):
        import av

        height, width, channels = shape
        if channels != 3 or height % 2 or width % 2:
            raise ValueError(f"H.264 expects even RGB image dimensions; got {shape}")
        self.container = av.open(str(path), mode="w", options={"movflags": "+faststart"})
        self.stream = self.container.add_stream("libx264", rate=30)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "18", "preset": "fast"}
        self.frames = 0
        self.closed = False

    def write(self, pixels):
        import av

        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format="rgb24")
        frame.pts = self.frames
        frame.time_base = Fraction(1, 30)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        if not self.closed:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.closed = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_replay/render"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="Exclusive end; defaults to the trajectory length")
    parser.add_argument("--robot-base-z-offset", type=float, default=0.,
                        help="Diagnostic mount offset in metres; must match the captured trajectory")
    parser.add_argument("--robot-base-offset", type=float, nargs=3, default=(0., 0., 0.),
                        metavar=("X", "Y", "Z"),
                        help="Diagnostic XYZ mount offset in metres; Z adds to --robot-base-z-offset")
    parser.add_argument("--validate-only", action="store_true", help="Validate the NPZ on CPU and exit")
    args = parser.parse_args()
    base_offset = np.asarray(args.robot_base_offset) + [0., 0., args.robot_base_z_offset]
    if not np.isfinite(base_offset).all():
        parser.error("robot base offsets must be finite")
    has_mount_offset = bool(np.any(base_offset != 0))
    data = load_trajectory(args.trajectory)
    count = len(data["time_s"])
    end = count if args.end_frame is None else args.end_frame
    if not 0 <= args.start_frame < end <= count:
        parser.error(f"Require 0 <= start-frame < end-frame <= {count}")
    report = {
        "complete": False,
        "trajectory": str(args.trajectory.resolve()),
        "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "frame_range_start_inclusive_end_exclusive": [args.start_frame, end],
        "frames_expected": end - args.start_frame,
        "fps": 30,
        "quaternion_order": "xyzw",
        "joint_names": list(JOINT_NAMES),
        "physics_steps_during_render": 0,
        "cube_motion_source": "Captured free-physics cube_pose_world_xyzw only",
        "state_timing": "Actual state before the respective action",
        "robot_base_offset_m": base_offset.tolist(),
        "robot_base_z_offset_m": float(base_offset[2]),
        "additional_robot_base_z_offset_m": args.robot_base_z_offset,
        "robot_base_offset_definition": "Additional CLI correction added to the configured scene mount",
        "scene_mount_assumption": ("Diagnostic mount offset; not the original scene mount"
                                   if has_mount_offset else "Configured scene mount (no CLI correction)"),
    }
    if args.validate_only:
        print(json.dumps(report, indent=2))
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    os.environ.update(S2R_ARM="so101", S2R_SCENE="rpd_meeting_room", S2R_TASK="pick_place_cube")
    app = env = None
    writers = {}
    try:
        from sim.app import launch, close_app

        app = launch(headless=True, enable_cameras=True)
        import gymnasium as gym
        from PIL import Image, ImageDraw
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import sim.envs as envs
        from sim.envs.mdp import as_torch
        from sim.envs.scene_builder import drop_unused_camera_outputs
        from sim.recording.replay_scene import SceneReplayer
        from core.arms.spec import load_arm_spec
        from core.scenes import load_scene_spec, load_mount_spec

        mount = load_mount_spec(load_scene_spec(), load_arm_spec()[0])

        env_id = envs.TASK.gym_id("teleop")
        cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1, use_fabric=True)
        cfg.seed = 101
        cfg.events.reset_from_starts = None
        base_position = cfg.scene.robot.init_state.pos
        cfg.scene.robot.init_state.pos = tuple(np.asarray(base_position) + base_offset)
        report["configured_robot_base_position_m"] = list(cfg.scene.robot.init_state.pos)
        report["configured_mount_correction_m"] = list(mount.translation_correction_m)
        report["configured_mount_correction_source"] = mount.source
        drop_unused_camera_outputs(cfg, depth=False, instance_id_seg=False)
        env = gym.make(env_id, cfg=cfg)
        env.reset()
        raw = env.unwrapped
        physics_steps_at_start = raw.sim._physics_step_count
        replayer = SceneReplayer(raw, None, None, CAMERA_NAMES)
        robot, cube = replayer.robot, replayer.objects["blue_cube"]
        joint_ids, resolved_names = robot.find_joints(list(JOINT_NAMES), preserve_order=True)
        if tuple(resolved_names) != JOINT_NAMES:
            raise ValueError(f"Live joint order differs: {resolved_names}")
        env_ids = torch.tensor([0], device=raw.device)
        samples = set((args.start_frame, (args.start_frame + end - 1) // 2, end - 1))
        report.update(frames_written=0, max_base_position_error_m=0., max_base_angle_error_rad=0.,
                      max_cube_position_write_error_m=0., max_cube_angle_write_error_rad=0.,
                      max_joint_write_error_rad=0., sample_image_stats=[])
        with torch.inference_mode():
            for index in range(args.start_frame, end):
                q = torch.as_tensor(data["joint_pos_rad"][index], device=raw.device,
                                    dtype=torch.float32).reshape(1, 6)
                robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q),
                                                    joint_ids=joint_ids, env_ids=env_ids)
                cube_pose = torch.as_tensor(data["cube_pose_world_xyzw"][index], device=raw.device,
                                            dtype=torch.float32).reshape(1, 7)
                cube.write_root_pose_to_sim_index(root_pose=cube_pose, env_ids=env_ids)
                cube.write_root_velocity_to_sim_index(root_velocity=q.new_zeros((1, 6)), env_ids=env_ids)
                raw.sim.forward()
                raw.scene.update(0.)
                base_pose = torch.cat((as_torch(robot.data.body_link_pos_w)[0, replayer.base_body_id],
                                       as_torch(robot.data.body_link_quat_w)[0, replayer.base_body_id])).cpu().numpy()
                distance, angle = pose_error(base_pose, data["robot_base_pose_world_xyzw"][index])
                report["max_base_position_error_m"] = max(report["max_base_position_error_m"], distance)
                report["max_base_angle_error_rad"] = max(report["max_base_angle_error_rad"], angle)
                if distance > 1e-4 or angle > 1e-3:
                    raise ValueError(f"Frame {index}: live fixed base differs by {distance} m, {angle} rad")
                actual_cube = torch.cat((as_torch(cube.data.root_pos_w)[0],
                                         as_torch(cube.data.root_quat_w)[0])).cpu().numpy()
                cube_distance, cube_angle = pose_error(actual_cube, data["cube_pose_world_xyzw"][index])
                joint_error = float(torch.max(torch.abs(as_torch(robot.data.joint_pos)[0, joint_ids] - q[0])).cpu())
                report["max_cube_position_write_error_m"] = max(report["max_cube_position_write_error_m"], cube_distance)
                report["max_cube_angle_write_error_rad"] = max(report["max_cube_angle_write_error_rad"], cube_angle)
                report["max_joint_write_error_rad"] = max(report["max_joint_write_error_rad"], joint_error)
                if cube_distance > 1e-5 or cube_angle > 1e-3 or joint_error > 1e-5:
                    raise ValueError(f"Frame {index}: captured pose did not reach the live render state")
                images = {name: as_torch(pixels).detach().cpu().numpy().astype(np.uint8)
                          for name, pixels in replayer.render_images().items()}
                for name, pixels in images.items():
                    if pixels.ndim != 3 or pixels.shape[2] != 3:
                        raise ValueError(f"Camera {name} returned shape {pixels.shape}")
                    if name not in writers:
                        writers[name] = VideoWriter(args.output_dir / f"{name}.mp4", pixels.shape)
                    writers[name].write(pixels)
                    if index in samples:
                        Image.fromarray(pixels).save(args.output_dir / f"{name}-{index:06d}.png")
                        report["sample_image_stats"].append({"source_frame": index, "camera": name,
                            "spatial_std_rgb": pixels.reshape(-1, 3).std(axis=0).tolist(),
                            "sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
                combined = Image.fromarray(np.concatenate([images[name] for name in CAMERA_NAMES], axis=1))
                draw = ImageDraw.Draw(combined)
                for x, label in ((6, "Sim external"), (images[CAMERA_NAMES[0]].shape[1] + 6, "Sim gripper")):
                    draw.text((x, 6), label, fill="white", stroke_width=2, stroke_fill="black")
                combined_pixels = np.asarray(combined)
                if "side_by_side" not in writers:
                    writers["side_by_side"] = VideoWriter(args.output_dir / "side_by_side.mp4", combined_pixels.shape)
                writers["side_by_side"].write(combined_pixels)
                report["frames_written"] += 1
                if index == args.start_frame or (index + 1) % 60 == 0 or index == end - 1:
                    print(f"[RENDER] frame={index} saved={report['frames_written']}/{end - args.start_frame}", flush=True)
        report["physics_steps_during_render"] = raw.sim._physics_step_count - physics_steps_at_start
        if report["physics_steps_during_render"] != 0:
            raise RuntimeError("Rendering advanced physics")
        report["video_frames_decoded"] = {}
        import av

        for name, writer in writers.items():
            writer.close()
            with av.open(str(args.output_dir / f"{name}.mp4")) as video:
                decoded = sum(1 for _ in video.decode(video=0))
                if video.streams.video[0].codec_context.name != "h264":
                    raise RuntimeError(f"{name} did not encode as H.264")
            report["video_frames_decoded"][name] = decoded
            if decoded != report["frames_expected"]:
                raise RuntimeError(f"{name} decoded {decoded} frames; expected {report['frames_expected']}")
        report["complete"] = report["frames_written"] == report["frames_expected"]
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            for writer in writers.values():
                writer.close()
            report["video_frames_submitted"] = {name: writer.frames for name, writer in writers.items()}
            if env is not None:
                env.close()
            # Kit closes the process; persist verification before requesting shutdown.
            (args.output_dir / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            if app is not None:
                close_app(app, success=report.get("complete", False))


if __name__ == "__main__":
    main()
