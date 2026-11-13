"""Render one dataset episode in its native scene, with both recorded cameras.

Writes joint observations and every tracked object pose; does not step physics
or synthesize a new trajectory. The source dataset is read-only.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
from sim.app import add_launch_arguments, apply_selection, launch, close_app


class VideoWriter:
    def __init__(self, path, fps, width, height):
        import av
        self.av = av
        self.container = av.open(str(path), 'w', options={'movflags': '+faststart'})
        self.stream = self.container.add_stream('libx264', rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '18', 'preset': 'fast'}
        self.fps, self.frames = fps, 0

    def write(self, pixels):
        frame = self.av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels), format='rgb24')
        frame.pts, frame.time_base = self.frames, Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--repo_id', default='local/stack_bowls_one_episode')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--scene', default='rpd_meeting_room')
    parser.add_argument('--variant', default='teleop')
    parser.add_argument('--export_controls', action='store_true',
                        help='Export lossless robot/object foreground masks and individual RGB videos')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Limit a diagnostic replay; omitted for the complete episode')
    parser.add_argument('--resolution_scale', type=int, choices=(1, 2, 3, 4), default=1,
                        help='Render more pixels at the same camera framing and calibrated field of view')
    add_launch_arguments(parser)
    if not any(action.dest=='headless' for action in parser._actions):
        parser.add_argument('--headless',action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    apply_selection(args)
    from core.arms.spec import load_arm_spec
    from core.tasks.loader import load_task
    from core.datasets.sources import load_source
    task, arm = load_task(args.task), load_arm_spec(args.arm)[0]
    sources = load_source(args.dataset, args.repo_id, set(), arm=arm, task=task)
    source = next((ep for ep in sources if ep.index == args.episode), None)
    if source is None:
        raise ValueError(f'Dataset has no episode {args.episode}')
    fps = int(source.fps)
    frame_count = source.length if args.max_frames is None else min(source.length, args.max_frames)
    if frame_count < 1:
        raise ValueError('Replay requires at least one frame')
    if source.fps != fps or fps < 1:
        raise ValueError('Video rendering requires a positive integer dataset FPS')
    args.output.mkdir(parents=True)
    metadata = args.dataset / 'task_meta' / f'episode_{args.episode:06d}.json'
    report = dict(complete=False, dataset=str(args.dataset.resolve()), episode=args.episode,
        task=task.name, arm=arm.name, scene=args.scene, instruction=source.task,
        tracked_objects=list(source.objects_base), fps=fps, expected_frames=frame_count,
        source_sidecar_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        method='Native Isaac Sim replay of recorded observations and all tracked object poses',
        render_resolution_scale=args.resolution_scale,
        physics_steps_during_render=0)
    app = env = writer = None
    camera_writers = {}
    success = False
    try:
        app = launch(args, enable_cameras=True)
        import gymnasium as gym
        import torch
        from PIL import Image, ImageDraw
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
        import sim.envs as envs
        from sim.envs.scene_builder import drop_unused_camera_outputs
        from sim.datagen.dr_replay import make_mapping, discover_sim_cameras
        from sim.recording.replay_scene import SceneReplayer
        from sim.envs.mdp import as_torch
        from isaaclab.sim.utils import find_matching_prim_paths
        from core.scenes.anchors import quaternion_matrix
        envs.register_envs(scene=args.scene, task=args.task)
        cfg = parse_env_cfg(task.gym_id(args.variant), device=args.device, num_envs=1, use_fabric=True)
        cfg.events.reset_from_starts = None
        if hasattr(cfg.events, 'check_eval_order'):
            cfg.events.check_eval_order = None
        cfg.terminations = None
        drop_unused_camera_outputs(cfg, depth=False, instance_id_seg=args.export_controls)
        if args.resolution_scale != 1:
            for sensor_cfg in vars(cfg.scene).values():
                if not hasattr(sensor_cfg, 'renderer_cfg') or not hasattr(sensor_cfg, 'width'):
                    continue
                scale = args.resolution_scale
                sensor_cfg.width *= scale
                sensor_cfg.height *= scale
                intrinsics = getattr(sensor_cfg.spawn, 'intrinsics', None)
                if intrinsics is not None:
                    sensor_cfg.spawn.intrinsics = {
                        **intrinsics,
                        'width': sensor_cfg.width, 'height': sensor_cfg.height,
                        'fx': intrinsics['fx']*scale, 'fy': intrinsics['fy']*scale,
                        # Preserve the same rays with RTX's half-pixel offset.
                        'cx': (intrinsics['cx']+.5)*scale-.5,
                        'cy': (intrinsics['cy']+.5)*scale-.5,
                    }
        if args.export_controls:
            for sensor_cfg in vars(cfg.scene).values():
                if hasattr(sensor_cfg, 'renderer_cfg') and sensor_cfg.renderer_cfg is not None:
                    sensor_cfg.renderer_cfg.colorize_instance_id_segmentation = False
        env = gym.make(task.gym_id(args.variant), cfg=cfg)
        env.reset()
        raw = env.unwrapped
        cameras = list(discover_sim_cameras(env))
        mapping = make_mapping(env, arm=arm, source_root=args.dataset)
        replay = SceneReplayer(raw, mapping, None, cameras, task=task.name, arm=arm,
                               render_passes=8 if args.resolution_scale > 1 else 2)
        protected_roots = [raw.scene.env_prim_paths[0] + '/Robot']
        for asset in replay.objects.values():
            paths = find_matching_prim_paths(asset.cfg.prim_path)
            if len(paths) != 1:
                raise RuntimeError(f'Expected one tracked object prim: {asset.cfg.prim_path}: {paths}')
            protected_roots.extend(paths)
        report['protected_prim_roots'] = protected_roots
        start_step = raw.sim.get_physics_step_count()
        report['cameras'] = cameras
        samples = {0, frame_count//2, frame_count-1}
        joint_errors = np.zeros(len(arm.action_joints))
        position_error = 0.
        report['joint_names'] = list(arm.joint_names)
        report['joint_error_units'] = ['rad' if j.type=='revolute' else 'm' for j in arm.action_joints]
        with torch.inference_mode():
            for i in range(frame_count):
                replay.pose_frame(source.states[i], {name:p[i] for name,p in source.objects_base.items()})
                raw.sim.render()
                ensure_isaac_rtx_render_update(force=True)
                images = replay.render_images()
                measured = as_torch(replay.robot.data.joint_pos)[0,replay.joint_ids].cpu().numpy()
                desired = mapping.to_sim(source.states[i]).cpu().numpy().reshape(-1)
                joint_errors = np.maximum(joint_errors, np.abs(measured-desired))
                base_p = as_torch(replay.robot.data.body_link_pos_w)[0,replay.base_body_id].cpu().numpy()
                base_q = as_torch(replay.robot.data.body_link_quat_w)[0,replay.base_body_id].cpu().numpy()
                for name, asset in replay.objects.items():
                    expected = base_p + quaternion_matrix(base_q) @ source.objects_base[name][i,:3]
                    actual = as_torch(asset.data.root_pos_w)[0].cpu().numpy()
                    position_error = max(position_error, float(np.linalg.norm(actual-expected)))
                if np.any(joint_errors > 1e-5) or position_error > 1e-5:
                    raise RuntimeError('Rendered pose differs from the recorded trajectory')
                arrays = []
                for camera in cameras:
                    a = images[camera]
                    if hasattr(a, 'detach'):
                        a = a.detach().cpu().numpy()
                    if a.ndim == 4:
                        a = a[0]
                    a = np.asarray(a[...,:3], dtype=np.uint8)
                    arrays.append(a)
                    if args.export_controls:
                        control_dir = args.output / 'controls' / camera
                        mask_dir = control_dir / 'foreground'
                        mask_dir.mkdir(parents=True, exist_ok=True)
                        sensor = raw.scene['camera_' + camera]
                        kind = 'instance_id_segmentation_fast'
                        labels = sensor.data.info[kind]['idToLabels']
                        ids = []
                        for instance_id, label in labels.items():
                            path = str(label)
                            if any(path == root or path.startswith(root + '/') for root in protected_roots):
                                ids.append(int(instance_id))
                        if not ids:
                            raise RuntimeError(f'No protected segmentation labels for {camera}: {labels}')
                        segments = as_torch(sensor.data.output[kind])[0].cpu().numpy().squeeze(-1)
                        mask = np.isin(segments, ids).astype(np.uint8) * 255
                        Image.fromarray(mask).save(mask_dir / f'{i:06d}.png')
                        if camera not in camera_writers:
                            camera_writers[camera] = VideoWriter(control_dir/'rgb.mp4', fps, a.shape[1], a.shape[0])
                            (control_dir/'instance-labels.json').write_text(json.dumps(labels, indent=2)+'\n')
                        camera_writers[camera].write(a)
                height = max(a.shape[0] for a in arrays)
                width = sum(a.shape[1] for a in arrays)
                canvas = Image.new('RGB', (width, height+48), '#15191f')
                draw = ImageDraw.Draw(canvas)
                left = 0
                for camera, a in zip(cameras, arrays):
                    canvas.paste(Image.fromarray(a), (left,48))
                    draw.text((left+10,29), camera, fill='white')
                    left += a.shape[1]
                draw.text((10,8), f'{source.task}  |  episode {args.episode}  |  {i/fps:.1f}s', fill='white')
                pixels = np.asarray(canvas)
                if writer is None:
                    writer = VideoWriter(args.output/'episode.mp4', fps, width, height+48)
                writer.write(pixels)
                if i in samples:
                    canvas.save(args.output/f'frame_{i:06d}.png')
                if i % 100 == 0:
                    print(f'[RENDER] {i}/{frame_count}', flush=True)
        report['physics_steps_during_render'] = raw.sim.get_physics_step_count()-start_step
        report['frames_written'] = writer.frames
        report['maximum_joint_write_errors_sim'] = joint_errors.tolist()
        report['maximum_object_position_write_error_m'] = position_error
        if report['physics_steps_during_render'] != 0 or writer.frames != frame_count:
            raise RuntimeError('Replay changed physics time or lost frames')
        if args.export_controls:
            report['foreground_masks'] = {}
            for camera in cameras:
                control_dir = args.output / 'controls' / camera
                masks = np.stack([np.asarray(Image.open(control_dir/'foreground'/f'{i:06d}.png'))
                                  for i in range(frame_count)])
                path = control_dir/'foreground.npz'
                np.savez_compressed(path, masks)
                report['foreground_masks'][camera] = dict(frames=frame_count, shape=list(masks.shape),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        success = report['complete'] = True
    finally:
        try:
            if writer is not None:
                writer.close()
            for camera_writer in camera_writers.values():
                camera_writer.close()
            if env is not None:
                env.close()
            (args.output/'render-report.json').write_text(json.dumps(report, indent=2)+'\n')
        finally:
            if app is not None:
                close_app(app, success=success)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
