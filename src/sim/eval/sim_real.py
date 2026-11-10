"""Matched-start, real-then-sim policy trials using the existing browser GUI."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import subprocess
from pathlib import Path
import time

import numpy as np


def limited_target(action, state, keys, ranges, *, first, start_limit, step_limit):
    from core.rollout.real_rollout import clamp_action

    state = np.asarray(state, dtype=np.float32)
    if state.shape != (len(keys),) or not np.isfinite(state).all():
        raise ValueError('Expected finite measured joint positions in arm order')
    target = clamp_action(action, keys, ranges)
    if first and np.max(np.abs(target - state)) > start_limit:
        raise ValueError(f'First policy target exceeds {start_limit} raw units from the starting arm pose')
    return np.clip(target, state - step_limit, state + step_limit)


def position_with_leader(leader, follower, measured, keys, ranges, *, enabled, motion, step_limit):
    """Operator positioning only; never read/send leader targets during a policy trial or after Stop."""
    if not enabled or leader is None:
        return np.asarray(measured, dtype=np.float32)
    values = limited_target(leader.get_action(), measured, keys, ranges,
                            first=False, start_limit=0, step_limit=step_limit)
    if motion:
        follower.send_action(dict(zip(keys, map(float, values))))
    return values


def connect_trial_policies(factory, executor):
    """Connect a complete pair, closing partial connections if startup fails."""
    connected = []
    try:
        for _ in range(2):
            policy = factory()
            connected.append(policy)
            executor.submit(policy.connect).result()
        for policy in connected:
            executor.submit(policy.reset).result()
        return connected
    except Exception:
        for policy in connected:
            close = getattr(policy, 'close', None)
            if close is not None:
                executor.submit(close).result()
        raise


def parse_args(argv=None):
    from core.rollout.real_rollout import build_parser
    from core.utils.config import parse_yaml_args
    from core.tasks.loader import load_task
    from sim.app import add_launch_arguments

    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--leader-port', default=os.getenv('TELEOP_PORT'))
    parser.add_argument('--leader-id', default=os.getenv('TELEOP_ID'))
    parser.add_argument('--no-leader', action='store_true', help='Disable operator positioning through the leader')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--shadow', action='store_true', help='Preview/infer without sending real arm commands')
    parser.add_argument('--backend', choices=('s2r', 'local-lerobot'), default='s2r')
    parser.add_argument('--gui-host', default='127.0.0.1')
    parser.add_argument('--gui-port', type=int, default=8765)
    parser.add_argument('--preview-seconds', type=float, default=0,
                        help='Exit after a preview; combine with --shadow to prevent physical motion')
    add_launch_arguments(parser)
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if not any(arg.startswith('--config_path') for arg in argv):
        argv = ['--config_path', 'configs/eval/sim_real.yaml', *argv]
    args = parse_yaml_args(parser, argv, json_fields=('rename_map',))
    if args.lang_instruction is None:
        args.lang_instruction = load_task(args.task).instruction
    if args.shadow:
        args.enable_motion = False
    if args.backend == 'local-lerobot' and args.policy != 'molmoact2':
        parser.error('local-lerobot backend serves MolmoAct2 only')
    return args


class TrialVideo:
    def __init__(self, path, fps):
        import cv2
        self.cv2 = cv2
        self.fps, self.frames, self.last_frame = fps, 0, None
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f'Video already exists: {self.path}')
        self.error_path = self.path.with_suffix('.ffmpeg.log')
        self.error_log = self.error_path.open('wb')
        # Kit's OpenCV lacks a usable MP4 writer. Encode in the system FFmpeg
        # process, isolated from Kit's loaded codec libraries.
        environment = dict(os.environ)
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('LD_PRELOAD', None)
        try:
            self.writer = subprocess.Popen([
                '/usr/bin/ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', '1280x480',
                '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                '-preset', 'ultrafast', '-crf', '23', '-threads', '2', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.error_log,
                env=environment, bufsize=1024 * 1024)
        except BaseException:
            self.error_log.close()
            raise
        self.closed = False

    def _write_frame(self, frame):
        try:
            self.writer.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as error:
            self.writer.wait(timeout=10)
            detail = self.error_path.read_text(errors='replace').strip()
            raise RuntimeError(f'FFmpeg failed for {self.path}: {detail}') from error

    def write(self, images, label, elapsed):
        cv2 = self.cv2
        frame = np.concatenate([cv2.resize(np.asarray(images[key])[..., :3], (640, 480))
                                for key in ('room', 'wrist')], axis=1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        expected = int(elapsed * self.fps)
        while self.frames < expected:
            self._write_frame(self.last_frame if self.last_frame is not None else frame)
            self.frames += 1
        self._write_frame(frame)
        self.frames += 1
        self.last_frame = frame

    def close(self, elapsed=None):
        if self.closed:
            return
        if elapsed is not None and self.last_frame is not None:
            while self.frames < round(elapsed * self.fps):
                self._write_frame(self.last_frame)
                self.frames += 1
        self.closed = True
        try:
            try:
                self.writer.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.writer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.writer.kill()
                self.writer.wait()
                raise RuntimeError(f'FFmpeg did not finish {self.path}')
            if code:
                detail = self.error_path.read_text(errors='replace').strip()
                raise RuntimeError(f'FFmpeg failed for {self.path}: {detail}')
        finally:
            self.error_log.close()


def run(args, app):
    import gymnasium as gym
    import torch
    import sim.envs
    from core.arms.mapping import JointMapping
    from core.policy_client.contract import ArmInfo
    from core.rollout.real_rollout import connect_follower, joint_ranges, make_policy, camera_specs
    from sim.teleop.agent import _env_id, _make_env_cfg, _sim_cameras
    from sim.teleop.config import AgentConfig
    from sim.teleop.loop import gui_frames, gui_panes, object_pose_in_base
    from sim.datagen.spawner import ObjectSpawner
    from sim.recording.web_gui import RecordingWebGui
    from sim.recording.recording_setup import cleanup, disconnect
    from sim.envs.mdp._common import as_torch

    env = follower = leader = gui = spawner = video = log = None
    policies = []
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paired-policy')
    future = None
    future_side = None
    output = Path(args.output or f'outputs/eval-sim-real/{datetime.now():%Y%m%d_%H%M%S}')
    output.mkdir(parents=True, exist_ok=True)
    report = {'arm': args.arm, 'scene': args.scene, 'task': args.task, 'policy': args.policy,
              'backend': args.backend, 'checkpoint_label': args.checkpoint,
              'seconds_per_trial': args.episode_seconds, 'motion_enabled': args.enable_motion,
              'real_object_pose': 'manually matched at jaw; not measured', 'trials': [],
              'calibration': args.calibration_selection.provenance}
    try:
        cfg = AgentConfig(arm=args.arm, scene=args.scene, task=args.task, variant=args.variant, fps=args.fps)
        env_id = _env_id(cfg)
        env = gym.make(env_id, cfg=_make_env_cfg(cfg, env_id, args.device, args.calibration_selection))
        obs, _ = env.reset()
        arm = args.arm_spec
        mapping = JointMapping.from_arm(arm, args.calibration_selection, device=env.unwrapped.device)
        robot = env.unwrapped.scene['robot']
        joint_ids = robot.find_joints(list(arm.usd_joint_names), preserve_order=True)[0]
        base_id = robot.find_bodies(arm.links.base)[0][0]
        ids = torch.tensor([0], device=env.unwrapped.device)
        spawner = ObjectSpawner(env, task=sim.envs.TASK)
        sim_cameras = _sim_cameras(env)
        real_cameras = camera_specs(args)
        follower = connect_follower(args)
        if not args.no_leader:
            from sim.teleop.devices import connect_device
            leader = connect_device('leader', args.leader_port, args.leader_id, cameras={}, fps=args.fps,
                                    selection=args.calibration_selection, calibrate=False)
        keys = list(arm.feature_names)
        ranges = joint_ranges(arm, args.calibration_selection.mapping)
        real_names = json.loads(args.rename_map)
        sim_names = {'realsense_rgb': 'room', 'wrist_cam': 'wrist'}
        gui = RecordingWebGui(gui_panes(sim_cameras, real_cameras), host=args.gui_host, port=args.gui_port)
        gui.mode = 'eval-sim-real'
        print(f'[eval-sim-real] Open {gui.url}; output: {output}', flush=True)
        print('[eval-sim-real] Place the real cube beside the OPEN fixed jaw, Spawn, then Start. '
              'Real runs first, then sim. Stop cancels both. Physical arm holds after its trial.', flush=True)
        phase = 'idle'
        positioning = leader is not None
        ready = False
        start_raw = start_objects = None
        phase_started = 0.0
        steps = 0
        began = time.monotonic()
        status = 'Leader positioning: move the leader, open the gripper, then Spawn and Start.' if positioning else (
            'Read-only positioning: place the real arm/cube, then Spawn and Start.')

        def mirror(raw):
            target = mapping.to_sim(torch.as_tensor(raw, device=env.unwrapped.device)).unsqueeze(0)
            robot.write_joint_state_to_sim_index(position=target, velocity=torch.zeros_like(target),
                                                joint_ids=joint_ids, env_ids=ids)
            robot.set_joint_position_target_index(target=target, joint_ids=joint_ids, env_ids=ids)
            return target

        def finish(reason):
            nonlocal video, log, phase, ready, positioning
            if phase in ('real', 'sim'):
                report['trials'].append({'side': phase, 'status': reason, 'steps': steps,
                                        'wall_seconds': time.monotonic() - phase_started})
            if video:
                video.close(min(time.monotonic() - phase_started, args.episode_seconds))
                video = None
            if log:
                log.close()
                log = None
            (output / 'report.json').write_text(json.dumps(report, indent=2))
            phase, ready, positioning = 'idle', False, False

        def begin(side):
            nonlocal phase, phase_started, steps, video, log
            phase, steps, phase_started = side, 0, time.monotonic()
            gui.set_state(True, False, f'{side.upper()} trial starting. Stop cancels both trials.')
            video = TrialVideo(output / f'{side}.mp4', args.fps)
            log = (output / f'{side}.jsonl').open('w')

        with torch.inference_mode():
            while app.is_running():
                tick = time.monotonic()
                requests = gui.consume_requests()
                if requests.get('exit'):
                    finish('user_exit')
                    break
                if requests.get('stop') or requests.get('discard'):
                    finish('user_stop')
                    status = 'Stopped. Reposition the real arm/cube and Spawn for a new trial.'
                observation = follower.get_observation()
                real_state = np.asarray([observation[key] for key in keys], dtype=np.float32)
                real_images = {real_names[name]: observation[name] for name in real_cameras}
                if phase == 'idle':
                    positioned = position_with_leader(
                        leader, follower, real_state, keys, ranges,
                        enabled=positioning and not requests.get('start'), motion=args.enable_motion,
                        step_limit=args.max_relative_target)
                    # Physical readback is authoritative when moving the real arm.
                    target = mirror(real_state if args.enable_motion or not positioning else positioned)
                    if requests.get('spawn'):
                        if any(real_state[i] < 40 for i, joint in enumerate(arm.joints) if joint.role == 'gripper'):
                            status = 'Open the gripper before spawning; the cube needs jaw clearance.'
                        else:
                            env.step(target)
                            spawner.spawn()
                            ready = True
                            status = (f'Match the real cube to the sim jaw placement. Start: real {args.episode_seconds / 60:g} min, '
                                      f'then sim {args.episode_seconds / 60:g} min.')
                    if requests.get('start') and ready:
                        if report['trials']:
                            status = 'This comparison is saved. Exit and relaunch to start another without overwriting it.'
                            ready = False
                        else:
                            positioning = False
                            if args.enable_motion:
                                follower.send_action(dict(zip(keys, map(float, real_state))))
                            start_raw = real_state.copy()
                            roundtrip = mapping.to_raw(mapping.to_sim(start_raw)).cpu().numpy()
                            if not np.allclose(roundtrip, start_raw, atol=0.1):
                                ready = False
                                status = 'Real start exceeds simulated joint limits; choose a pose within both ranges.'
                                gui.set_state(False, False, status)
                                continue
                            start_objects = {name: as_torch(env.unwrapped.scene[name].data.root_state_w).clone()
                                             for name in sim.envs.TASK.spec.tracked_objects}
                            report['start_raw'] = start_raw.tolist()
                            report['joint_names'] = list(arm.joint_names)
                            report['sim_objects_start_base_wxyz'] = {
                                name: object_pose_in_base(env, name, base_id).cpu().tolist()
                                for name in start_objects}
                            gui.set_state(True, False, 'Connecting policy; Stop cancels before motion.')
                            if not policies:
                                def factory():
                                    if args.backend == 'local-lerobot':
                                        from policies.molmoact2.local_tcp import LocalLeRobotPolicy
                                        return LocalLeRobotPolicy(
                                            ArmInfo.from_arm(arm, args.calibration_selection.mapping),
                                            port=args.policy_port or 5005, instruction=args.lang_instruction,
                                            horizon=args.action_horizon or 30)
                                    return make_policy(args)
                                try:
                                    policies = connect_trial_policies(factory, executor)
                                except Exception as error:
                                    status = f'Policy unavailable: {error}. Arm held. Start again when the server is ready.'
                                    report['last_connection_error'] = str(error)
                                    gui.set_state(False, True, status)
                                    print(status, flush=True)
                                    continue
                            begin('real')
                if phase in ('real', 'sim') and tick - phase_started >= args.episode_seconds:
                    previous = phase
                    finish('time_limit')
                    if previous == 'real':
                        target = mirror(start_raw)
                        for name, state in start_objects.items():
                            obj = env.unwrapped.scene[name]
                            obj.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=ids)
                            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(state[:, 7:13]), env_ids=ids)
                        env.unwrapped.sim.forward()
                        env.unwrapped.scene.update(0.0)
                        env.unwrapped.sim.render()
                        obs = env.unwrapped.observation_manager.compute()
                        begin('sim')
                    else:
                        status = 'Finished: real.mp4, sim.mp4 and joint logs saved. Exit to close.'
                visual = obs.get('visual', {})
                sim_images = {sim_names.get(name, name): visual[f'rgb_{name}'][0].cpu().numpy()
                              for name in sim_cameras}
                sim_state = mapping.to_raw(obs['policy']['joint_pos_obs'][0]).cpu().numpy()
                if phase in ('real', 'sim'):
                    side = phase
                    state, images = (real_state, real_images) if side == 'real' else (sim_state, sim_images)
                    try:
                        elapsed = time.monotonic() - phase_started
                        video.write(images, f'{side.upper()} {elapsed:.1f}s / {args.episode_seconds:.0f}s', elapsed)
                        if future is not None and future.done():
                            completed, future = future, None
                            action = completed.result()
                            if future_side == side:
                                values = limited_target(action, state, keys, ranges, first=steps == 0,
                                                        start_limit=args.max_start_error,
                                                        step_limit=args.max_relative_target)
                                if side == 'real':
                                    target = mapping.to_sim(start_raw).unsqueeze(0)
                                    if args.enable_motion:
                                        follower.send_action(dict(zip(keys, map(float, values))))
                                else:
                                    target = mapping.to_sim(values).unsqueeze(0)
                                log.write(json.dumps({'seconds': elapsed, 'state': state.tolist(),
                                                      'target': values.tolist(),
                                                      'motion': side == 'sim' or args.enable_motion}) + '\n')
                                log.flush()
                                steps += 1
                        if future is None:
                            policy = policies[0 if side == 'real' else 1]
                            future = executor.submit(policy.get_raw_action, state.copy(),
                                                     {key: value.copy() for key, value in images.items()})
                            future_side = side
                        status = f'{side.upper()} trial: {elapsed:.1f}/{args.episode_seconds:.0f}s; '
                        status += ('physical arm holding; sim running' if side == 'sim' else
                                   'real motion enabled' if args.enable_motion else 'SHADOW: real motion disabled')
                    except Exception as error:
                        finish(f'error: {error}')
                        status = f'Stopped: {error}'
                        print(status, flush=True)
                spawner.hold()
                obs, *_ = env.step(target)
                if not torch.isfinite(obs['policy']['joint_pos_obs']).all():
                    raise RuntimeError('Simulation joints became non-finite; comparison stopped')
                gui.update_images(gui_frames(obs.get('visual'), sim_cameras, observation, real_cameras))
                gui.set_state(phase != 'idle', ready and not report['trials'], status, positioning=positioning)
                if args.preview_seconds and time.monotonic() - began >= args.preview_seconds:
                    import cv2
                    frames = gui_frames(obs.get('visual'), sim_cameras, observation, real_cameras)
                    report['preview'] = {}
                    for name, frame in frames.items():
                        pixels = frame.cpu().numpy() if hasattr(frame, 'cpu') else np.asarray(frame)
                        report['preview'][name] = {'shape': list(pixels.shape), 'std': float(pixels.std())}
                        cv2.imwrite(str(output / (name.replace(': ', '-') + '.png')),
                                    cv2.cvtColor(pixels[..., :3], cv2.COLOR_RGB2BGR))
                    finish('preview_complete')
                    break
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if video:
            cleanup('Trial video close', video.close)
        if log:
            log.close()
        (output / 'report.json').write_text(json.dumps(report, indent=2))
        for policy in policies:
            if hasattr(policy, 'close'):
                cleanup('Policy close', policy.close)
        if gui:
            cleanup('GUI close', gui.destroy)
        cleanup('Leader disconnect', lambda: disconnect(leader, 'physical leader'))
        cleanup('Follower disconnect', lambda: disconnect(follower, 'physical follower'))
        if spawner:
            cleanup('Spawner close', spawner.close)
        if env:
            cleanup('Environment close', env.close)


def main(argv=None):
    from core.rollout.real_rollout import validate
    from sim.app import apply_selection, launch, close_app

    args = parse_args(argv)
    apply_selection(args)
    errors = validate(args)
    if not args.no_leader:
        if not args.leader_port or not args.leader_id:
            errors.append('Leader positioning requires --leader-port and --leader-id; or use --no-leader')
        elif os.path.realpath(args.leader_port) == os.path.realpath(args.follower_port):
            errors.append('Leader and follower must use different serial devices')
    if errors:
        raise ValueError('; '.join(errors))
    if not args.no_leader:
        from core.arms.calibration import resolve_calibration
        args.calibration_selection = resolve_calibration(
            args.arm_spec, unit=args.unit, follower_id=args.follower_id, leader_id=args.leader_id)
    app = launch(args, enable_cameras=True)
    try:
        run(args, app)
    finally:
        close_app(app)


if __name__ == '__main__':
    main()
