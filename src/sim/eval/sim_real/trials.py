"""Matched-start, real-then-sim policy trials using the existing browser GUI."""
import json
import os
import subprocess
from pathlib import Path

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


def connect_trial_policies(factory, executor=None):
    """Connect a complete pair, closing partial connections if startup fails."""
    connected = []
    call = (lambda fn: executor.submit(fn).result()) if executor else (lambda fn: fn())
    try:
        for _ in range(2):
            policy = factory()
            connected.append(policy)
            call(policy.connect)
        for policy in connected:
            call(policy.reset)
        return connected
    except Exception:
        for policy in connected:
            close = getattr(policy, 'close', None)
            if close is not None:
                call(close)
        raise


def parse_args(argv=None):
    from core.rollout.real_rollout import build_parser
    from utils.cli import parse_yaml_args
    from core.tasks.loader import load_task
    from sim import add_launch_arguments

    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--leader-port', default=os.getenv('TELEOP_PORT'))
    parser.add_argument('--leader-id', default=os.getenv('TELEOP_ID'))
    parser.add_argument('--no-leader', action='store_true', help='Disable operator positioning through the leader')
    parser.add_argument('--no-object', action='store_true',
                        help='Allow Start without Spawn and skip task object spawning/restoration')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--shadow', action='store_true', help='Preview/infer without sending real arm commands')
    parser.add_argument('--backend', choices=('s2r', 'local-lerobot'), default='s2r')
    parser.add_argument('--gui-host', default='127.0.0.1')
    parser.add_argument('--gui-port', type=int, default=8765)
    parser.add_argument('--preview-seconds', type=float, default=0,
                        help='Exit after a preview; combine with --shadow to prevent physical motion')
    parser.add_argument('--policy_options', default='{}', help='JSON mapping of per-policy client options')
    from core.rollout.testing_gui import hardware_defaults
    hardware = hardware_defaults()
    parser.set_defaults(camera_gripper=hardware.get('CAMERA_GRIPPER'),
                        camera_external=hardware.get('CAMERA_EXTERNAL'),
                        camera_width=int(hardware.get('CAMERA_WIDTH', '640')),
                        camera_height=int(hardware.get('CAMERA_HEIGHT', '480')),
                        camera_fourcc=hardware.get('CAMERA_FOURCC') or None)
    add_launch_arguments(parser)
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if not any(arg.startswith('--config_path') for arg in argv):
        argv = ['--config_path', 'workspace/configs/eval/sim_real.yaml', *argv]
    args = parse_yaml_args(parser, argv, json_fields=('rename_map', 'policy_options'))
    args.policy_options = json.loads(args.policy_options)
    if not isinstance(args.policy_options, dict) or any(not isinstance(value, dict) for value in args.policy_options.values()):
        parser.error('policy_options must map policy names to option dictionaries')
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
        panels = []
        for key in ('room', 'wrist'):
            if key in images:
                panel = cv2.resize(np.asarray(images[key])[..., :3], (640, 480))
            else:
                panel = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(panel, f'{key} camera disabled', (110, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (180, 180, 180), 2)
            panels.append(panel)
        frame = np.concatenate(panels, axis=1)
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
    from sim.eval.sim_real.paired import run as run_interactive
    return run_interactive(args, app)


def main(argv=None):
    from core.rollout.real_rollout import validate
    from sim import apply_selection, launch, close_app

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
        from utils.arms.calibration import resolve_calibration
        args.calibration_selection = resolve_calibration(
            args.arm_spec, unit=args.unit, follower_id=args.follower_id, leader_id=args.leader_id)
    app = launch(args, enable_cameras=True)
    try:
        run(args, app)
    finally:
        close_app(app)


if __name__ == '__main__':
    main()
