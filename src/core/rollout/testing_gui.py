"""Real-arm policy testing console. No simulator starts and no device opens until Connect."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import errno
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import uuid

import numpy as np

from core.arms.spec import load_arm_spec
from core.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK, repo_dir
from core.policy_client.discovery import discover
from core.rollout import real_rollout
from core.utils.config import parse_yaml_args


HARDWARE_ENV = ('ROBOT_PORT', 'ROBOT_ID', 'CAMERA_GRIPPER', 'CAMERA_EXTERNAL',
                'CAMERA_WIDTH', 'CAMERA_HEIGHT', 'CAMERA_FOURCC', 'CONTROL_FPS')


def hardware_defaults():
    """Current local hardware settings override stale values in a long-lived container."""
    values = {key: os.environ[key] for key in HARDWARE_ENV if key in os.environ}
    path = repo_dir('configs').parent / '.env'
    if path.is_file():
        for line in path.read_text().splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() in HARDWARE_ENV:
                words = shlex.split(value, comments=True)
                values[key.strip()] = ' '.join(words)
    return values


def parse_args(argv=None):
    env = hardware_defaults()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config_path', type=Path)
    p.add_argument('--arm', default=os.environ.get('S2R_ARM', DEFAULT_ARM))
    p.add_argument('--scene', default=os.environ.get('S2R_SCENE', DEFAULT_SCENE))
    p.add_argument('--task', default=os.environ.get('S2R_TASK', DEFAULT_TASK))
    p.add_argument('--unit', default=os.environ.get('S2R_UNIT'))
    p.add_argument('--expected_follower_sha256', help='Refuse a different physical follower calibration')
    p.add_argument('--policy', default='molmoact2')
    p.add_argument('--policy_host', default='localhost')
    p.add_argument('--policy_port', type=int)
    p.add_argument('--policy_options', default='{}', help='JSON mapping of policy names to client options')
    p.add_argument('--policy_option', action='append', default=[])
    p.add_argument('--lang_instruction', default='')
    p.add_argument('--follower_port', default=env.get('ROBOT_PORT'))
    p.add_argument('--follower_id', default=env.get('ROBOT_ID'))
    p.add_argument('--camera_gripper', default=env.get('CAMERA_GRIPPER'))
    p.add_argument('--camera_external', default=env.get('CAMERA_EXTERNAL'))
    p.add_argument('--camera_width', type=int, default=int(env.get('CAMERA_WIDTH', '640')))
    p.add_argument('--camera_height', type=int, default=int(env.get('CAMERA_HEIGHT', '480')))
    p.add_argument('--camera_fourcc', default=env.get('CAMERA_FOURCC') or None)
    p.add_argument('--rename_map', default=real_rollout.DEFAULT_RENAME_MAP)
    p.add_argument('--fps', type=int, default=int(env.get('CONTROL_FPS', '30')))
    p.add_argument('--episode_seconds', type=float, default=30.)
    p.add_argument('--action_horizon', type=int)
    p.add_argument('--max_start_error', type=float, default=20.)
    p.add_argument('--max_relative_target', type=float, default=10.)
    p.add_argument('--output', type=Path, default=Path('outputs/real_testing'))
    p.add_argument('--gui_host', default='127.0.0.1')
    p.add_argument('--gui_port', type=int, default=8766)
    p.add_argument('--preview_fps', type=float, default=8.)
    args = parse_yaml_args(p, argv, json_fields=('rename_map', 'policy_options'))
    if args.expected_follower_sha256 is not None and (
        len(args.expected_follower_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.expected_follower_sha256)
    ):
        p.error('expected_follower_sha256 must be a lowercase SHA-256 hash')
    if any('=' not in item for item in args.policy_option):
        p.error('policy_option must use KEY=VALUE')
    args.policy_options = json.loads(args.policy_options)
    if not isinstance(args.policy_options, dict) or any(
        not isinstance(k, str) or not isinstance(v, dict) for k, v in args.policy_options.items()
    ):
        p.error('policy_options must map policy names to option dictionaries')
    if not 1 <= args.fps <= 120 or not 0 < args.preview_fps <= 30:
        p.error('fps must be 1–120 and preview_fps must be in (0, 30]')
    if not math.isfinite(args.episode_seconds) or not .1 <= args.episode_seconds <= 600:
        p.error('episode_seconds must be in [0.1, 600]')
    if any(not math.isfinite(v) or v <= 0 for v in (args.max_start_error, args.max_relative_target)):
        p.error('motion limits must be finite and positive')
    if args.action_horizon is not None and args.action_horizon < 1:
        p.error('action_horizon must be positive')
    args.rename_map = json.loads(args.rename_map)
    if not isinstance(args.rename_map, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v for k, v in args.rename_map.items()
    ) or len(set(args.rename_map.values())) != len(args.rename_map):
        p.error('rename_map must contain distinct nonempty policy camera names')
    # This is an explicitly operated real-arm GUI. Run authorizes movement;
    # Connect alone never issues a policy action, and disconnect retains torque.
    args.enable_motion, args.release_on_exit, args.episodes = True, False, 1
    args.checkpoint = ''
    return args


def close_policy(policy):
    if policy is not None:
        close = getattr(policy, 'close', None) or getattr(policy, 'close_timing', None)
        if close:
            close()


class RealArmSession:
    """One owner for robot I/O; one other thread owns every policy socket operation."""

    def __init__(self, args, *, follower_factory=real_rollout.connect_follower,
                 policy_factory=real_rollout.make_policy, manifests=None, validator=real_rollout.validate,
                 heartbeat_timeout=3.):
        self.args = args
        self.arm = load_arm_spec(args.arm)[0]
        self.manifests = discover() if manifests is None else manifests
        self._follower_factory, self._policy_factory, self._validator = follower_factory, policy_factory, validator
        self._lock = threading.RLock()
        self._motion_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='real-policy')
        self._robot = self._policy = self._future = self._kind = self._run = None
        self._connect_requested = self._disconnect_requested = False
        self._heartbeat_timeout = heartbeat_timeout
        self._frames, self._frame_seq, self._history = {}, 0, []
        self._phase, self._status, self._error = 'disconnected', 'Connect the real arm to begin.', ''
        self._last_preview = 0.
        self._thread = threading.Thread(target=self._loop, name='real-arm-io', daemon=True)
        self._thread.start()

    def catalog(self):
        cameras = {name for name in ('external', 'gripper') if getattr(self.args, 'camera_' + name)}
        inputs = {self.args.rename_map.get(name) for name in cameras}
        policies = []
        for name, m in self.manifests.items():
            if not m.language:
                continue
            reason = ''
            if m.action_dim != 'arm' and m.action_dim != len(self.arm.action_joints):
                reason = f'Requires {m.action_dim} joints; {self.arm.name} has {len(self.arm.action_joints)}.'
            missing = set(m.cameras) - inputs
            if missing:
                reason = 'Missing camera inputs: ' + ', '.join(sorted(missing))
            policies.append(dict(name=name, description=m.description, port=m.default_port,
                                 compatible=not reason, reason=reason, language=m.language))
        return dict(arm=self.arm.name, follower_id=self.args.follower_id or '',
                    follower_port=self.args.follower_port or '', cameras=sorted(cameras),
                    camera_devices={n: getattr(self.args, 'camera_' + n) for n in sorted(cameras)},
                    policies=policies, defaults=dict(policy=self.args.policy, host=self.args.policy_host,
                    port=self.args.policy_port, instruction=self.args.lang_instruction,
                    episode_seconds=self.args.episode_seconds), fps=self.args.fps)

    def snapshot(self):
        with self._lock:
            run = self._run
            elapsed = time.monotonic() - run['started'] if run and run.get('started') else 0.
            return dict(phase=self._phase, status=self._status, error=self._error,
                        connected=self._robot is not None, busy=self._run is not None or self._connect_requested or self._phase=='disconnecting',
                        elapsed=round(elapsed, 1), steps=run['steps'] if run else 0,
                        duration=run['seconds'] if run else self.args.episode_seconds,
                        owner=run['owner'] if run else None,
                        frame_seq=self._frame_seq, cameras=sorted(self._frames), history=list(self._history))

    def frame(self, name):
        with self._lock:
            return self._frames.get(name)

    def request_connect(self, client_id=None):
        with self._lock:
            if self._robot is not None or self._connect_requested or self._run or self._phase=='disconnecting':
                raise ValueError('The arm is already connected or a connection is in progress.')
            self._connect_requested = True
            self._phase, self._status, self._error = 'connecting_robot', 'Connecting the real arm…', ''

    def request_run(self, body):
        name, prompt, owner = body.get('policy'), body.get('prompt'), body.get('client_id')
        rows = {p['name']: p for p in self.catalog()['policies']}
        if name not in rows or not rows[name]['compatible']:
            raise ValueError(rows.get(name, {}).get('reason') or 'Choose an available policy.')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError('Enter a task prompt (1–4000 characters).')
        if not isinstance(owner, str) or not 1 <= len(owner) <= 128:
            raise ValueError('A browser session ID is required.')
        host, port, seconds = body.get('host', self.args.policy_host), body.get('port'), body.get('seconds')
        if not isinstance(host, str) or not host.strip() or len(host) > 255 or any(c.isspace() or c == '/' for c in host):
            raise ValueError('Enter a policy server hostname or IP address.')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Policy port must be an integer from 1 to 65535.')
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or not .1 <= seconds <= 600:
            raise ValueError('Run duration must be between 0.1 and 600 seconds.')
        with self._lock:
            if self._robot is None or self._run or self._disconnect_requested or self._phase!='ready':
                raise ValueError('Connect the arm and finish the current run first.')
            self._run = dict(policy=name, prompt=prompt.strip(), owner=owner, host=host.strip(), port=port,
                             seconds=float(seconds), cancel=threading.Event(), steps=0,
                             heartbeat=time.monotonic(), started=None, stop_status='stopped', detail='')
            self._phase, self._status, self._error = 'queued', 'Connecting the selected policy…', ''

    def heartbeat(self, owner):
        with self._lock:
            if self._run and self._run['owner'] == owner:
                self._run['heartbeat'] = time.monotonic()

    def request_stop(self, *, disconnect=False):
        # Serialize the acknowledgement with dispatch, so a late inference reply
        # cannot send a new command after this request has returned.
        with self._motion_lock, self._lock:
            if self._run:
                self._run['cancel'].set()
                self._phase, self._status = 'stopping', 'Stopping; the arm holds its last position.'
            if disconnect:
                self._disconnect_requested = True

    def _set(self, phase, status, error=''):
        with self._lock:
            self._phase, self._status, self._error = phase, status, error

    def _connect(self):
        args = argparse.Namespace(**vars(self.args))
        args.rename_map = json.dumps(self.args.rename_map)
        errors = self._validator(args)
        if errors:
            raise ValueError('; '.join(errors))
        if (args.expected_follower_sha256 is not None and
                args.calibration_selection.provenance['follower_calibration']['sha256'] != args.expected_follower_sha256):
            raise ValueError('Follower calibration SHA-256 differs from expected_follower_sha256; refusing to connect.')
        robot = self._follower_factory(args)
        with self._lock:
            self._robot, self._device_args = robot, args
            self._connect_requested = False
        self._set('ready', 'Real arm connected. Enter a prompt and run a policy.')

    def _prepare_policy(self, run):
        args = argparse.Namespace(**vars(self._device_args))
        options = dict(self.args.policy_options.get(run['policy'], {}))
        if run['policy'] == self.args.policy:
            options.update(option.split('=', 1) for option in self.args.policy_option)
        args.policy, args.lang_instruction = run['policy'], run['prompt']
        args.policy_host, args.policy_port = run['host'], run['port']
        args.policy_option = [f'{k}={v}' for k, v in options.items()]
        policy = self._policy_factory(args)
        try:
            policy.connect()
            policy.reset()
            run['server_info'] = getattr(policy, 'server_info', None)
            run['joint_transform'] = getattr(policy, 'joint_transform', None)
            if run['cancel'].is_set():
                close_policy(policy)
                return None
            return policy
        except BaseException:
            close_policy(policy)
            raise

    def _begin(self):
        run = self._run
        run['id'] = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
        run['folder'] = self.args.output / run['id']
        run['folder'].mkdir(parents=True)
        run['trace'] = (run['folder'] / 'actions.jsonl').open('w', buffering=1)
        if run['cancel'].is_set():
            self._finish('stopped', run['detail'])
            return
        self._future = self._executor.submit(self._prepare_policy, run)
        self._kind = 'prepare'
        self._set('connecting_policy', 'Connecting the selected policy…')

    def _finish(self, status, detail=''):
        self._run['result'] = (status, detail)
        policy, self._policy = self._policy, None
        self._future = self._executor.submit(close_policy, policy)
        self._kind = 'close'
        self._set('stopping' if status == 'stopped' else 'finishing', 'Holding position; saving the run…')

    def _complete(self):
        run = self._run
        status, detail = run['result']
        seconds = round(time.monotonic() - run['started'], 3) if run['started'] else 0.
        report = dict(schema='s2r_real_testing_v1', id=run['id'], arm=self.arm.name, policy=run['policy'],
                      instruction=run['prompt'], policy_host=run['host'], policy_port=run['port'],
                      status=status, detail=detail, steps=run['steps'], seconds=seconds,
                      outcome='unlabeled', motion=True, fps=self.args.fps,
                      max_start_error=self.args.max_start_error, max_relative_target=self.args.max_relative_target,
                      policy_server=run.get('server_info'), joint_transform=run.get('joint_transform'),
                      calibration=self._device_args.calibration_selection.provenance)
        try:
            if 'trace' in run:
                run['trace'].close()
            (run['folder'] / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        except (OSError, TypeError) as error:
            status, detail = 'report_error', f'Could not save the run report: {error}'
            report.update(status=status, detail=detail)
        summary = {k: report[k] for k in ('id', 'policy', 'instruction', 'status', 'detail', 'steps', 'seconds')}
        with self._lock:
            self._history.insert(0, summary)
            self._history = self._history[:20]
            self._run = self._future = self._kind = None
        self._set('ready', 'Run finished. The arm holds its last position.' if status == 'completed'
                  else 'Stopped. The arm holds its last position.' if status == 'stopped' else 'Run ended; check the error below.',
                  detail if status not in ('completed', 'stopped') else '')

    def report(self, run_id):
        if not run_id or '/' in run_id or '\\' in run_id or run_id in ('.', '..'):
            return None
        path = self.args.output / run_id / 'report.json'
        return path.read_bytes() if path.is_file() else None

    def _preview(self, observation):
        now = time.monotonic()
        if now - self._last_preview < 1 / self.args.preview_fps:
            return
        from PIL import Image
        frames = {}
        for name in self.catalog()['cameras']:
            pixels = np.asarray(observation[name]).copy()
            if pixels.ndim != 3 or pixels.shape[2] < 3:
                raise ValueError(f'Invalid RGB image from {name}')
            buffer = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(pixels[..., :3], dtype=np.uint8)).save(buffer, 'JPEG', quality=80)
            frames[name] = buffer.getvalue()
        with self._lock:
            self._frames, self._frame_seq = frames, self._frame_seq + 1
        self._last_preview = now

    def _drive(self, observation):
        run = self._run
        now = time.monotonic()
        if not run['cancel'].is_set() and now - run['heartbeat'] > self._heartbeat_timeout:
            run['cancel'].set()
            run['detail'] = 'Browser heartbeat lost.'
        if not run['cancel'].is_set() and run['started'] and now - run['started'] >= run['seconds']:
            run['stop_status'] = 'completed'
            run['cancel'].set()
        if self._kind == 'close':
            if self._future.done():
                self._future.result()
                self._complete()
            return
        if self._future is not None and self._future.done():
            kind, future = self._kind, self._future
            self._future = self._kind = None
            try:
                value = future.result()
                if kind == 'prepare':
                    self._policy = value
                    if not run['cancel'].is_set():
                        run['started'] = time.monotonic()
                        self._set('running', 'Policy is controlling the real arm.')
                elif not run['cancel'].is_set():
                    keys = list(self.arm.feature_names)
                    state = np.array([observation[key] for key in keys], dtype=np.float32)
                    target = real_rollout.clamp_action(value, keys, self._ranges)
                    if run['steps'] == 0 and float(np.max(np.abs(target - state))) > self.args.max_start_error:
                        self._finish('start_too_far', 'The first policy target is too far from the current arm pose.')
                        return
                    with self._motion_lock:
                        if not run['cancel'].is_set() and not self._shutdown.is_set():
                            sent = self._robot.send_action(dict(zip(keys, target.tolist())))
                            step = run['steps']
                            run['steps'] += 1
                            run['trace'].write(json.dumps(dict(step=step, t=round(now - run['started'], 4),
                                state=state.tolist(), requested={k:float(value[k]) for k in keys},
                                target=target.tolist(), sent=True, driver_action={k:float(v) for k,v in sent.items()} if isinstance(sent,dict) else None)) + '\n')
            except Exception as error:
                self._finish('policy_error' if kind == 'prepare' else 'action_error', str(error))
                return
        if run['cancel'].is_set():
            self._set('stopping', 'Stopping; the arm holds its last position.')
            if self._future is None:
                self._finish(run['stop_status'], run['detail'])
            return
        if self._policy is not None and self._future is None:
            state = np.array([observation[key] for key in self.arm.feature_names], dtype=np.float32)
            images = {self.args.rename_map[name]: np.asarray(observation[name]).copy() for name in self.catalog()['cameras']}
            self._future = self._executor.submit(self._policy.get_raw_action, state.copy(), images)
            self._kind = 'action'

    def _disconnect(self):
        self._set('disconnecting', 'Disconnecting the real arm…')
        robot, self._robot = self._robot, None
        if robot is not None:
            robot.disconnect()
        with self._lock:
            self._frames, self._frame_seq = {}, self._frame_seq + 1
        self._disconnect_requested = False
        self._set('disconnected', 'Real arm disconnected. Torque was not released.')

    def _loop(self):
        try:
            while not self._shutdown.is_set():
                start = time.monotonic()
                try:
                    if self._connect_requested:
                        self._connect()
                        self._ranges = real_rollout.joint_ranges(self.arm, self._device_args.calibration_selection.mapping)
                    if self._run and 'id' not in self._run:
                        self._begin()
                    if self._robot is not None:
                        obs = self._robot.get_observation()
                        if not np.isfinite([obs[key] for key in self.arm.feature_names]).all():
                            raise ValueError('Robot returned non-finite joint positions.')
                        self._preview(obs)
                        if self._run:
                            self._drive(obs)
                    if self._disconnect_requested and self._run is None:
                        self._disconnect()
                except Exception as error:
                    self._connect_requested = False
                    if self._run and self._kind != 'close':
                        self._run['cancel'].set()
                        self._run['stop_status'], self._run['detail'] = 'robot_error', str(error)
                        # Keep draining the pending policy result without dispatch.
                        if self._future is None or self._future.done():
                            if self._kind == 'prepare' and self._future is not None:
                                try:
                                    self._policy = self._future.result()
                                except Exception:
                                    pass
                            self._finish('robot_error', str(error))
                    elif self._kind == 'close' and self._future.done():
                        self._complete()
                    self._error = str(error)
                    if self._run is None:
                        try:
                            self._disconnect()
                        except Exception:
                            pass
                        self._set('disconnected', 'Connection failed. Check the error below.', str(error))
                period = 1 / self.args.fps if self._run else 1 / self.args.preview_fps
                self._shutdown.wait(max(0., period - (time.monotonic() - start)))
        finally:
            if self._run:
                self._run['cancel'].set()
            if self._future is not None:
                try:
                    value = self._future.result()
                    if self._kind == 'prepare':
                        self._policy = value
                except Exception:
                    pass
            self._executor.submit(close_policy, self._policy).result()
            self._executor.shutdown(wait=True, cancel_futures=True)
            if self._run and 'trace' in self._run:
                self._run.setdefault('result', ('stopped', 'Console closed.'))
                try:
                    self._complete()
                except Exception:
                    if self._run:
                        self._run['trace'].close()
            try:
                self._disconnect()
            except Exception:
                pass

    def close(self):
        self.request_stop(disconnect=True)
        self._shutdown.set()
        self._thread.join(timeout=35.)


def main(argv=None):
    from core.rollout.testing_http import TestingHttpServer
    args = parse_args(argv)
    session = RealArmSession(args)
    try:
        server = TestingHttpServer(session, host=args.gui_host, port=args.gui_port)
    except BaseException as error:
        session.close()
        if isinstance(error, OSError) and error.errno == errno.EADDRINUSE:
            print(f'Cannot start the GUI: port {args.gui_port} is already in use.\n'
                  f'If the GUI is already running, open http://{args.gui_host}:{args.gui_port}\n'
                  'To use another port, add --gui_port 8767 to your command.', file=sys.stderr)
            return 1
        raise
    print(f'Real-arm testing GUI: http://{args.gui_host}:{server.server_address[1]}', flush=True)
    print('No hardware is connected until Connect is clicked. Run sends policy actions to the real arm.', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        session.request_stop()
        server.server_close()
        session.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
