"""HTTP/control-loop bridge for matched real/simulation evaluation; no device I/O."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from core.policy_client.discovery import discover
from core.tasks.loader import load_task
from core.rollout.testing_http import TestingHttpServer


def operator_stage_label(task_stages, stage):
    """A human-selected furthest milestone; never an automatic success claim."""
    order = ('not_reached', *task_stages)
    if stage not in order:
        raise ValueError('Choose a stage defined by the selected task.')
    index = order.index(stage)
    return dict(stage=stage, stages={name: i < index for i, name in enumerate(task_stages)},
                outcome='success' if task_stages and stage == task_stages[-1] else 'incomplete',
                outcome_source='operator')


def trial_arguments(args, body):
    """Freeze one selection for both sides; per-policy options follow that selection."""
    selected = argparse.Namespace(**vars(args))
    selected.policy = body['policy']
    selected.lang_instruction = body['prompt']
    selected.policy_host, selected.policy_port = body['host'], body['port']
    selected.episode_seconds = body['seconds']
    options = dict(args.policy_options.get(selected.policy, {}))
    if selected.policy == args.policy:
        options.update(item.split('=', 1) for item in args.policy_option)
    selected.policy_option = [f'{key}={value}' for key, value in options.items()]
    return selected


class PairedEvaluationGui:
    def __init__(self, args, camera_names, output, *, manifests=None):
        self.args, self.output = args, Path(output)
        self._names = list(camera_names)
        self._manifests = discover() if manifests is None else manifests
        self._lock = threading.RLock()
        self.motion_lock = threading.Lock()
        self.cancel = threading.Event()
        self.task_stages = load_task(args.task).stages
        self._trial_end = None
        self._requests, self._frames, self._history = {}, {}, []
        self._seq = 0
        self._owner = None
        self._heartbeat = self._last_frame = 0.
        self._state = dict(phase='disconnected', connected=False, can_start=False, busy=False,
                           status='Connect the real arm to prepare a comparison.', error='',
                           elapsed=0., steps=0, duration=args.episode_seconds, positioning=False)
        self._server = TestingHttpServer(self, host=args.gui_host, port=args.gui_port)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f'http://{args.gui_host}:{self._server.server_address[1]}'

    def catalog(self):
        arm = self.args.arm_spec
        real_inputs = {json.loads(self.args.rename_map).get(name.removeprefix('Real: '))
                       for name in self._names if name.startswith('Real: ')}
        sim_inputs = {{'realsense_rgb': 'room', 'wrist_cam': 'wrist'}.get(name.removeprefix('Sim: '),
                      name.removeprefix('Sim: ')) for name in self._names if name.startswith('Sim: ')}
        rows = []
        for name, manifest in self._manifests.items():
            if not manifest.language:
                continue
            reason = ''
            if manifest.action_dim not in ('arm', len(arm.action_joints)):
                reason = f'Requires {manifest.action_dim} joints; {arm.name} has {len(arm.action_joints)}.'
            missing = set(manifest.cameras) - (real_inputs & sim_inputs)
            if missing:
                reason = 'Missing inputs on one comparison side: ' + ', '.join(sorted(missing))
            if self.args.backend == 'local-lerobot' and name != 'molmoact2':
                reason = 'The local-lerobot backend serves MolmoAct2 only.'
            rows.append(dict(name=name, description=manifest.description, port=manifest.default_port,
                             compatible=not reason, reason=reason, language=True))
        views = [name for name in ('Real: external', 'Sim: realsense_rgb') if name in self._names]
        views += [name for name in self._names if name not in views]
        return dict(mode='eval-sim-real', arm=arm.name, scene=self.args.scene, task=self.args.task,
                    follower_id=self.args.follower_id, follower_port=self.args.follower_port,
                    cameras=self._names, default_views=views[:2], policies=rows,
                    camera_devices={name: getattr(self.args, 'camera_' + name)
                                    for name in ('external', 'gripper') if getattr(self.args, 'camera_' + name)},
                    requires_spawn=not self.args.no_object, motion_enabled=self.args.enable_motion,
                    stages=[dict(value=name, label=name.replace('_', ' ').capitalize())
                            for name in ('not_reached', *self.task_stages)],
                    defaults=dict(policy=self.args.policy, host=self.args.policy_host, port=self.args.policy_port,
                                  instruction=self.args.lang_instruction, episode_seconds=self.args.episode_seconds),
                    fps=self.args.fps)

    def snapshot(self):
        with self._lock:
            return dict(self._state, busy=self._state['busy'] or bool(self._requests), owner=self._owner,
                        ending_side=self._trial_end['side'] if self._trial_end else None,
                        frame_seq=self._seq, cameras=sorted(self._frames), history=list(self._history))

    def publish(self, phase, status, *, connected, can_start=False, elapsed=0., steps=0,
                duration=None, error='', positioning=False):
        with self._lock:
            self._state.update(phase=phase, status=status, connected=connected, can_start=can_start,
                               busy=phase not in ('ready', 'disconnected'), elapsed=elapsed, steps=steps,
                               duration=duration or self.args.episode_seconds, error=error, positioning=positioning)

    def request_connect(self, client_id=None):
        with self._lock:
            if self._state['connected'] or self._state['busy'] or self._requests:
                raise ValueError('A connection or comparison is already in progress.')
            self._requests['connect'] = True
            if isinstance(client_id, str) and 1 <= len(client_id) <= 128:
                self._owner, self._heartbeat = client_id, time.monotonic()
            self._state.update(phase='connecting_robot', busy=True, status='Connecting the real arm…', error='')

    def request_run(self, body):
        policies = {row['name']: row for row in self.catalog()['policies']}
        name, prompt = body.get('policy'), body.get('prompt')
        if not isinstance(name, str) or name not in policies or not policies[name]['compatible']:
            raise ValueError(policies.get(name, {}).get('reason') if isinstance(name, str) and name in policies
                             else 'Choose a compatible policy.')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError('Enter a task prompt (1–4000 characters).')
        host, port, seconds, owner = body.get('host'), body.get('port'), body.get('seconds'), body.get('client_id')
        if not isinstance(host, str) or not host.strip() or len(host) > 255 or any(c.isspace() or c == '/' for c in host):
            raise ValueError('Enter a policy server hostname or IP address.')
        if self.args.backend == 'local-lerobot' and host.strip() not in ('localhost', '127.0.0.1'):
            raise ValueError('The local-lerobot backend requires a loopback policy server.')
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Policy port must be an integer from 1 to 65535.')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not .1 <= seconds <= 600:
            raise ValueError('Run duration per side must be between 0.1 and 600 seconds.')
        if not isinstance(owner, str) or not 1 <= len(owner) <= 128:
            raise ValueError('A browser session ID is required.')
        with self._lock:
            if not self._state['connected'] or not self._state['can_start'] or self._state['busy'] or self._requests:
                raise ValueError('Connect the arm and Spawn an object before starting the comparison.')
            self._requests['run'] = dict(policy=name, prompt=prompt.strip(), host=host.strip(), port=port,
                                         seconds=float(seconds))
            self._owner, self._heartbeat = owner, time.monotonic()
            self.cancel.clear()
            self._trial_end = None
            self._state.update(phase='queued', busy=True, error='', status='Preparing the real → sim comparison…')

    def request_spawn(self):
        with self._lock:
            if not self._state['connected'] or self._state['busy'] or self._requests or self.args.no_object:
                raise ValueError('Spawn is available after connecting, between comparisons.')
            self._requests['spawn'] = True
            self._state.update(phase='spawning', busy=True, error='', status='Spawning the simulation object…')

    def request_stop(self, *, disconnect=False):
        with self.motion_lock, self._lock:
            self.cancel.set()
            self._requests['stop'] = True
            if disconnect:
                self._requests['disconnect'] = True

    def request_end_trial(self, body):
        side, stage = body.get('side'), body.get('stage')
        if side not in ('real', 'sim'):
            raise ValueError('Choose the active comparison side.')
        label = operator_stage_label(self.task_stages, stage)
        # As with Stop, acknowledge only after any current dispatch finishes.
        # The control loop checks this latch before sending another command.
        with self.motion_lock, self._lock:
            if (self._state['phase'] != side or not self._state['connected'] or self.cancel.is_set()
                    or (self._trial_end and self._trial_end['side'] == side)):
                raise ValueError('Stage selection is available only during the active trial.')
            self._trial_end = dict(label, side=side, at=time.monotonic(),
                                   confirmed_at=datetime.now(timezone.utc).isoformat())
            self._requests['end_trial'] = True
            self._state.update(phase='ending_' + side, busy=True,
                               status=f'{side.capitalize()} stage recorded. Holding position; ending this trial…')

    def trial_end_for(self, side):
        with self._lock:
            return dict(self._trial_end) if self._trial_end and self._trial_end['side'] == side else None

    def heartbeat(self, owner):
        with self._lock:
            if owner == self._owner:
                self._heartbeat = time.monotonic()

    @property
    def cancelled(self):
        with self._lock:
            if self._owner and time.monotonic() - self._heartbeat > 3.:
                self.cancel.set()
            return self.cancel.is_set()

    def consume_requests(self):
        with self._lock:
            result, self._requests = self._requests, {}
            return result

    def update_images(self, frames):
        now = time.monotonic()
        if now - self._last_frame < .125:
            return
        encoded = {}
        for name, frame in frames.items():
            if frame is None:
                continue
            if hasattr(frame, 'detach'):
                frame = frame.detach().cpu().numpy()
            buffer = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(np.asarray(frame)[..., :3], dtype=np.uint8)).save(buffer, 'JPEG', quality=80)
            encoded[name] = buffer.getvalue()
        with self._lock:
            self._frames, self._seq = encoded, self._seq + 1
        self._last_frame = now

    def frame(self, name):
        with self._lock:
            return self._frames.get(name)

    def clear_real_frames(self):
        with self._lock:
            self._owner = None
            self._frames = {key: value for key, value in self._frames.items() if key.startswith('Sim: ')}
            self._seq += 1

    def record_report(self, report):
        stages = {row['side']: row.get('stage') for row in report['trials']}
        seconds = {row['side']: row['wall_seconds'] for row in report['trials']}
        summary = dict(id=report['id'], policy=report['policy'], instruction=report['instruction'],
                       status=report['status'], detail=report.get('detail', ''),
                       real_stage=stages.get('real'), sim_stage=stages.get('sim'),
                       real_seconds=seconds.get('real'), sim_seconds=seconds.get('sim'),
                       seconds=sum(row['wall_seconds'] for row in report['trials']))
        with self._lock:
            self._history.insert(0, summary)
            self._history = self._history[:20]
            self._owner = None

    def report(self, run_id):
        if not run_id or '/' in run_id or '\\' in run_id or run_id in ('.', '..'):
            return None
        path = self.output / run_id / 'report.json'
        return path.read_bytes() if path.is_file() else None

    def destroy(self):
        self.request_stop(disconnect=True)
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.)
