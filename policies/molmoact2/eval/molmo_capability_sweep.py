#!/usr/bin/env python3
"""Unassisted Molmo research runner; smoke outputs/state stay in RAM.

Waits for a specified existing launcher without interrupting it, freezes a
source snapshot, and compares one original start per zone under identical
milestone deadlines. It never edits the normal config or existing reports.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CREPO = '/workspace/Sim-to-Real-SO-101-Workshop'
MARKER = '[SMOKE_RESULT] '
FAILURE_LINE = re.compile(r'\b(?:error|exception|traceback|fatal)\b|(?:^|\s)[\w.]+(?:Error|Exception):', re.I)


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def process_identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        # comm may contain spaces, so parse fields after its closing parenthesis.
        start = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        return start, (proc / 'cmdline').read_bytes(), str((proc / 'cwd').resolve())
    except (FileNotFoundError, ProcessLookupError):
        return None


def port_free(port):
    try:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False


def validate_result(report, expected, horizon, options=None, camera_inputs=None):
    if not report.get('complete') or report.get('requested_episodes') != len(expected):
        raise ValueError('Incomplete smoke trial is not a scored candidate')
    if report.get('checkpoint') != 'molmoact2_so101':
        raise ValueError('Trial did not use the released SO-101 checkpoint')
    if report.get('action_horizon') != horizon or report.get('action_repeat') != 1:
        raise ValueError('Unexpected execution timing')
    if report.get('episode_length_s') != 210:
        raise ValueError('Unexpected absolute smoke ceiling')
    cameras = (list(camera_inputs) if camera_inputs is not None else
               ['room', 'wrist'])
    if report.get('policy_cameras') != cameras:
        raise ValueError('Unexpected camera inputs')
    settings = report.get('resume_settings', {})
    if options is not None and settings.get('policy_options') != {key: str(value) for key, value in options.items()}:
        raise ValueError('Runtime policy options differ from the candidate')
    if settings.get('grasp', {}).get('gripper_latch'):
        raise ValueError('Assisted gripper control is forbidden')
    smoothing = settings.get('smoothing', {})
    if any(smoothing.get(key, default) != default for key, default in (
            ('prefetch_steps', 0), ('blend_steps', 0), ('ema_alpha', 1.),
            ('temporal_ensemble', 0), ('smooth_gripper', False))):
        raise ValueError('Action smoothing is forbidden')
    protocol = report.get('smoke_test', {})
    if protocol != dict(selection='one_per_zone', reach_timeout_s=60.0, progress_timeout_s=30.0):
        raise ValueError('Trial changed the agreed smoke protocol')
    rows = report.get('episodes', [])
    if len(rows) != len(expected):
        raise ValueError('Unexpected episode count')
    for row, start in zip(rows, expected):
        if (row['start_id'], row['zone'], row['policy_seed']) != (start['id'], start['zone'], start['policy_seed']):
            raise ValueError('Start or seed changed between candidates')
        budget = row.get('smoke_deadline') or {}
        if budget.get('reach_timeout_s') != 60 or budget.get('progress_timeout_s') != 30:
            raise ValueError('Episode omitted the progress deadline')
    return report


def rank_result(item):
    rows = item['report']['episodes']
    successes = [row for row in rows if row['success']]
    mean_success_time = sum(row['elapsed_sim_s'] for row in successes) / len(successes) if successes else 0.
    # Fast failures are not a speed improvement. Time only completed placements.
    return (len(successes), sum(row['stages']['grasped'] for row in rows), -mean_success_time)


class Investigation:
    def __init__(self, args, camera_inputs=None):
        import yaml
        if camera_inputs is not None and (
                not isinstance(camera_inputs, (tuple, list)) or not camera_inputs
                or any(c not in ('room', 'wrist') for c in camera_inputs)
                or len(set(camera_inputs)) != len(camera_inputs)):
            raise ValueError('Research camera inputs must be unique supported camera names')
        self.camera_inputs = tuple(camera_inputs) if camera_inputs is not None else ('room', 'wrist')
        self.args = args
        self.root = args.state_dir.resolve()
        if not str(self.root).startswith('/dev/shm/'):
            raise ValueError('Research smoke state must be RAM-backed under /dev/shm')
        self.root.mkdir(parents=True, exist_ok=False)
        self.cfg = yaml.safe_load((ROOT / 'configs/eval/pick_place_molmo_so101_smoke.yaml').read_text())
        if self.cfg['policy']['checkpoint'] != 'molmoact2_so101':
            raise ValueError('Released SO-101 checkpoint required')
        self.server = None
        self.server_key = None
        self.viewer = None
        self.client = None
        self.run_id = None
        self.results = []
        self.current = None
        self.phase = 'preparing'
        self.wait_identity = process_identity(args.wait_pid) if args.wait_pid else None
        if args.wait_pid and self.wait_identity is not None:
            _, command, cwd = self.wait_identity
            if b'docker/eval.sh' not in command or cwd != str(args.control_repo.resolve()):
                raise ValueError('Wait PID is not the expected existing evaluation launcher')
        self.snapshot = self.root / 'runtime'
        self.snapshot.mkdir()
        shutil.copytree(ROOT / 'source', self.snapshot / 'source',
                        ignore=shutil.ignore_patterns('assets', '__pycache__', '*.pyc'))
        assets = self.snapshot / 'source/sim_to_real_so101/assets'
        assets.symlink_to(CREPO + '/source/sim_to_real_so101/assets')
        shutil.copytree(ROOT / 'policies', self.snapshot / 'policies',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        if camera_inputs is not None:
            # Only the new private snapshot changes, before its source digest
            # or any process is created. The original manifest stays intact.
            manifest = self.snapshot / 'policies/molmoact2/policy.toml'
            contents, count = re.subn(r'(?m)^cameras\s*=.*$',
                'cameras = ' + json.dumps(list(self.camera_inputs)), manifest.read_text())
            if count != 1:
                raise ValueError('Expected exactly one camera declaration in the private manifest')
            manifest.write_text(contents)
        for name in ('datasets', 'eval_sets', 'configs'):
            (self.snapshot / name).symlink_to(CREPO + '/' + name)
        digest = hashlib.sha256()
        for path in sorted(path for folder in ('source', 'policies')
                           for path in (self.snapshot / folder).rglob('*')
                           if path.is_file() and path.suffix in ('.py', '.toml', '.lock')):
            digest.update(str(path.relative_to(self.snapshot)).encode())
            digest.update(path.read_bytes())
        self.source_digest = digest.hexdigest()
        sys.path.insert(0, str(ROOT / 'source'))
        from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts
        self.expected_by_seed = {seed: EvalSetStarts(ROOT / 'eval_sets/pick_place_v1_seed1984.json',
                                 seed=seed, selection='one_per_zone').schedule for seed in (1984, 9182, 4173)}
        self.save()

    def save(self, phase=None):
        if phase:
            self.phase = phase
        payload = dict(status=self.phase, pid=os.getpid(), heartbeat_unix_s=time.time(),
                       waiting_for_pid=self.args.wait_pid, source_digest=self.source_digest,
                       runtime_snapshot=str(self.snapshot), current=self.current, results=self.results,
                       manifest_camera_inputs=list(self.camera_inputs),
                       scope='Released SO101 weights; model-generated actions; no assistance; one example per zone')
        temporary = self.root / 'state.tmp'
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.root / 'state.json')

    def wait_for_existing(self):
        if self.wait_identity:
            print('[RESEARCH] Waiting for the existing evaluation launcher to finish; it will not be interrupted.', flush=True)
            while process_identity(self.args.wait_pid) == self.wait_identity:
                self.save('waiting_for_existing_evaluation')
                time.sleep(30)
        while not port_free(5556):
            self.save('waiting_for_policy_port')
            time.sleep(30)
        self.save('ready')

    def start_viewer(self):
        if not port_free(9876):
            raise RuntimeError('Rerun port 9876 is occupied; refusing to replace another viewer')
        self.viewer = subprocess.Popen([sys.executable, str(ROOT / 'docker/rerun_viewer.py'),
                                        '--container', 'teleop', '--port', '9876'])
        deadline = time.monotonic() + 60
        while port_free(9876):
            if self.viewer.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Owned Rerun viewer failed to start')
            time.sleep(.5)

    def wait_for_gpu_memory(self):
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                       '--format=csv,noheader,nounits'], text=True).strip())
            if used <= 5000:
                return
            self.save('waiting_for_gpu_memory')
            time.sleep(30)

    def stop_server(self):
        if self.server:
            subprocess.run(['docker', 'rm', '-f', self.server], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.server = self.server_key = None

    def start_server(self, options):
        # Pan conversion is client-side; its presence must not require reloading weights.
        core = {key: value for key, value in options.items()
                if key not in ('pan_model_scale', 'pan_model_offset', 'include_original_room')}
        key = json.dumps(core, sort_keys=True)
        if self.server_key == key:
            return
        self.stop_server()
        if not port_free(5556):
            raise RuntimeError('Policy port occupied by another process')
        self.server = 'molmo-capability-' + str(os.getpid()) + '-' + str(time.time_ns())
        # Keep a short-lived failed server until its exit state and logs have
        # been read. close()/stop_server() removes only this owned container.
        command = ['docker', 'run', '-d', '--name', self.server, '--network', 'host', '--gpus', 'all',
                   '-v', str(self.snapshot / 'policies') + ':/policies:ro', '-v', 'policy-cache:/opt/cache',
                   '-v', str(Path.home() / '.cache/huggingface') + ':/root/.cache/huggingface',
                   '-v', '/external_storage/molmoact2:/workspace/models:ro', 'policy-server:latest',
                   '--policy', 'molmoact2', '--checkpoint', '/workspace/models/molmoact2_so101', '--port', '5556']
        for name, value in core.items():
            command.extend(['--option', f'{name}={value}'])
        run(command, stdout=subprocess.DEVNULL)
        end = time.monotonic() + 900
        while True:
            inspection = subprocess.run(['docker', 'inspect', '-f', '{{json .State}}', self.server],
                                        capture_output=True, text=True)
            state = json.loads(inspection.stdout) if inspection.returncode == 0 else {}
            expired = time.monotonic() > end
            if not state.get('Running') or expired:
                tail = subprocess.run(['docker', 'logs', '--tail', '80', self.server], capture_output=True, text=True)
                reason = 'startup timed out' if expired else 'server exited before becoming ready'
                details = inspection.stdout.strip() or inspection.stderr.strip()
                raise RuntimeError(f'Owned policy failed to load ({reason}): {details}\n'
                                   + tail.stdout + tail.stderr)
            if not port_free(5556):
                break
            if self.viewer.poll() is not None:
                raise RuntimeError('Owned Rerun viewer exited')
            time.sleep(1)
        self.server_key = key

    def cleanup_client(self):
        if self.client is not None:
            if self.client.poll() is None:
                self.client.terminate()
                try:
                    self.client.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.client.kill()
                    self.client.wait()
            self.client = None
        if self.run_id:
            run(['docker', 'exec', 'teleop', '/workspace/isaaclab/_isaac_sim/python.sh',
                 CREPO + '/docker/eval_cleanup.py', self.run_id], stdout=subprocess.DEVNULL)
            self.run_id = None

    def trial(self, label, options, horizon=30, seed=1984, instruction=None):
        cameras = list(self.camera_inputs)
        include_original = str(options.get('include_original_room', False)).lower()
        if include_original != 'false':
            raise ValueError('Research trials use the retained room/wrist cameras only')
        self.current = dict(label=label, horizon=horizon, seed=seed, options=options,
                            instruction=instruction or self.cfg['eval']['lang'],
                            camera_inputs=cameras)
        self.save('loading_candidate')
        self.start_server(options)
        self.run_id = 'molmo-capability-' + str(os.getpid()) + '-' + str(time.time_ns())
        entry = ('import sys,runpy;root=sys.argv.pop(1);sys.path[:0]=[root,root+"/source"];'
                 'runpy.run_module("sim_to_real_so101.scripts.lerobot_eval",run_name="__main__")')
        command = ['docker', 'exec', '-e', 'POLICIES_DIR=' + str(self.snapshot / 'policies'),
                   '-e', 'SO101_EVAL_RUN_ID=' + self.run_id, '-e', 'PYTHONUNBUFFERED=1',
                   '-e', 'RERUN_CONNECT_URL=rerun+http://127.0.0.1:9876/proxy',
                   'teleop', '/workspace/isaaclab/_isaac_sim/python.sh', '-c', entry, str(self.snapshot),
                   '--task', 'Lerobot-So101-Teleop-Pick-Place-Eval', '--num_envs', '1', '--headless',
                   '--cube_starts', self.cfg['eval']['dataset'] + '/pick_place_meta',
                   '--start_mode', 'cycle', '--random_fraction', '0', '--robot_start', 'recorded',
                   '--robot_usd', self.cfg['eval']['robot_usd'], '--checkpoint', 'molmoact2_so101',
                   '--rename_map', json.dumps(self.cfg['eval']['rename_map']), '--policy', 'molmoact2',
                   '--policy_host', 'localhost', '--policy_port', '5556', '--action_horizon', str(horizon),
                   '--action_repeat', '1', '--episode_length_s', '210', '--render_warmup', '0', '--seed', str(seed),
                   '--eval_set', self.cfg['eval']['eval_set'], '--repeats', '1', '--splits', 'id,ood,yaw,train',
                   '--lang_instruction', instruction or self.cfg['eval']['lang'], '--smoke-test',
                   '--smoke-selection', 'one_per_zone', '--smoke-reach-timeout-s', '60', '--smoke-progress-timeout-s', '30']
        for name, value in options.items():
            command.extend(['--policy_option', f'{name}={value}'])
        self.save('running_candidate')
        print('[CANDIDATE_START] ' + json.dumps(self.current), flush=True)
        report, pending, last_progress = None, b'', 0.
        tail = deque(maxlen=25)
        diagnostics = deque(maxlen=40)
        self.client = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while data := os.read(self.client.stdout.fileno(), 65536):
                if self.viewer.poll() is not None:
                    raise RuntimeError('Owned Rerun viewer exited')
                records = re.split(b'[\r\n]', pending + data)
                pending = records.pop()
                for record in records:
                    line = record.decode('utf-8', 'replace').strip()
                    if not line:
                        continue
                    if MARKER in line:
                        report = json.loads(line.split(MARKER, 1)[1])
                        continue
                    tail.append(line)
                    episode = re.search(r'\[EPISODE \d+\]', line)
                    if episode:
                        # tqdm can leave its current progress line directly
                        # before the evaluator's episode summary.
                        summary = line[episode.start():]
                        self.current['last_episode'] = summary
                        self.save()
                        print(summary, flush=True)
                    elif line.startswith('Rollout') and time.monotonic() - last_progress >= 30:
                        last_progress = time.monotonic()
                        print(line, flush=True)
                    elif FAILURE_LINE.search(line):
                        diagnostics.append(line)
                        print(line, flush=True)
                    elif line.startswith('[SMOKE TEST]'):
                        print(line, flush=True)
            code = self.client.wait()
            if code or report is None or not report.get('complete'):
                context = list(dict.fromkeys(diagnostics or tail))
                if self.server:
                    try:
                        server_log = subprocess.run(['docker', 'logs', '--tail', '80', self.server],
                                                    capture_output=True, text=True, timeout=10)
                        context.extend(('Owned policy server:', server_log.stdout, server_log.stderr))
                    except (OSError, subprocess.TimeoutExpired) as error:
                        context.append('Could not read owned policy diagnostics: ' + str(error))
                raise RuntimeError(f'Candidate exited {code} without a complete result: ' + '\n'.join(context))
            validate_result(report, self.expected_by_seed[seed], horizon, options, cameras)
            result = dict(self.current, report=report)
            self.results.append(result)
            self.current = None
            self.save('candidate_complete')
            print('[CANDIDATE_RESULT] ' + json.dumps(dict(label=label, seed=seed,
                  successes=report['overall']['successes'], episodes=report['overall']['episodes'])), flush=True)
            return result
        finally:
            self.cleanup_client()

    def close(self):
        try:
            self.cleanup_client()
        finally:
            self.stop_server()
            self.close_viewer()

    def close_viewer(self):
        if self.viewer is not None and self.viewer.poll() is None:
            self.viewer.terminate()
            try:
                self.viewer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.viewer.kill()
                self.viewer.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--control-repo', type=Path, default=Path('/home/rpuser1/Sim-to-Real-SO-101-Workshop'))
    args = parser.parse_args()
    investigation = Investigation(args)
    def stop(signum, frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        investigation.wait_for_existing()
        investigation.wait_for_gpu_memory()
        investigation.start_viewer()
        baseline = dict(investigation.cfg['policy']['options'])
        fp32 = dict(baseline, action_compute_dtype='float32')
        native = {key: value for key, value in fp32.items() if key not in ('pan_model_scale', 'pan_model_offset')}
        candidates = [('baseline_h30', baseline, 30), ('fp32_compute_h30', fp32, 30),
                      ('fp32_compute_h10', fp32, 10), ('fp32_compute_steps20_h30', dict(fp32, num_steps=20), 30),
                      ('native_fp32_h30', native, 30), ('baseline_h10', baseline, 10)]
        for label, options, horizon in candidates:
            investigation.trial(label, options, horizon)
        if max(item['report']['overall']['successes'] for item in investigation.results) <= investigation.results[0]['report']['overall']['successes']:
            investigation.trial('fp32_explicit_task_instruction', fp32, instruction=
                'Move the arm towards the blue cube, grasp it, lift it up, and drop it into the white box.')
        ranked = sorted(investigation.results, key=rank_result, reverse=True)
        leader = ranked[0]
        # Paired verification tests both the control and the leader under the
        # same extra seeds; it never silently increases any time budget.
        for seed in (9182, 4173):
            investigation.trial('verify_baseline_h30', baseline, 30, seed)
            if leader['label'] != 'baseline_h30':
                investigation.trial('verify_' + leader['label'], leader['options'], leader['horizon'], seed,
                                     instruction=leader['instruction'])
        investigation.save('planned_trials_complete')
        print('[RESEARCH] Planned comparisons complete; results require analysis and premortem before promotion.', flush=True)
    except BaseException as error:
        investigation.current = dict(investigation.current or {}, error=repr(error))
        investigation.save('interrupted_or_failed')
        raise
    finally:
        investigation.close()


if __name__ == '__main__':
    main()
