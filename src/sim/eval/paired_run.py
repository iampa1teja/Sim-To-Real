"""Interactive matched-start comparisons driven by the testing-console controls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import threading
import time
import uuid

import numpy as np



def _local_tcp():
    """Load policies/molmoact2/local_tcp.py by path; the policies folder is not an importable package."""
    import importlib.util
    import sys
    from core.policy_client.discovery import get_manifest

    module_name = '_s2r_policy_molmoact2_local_tcp'
    if module_name not in sys.modules:
        definition = importlib.util.spec_from_file_location(
            module_name, Path(get_manifest('molmoact2').path) / 'local_tcp.py')
        module = importlib.util.module_from_spec(definition)
        sys.modules[module_name] = module
        try:
            definition.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    return sys.modules[module_name]


def run(args, app):
    import gymnasium as gym
    import torch
    import sim.envs
    from core.arms.mapping import JointMapping
    from core.policy_client.contract import ArmInfo
    from core.rollout.paired_gui import PairedEvaluationGui, trial_arguments
    from core.rollout.real_rollout import connect_follower, joint_ranges, make_policy, camera_specs
    from sim.teleop.agent import _env_id, _make_env_cfg, _sim_cameras
    from sim.teleop.config import AgentConfig
    from sim.teleop.loop import gui_frames, gui_panes, object_pose_in_base
    from sim.datagen.spawner import ObjectSpawner
    from sim.recording.recording_setup import cleanup, disconnect
    from sim.envs.mdp._common import as_torch
    from sim.eval.sim_real import TrialVideo, limited_target, position_with_leader, connect_trial_policies

    env = follower = leader = gui = spawner = video = log = None
    policies, comparisons = [], []
    future = future_side = comparison = selected = folder = None
    active_side = None
    last_real_goal = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='paired-policy')
    output = Path(args.output or f'outputs/eval-sim-real/{datetime.now():%Y%m%d_%H%M%S}')
    output.mkdir(parents=True, exist_ok=True)
    phase, ready, positioning, want_disconnect = 'disconnected', False, False, False
    phase_started = 0.
    steps = 0
    status, error = 'Connect the real arm to prepare a comparison.', ''
    began = time.monotonic()
    stop_event = threading.Event()
    previous_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda *_: stop_event.set())

    def close_policies():
        for policy in policies:
            close = getattr(policy, 'close', None)
            if close:
                cleanup('Policy close', close)

    def close_side(reason):
        nonlocal active_side, video, log
        if active_side:
            label = gui.trial_end_for(active_side)
            ended_at = label['at'] if label else time.monotonic()
            elapsed = max(0., min(ended_at - phase_started, selected.episode_seconds))
            row = dict(side=active_side, status='operator_end' if label else reason, steps=steps, wall_seconds=elapsed)
            if label:
                row.update({key: value for key, value in label.items() if key not in ('side', 'at')})
            comparison['trials'].append(row)
            (folder / 'report.json').write_text(json.dumps(comparison, indent=2) + '\n')
            if video:
                video.close(elapsed)
            if log:
                log.close()
        active_side = video = log = None

    def complete(reason, detail=''):
        nonlocal comparison, policies, phase, ready, positioning, status, future, future_side
        close_side(reason)
        if future is not None:
            try:
                future.result()
            except Exception:
                pass
            future = future_side = None
        executor.submit(close_policies).result()
        policies = []
        if comparison:
            comparison.update(status=reason, detail=detail)
            labels = {row['side']: row.get('stage') for row in comparison['trials']}
            comparison['stage_comparison'] = dict(real=labels.get('real'), sim=labels.get('sim'),
                same_stage=labels['real'] == labels['sim'] if labels.get('real') and labels.get('sim') else None,
                source='operator')
            (folder / 'report.json').write_text(json.dumps(comparison, indent=2) + '\n')
            gui.record_report(comparison)
            comparisons.append(dict(id=comparison['id'], policy=comparison['policy'],
                                    instruction=comparison['instruction'], status=reason))
            (output / 'report.json').write_text(json.dumps(dict(arm=args.arm, scene=args.scene,
                                                              task=args.task, comparisons=comparisons), indent=2) + '\n')
        comparison = None
        phase = 'ready' if follower else 'disconnected'
        ready, positioning = args.no_object, False
        status = ('Finished both trials. Reposition and Spawn for another comparison.' if reason == 'completed'
                  else f'Stopped. {detail}'.strip())

    def begin(side):
        nonlocal phase, active_side, phase_started, steps, video, log, last_real_goal
        phase = active_side = side
        phase_started, steps = time.monotonic(), 0
        if side == 'real':
            last_real_goal = None
        gui.publish(side, f'{side.upper()} trial starting.', connected=follower is not None,
                    duration=selected.episode_seconds)
        video = TrialVideo(folder / f'{side}.mp4', args.fps)
        log = (folder / f'{side}.jsonl').open('w', buffering=1)

    try:
        cfg = AgentConfig(arm=args.arm, scene=args.scene, task=args.task, variant=args.variant, fps=args.fps)
        env_id = _env_id(cfg)
        env = gym.make(env_id, cfg=_make_env_cfg(cfg, env_id, args.device, args.calibration_selection))
        obs, _ = env.reset()
        arm = args.arm_spec
        mapping = JointMapping.from_arm(arm, args.calibration_selection, device=env.unwrapped.device)
        robot = env.unwrapped.scene['robot']
        joint_ids = robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)[0]
        base_id = robot.find_bodies(arm.links.base)[0][0]
        ids = torch.tensor([0], device=env.unwrapped.device)
        spawner = ObjectSpawner(env, task=sim.envs.TASK)
        sim_cameras, real_cameras = _sim_cameras(env), camera_specs(args)
        keys = list(arm.feature_names)
        ranges = joint_ranges(arm, args.calibration_selection.mapping)
        real_names = json.loads(args.rename_map)
        sim_names = {'realsense_rgb': 'room', 'wrist_cam': 'wrist'}
        gui = PairedEvaluationGui(args, gui_panes(sim_cameras, real_cameras), output)
        print(f'[eval-sim-real] Open {gui.url}; no physical device opens until Connect.', flush=True)
        target = obs['policy']['joint_pos_obs'].clone()
        start_raw = start_objects = None

        def mirror(raw):
            value = mapping.to_sim(torch.as_tensor(raw, device=env.unwrapped.device)).unsqueeze(0)
            robot.write_joint_state_to_sim_index(position=value, velocity=torch.zeros_like(value),
                                                joint_ids=joint_ids, env_ids=ids)
            robot.set_joint_position_target_index(target=value, joint_ids=joint_ids, env_ids=ids)
            return value

        with torch.inference_mode():
            while not stop_event.is_set() and app.is_running():
                tick = time.monotonic()
                requests = gui.consume_requests()
                if requests.get('exit'):
                    gui.cancel.set()
                    break
                want_disconnect |= bool(requests.get('disconnect'))
                if requests.get('stop'):
                    positioning = False
                    if comparison:
                        phase = 'stopping'
                    else:
                        status = 'Stopped. The real arm holds its last position.'
                if requests.get('connect') and follower is None:
                    try:
                        gui.cancel.clear()
                        follower = connect_follower(args)
                        if not args.no_leader:
                            from sim.teleop.devices import connect_device
                            leader = connect_device('leader', args.leader_port, args.leader_id, cameras={}, fps=args.fps,
                                                    selection=args.calibration_selection, calibrate=False)
                        phase, ready, positioning, error = 'ready', args.no_object, leader is not None, ''
                        status = ('Leader positioning enabled. Open the gripper, place the cube and Spawn.' if positioning
                                  else 'Connected. Position the real arm/cube, then Spawn the simulation object.')
                    except Exception as exc:
                        disconnect(leader, 'physical leader'); leader = None
                        disconnect(follower, 'physical follower'); follower = None
                        phase, status, error = 'disconnected', 'Connection failed. Check the error below.', str(exc)

                observation = follower.get_observation() if follower else {}
                real_state = np.asarray([observation[key] for key in keys], dtype=np.float32) if follower else None
                if real_state is not None and not np.isfinite(real_state).all():
                    raise RuntimeError('Real arm returned non-finite joint positions')
                real_images = {real_names[name]: observation[name] for name in real_cameras} if follower else {}
                if comparison and gui.cancelled:
                    phase = 'stopping'

                if phase == 'ready' and follower:
                    with gui.motion_lock:
                        positioned = position_with_leader(leader, follower, real_state, keys, ranges,
                            enabled=positioning and not requests.get('run') and not gui.cancelled,
                            motion=args.enable_motion, step_limit=args.max_relative_target)
                    target = mirror(real_state if args.enable_motion or not positioning else positioned)
                    if requests.get('spawn') and not args.no_object:
                        if any(real_state[i] < 40 for i, joint in enumerate(arm.action_joints) if joint.role == 'gripper'):
                            status, error = 'Open the gripper before spawning.', 'The cube needs clearance between the jaws.'
                        else:
                            env.step(target)
                            spawner.spawn()
                            ready, error = True, ''
                            status = 'Object spawned. Match the real cube placement, then Run real → sim.'
                    if requests.get('run') and ready:
                        selected = trial_arguments(args, requests['run'])
                        start_raw = real_state.copy()
                        roundtrip = mapping.to_raw(mapping.to_sim(start_raw)).cpu().numpy()
                        if not np.allclose(roundtrip, start_raw, atol=.1):
                            status, error = 'Choose a start pose within both real and simulation limits.', 'Start pose exceeds the shared ranges.'
                        else:
                            positioning, ready, error = False, False, ''
                            start_objects = {name: as_torch(env.unwrapped.scene[name].data.root_state_w).clone()
                                             for name in (() if args.no_object else sim.envs.TASK.spec.tracked_objects)}
                            run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
                            folder = output / run_id
                            folder.mkdir()
                            comparison = dict(schema='s2r_sim_real_testing_v1', id=run_id, arm=args.arm,
                                scene=args.scene, task=args.task, policy=selected.policy, instruction=selected.lang_instruction,
                                policy_host=selected.policy_host, policy_port=selected.policy_port,
                                backend=selected.backend, checkpoint_label=args.checkpoint,
                                seconds_per_trial=selected.episode_seconds, fps=args.fps,
                                motion_enabled=args.enable_motion, outcome='unlabeled', status='running', trials=[],
                                real_object_pose='manually matched at jaw; not measured',
                                calibration=args.calibration_selection.provenance,
                                start_raw=start_raw.tolist(), joint_names=list(arm.joint_names),
                                sim_objects_start_base_wxyz={name: object_pose_in_base(env, name, base_id).cpu().tolist()
                                                             for name in start_objects})
                            def factory():
                                if selected.backend == 'local-lerobot':
                                    return _local_tcp().LocalLeRobotPolicy(ArmInfo.from_arm(arm, args.calibration_selection.mapping),
                                        port=selected.policy_port or 5005,
                                        instruction=selected.lang_instruction, horizon=selected.action_horizon or 30)
                                return make_policy(selected)
                            future = executor.submit(connect_trial_policies, factory)
                            future_side, phase = 'connect', 'connecting_policy'
                            status = 'Connecting the selected policy for both trials; Stop cancels before motion.'

                if phase == 'connecting_policy' and future is not None and future.done():
                    try:
                        policies, future = future.result(), None
                        comparison['policy_servers'] = [getattr(policy, 'server_info', None) for policy in policies]
                        comparison['joint_transforms'] = [getattr(policy, 'joint_transform', None) for policy in policies]
                        if gui.cancelled:
                            phase = 'stopping'
                        else:
                            begin('real')
                    except Exception as exc:
                        future = None
                        error = str(exc)
                        complete('error', f'Policy unavailable: {exc}')

                if phase == 'stopping':
                    if future is None or future.done():
                        if future is not None:
                            try:
                                value = future.result()
                                if future_side == 'connect':
                                    policies = value
                            except Exception:
                                pass
                            future = None
                        complete('error' if error else 'stopped', error or 'Operator Stop or browser heartbeat timeout.')
                    else:
                        status = 'Stopping; waiting for the pending reply. The real arm holds position.'

                if phase in ('real', 'sim') and (
                        gui.trial_end_for(phase) or tick - phase_started >= selected.episode_seconds):
                    previous = phase
                    label = gui.trial_end_for(previous)
                    if previous == 'real' and label:
                        with gui.motion_lock:
                            if not gui.cancelled and args.enable_motion:
                                # Freeze arm motion without relaxing the last gripper goal.
                                hold = {key: (last_real_goal[key] if last_real_goal and joint.role == 'gripper'
                                              else float(real_state[i]))
                                        for i, (key, joint) in enumerate(zip(keys, arm.action_joints))}
                                bounded_hold = limited_target(hold, real_state, keys, ranges, first=False,
                                    start_limit=args.max_start_error, step_limit=args.max_relative_target)
                                follower.send_action(dict(zip(keys, map(float, bounded_hold))))
                    close_side('operator_end' if label else 'time_limit')
                    if previous == 'real':
                        target = mirror(start_raw)
                        for name, state in start_objects.items():
                            obj = env.unwrapped.scene[name]
                            obj.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=ids)
                            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(state[:, 7:13]), env_ids=ids)
                        env.unwrapped.sim.forward()
                        env.unwrapped.scene.update(0.)
                        env.unwrapped.sim.render()
                        obs = env.unwrapped.observation_manager.compute()
                        begin('sim')
                    else:
                        complete('completed')

                visual = obs.get('visual', {})
                sim_images = {sim_names.get(name, name): visual[f'rgb_{name}'][0].cpu().numpy() for name in sim_cameras}
                sim_state = mapping.to_raw(obs['policy']['joint_pos_obs'][0]).cpu().numpy()
                if phase in ('real', 'sim'):
                    state, images = (real_state, real_images) if phase == 'real' else (sim_state, sim_images)
                    try:
                        elapsed = time.monotonic() - phase_started
                        video.write(images, f'{phase.upper()} {elapsed:.1f}s / {selected.episode_seconds:.0f}s', elapsed)
                        if future is not None and future.done():
                            completed, future = future, None
                            if future_side != phase:
                                # A result from the ended side must not affect the next side.
                                completed.exception()
                                action = None
                            else:
                                action = completed.result()
                            if future_side == phase and not gui.cancelled and not gui.trial_end_for(phase):
                                values = limited_target(action, state, keys, ranges, first=steps == 0,
                                    start_limit=args.max_start_error, step_limit=args.max_relative_target)
                                with gui.motion_lock:
                                    accepted = not gui.cancelled and not gui.trial_end_for(phase)
                                    if accepted:
                                        if phase == 'real' and args.enable_motion:
                                            command = dict(zip(keys, map(float, values)))
                                            sent = follower.send_action(command)
                                            last_real_goal = {key: float(sent[key]) for key in keys} if isinstance(sent, dict) else command
                                        elif phase == 'sim':
                                            target = mapping.to_sim(values).unsqueeze(0)
                                if accepted:
                                    log.write(json.dumps(dict(seconds=elapsed, state=state.tolist(), target=values.tolist(),
                                                              motion=phase == 'sim' or args.enable_motion)) + '\n')
                                    steps += 1
                        if future is None and not gui.cancelled and not gui.trial_end_for(phase):
                            policy = policies[0 if phase == 'real' else 1]
                            future = executor.submit(policy.get_raw_action, state.copy(), {k: v.copy() for k, v in images.items()})
                            future_side = phase
                        status = f'{phase.upper()} trial: {elapsed:.1f}/{selected.episode_seconds:.0f}s. '
                        status += ('Real arm holding; simulation running.' if phase == 'sim' else
                                   'Real motion enabled.' if args.enable_motion else 'Shadow mode: real motion disabled.')
                    except Exception as exc:
                        error = str(exc)
                        # Drain any outstanding request before closing its socket.
                        gui.cancel.set()
                        phase = 'stopping'
                        status = f'Stopping: {exc}'

                if want_disconnect and comparison is None and future is None:
                    disconnect(leader, 'physical leader'); leader = None
                    disconnect(follower, 'physical follower'); follower = None
                    observation = {}
                    gui.clear_real_frames()
                    phase, ready, positioning, want_disconnect = 'disconnected', False, False, False
                    status = 'Real arm disconnected. Motor torque was not released.'
                spawner.hold()
                obs, *_ = env.step(target)
                if not torch.isfinite(obs['policy']['joint_pos_obs']).all():
                    raise RuntimeError('Simulation joints became non-finite; comparison stopped')
                gui.update_images(gui_frames(obs.get('visual'), sim_cameras, observation, real_cameras))
                gui.publish(phase, status, connected=follower is not None, can_start=ready,
                            elapsed=time.monotonic() - phase_started if active_side else 0., steps=steps if active_side else 0,
                            duration=selected.episode_seconds if selected else args.episode_seconds,
                            error=error, positioning=positioning)
                if args.preview_seconds and time.monotonic() - began >= args.preview_seconds:
                    break
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if gui:
            gui.cancel.set()
        if future is not None:
            try:
                value = future.result()
                if future_side == 'connect':
                    policies = value
            except Exception:
                pass
        if comparison:
            cleanup('Comparison close', lambda: complete('error' if error else 'stopped', error or 'Evaluation closed.'))
        executor.submit(close_policies).result()
        executor.shutdown(wait=True, cancel_futures=True)
        if gui:
            cleanup('GUI close', gui.destroy)
        cleanup('Leader disconnect', lambda: disconnect(leader, 'physical leader'))
        cleanup('Follower disconnect', lambda: disconnect(follower, 'physical follower'))
        if spawner:
            cleanup('Spawner close', spawner.close)
        if env:
            cleanup('Environment close', env.close)
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
