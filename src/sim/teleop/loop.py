"""Teleop ticks and recording request transitions, separate from device/environment setup."""
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import time
from typing import Any

import torch


def object_pose_in_base(env, name, base_id):
    """Capture one base-relative object pose on device; sidecars use WXYZ."""
    import isaaclab.utils.math as math_utils
    from sim.envs.mdp._common import as_torch

    scene = env.unwrapped.scene
    robot, obj = scene['robot'], scene[name]
    position, quaternion = math_utils.subtract_frame_transforms(
        as_torch(robot.data.body_link_pos_w)[:, base_id], as_torch(robot.data.body_link_quat_w)[:, base_id],
        as_torch(obj.data.root_pos_w), as_torch(obj.data.root_quat_w),
    )
    return torch.cat((position, quaternion[:, 3:4], quaternion[:, :3]), dim=-1)[0].clone()


def _write_sidecar(root, index, *, poses, sizes, fps, task, arm, reference_frame):
    from core.datasets.task_meta import save_task_trajectory

    save_task_trajectory(root, index, poses, sizes, fps, task=task, arm=arm, reference_frame=reference_frame)


def save_episode(recorders, trajectory, tracked, fps, *, task, arm):
    """Commit every tracked trajectory on the synchronized dataset frame timeline."""
    if tracked is not None and len(trajectory) != recorders.frame_count:
        raise ValueError(f'Object trajectory has {len(trajectory)} frames; datasets have {recorders.frame_count}.')
    sidecar = None
    if tracked is not None:
        poses = {name: torch.stack([frame[name] for frame in trajectory]).cpu().tolist()
                 for name in tracked} if trajectory else {name: [] for name in tracked}
        sidecar = partial(_write_sidecar, poses=poses, sizes=tracked, fps=fps, task=task.name, arm=arm.name,
                          reference_frame=f'robot base link ({arm.links.base})')
    index = recorders.save_episode(sidecar)
    trajectory.clear()
    return index


GUI_REQUESTS = {'start': 'start', 'stop': 'end', 'save': 'end', 'exit': 'stop', 'discard': 'rerecord',
                'spawn': 'spawn'}


def gui_panes(sim_cameras, real_cameras):
    return [f'Sim: {name}' for name in sim_cameras] + [f'Real: {name}' for name in real_cameras]


def gui_frames(visual_obs, sim_cameras, follower_observation, real_cameras):
    frames = {}
    for name in sim_cameras:
        images = (visual_obs or {}).get(f'rgb_{name}')
        frames[f'Sim: {name}'] = images[0] if images is not None else None
    for name in real_cameras:
        frames[f'Real: {name}'] = (follower_observation or {}).get(name)
    return frames


@dataclass(frozen=True)
class LoopResources:
    env: Any
    env_id: str
    leader: Any
    mapping: Any
    arm: Any
    task: Any
    controls: Any
    spawner: Any
    sim_cameras: dict
    real_cameras: dict
    follower: Any = None
    recorders: Any = None
    real_pipeline: Any = None
    gui: Any = None
    telemetry: Any = None
    tracked: dict | None = None
    base_id: int | None = None


class RecordingPhase(Enum):
    BUFFER = 'buffer'
    RECORDING = 'recording'
    STOPPED = 'stopped'


@dataclass(frozen=True)
class RequestTransition:
    stop: bool = False
    invalidate_observation: bool = False
    reset_roll: bool = False


@dataclass
class RecordingController:
    cfg: Any
    resources: LoopResources
    phase: RecordingPhase = RecordingPhase.BUFFER
    note: str = ''
    trajectory: list = field(default_factory=list)

    @property
    def recording(self):
        return self.phase is RecordingPhase.RECORDING

    def consume_requests(self):
        ctx = self.resources
        requests = dict(ctx.controls.consume_requests())
        if ctx.gui is not None:
            gui_requests = ctx.gui.consume_requests()
            for button, request in GUI_REQUESTS.items():
                if gui_requests.get(button):
                    requests[request] = True
            if gui_requests.get('rerecord') and ctx.recorders is not None and not self.recording:
                ctx.gui.set_state(active=False, can_start=False, busy=True,
                                  status='Preparing a safe copy for re-recording...')
                try:
                    ctx.recorders.begin(gui_requests['rerecord_episode'])
                    self.note = f'Re-recording episode {ctx.recorders.target}. Discard keeps the original.'
                except (ValueError, OSError) as exc:
                    self.note = f'Cannot re-record: {exc}'
        return requests

    def _save(self):
        ctx = self.resources
        return save_episode(ctx.recorders, self.trajectory, ctx.tracked, self.cfg.fps, task=ctx.task, arm=ctx.arm)

    def _discard(self):
        self.resources.recorders.cancel_episode()
        self.trajectory.clear()

    def handle(self, requests):
        """Apply stop/discard/save/reset/start precedence before the next device read."""
        ctx = self.resources
        invalidate = False
        if requests.get('spawn'):
            try:
                ctx.spawner.spawn(self.cfg.sim_reset_object_name, offset=tuple(self.cfg.sim_reset_object_offset))
                invalidate = True
            except ValueError as exc:
                print(f'[WARNING]: Spawn failed: {exc}')

        if requests.get('stop'):
            if self.recording:
                self._discard() if requests.get('rerecord') else self._save()
                ctx.controls.set_recording(False)
            self.phase = RecordingPhase.STOPPED
            return RequestTransition(stop=True, invalidate_observation=invalidate)

        resetting = any(requests.get(key) for key in ('rerecord', 'end', 'reset'))
        if resetting:
            if self.recording and requests.get('rerecord'):
                self._discard()
                self.note = 'Episode discarded.'
            elif self.recording:
                replacing = ctx.recorders.target
                index = self._save()
                if index is None:
                    self.note = 'Nothing recorded; episode not saved.'
                elif replacing is not None:
                    self.note = f'Replaced episode {index}; the original dataset was backed up.'
                else:
                    self.note = f'Saved episode {index}.'
            elif ctx.recorders is not None and ctx.recorders.target is not None:
                self._discard()
                self.note = 'Re-record cancelled; original episode kept.'
            self.phase = RecordingPhase.BUFFER
            ctx.env.reset()
            ctx.spawner.reset()
            ctx.controls.set_recording(False)
            print('[INFO]: RESET/BUFFER phase. Reposition the real scene and leader, then start the next episode.')
            invalidate = True

        if requests.get('start') and not self.recording and not resetting:
            if ctx.recorders is None:
                print('[WARNING]: Recording is disabled; restart with the dataset settings before pressing S.')
            else:
                if self.cfg.sim_reset_object_name:
                    ctx.spawner.spawn(self.cfg.sim_reset_object_name, offset=tuple(self.cfg.sim_reset_object_offset))
                self.phase = RecordingPhase.RECORDING
                self.note = ''
                ctx.controls.set_recording(True)
                print(f'[INFO]: Recording episode {ctx.recorders.episode_index}.')
                invalidate = True
        return RequestTransition(invalidate_observation=invalidate, reset_roll=resetting)

    def gui_status(self):
        recorders = self.resources.recorders
        if self.recording:
            return f'\u25cf Recording episode {recorders.episode_index}: {recorders.frame_count / self.cfg.fps:.1f} s'
        if recorders is None:
            return 'Recording is disabled (set dataset.repo_id, dataset.root and dataset.task_name).'
        return f'{self.note} Start recording when ready.' if self.note else 'Start recording when ready.'


def run_loop(cfg, simulation_app, ctx: LoopResources):
    """Run input, recording and physics in their established per-frame order."""
    from sim.recording.lerobot_recorder import split_visual_obs
    from sim.teleop.devices import real_dataset_frame

    controller = RecordingController(cfg, ctx)
    env, mapping = ctx.env, ctx.mapping
    roll_spec = ctx.arm.wrist_roll
    roll = mapping.names.index(roll_spec.joint) if roll_spec and roll_spec.relative_in_teleop else None
    if roll is not None:
        roll_min, roll_max = torch.deg2rad(mapping.clamp_mins[roll]), torch.deg2rad(mapping.clamp_maxs[roll])
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    leader_roll_start = sim_roll_start = previous_target = previous_leader_action = obs = None
    with torch.inference_mode():
        while simulation_app.is_running():
            loop_start = time.perf_counter()
            if ctx.telemetry is not None:
                ctx.telemetry.observe(ctx.follower.get_observation(), mapping, env.unwrapped.scene['robot'],
                                      previous_target, previous_command=previous_leader_action)
            if ctx.spawner.poll():
                obs = None
            transition = controller.handle(controller.consume_requests())
            if transition.stop:
                break
            if transition.invalidate_observation:
                obs = None
            if transition.reset_roll:
                leader_roll_start = None
            try:
                leader_action = ctx.leader.get_action()
            except Exception as exc:
                raise RuntimeError(f'Failed to read the physical leader: {exc}') from exc
            target = mapping.to_sim(mapping.raw_from_actions(leader_action))
            if obs is None:
                obs = env.unwrapped.observation_manager.compute()
            if roll is not None:
                if leader_roll_start is None:
                    leader_roll_start = target[roll].clone()
                    sim_roll_start = obs['policy']['joint_pos_obs'][0, roll].clone()
                target[roll] = torch.clamp(sim_roll_start + target[roll] - leader_roll_start, roll_min, roll_max)
            follower_observation = None
            if ctx.follower is not None and (controller.recording or ctx.gui is not None):
                try:
                    follower_observation = ctx.follower.get_observation()
                except Exception as exc:
                    raise RuntimeError(f'Follower observation failed; this episode is discarded: {exc}') from exc
            if controller.recording:
                visual_obs = obs.get('visual')
                if visual_obs is None:
                    raise RuntimeError(f"{ctx.env_id} has no 'visual' observation group to record.")
                rgb, depth, seg = split_visual_obs(visual_obs, ctx.sim_cameras)
                recorder = ctx.recorders.recorders['sim']
                frames = {'sim': recorder.make_sim_frame(
                    mapping.to_raw(target), mapping.to_raw(obs['policy']['joint_pos_obs'][0]), rgb)}
                auxiliary = {'sim': recorder.make_auxiliary_frame(rgb, depth, seg)}
                if ctx.follower is not None:
                    frames['real'] = real_dataset_frame(ctx.real_pipeline, leader_action, follower_observation,
                                                        cfg.dataset.task_name)
                ctx.recorders.add_frames(frames, auxiliary)
                if ctx.tracked is not None:
                    controller.trajectory.append({name: object_pose_in_base(env, name, ctx.base_id) for name in ctx.tracked})
            if ctx.gui is not None:
                ctx.gui.set_state(
                    active=controller.recording, can_start=ctx.recorders is not None and not controller.recording,
                    status=controller.gui_status(), recording=controller.recording,
                    encode_progress='Videos are encoded while recording.',
                    saved_episodes=ctx.recorders.saved_episodes if ctx.recorders is not None else 0,
                    rerecord_episode=ctx.recorders.target if ctx.recorders is not None else None,
                )
                ctx.gui.update_images(gui_frames(obs.get('visual'), ctx.sim_cameras, follower_observation, ctx.real_cameras))
            if ctx.follower is not None:
                try:
                    ctx.follower.send_action(leader_action)
                except Exception as exc:
                    raise RuntimeError(f'Follower action write failed; this episode is discarded: {exc}') from exc
            actions[:] = target
            ctx.spawner.hold()
            try:
                obs, _, _, _, _ = env.step(actions)
            except Exception as exc:
                raise RuntimeError(f'Isaac Sim action application failed: {exc}') from exc
            if not torch.isfinite(obs['policy']['joint_pos_obs']).all():
                raise RuntimeError(
                    'Simulation physics produced nonfinite joint positions; stopping teleop and discarding '
                    'the uncommitted episode. This is a simulation-state failure, not a camera or dataset error.'
                )
            previous_target, previous_leader_action = target.clone(), dict(leader_action)
            remaining = (1.0 / cfg.fps) - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
