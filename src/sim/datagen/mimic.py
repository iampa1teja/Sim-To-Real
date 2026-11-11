"""Isaac Lab Mimic source segmentation and retargeting with the shared S2R recorder.

The upstream DataGenerator owns selection, rigid transforms, interpolation and
subtask sequencing. This module supplies the selected arm's IK and the task's
stage annotations. Imports of runtime environment classes remain lazy until Kit
has been launched. Ordinary Mimic is used; SkillGen/CuRobo is not required.
"""
from copy import deepcopy
import asyncio
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

# Contact-rich demonstrations need command recovery well below the squeeze
# clearance. These are local FK iterations, without advancing physics or PI.
COMMAND_INVERSE_OPTIONS = dict(max_iterations=32, position_tolerance_m=1e-8,
                               orientation_tolerance_rad=1e-7)


@dataclass(frozen=True)
class MimicSpec:
    subtasks: tuple[dict, ...]
    select_source_per_subtask: bool = True

    @classmethod
    def from_task(cls, task, overrides=None):
        values = {**task.spec.raw.get('mimic', {}), **(overrides or {})}
        rows = values.get('subtasks')
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Task {task.name!r} needs [[task.mimic.subtasks]] for Isaac Lab Mimic')
        signals = []
        for index, row in enumerate(rows):
            reference, signal = row.get('object_ref'), row.get('subtask_term_signal')
            if reference is not None and reference not in task.spec.objects:
                raise ValueError(f'Mimic subtask {index} references unknown task object {reference!r}')
            if signal is not None and signal not in task.stages:
                raise ValueError(f'Mimic subtask {index} references unknown task stage {signal!r}')
            if index == len(rows) - 1:
                if signal is not None or tuple(row.get('subtask_term_offset_range', (0, 0))) != (0, 0):
                    raise ValueError('The final Mimic subtask must have no termination signal or offset')
            elif signal is None or signal in signals:
                raise ValueError('Non-final Mimic subtasks need unique task termination signals')
            signals.append(signal)
        return cls(tuple(deepcopy(rows)), bool(values.get('select_source_per_subtask', True)))

    def configure(self, cfg, eef_name, seed):
        from isaaclab.envs import SubTaskConfig

        cfg.datagen_config.seed = seed
        cfg.datagen_config.use_skillgen = False
        cfg.datagen_config.use_navigation_controller = False
        cfg.datagen_config.generation_guarantee = False
        cfg.datagen_config.generation_select_src_per_subtask = self.select_source_per_subtask
        cfg.datagen_config.generation_interpolate_from_last_target_pose = True
        cfg.task_constraint_configs = []
        cfg.subtask_configs = {eef_name: [SubTaskConfig(**{
            'action_noise': 0.0, 'num_interpolation_steps': 5,
            'selection_strategy': 'nearest_neighbor_object', 'selection_strategy_kwargs': {'nn_k': 1}, **row,
        }) for row in self.subtasks]}


def episode_from_annotations(actions, frames, *, eef_name, device='cpu', mapping=None):
    """Create SDK EpisodeData in simulator units; decode raw parquet with mapping."""
    import torch
    from isaaclab.utils.datasets import EpisodeData

    actions = mapping.to_sim(actions) if mapping is not None else actions
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    if actions.ndim != 2 or len(actions) != len(frames) or len(frames) < 2 or not torch.isfinite(actions).all():
        raise ValueError('Mimic source actions and annotations must have matching finite (T,J) frames')
    names = tuple(frames[0]['object_pose'])
    signals = tuple(frames[0]['subtask_term_signals'])
    if any(tuple(frame['object_pose']) != names or tuple(frame['subtask_term_signals']) != signals for frame in frames):
        raise ValueError('Mimic annotation object and signal names must stay consistent')

    def poses(values):
        data = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)
        if data.shape != (len(frames), 4, 4) or not torch.isfinite(data).all():
            raise ValueError('Mimic source poses must have finite shape (T,4,4)')
        if not torch.allclose(data[:, 3], data.new_tensor([0, 0, 0, 1]).expand(len(frames), -1)):
            raise ValueError('Mimic source poses must be homogeneous transforms')
        rotations = data[:, :3, :3]
        identity = torch.eye(3, device=device).expand(len(frames), -1, -1)
        if not torch.allclose(rotations.transpose(1, 2) @ rotations, identity, atol=1e-4, rtol=0):
            raise ValueError('Mimic source rotations must be orthonormal')
        if not torch.allclose(torch.linalg.det(rotations), data.new_ones(len(frames)), atol=1e-4, rtol=0):
            raise ValueError('Mimic source rotations must be right-handed')
        return data

    term_signals = {}
    for name in signals:
        flags = torch.tensor([frame['subtask_term_signals'][name] for frame in frames], dtype=torch.bool, device=device)
        if ((~flags[1:]) & flags[:-1]).any():
            raise ValueError(f'Mimic source signal {name!r} must be latched')
        term_signals[name] = flags.unsqueeze(1)
    episode = EpisodeData()
    episode.data = {'actions': actions, 'obs': {'datagen_info': {
        'eef_pose': {eef_name: poses([frame['eef_pose'] for frame in frames])},
        'target_eef_pose': {eef_name: poses([frame['target_eef_pose'] for frame in frames])},
        'object_pose': {name: poses([frame['object_pose'][name] for frame in frames]) for name in names},
        'subtask_term_signals': term_signals,
    }}}
    return episode


def make_mimic_cfg(cfg, task, arm, spec, seed):
    """Preserve the task-applied scene config while adding the real SDK Mimic mixin."""
    from isaaclab.envs import MimicEnvCfg
    from isaaclab.utils import configclass
    from sim.envs.scene_builder import SceneEnvCfg

    @configclass
    class S2RMimicCfg(SceneEnvCfg, MimicEnvCfg):
        def __post_init__(self):
            # This wraps a fully built task config. Re-running the base scene
            # builders here (or during replace/copy) would reset its mount and
            # task-specific actions/events.
            pass

    S2RMimicCfg.to_dict = SceneEnvCfg.to_dict
    values = deepcopy(vars(cfg))
    result = S2RMimicCfg()
    for name, value in values.items():
        setattr(result, name, value)
    spec.configure(result, arm.links.ee, seed)
    result.recorders = None  # SuccessRecorder owns the only LeRobot commit/filter path.
    result.terminations = None  # The shared rollout observes the exact terminal frame.
    result.class_type = 'sim.datagen.mimic:S2RMimicEnv'
    return result


def _tensor(value):
    import torch

    return value if isinstance(value, torch.Tensor) else value.torch


def __getattr__(name):
    if name != 'S2RMimicEnv':
        raise AttributeError(name)
    import torch
    import isaaclab.utils.math as math_utils
    from isaaclab.envs import ManagerBasedRLMimicEnv

    class S2RMimicEnv(ManagerBasedRLMimicEnv):
        """The SDK environment interface, backed by the same generic native IK."""

        def _base(self):
            robot = self.scene['robot']
            index = robot.find_bodies(self.s2r_arm.links.base)[0][0]
            return _tensor(robot.data.body_link_pos_w)[:, index], _tensor(robot.data.body_link_quat_w)[:, index]

        def _relative_pose(self, position, orientation, env_ids=None):
            base_position, base_orientation = self._base()
            position, orientation = math_utils.subtract_frame_transforms(
                base_position, base_orientation, position, orientation)
            result = math_utils.make_pose(position, math_utils.matrix_from_quat(orientation))
            return result if env_ids is None else result[env_ids]

        def get_robot_eef_pose(self, eef_name, env_ids=None):
            if eef_name != self.s2r_arm.links.ee:
                raise ValueError(f'Unknown Mimic end effector {eef_name!r}')
            return self._relative_pose(*self.s2r_ik.current_pose(), env_ids=env_ids)

        def get_object_poses(self, env_ids=None):
            return {name: self._relative_pose(_tensor(self.scene[name].data.root_pos_w),
                                              _tensor(self.scene[name].data.root_quat_w), env_ids)
                    for name in self.s2r_task.spec.objects}

        def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict,
                                      action_noise_dict=None, env_id=0):
            if env_id != 0 or self.num_envs != 1:
                raise ValueError('The shared LeRobot Mimic recorder currently runs one environment')
            eef_name = self.s2r_arm.links.ee
            target = target_eef_pose_dict[eef_name].clone()
            noise = float((action_noise_dict or {}).get(eef_name) or 0.0)
            if noise:
                target[:3, 3] += torch.randn_like(target[:3, 3]) * noise
            # These poses encode FK of the source's executed joint commands,
            # including any load compensation already applied by its controller.
            # Recover bounded joint targets locally; another feedback/PI update
            # would add that compensation again and distort the demonstration.
            measured = _tensor(self.s2r_ik.robot.data.joint_pos)[:, self.s2r_ik.joint_ids]
            solved = self.s2r_fk.inverse(target.detach().cpu().numpy(), measured[0].detach().cpu().numpy(),
                                         options=self.s2r_ik.options, **COMMAND_INVERSE_OPTIONS)
            action = solved.joint_positions.to(dtype=measured.dtype, device=self.device)
            indices = [index for index, joint in enumerate(self.s2r_arm.action_joints) if joint.role == 'gripper']
            grip = torch.as_tensor(gripper_action_dict[eef_name], dtype=action.dtype, device=action.device).flatten()
            if grip.shape != (len(indices),):
                raise ValueError('Mimic gripper action differs from the declared gripper joints')
            action[0, indices] = grip
            self.s2r_last_target = target
            self.s2r_last_gripper = grip
            return action

        def action_to_target_eef_pose(self, action):
            values = _tensor(action).detach().cpu().numpy()
            if values.ndim == 1:
                values = values[None]
            poses = np.stack([self.s2r_fk.fk(row) for row in values])
            return {self.s2r_arm.links.ee: torch.as_tensor(poses, dtype=torch.float32, device=self.device)}

        def actions_to_gripper_actions(self, actions):
            indices = [index for index, joint in enumerate(self.s2r_arm.action_joints) if joint.role == 'gripper']
            return {self.s2r_arm.links.ee: actions[..., indices]}

        def get_subtask_term_signals(self, env_ids=None):
            stages = self.s2r_rollout.recorder.trace.stages
            return {name: torch.tensor([bool(stages[name])], device=self.device) for name in self.s2r_task.stages}

    S2RMimicEnv.__qualname__ = name
    globals()[name] = S2RMimicEnv
    return S2RMimicEnv


def parse_args(argv=None):
    from sim.datagen import scripted

    parser = scripted.build_parser()
    parser.description = __doc__
    for action in parser._actions:
        if action.dest == 'source':
            action.choices = ('mimic',)
            action.default = 'mimic'
    parser.add_argument('--source_dataset', type=Path, help='Successful S2R dataset with datagen_meta annotations')
    parser.add_argument('--input_file', type=Path, help='Alternatively, an SDK-annotated Mimic HDF5 source')
    parser.add_argument('--mimic_options', default='{}', help='JSON object overriding task.mimic options')
    parser.add_argument('--diagnostic_trace', action='store_true',
                        help='Save existing frame annotations, post-step signals and joint buffers beside --report')
    args = scripted.parse_args(argv, parser=parser)
    args.mimic_options = json.loads(args.mimic_options) if isinstance(args.mimic_options, str) else args.mimic_options
    if not isinstance(args.mimic_options, dict):
        parser.error('mimic_options must be a JSON object')
    if (args.source_dataset is None) == (args.input_file is None):
        parser.error('Select exactly one --source_dataset or --input_file')
    source = args.source_dataset or args.input_file
    if not source.exists():
        parser.error(f'Mimic source does not exist: {source}')
    if args.source_dataset is not None and not (source / 'datagen_meta').is_dir():
        parser.error('source_dataset needs aligned datagen_meta annotations from a successful generated/replayed rollout')
    destination = args.destination.resolve()
    if destination == source.resolve() or destination.is_relative_to(source.resolve()):
        parser.error('Mimic destination must be independent of its source')
    if args.diagnostic_trace:
        if args.report is None:
            parser.error('diagnostic_trace requires --report outside the dataset destination')
        if args.report.resolve().is_relative_to(destination):
            parser.error('diagnostic_trace report must be outside the dataset destination')
    return args


async def load_source_pool(context, spec):
    """Use upstream segmentation for either SDK HDF5 or shared LeRobot annotations."""
    import hashlib
    from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
    from core.arms.mapping import JointMapping
    from core.datasets.sources import load_source, recorded_joint_mapping

    env = context.raw
    pool = DataGenInfoPool(env, env.cfg, env.device)
    if context.args.input_file is not None:
        pool.load_from_dataset_file(str(context.args.input_file))
        provenance = {'input_file': str(context.args.input_file.resolve()),
                      'sha256': hashlib.sha256(context.args.input_file.read_bytes()).hexdigest()}
    else:
        root = context.args.source_dataset
        report_path = root / 'meta/datagen.json'
        source_report = json.loads(report_path.read_text()) if report_path.is_file() else {}
        offset = source_report.get('control_offset_m')
        if offset is None or not np.allclose(offset, context.ik.position_offset_m, atol=1e-9, rtol=0):
            raise ValueError('Mimic source control_offset_m must match the selected TCP; regenerate matching annotations')
        saved, _ = recorded_joint_mapping(root, context.arm)
        source_mapping = JointMapping.from_arm(context.arm, saved or context.calibration.mapping, device=env.device)
        episodes = load_source(root, 'local/mimic_source', set(), arm=context.arm, task=context.task)
        hashes = {}
        for source in episodes:
            path = root / 'datagen_meta' / f'episode_{source.index:06d}.json'
            payload = json.loads(path.read_text())
            if not context.task.is_success(payload.get('stages', {})):
                raise ValueError(f'Mimic source episode {source.index} did not pass its task success check')
            frames = payload['frames']
            episode = episode_from_annotations(source.actions, frames, eef_name=context.arm.links.ee,
                                                device=env.device, mapping=source_mapping)
            for row in spec.subtasks[:-1]:
                signal = row['subtask_term_signal']
                flags = episode.data['obs']['datagen_info']['subtask_term_signals'][signal].flatten()
                if bool(flags[0]) or not bool(flags.any()):
                    raise ValueError(f'Mimic source episode {source.index} needs an observed 0-to-1 transition for {signal}')
            await pool.add_episode(episode)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance = {'source_dataset': str(root.resolve()), 'annotation_sha256': hashes,
                      'source_joint_mapping': saved or context.calibration.mapping,
                      'control_offset_m': list(context.ik.position_offset_m)}
    if not pool.num_datagen_infos:
        raise ValueError('Mimic source contains no annotated demonstrations')
    return pool, provenance


def _success(env):
    import torch

    success = env.s2r_task.is_success(env.s2r_rollout.recorder.trace.stages)
    return torch.tensor([success], dtype=torch.bool, device=env.device)


class _DiagnosticTrace:
    """Copy buffers already observed by the shared executor; never step or evaluate a task."""

    def __init__(self, context, attempt, seed):
        report = Path(context.args.report)
        self.path = report.with_name(report.stem + '-diagnostics') / f'attempt_{attempt:04d}.json'
        if self.path.resolve().is_relative_to(Path(context.args.destination).resolve()):
            raise ValueError('Mimic diagnostic traces must be outside the dataset destination')
        self.data = {'schema_version': 1, 'source': 'isaaclab_mimic', 'attempt': attempt, 'seed': seed,
                     'joint_names': list(context.arm.joint_names),
                     'sim_joint_units': ['rad' if joint.type == 'revolute' else 'm' for joint in context.arm.action_joints],
                     'alignment': 'annotations are pre-action; states, signals and measured joints are post-step',
                     'initial_state': None, 'initial_joint_positions_sim': None, 'steps': []}

    @staticmethod
    def _joints(rollout):
        from sim.datagen.rollout import LiveRollout

        return LiveRollout._numpy(rollout.robot.data.joint_pos)[0, rollout.joint_ids].copy().tolist()

    def begin(self, rollout):
        from sim.datagen.rollout import _json_state

        self.data['initial_state'] = _json_state(rollout.snapshot())
        self.data['initial_joint_positions_sim'] = self._joints(rollout)

    def append(self, rollout, target):
        from sim.datagen.rollout import LiveRollout, _json_state

        state = _json_state(rollout.last_state)
        self.data['steps'].append({
            'frame_index': rollout.recorder.steps - 1,
            'commanded_joint_positions_sim': LiveRollout._numpy(target).reshape(-1).copy().tolist(),
            'measured_joint_positions_sim': self._joints(rollout),
            'task_signals': deepcopy(state['signals']), 'state': state,
            'latched_stages': dict(rollout.recorder.trace.stages),
        })

    def write(self, rollout, result, error):
        from sim.datagen.rollout import write_json

        self.data['annotations'] = [] if rollout is None else deepcopy(rollout.recorder.annotations)
        self.data['result'], self.data['error'] = result, error
        write_json(self.path, self.data)


def generate_attempt(context, pool, *, attempt, seed, event_loop):
    """Pump SDK generation, then execute reset/render/physics outside asyncio.

    This follows Isaac Lab Mimic's synchronous ``generation.env_loop``. Kit
    ticks its own async engine during rendering, so environment operations
    cannot run inside a different loop's coroutine.
    """
    import contextlib
    import torch
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from sim.datagen.scripted import _matrix_pose

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError('Mimic physics must run synchronously outside an asyncio coroutine')

    diagnostic = (_DiagnosticTrace(context, attempt, seed)
                  if getattr(context.args, 'diagnostic_trace', False) else None)
    env = context.raw
    reset_queue, action_queue = asyncio.Queue(), asyncio.Queue()
    generator = DataGenerator(env, src_demo_datagen_info_pool=pool)
    future = event_loop.create_task(generator.generate(0, TerminationTermCfg(func=_success),
        env_reset_queue=reset_queue, env_action_queue=action_queue, export_demo=False))
    rollout, reason, result, error = None, 'mimic_trajectory_finished', None, None
    try:
        with torch.inference_mode():
            while not future.done():
                # Only SDK selection/retargeting and queue waits run here.
                # run_until_complete returns before any operation that can
                # update Kit (including reset warmup and recorder teardown).
                event_loop.run_until_complete(asyncio.sleep(0))
                while not reset_queue.empty():
                    env_id = reset_queue.get_nowait()
                    if env_id != 0:
                        raise ValueError('Mimic shared recorder supports one environment')
                    rollout = context.begin_attempt(seed)
                    env.s2r_rollout = rollout
                    if diagnostic is not None:
                        diagnostic.begin(rollout)
                    reset_queue.task_done()
                while not action_queue.empty():
                    env_id, target = action_queue.get_nowait()
                    if env_id != 0:
                        raise ValueError('Mimic shared recorder supports one environment')
                    if rollout is None:
                        raise RuntimeError('Mimic requested an action before reset')
                    if rollout.recorder.steps >= context.args.max_steps:
                        action_queue.task_done()
                        reason = 'max_steps'
                        future.cancel()
                        break
                    # The shared recorder replaces this request with FK of the
                    # actual dispatched joints for target_eef_pose annotation.
                    base = env.s2r_last_target.detach().cpu().numpy()
                    base_pose = _matrix_pose(base)
                    position, orientation = env._base()
                    from core.scenes import quaternion_multiply
                    from core.benchmark.geometry import rotation_matrix
                    from core.tasks.base import Pose
                    p = position[0].detach().cpu().numpy()
                    q = orientation[0].detach().cpu().numpy()
                    world = p + rotation_matrix((q[3], q[0], q[1], q[2])) @ base_pose.position
                    requested = Pose.from_lists(world, quaternion_multiply(q, base_pose.quat_xyzw))
                    rollout.step(target, target_eef_pose=requested, gripper_action=env.s2r_last_gripper,
                                 phase='isaaclab_mimic')
                    if diagnostic is not None:
                        diagnostic.append(rollout, target)
                    action_queue.task_done()
            if future.cancelled():
                if reason != 'max_steps':
                    raise asyncio.CancelledError()
            else:
                sdk_result = future.result()
                if sdk_result['success'] != context.task.is_success(rollout.recorder.trace.stages):
                    raise RuntimeError('SDK Mimic success disagrees with the shared task trace')
        if rollout is None:
            raise RuntimeError('Mimic did not begin a rollout')
        result = rollout.recorder.finish(attempt=attempt, seed=seed, reason=reason)
        if diagnostic is not None:
            result['diagnostic_trace'] = str(diagnostic.path)
        return result
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        if rollout is not None and not rollout.recorder.closed:
            rollout.recorder.abort()
        raise
    finally:
        if not future.done():
            future.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                event_loop.run_until_complete(future)
        finally:
            if diagnostic is not None:
                diagnostic.write(rollout, result, error)


def main(argv=None):
    from sim.app import apply_selection, close_app, launch
    from sim.datagen.scripted import make_report, publish_report, setup_context
    from core.tasks.loader import load_task

    args = parse_args(argv)
    apply_selection(args)
    task = load_task(args.task)
    spec = MimicSpec.from_task(task, args.mimic_options)
    app = launch(args, enable_cameras=True)
    context = report = None
    succeeded = False
    try:
        import importlib.metadata
        import sys
        import torch

        def env_factory(cfg):
            config = make_mimic_cfg(cfg, task, cfg.arm_spec, spec, args.seed)
            return getattr(sys.modules[__name__], 'S2RMimicEnv')(cfg=config)

        context = setup_context(args, env_factory=env_factory, source='mimic')
        env = context.raw
        env.s2r_arm, env.s2r_task = context.arm, context.task
        env.s2r_ik, env.s2r_fk = context.ik, context.kinematics
        loop = asyncio.new_event_loop()
        try:
            pool, provenance = loop.run_until_complete(load_source_pool(context, spec))
            provenance.update(engine='isaaclab_mimic.datagen.DataGenerator',
                              sdk_version=importlib.metadata.version('isaaclab_mimic'),
                              subtasks=list(spec.subtasks))
            report = make_report(context, source_provenance=provenance)
            report['ik_controller']['command_inverse'] = dict(COMMAND_INVERSE_OPTIONS)
            publish_report(context, report)
            with torch.inference_mode():
                for attempt in range(args.attempts):
                    row = generate_attempt(context, pool, attempt=attempt, seed=args.seed + attempt,
                                           event_loop=loop)
                    report['episodes'].append(row)
                    publish_report(context, report)
                    print('[MIMIC_ATTEMPT] ' + json.dumps(row), flush=True)
                    if args.episodes and report['accepted_episodes'] >= args.episodes:
                        break
            report['attempts_completed'] = len(report['episodes'])
            report['request_fulfilled'] = (report['accepted_episodes'] >= args.episodes if args.episodes
                                          else report['attempts_completed'] == args.attempts)
            report['complete'] = report['request_fulfilled']
            publish_report(context, report)
            succeeded = report['accepted_episodes'] > 0 and (not args.episodes or report['accepted_episodes'] >= args.episodes)
            return 0 if succeeded else 1
        finally:
            loop.close()
    finally:
        try:
            if context is not None:
                if report is not None:
                    publish_report(context, report)
                context.close()
        finally:
            close_app(app, success=succeeded)


if __name__ == '__main__':
    raise SystemExit(main())
