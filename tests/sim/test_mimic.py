"""Real Isaac Lab Mimic segments and retargets S2R annotations on CPU."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.tasks.loader import load_task
from core.tasks.base import Pose, StepState
from sim.datagen.mimic import MimicSpec, episode_from_annotations


def source_frames():
    frames = []
    for index in range(12):
        actual = np.eye(4)
        actual[0, 3] = .1 + .001 * index
        target = actual.copy()
        target[0, 3] += .001
        cube, box = np.eye(4), np.eye(4)
        cube[0, 3], box[1, 3] = .1, .2
        frames.append({'eef_pose': actual.tolist(), 'target_eef_pose': target.tolist(),
                       'object_pose': {'blue_cube': cube.tolist(), 'white_box': box.tolist()},
                       'subtask_term_signals': {'grasped': index >= 3, 'placed': index >= 8}})
    return frames


def test_actual_mimic_pool_segments_then_retargets_reference_object():
    pytest.importorskip("isaaclab")
    from isaaclab.envs import MimicEnvCfg
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

    task = load_task('pick_place_cube')
    cfg = MimicEnvCfg()
    spec = MimicSpec.from_task(task)
    spec.configure(cfg, 'tool', 1984)
    current_object = torch.eye(4)[None]
    current_object[0, 0, 3] = .3
    env = SimpleNamespace(cfg=cfg, device='cpu',
        get_object_poses=lambda env_ids: {'blue_cube': current_object},
        get_robot_eef_pose=lambda **kwargs: torch.eye(4)[None],
        actions_to_gripper_actions=lambda actions: {'tool': actions[..., -1:]})
    actions = np.zeros((12, 4), dtype=np.float32)
    actions[:, -1] = np.arange(12) / 12
    episode = episode_from_annotations(actions, source_frames(), eef_name='tool')
    pool = DataGenInfoPool(env, cfg, 'cpu')
    asyncio.run(pool.add_episode(episode))
    assert pool.subtask_boundaries['tool'] == [[(0, 4), (4, 9), (9, 12)]]
    generator = DataGenerator(env, src_demo_datagen_info_pool=pool)
    trajectory = generator.generate_eef_subtask_trajectory(0, 'tool', 0,
        generator.randomize_subtask_boundaries(), {}, {'tool': None})
    waypoints = trajectory.get_full_sequence()
    assert len(waypoints) == 5
    # Actual SDK retargeting moves every source control target by +0.2m with its object.
    np.testing.assert_allclose([waypoint.pose[0, 3] for waypoint in waypoints],
                               [.3, .301, .302, .303, .304], atol=1e-7, rtol=0)
    np.testing.assert_allclose([waypoint.gripper_action[0] for waypoint in waypoints],
                               [0, 0, 1 / 12, 2 / 12, 3 / 12], atol=1e-7, rtol=0)
    assert generator.__class__.__module__ == 'isaaclab_mimic.datagen.data_generator'
    assert not torch.cuda._initialized


def test_source_annotation_requires_latched_aligned_signals():
    pytest.importorskip("isaaclab")
    frames = source_frames()
    frames[5]['subtask_term_signals']['grasped'] = False
    with pytest.raises(ValueError, match='must be latched'):
        episode_from_annotations(np.zeros((12, 4)), frames, eef_name='tool')
    with pytest.raises(ValueError, match='matching finite'):
        episode_from_annotations(np.zeros((11, 4)), source_frames(), eef_name='tool')


def test_subtasks_declare_objects_and_nonfinal_stage_boundaries():
    task = load_task('pick_place_cube')
    rows = deepcopy(task.spec.raw['mimic']['subtasks'])
    rows[0]['object_ref'] = 'undeclared'
    with pytest.raises(ValueError, match='unknown task object'):
        MimicSpec.from_task(task, {'subtasks': rows})
    rows = deepcopy(task.spec.raw['mimic']['subtasks'])
    rows[-1]['subtask_term_signal'] = 'success'
    with pytest.raises(ValueError, match='final Mimic subtask'):
        MimicSpec.from_task(task, {'subtasks': rows})


def test_mimic_recovers_command_pose_without_reapplying_feedback_bias():
    pytest.importorskip("isaaclab")
    from sim.datagen import mimic
    from sim.actions.ik import IKOptions

    target = torch.eye(4)
    target[:3, 3] = torch.tensor([.1, -.2, .3])
    options = IKOptions()
    calls = []

    def inverse(pose, measured, *, options, max_iterations, position_tolerance_m, orientation_tolerance_rad):
        calls.append(options)
        assert max_iterations == 32
        assert position_tolerance_m == 1e-8
        assert orientation_tolerance_rad == 1e-7
        np.testing.assert_array_equal(pose, target.numpy())
        np.testing.assert_array_equal(measured, [3., 1., 4., 2.])
        return SimpleNamespace(joint_positions=torch.tensor([[3.01, 1., 4.02, 2.003]], dtype=torch.float64))

    arm = SimpleNamespace(links=SimpleNamespace(ee='tool'),
                          action_joints=[SimpleNamespace(role=role) for role in ('arm', 'gripper', 'arm', 'arm')])
    ik = SimpleNamespace(options=options, joint_ids=[2, 0, 3, 1],
                         robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[1., 2., 3., 4.]]))),
                         solve=lambda *a, **k: pytest.fail('Mimic must not reapply the scripted feedback controller'))
    env = SimpleNamespace(num_envs=1, device='cpu', s2r_arm=arm, s2r_ik=ik,
                          s2r_fk=SimpleNamespace(inverse=inverse))
    action = mimic.S2RMimicEnv.target_eef_pose_to_action(env, {'tool': target}, {'tool': torch.tensor([.25])})
    torch.testing.assert_close(action, torch.tensor([[3.01, .25, 4.02, 2.003]]))
    assert calls == [options] and action.dtype == torch.float32
    torch.testing.assert_close(env.s2r_last_target, target)
    torch.testing.assert_close(env.s2r_last_gripper, torch.tensor([.25]))


@pytest.mark.parametrize('max_steps,accepted,reason,step_error', [
    (100, True, 'mimic_trajectory_finished', False),
    (2, False, 'max_steps', False),
    (100, False, 'physical_step_error', True),
])
@pytest.mark.parametrize('diagnostic_trace', [False, True])
def test_actual_sdk_queue_execution_keeps_physics_outside_asyncio(
        max_steps, accepted, reason, step_error, diagnostic_trace, tmp_path, monkeypatch):
    pytest.importorskip("isaaclab")
    from isaaclab.envs import MimicEnvCfg
    from isaaclab.managers import RecorderManager
    from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
    from sim.datagen.mimic import generate_attempt

    task = load_task('pick_place_cube')
    monkeypatch.setattr(task, 'signals', lambda env: pytest.fail('diagnostics must reuse existing task signals'))
    cfg = MimicEnvCfg()
    MimicSpec.from_task(task).configure(cfg, 'tool', 1984)
    cube, box = torch.eye(4)[None], torch.eye(4)[None]
    cube[0, 0, 3], box[0, 1, 3] = .3, .2
    env = SimpleNamespace(cfg=cfg, device='cpu',
        scene=SimpleNamespace(get_state=lambda **kwargs: {}),
        s2r_arm=SimpleNamespace(links=SimpleNamespace(ee='tool')), s2r_task=task,
        get_object_poses=lambda env_ids: {'blue_cube': cube, 'white_box': box},
        get_robot_eef_pose=lambda **kwargs: torch.eye(4)[None],
        actions_to_gripper_actions=lambda actions: {'tool': actions[..., -1:]},
        _base=lambda: (torch.zeros((1, 3)), torch.tensor([[0., 0., 0., 1.]])))
    env.recorder_manager = RecorderManager(None, env)
    lifecycle = []

    def assert_synchronous(operation):
        # Kit updates its own asyncio engine while rendering. A physical call
        # under the generator's running loop caused its nested-loop assertion.
        with pytest.raises(RuntimeError, match='no running event loop'):
            asyncio.get_running_loop()
        lifecycle.append(operation)

    def target_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        env.s2r_last_target = target_eef_pose_dict['tool']
        env.s2r_last_gripper = gripper_action_dict['tool']
        return torch.cat((env.s2r_last_target[:3, 3], env.s2r_last_gripper.flatten()))[None]

    env.target_eef_pose_to_action = target_to_action

    class Buffer:
        def __init__(self):
            self.steps, self.closed = 0, False
            self.trace = SimpleNamespace(stages={'success': False})
            self.annotations = []

        def finish(self, **kwargs):
            assert_synchronous('finish')
            self.closed = True
            return {'steps': self.steps, 'accepted': self.trace.stages['success'], **kwargs}

        def abort(self):
            assert_synchronous('abort')
            self.closed = True

    class Rollout:
        def __init__(self):
            self.recorder = Buffer()
            self.joint_ids = [1, 3, 0, 2]
            self.robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.tensor([[30., 10., 40., 20.]])))
            self.last_state = None

        def snapshot(self):
            assert_synchronous('initial_snapshot')
            return StepState({'object': Pose((1, 2, 3))}, Pose((0, 0, 1)), 0., {})

        def step(self, target, **kwargs):
            assert_synchronous('step')
            assert target.shape == (4,) and torch.isfinite(target).all()
            if step_error and self.recorder.steps == 1:
                raise RuntimeError('physical_step_error')
            self.recorder.steps += 1
            self.recorder.trace.stages['success'] = self.recorder.steps >= 7
            self.robot.data.joint_pos[0, self.joint_ids] = target - .05
            self.last_state = StepState({'object': Pose((1, 2, self.recorder.steps))}, Pose((0, 0, 1)), 0.,
                                        {'holding': False, 'success': self.recorder.steps >= 7,
                                         'physical_steps': self.recorder.steps})
            self.recorder.annotations.append({'eef_pose': torch.eye(4).tolist(),
                'target_eef_pose': env.s2r_last_target.tolist(),
                'subtask_term_signals': dict(self.recorder.trace.stages)})

    def begin_attempt(seed):
        assert_synchronous('reset')
        return Rollout()

    arm = SimpleNamespace(joint_names=('first', 'slide', 'third', 'grip'),
        action_joints=[SimpleNamespace(type=kind) for kind in ('revolute', 'prismatic', 'revolute', 'revolute')])
    report_path = tmp_path / 'mimic.json'
    diagnostic_path = tmp_path / 'mimic-diagnostics/attempt_0000.json'
    context = SimpleNamespace(raw=env, task=task, arm=arm,
                              args=SimpleNamespace(max_steps=max_steps, diagnostic_trace=diagnostic_trace,
                                  report=report_path, destination=tmp_path / 'dataset'),
                              begin_attempt=begin_attempt)
    pool = DataGenInfoPool(env, cfg, 'cpu')
    asyncio.run(pool.add_episode(episode_from_annotations(np.zeros((12, 4)), source_frames(), eef_name='tool')))

    loop = asyncio.new_event_loop()
    try:
        if step_error:
            with pytest.raises(RuntimeError, match='physical_step_error'):
                generate_attempt(context, pool, attempt=0, seed=1984, event_loop=loop)
            assert lifecycle[-1] == 'abort'
        else:
            result = generate_attempt(context, pool, attempt=0, seed=1984, event_loop=loop)
            assert result['accepted'] is accepted and result['reason'] == reason
            assert result['steps'] <= max_steps
            assert lifecycle[-1] == 'finish'
        assert lifecycle.count('reset') == 1
        assert not asyncio.all_tasks(loop)
    finally:
        loop.close()
    assert env.s2r_rollout.recorder.closed
    if diagnostic_trace:
        import json
        data = json.loads(diagnostic_path.read_text())
        assert data['initial_state']['objects']['object']['position'] == [1., 2., 3.]
        assert data['initial_joint_positions_sim'] == [10., 20., 30., 40.]
        assert data['sim_joint_units'] == ['rad', 'm', 'rad', 'rad']
        assert len(data['steps']) == len(data['annotations']) == env.s2r_rollout.recorder.steps
        assert lifecycle.count('initial_snapshot') == 1
        for index, row in enumerate(data['steps']):
            assert row['frame_index'] == index and row['task_signals']['physical_steps'] == index + 1
            np.testing.assert_allclose(row['measured_joint_positions_sim'],
                                       np.asarray(row['commanded_joint_positions_sim']) - .05, atol=1e-7)
        assert data['annotations'] == env.s2r_rollout.recorder.annotations
        if step_error:
            assert data['error'] == {'type': 'RuntimeError', 'message': 'physical_step_error'}
        else:
            assert data['result']['diagnostic_trace'] == str(diagnostic_path)
    else:
        assert not diagnostic_path.exists() and 'initial_snapshot' not in lifecycle
    assert not (tmp_path / 'dataset').exists()
    assert not torch.cuda._initialized


def test_mimic_attempt_refuses_physics_from_a_running_coroutine():
    pytest.importorskip("isaaclab")
    from sim.datagen.mimic import generate_attempt

    async def nested():
        with pytest.raises(RuntimeError, match='synchronously outside an asyncio coroutine'):
            generate_attempt(None, None, attempt=0, seed=1984, event_loop=asyncio.get_running_loop())

    asyncio.run(nested())


def test_source_raw_units_are_decoded_before_sdk_gripper_extraction():
    pytest.importorskip("isaaclab")
    from utils.arms.spec import load_arm_spec
    from utils.arms.mapping import JointMapping

    arm = load_arm_spec('so101')[0]
    mapping = JointMapping.from_arm(arm, arm.runtime_joint_limits, device='cpu')
    raw = np.zeros((12, len(arm.action_joints)), dtype=np.float32)
    raw[:, -1] = 100
    episode = episode_from_annotations(raw, source_frames(), eef_name=arm.links.ee, mapping=mapping)
    torch.testing.assert_close(episode.data['actions'], mapping.to_sim(raw))
    assert episode.data['actions'][0, -1] < 2  # Sim gripper radians, not raw normalized 100.


def test_source_reflections_are_rejected():
    pytest.importorskip("isaaclab")
    frames = source_frames()
    frames[0]['eef_pose'][0][0] = -1
    with pytest.raises(ValueError, match='right-handed'):
        episode_from_annotations(np.zeros((12, 4)), frames, eef_name='tool')


def test_mimic_entrypoint_uses_shared_selectors_and_source_validation(tmp_path):
    pytest.importorskip("isaaclab")
    from sim.datagen.mimic import parse_args

    source = tmp_path / 'source'
    (source / 'datagen_meta').mkdir(parents=True)
    config = tmp_path / 'mimic.yaml'
    config.write_text(f'''source: mimic
arm: so101
scene: rpd_meeting_room
task: pick_place_cube
unit: rpd_robot
source_dataset: {source}
destination: {tmp_path / 'generated'}
headless: true
mimic_options:
  select_source_per_subtask: false
''')
    args = parse_args(['--config_path', str(config)])
    assert args.source == 'mimic' and args.source_dataset == source
    assert args.mimic_options == {'select_source_per_subtask': False}
    assert args.unit == 'rpd_robot' and args.headless
    assert not args.diagnostic_trace
    traced = parse_args(['--config_path', str(config), '--diagnostic_trace', '--report', str(tmp_path / 'run.json')])
    assert traced.diagnostic_trace
    with pytest.raises(SystemExit):
        parse_args(['--config_path', str(config), '--diagnostic_trace', '--report',
                    str(tmp_path / 'generated/run.json')])
