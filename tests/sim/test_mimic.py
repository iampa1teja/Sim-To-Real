"""Real Isaac Lab Mimic segments and retargets S2R annotations on CPU."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.tasks.loader import load_task
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


@pytest.mark.parametrize('max_steps,accepted,reason', [(100, True, 'mimic_trajectory_finished'), (2, False, 'max_steps')])
def test_actual_sdk_async_queue_execution_and_bounded_cancellation(max_steps, accepted, reason):
    from isaaclab.envs import MimicEnvCfg
    from isaaclab.managers import RecorderManager
    from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
    from sim.datagen.mimic import generate_attempt

    task = load_task('pick_place_cube')
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

    def target_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        env.s2r_last_target = target_eef_pose_dict['tool']
        env.s2r_last_gripper = gripper_action_dict['tool']
        return torch.cat((env.s2r_last_target[:3, 3], env.s2r_last_gripper.flatten()))[None]

    env.target_eef_pose_to_action = target_to_action

    class Buffer:
        def __init__(self):
            self.steps, self.closed = 0, False
            self.trace = SimpleNamespace(stages={'success': False})

        def finish(self, **kwargs):
            self.closed = True
            return {'steps': self.steps, 'accepted': self.trace.stages['success'], **kwargs}

        def abort(self):
            self.closed = True

    class Rollout:
        def __init__(self):
            self.recorder = Buffer()

        def step(self, target, **kwargs):
            assert target.shape == (4,) and torch.isfinite(target).all()
            self.recorder.steps += 1
            self.recorder.trace.stages['success'] = self.recorder.steps >= 7

    context = SimpleNamespace(raw=env, task=task, args=SimpleNamespace(max_steps=max_steps),
                              begin_attempt=lambda seed: Rollout())
    pool = DataGenInfoPool(env, cfg, 'cpu')
    asyncio.run(pool.add_episode(episode_from_annotations(np.zeros((12, 4)), source_frames(), eef_name='tool')))

    async def run():
        return await asyncio.wait_for(generate_attempt(context, pool, attempt=0, seed=1984), timeout=5)

    result = asyncio.run(run())
    assert result['accepted'] is accepted and result['reason'] == reason
    assert result['steps'] <= max_steps
    assert env.s2r_rollout.recorder.closed
    assert not torch.cuda._initialized


def test_source_raw_units_are_decoded_before_sdk_gripper_extraction():
    from core.arms.spec import load_arm_spec
    from core.arms.mapping import JointMapping

    arm = load_arm_spec('so101')[0]
    mapping = JointMapping.from_arm(arm, arm.runtime_joint_limits, device='cpu')
    raw = np.zeros((12, len(arm.joints)), dtype=np.float32)
    raw[:, -1] = 100
    episode = episode_from_annotations(raw, source_frames(), eef_name=arm.links.ee, mapping=mapping)
    torch.testing.assert_close(episode.data['actions'], mapping.to_sim(raw))
    assert episode.data['actions'][0, -1] < 2  # Sim gripper radians, not raw normalized 100.


def test_source_reflections_are_rejected():
    frames = source_frames()
    frames[0]['eef_pose'][0][0] = -1
    with pytest.raises(ValueError, match='right-handed'):
        episode_from_annotations(np.zeros((12, 4)), frames, eef_name='tool')


def test_mimic_entrypoint_uses_shared_selectors_and_source_validation(tmp_path):
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
