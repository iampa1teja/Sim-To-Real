"""Schema migration keeps frozen trial identity, poses, codecs and pairing intact."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.arms.spec import load_arm_spec
from core.benchmark.compare import compare
from core.benchmark.eval_set import (EvalSetStarts, convert_eval_set, eval_set_digest,
                                     generate_eval_set, load_eval_set, main, validate_eval_set)
from core.benchmark.starts import RecordedStarts
from core.benchmark.smoke import MilestoneDeadline
from core.datasets.task_meta import save_task_trajectory
from core.tasks.loader import load_task

SETS = Path(__file__).resolve().parents[2] / 'workspace/tasks/pick_place_cube/eval_sets'
LEGACY = Path(__file__).resolve().parents[1] / 'fixtures/legacy_eval_set_v1.json'


def test_conversion_preserves_trial_pose_raw_state_and_legacy_identity(tmp_path):
    path = LEGACY
    original = load_eval_set(path)
    unchanged = deepcopy(original)
    converted = convert_eval_set(original, task='pick_place_cube')
    assert original == unchanged
    assert converted['legacy_eval_set_id'] == original['eval_set_id']
    assert converted['eval_set_id'] != original['eval_set_id']
    target = tmp_path / 'schema2.json'
    target.write_text(json.dumps(converted))
    old = EvalSetStarts(path, repeats=2, seed=777)
    new = EvalSetStarts(target, repeats=2, seed=777)
    assert len(old.schedule) == len(new.schedule)
    for index in range(len(old.schedule)):
        before_index, before, before_zone = old.sample()
        after_index, after, after_zone = new.sample()
        assert (before_index, before_zone) == (after_index, after_zone)
        assert before.keys() == after.keys()
        for name in before:
            for key in before[name]:
                np.testing.assert_array_equal(before[name][key], after[name][key])
        np.testing.assert_array_equal(old.robot_state(index), new.robot_state(index))
        for key in ('id', 'split', 'repeat', 'policy_seed', 'zone'):
            assert old.current[key] == new.current[key]
    arm = load_arm_spec()[0]
    for name, runtime_span in arm.runtime_joint_limits.items():
        assert converted['joint_mapping'][name] == original['scene']['joint_mapping'].get(name, runtime_span)
    assert set(converted['joint_mapping']) == set(converted['joint_names'])
    assert convert_eval_set(converted) == converted


def test_migration_verifies_legacy_hash_before_trusting_identity():
    original = load_eval_set(LEGACY)
    original['starts'][0]['x'] += .001
    with pytest.raises(ValueError, match='hash mismatch'):
        convert_eval_set(original)


@pytest.mark.parametrize('change,message', [
    (lambda data: data['starts'][0]['objects']['blue_cube'].update(position=[0, 1]), 'three finite'),
    (lambda data: data['starts'][0]['objects']['blue_cube'].update(orientation_wxyz=[0, 0, 0, 0]), 'nonzero'),
    (lambda data: data.update(joint_names=list(reversed(data['joint_names']))), 'joint order'),
    (lambda data: data['joint_mapping'].pop('gripper'), 'complete saved'),
])
def test_schema2_validates_pose_order_and_complete_device_codec(change, message):
    data = load_eval_set(SETS / 'pick_place_v1_seed1984.json')
    change(data)
    data['eval_set_id'] = eval_set_digest(data)
    with pytest.raises(ValueError, match=message):
        validate_eval_set(data)


def _report(data):
    return dict(schema_version=2, complete=True, eval_set=data, eval_set_id=data['eval_set_id'],
                task='S2R-pick_place_cube-eval', checkpoint='fixture', episode_length_s=15,
                robot_start='recorded', seed=1984, action_horizon=16,
                episodes=[dict(start_id='id-000', split='id', repeat=0, policy_seed=7, success=True)])


def test_old_and_converted_results_pair_through_preserved_identity():
    old = load_eval_set(LEGACY)
    new = convert_eval_set(old)
    pairs = compare([_report(old), _report(new)])
    assert pairs[0]['shared_starts'] == 1
    unrelated = deepcopy(new)
    unrelated.pop('legacy_eval_set_id')
    unrelated['parameters']['seed'] += 1
    unrelated['eval_set_id'] = eval_set_digest(unrelated)
    with pytest.raises(ValueError, match='same eval set'):
        compare([_report(old), _report(unrelated)])


def _dataset(tmp_path):
    arm = load_arm_spec()[0]
    task = load_task('stack_two_cubes')
    root = tmp_path / 'dataset'
    for index, offset in enumerate((0., .01, -.01)):
        poses = {'cube_a': np.array([[.11 + offset, -.12, .01, 1, 0, 0, 0]]),
                 'cube_b': np.array([[.22, .04 + offset, .01, 1, 0, 0, 0]])}
        save_task_trajectory(root, index, poses, {name: (.02, .02, .02) for name in poses}, 30,
                             task=task.name, arm=arm.name)
    chunk = root / 'data/chunk-000'
    chunk.mkdir(parents=True)
    pq.write_table(pa.table({'episode_index': [0, 1, 2], 'frame_index': [0, 0, 0],
                            'observation.state': [[0.] * len(arm.joints)] * 3}), chunk / 'part.parquet')
    scene = dict(base_position_world=[0., 0., 0.], base_quaternion_wxyz=[1., 0., 0., 0.],
                 table_plane_world=[0., 0., 0.], table_outline_base_xy=[[-1, -1], [1, -1], [1, 1], [-1, 1]],
                 objects={name: dict(size=[.02, .02, .02]) for name in poses}, joint_mapping=arm.runtime_joint_limits)
    return root, task, arm, scene


def test_two_object_recorded_starts_and_task_distribution_are_generic(tmp_path):
    root, task, arm, scene = _dataset(tmp_path)
    starts = RecordedStarts(root, seed=5, task=task, arm=arm, robot_states_root=root)
    index, objects, zone = starts.sample()
    assert objects.keys() == {'cube_a', 'cube_b'}
    assert len(zone) == 2
    original = starts.starts[index]['cube_a']['position'].copy()
    objects['cube_a']['position'][:] = 999
    np.testing.assert_array_equal(starts.starts[index]['cube_a']['position'], original)
    first = generate_eval_set(root, scene, task=task, arm=arm, seed=31, n_id=3, n_ood=0, n_yaw=1,
                              git_commit='fixture-revision')
    second = generate_eval_set(root, scene, task=task, arm=arm, seed=31, n_id=3, n_ood=0, n_yaw=1,
                               git_commit='fixture-revision')
    assert first['eval_set_id'] == second['eval_set_id']
    assert first['task'] == task.name
    assert all(set(start['objects']) == {'cube_a', 'cube_b'} for start in first['starts'])
    assert len(first['starts']) == 7


def test_eval_set_cli_accepts_explicit_provenance_without_git(tmp_path, monkeypatch):
    root, task, arm, scene = _dataset(tmp_path)
    scene_path = tmp_path / 'scene.json'
    scene_path.write_text(json.dumps(scene))
    monkeypatch.setattr('core.utils.provenance.git_commit', lambda *args, **kwargs: pytest.fail('git consulted'))
    output = tmp_path / 'eval.json'
    main(['--dataset', str(root), '--out', str(output), '--task', task.name, '--arm', arm.name,
          '--scene_json', str(scene_path), '--n_id', '2', '--n_ood', '0', '--n_yaw', '0',
          '--git-commit', 'fixture-revision'])
    assert load_eval_set(output, task=task)['git_commit'] == 'fixture-revision'


def test_smoke_progress_uses_declared_task_milestones():
    deadline = MilestoneDeadline(5, 3, milestones=('aligned', 'lowered'))
    assert deadline.maximum_duration_s == 11
    assert not deadline.observe(5, {'aligned': True})
    assert not deadline.observe(8, {'aligned': True, 'lowered': True})
    assert deadline.observe(11, {'aligned': True, 'lowered': True})
    assert deadline.summary()['milestone_times_s'] == {'aligned': 5., 'lowered': 8.}
