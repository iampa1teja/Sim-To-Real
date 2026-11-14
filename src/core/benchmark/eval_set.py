"""Versioned fixed-start benchmarks with deterministic schedules and explicit migration."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import numpy as np

from core.arms.spec import load_arm_spec
from core.arms.mapping import mapping_for_arm
from core.arms.calibration import validate_mapping
from core.utils.hashing import json_sha256
from .geometry import (circular_mean, face_yaw, hull, hull_distance, inside_polygon,
                       rotation_matrix, sample_yaw, supported_on_table, wrap_yaw)
from .starts import (RecordedStarts, copy_objects, objects_to_poses, poses_to_objects, primary_object, task_instance)

SPLITS = ('id', 'ood', 'yaw', 'train')


def normalize_scene(scene, task=None):
    """Adapt schema-1 geometry into the named-object scene contract at its read boundary."""
    task = task_instance(task)
    result = deepcopy(scene)
    primary = primary_object(task)
    result.setdefault('start_object', primary)
    objects = result.setdefault('objects', {})
    if 'cube_size_m' in scene:
        objects.setdefault(primary, {}).setdefault('size', list(scene['cube_size_m']))
    if 'box_position_world' in scene:
        candidates = [name for name in task.spec.objects if name != primary]
        if len(candidates) != 1:
            raise ValueError('Legacy scene geometry requires one unambiguous fixed object')
        name = candidates[0]
        rotation = rotation_matrix(scene['base_quaternion_wxyz'])
        position = rotation.T @ (np.asarray(scene['box_position_world']) - np.asarray(scene['base_position_world']))
        from core.scenes import quaternion_multiply

        base = scene['base_quaternion_wxyz']
        w, x, y, z = scene['box_quaternion_wxyz']
        relative = quaternion_multiply((-base[1], -base[2], -base[3], base[0]), (x, y, z, w))
        orientation = [relative[3], *relative[:3]]
        objects.setdefault(name, {}).update(position_world=scene['box_position_world'],
            orientation_world_wxyz=scene['box_quaternion_wxyz'], position=position.tolist(),
            orientation_wxyz=orientation, size=scene['box_size_m'], bottom_origin=True)
        if 'box_footprint_base_xy' in scene:
            objects[name]['footprint_base_xy'] = scene['box_footprint_base_xy']
    for name, values in objects.items():
        spec = task.spec.objects.get(name)
        if spec is not None and 'size' not in values:
            size = spec.size or spec.params.get('size_outer')
            if size is not None:
                values['size'] = list(size)
    return result


def eval_set_digest(data):
    keys = ('schema_version', 'parameters', 'scene', 'training_region', 'starts')
    if data.get('schema_version') == 2:
        keys += ('task', 'arm', 'joint_names', 'joint_mapping')
        if 'legacy_eval_set_id' in data:
            keys += ('legacy_eval_set_id',)
    return json_sha256({key: data[key] for key in keys})


def validate_eval_set(data, *, arm=None, task=None):
    version = data.get('schema_version')
    if (version not in (1, 2) or data.get('units') != 'm' or data.get('angle_units') != 'rad'):
        raise ValueError('Unsupported eval set schema or coordinate units')
    arm = arm or load_arm_spec(data.get('arm'))[0]
    expected_frame = 'robot base link (base)' if version == 1 else f'robot base link ({arm.links.base})'
    if data.get('reference_frame') != expected_frame:
        raise ValueError('Eval set reference frame differs from the selected arm base link')
    if data.get('eval_set_id') != eval_set_digest(data):
        raise ValueError('Eval set content hash mismatch')
    if version == 2:
        if data.get('arm') != arm.name or data.get('joint_names') != list(arm.joint_names):
            raise ValueError('Eval set arm or joint order differs from the selected arm')
        if not isinstance(data.get('task'), str) or not data['task']:
            raise ValueError('Eval set needs its task identity')
        if task is not None and data['task'] != task_instance(task).name:
            raise ValueError('Eval set task differs from the selected task')
        spans = data.get('joint_mapping')
        if not isinstance(spans, dict) or set(spans) != set(arm.joint_names):
            raise ValueError('Eval set needs a complete saved joint mapping')
        mapping_for_arm(arm, validate_mapping(arm, spans))
    ids = set()
    for start in data['starts']:
        if start['id'] in ids or start['split'] not in SPLITS:
            raise ValueError('Duplicate start ID or invalid split')
        ids.add(start['id'])
        if version == 1:
            if not np.isfinite([start['x'], start['y'], start['yaw']]).all():
                raise ValueError('Nonfinite start pose')
            if 'orientation_wxyz' in start:
                rotation_matrix(start['orientation_wxyz'])
        else:
            if not isinstance(start.get('objects'), dict) or not start['objects']:
                raise ValueError('Eval start needs named object poses')
            for name, pose in start['objects'].items():
                position = np.asarray(pose['position'], dtype=float)
                if not isinstance(name, str) or not name or position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError('Object positions must contain three finite metres')
                rotation_matrix(pose['orientation_wxyz'])
            if task is not None and set(start['objects']) - set(task_instance(task).spec.objects):
                raise ValueError('Eval start contains objects not declared by the task')
        state = np.asarray(start['arm_start_state'])
        if state.shape != (len(arm.joint_names),) or not np.isfinite(state).all():
            raise ValueError(f'Arm start state must contain {len(arm.joint_names)} finite values')
    if not ids:
        raise ValueError('Empty eval set')
    return data


def load_eval_set(path, *, arm=None, task=None):
    return validate_eval_set(json.loads(Path(path).read_text()), arm=arm, task=task)


def start_objects(start, *, task=None):
    if 'objects' in start:
        return copy_objects(start['objects'])
    name = primary_object(task)
    yaw = start['yaw']
    orientation = start.get('orientation_wxyz', [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    return {name: dict(position=np.array([start['x'], start['y'], 0.0]),
                       orientation_wxyz=np.asarray(orientation, dtype=float))}


def convert_eval_set(data, *, task=None, arm=None):
    """Return a schema-2 copy; never alter the source set or its saved legacy digest."""
    task = task_instance(task)
    arm = arm or load_arm_spec(data.get('arm'))[0]
    validate_eval_set(data, arm=arm, task=task)
    if data['schema_version'] == 2:
        return deepcopy(data)
    result = deepcopy(data)
    result.update(schema_version=2, task=task.name, arm=arm.name, joint_names=list(arm.joint_names),
                  reference_frame=f'robot base link ({arm.links.base})',
                  joint_mapping=mapping_for_arm(arm, data['scene'].get('joint_mapping', {})),
                  legacy_eval_set_id=data['eval_set_id'])
    for start in result['starts']:
        start['objects'] = _json_objects(start_objects(start, task=task))
        for key in ('x', 'y', 'yaw', 'orientation_wxyz'):
            start.pop(key, None)
    result['eval_set_id'] = eval_set_digest(result)
    return validate_eval_set(result, arm=arm, task=task)


def _json_objects(objects):
    return {name: {key: np.asarray(value).tolist() for key, value in pose.items()} for name, pose in objects.items()}


class EvalSetStarts:
    """Ordered trials with stable seeds, independent of the task and disk schema."""
    def __init__(self, path, splits=None, repeats=1, seed=1984, start_offset=0, selection='all', *, arm=None, task=None):
        self.task = task_instance(task)
        self.data = load_eval_set(path, arm=arm, task=self.task)
        if isinstance(splits, str):
            splits = splits.split(',')
        available = {start['split'] for start in self.data['starts']}
        if (type(repeats) is not int or repeats < 1 or
                (splits is not None and (len(splits) != len(set(splits)) or not set(splits) <= available))):
            raise ValueError('repeats must be positive; splits must be unique names present in eval set')
        indexed = [(index, start) for index, start in enumerate(self.data['starts'])
                   if splits is None or start['split'] in splits]
        self.starts = [start for _, start in indexed]
        if not self.starts:
            raise ValueError('No selected eval starts')
        self.schedule = [dict(start, repeat=repeat, policy_seed=int(
            np.random.SeedSequence([seed, repeat, index]).generate_state(1)[0]))
            for repeat in range(repeats) for index, start in indexed]
        if selection == 'one_per_zone':
            from .smoke import one_per_zone
            self.schedule = one_per_zone(self.schedule)
        elif selection != 'all':
            raise ValueError('Benchmark selection must be all or one_per_zone')
        if type(start_offset) is not int or not 0 <= start_offset <= len(self.schedule):
            raise ValueError('start_offset must be within the benchmark schedule')
        self.cursor, self.current = start_offset, None

    def sample(self, accepts_random=None):
        # Isaac performs one unused automatic reset on the final done step.
        index = min(self.cursor, len(self.schedule) - 1)
        self.current = self.schedule[index]
        self.cursor += 1
        return index, start_objects(self.current, task=self.task), self.current['zone']

    def robot_state(self, index):
        return np.asarray(self.schedule[index]['arm_start_state']).copy()


def _generate_recorded_region(dataset, scene, *, seed=1984, n_id=60, n_ood=20, n_yaw=20,
                      min_dist_m=.02, ood_margin_m=(.02, .05), yaw_mode='matched', reach_envelope=False,
                      git_commit, arm=None, task=None):
    arm = arm or load_arm_spec()[0]
    recorded = RecordedStarts(dataset, robot_states_root=dataset, arm=arm, task=task)
    primary = primary_object(task)
    poses = [(start[primary]['position'], start[primary]['orientation_wxyz']) for start in recorded.starts]
    xy = np.array([p[:2] for p, _ in poses])
    polygon = hull(xy)
    # Opt-in: keep id/ood starts inside the radial band of the recorded picks, so none is closer to the base or
    # farther out than an arm demonstrably reached (the original sampler only checks box, table and hull).
    reach = (float(np.linalg.norm(xy, axis=1).min()), float(np.linalg.norm(xy, axis=1).max())) if reach_envelope else None
    bearings = np.arctan2(xy[:, 1], xy[:, 0])
    # Pan coverage means the polar sweep of the recorded cube starts. Resting
    # first-frame arm states are used for resetting, not inverse kinematics.
    pan_offset = circular_mean(bearings, 2 * np.pi)
    pan = (bearings - pan_offset + np.pi) % (2 * np.pi) - np.pi
    pan_limits = [float(pan.min() - np.deg2rad(10)), float(pan.max() + np.deg2rad(10))]
    recorded_yaws = np.array([face_yaw(q) for _, q in poses])
    relative_yaws = wrap_yaw(recorded_yaws - bearings)
    mean_state = np.mean(recorded.robot_states, axis=0).tolist()
    rng = np.random.default_rng(seed)
    starts = []

    def candidate(p, yaw):
        q = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        return {primary: dict(position=np.r_[p, 0.0], orientation_wxyz=q)}

    def row(split, p, yaw, **extra):
        return dict(id=f'{split}-{sum(s["split"] == split for s in starts):03d}', split=split,
                    x=float(p[0]), y=float(p[1]), yaw=float(yaw), zone='OUT' if split == 'ood' else recorded.zone(p),
                    arm_start_state=mean_state.copy(),
                    rel_yaw=float(wrap_yaw(yaw - math.atan2(p[1], p[0]))), source_start_id=None, **extra)
    for split, count in (('id', n_id), ('ood', n_ood)):
        accepted = 0
        margin = ood_margin_m[1] if split == 'ood' else 0
        for _ in range(500000):
            if accepted == count:
                break
            p = rng.uniform(xy.min(0) - margin, xy.max(0) + margin)
            inside = inside_polygon(p, polygon)
            if split == 'id' and not inside:
                continue
            if split == 'ood':
                distance = hull_distance(p, polygon)
                estimated_pan = (np.arctan2(p[1], p[0]) - pan_offset + np.pi) % (2 * np.pi) - np.pi
                if inside or not ood_margin_m[0] <= distance <= ood_margin_m[1] or not pan_limits[0] <= estimated_pan <= pan_limits[1]:
                    continue
            if reach and not reach[0] <= np.linalg.norm(p) <= reach[1]:
                continue
            if np.linalg.norm(xy - p, axis=1).min() < min_dist_m:
                continue
            yaw, _, source = sample_yaw(p, xy, relative_yaws, yaw_mode, rng)
            if not task.start_is_valid(objects_to_poses(candidate(p, yaw)), scene) or not supported_on_table(p, yaw, scene):
                continue
            paired_yaw = rng.uniform(0, np.pi / 2) if split == 'id' and accepted < n_yaw else None
            if paired_yaw is not None and (not task.start_is_valid(objects_to_poses(candidate(p, paired_yaw)), scene) or not supported_on_table(p, paired_yaw, scene)):
                continue
            starts.append(row(split, p, yaw, **({'paired_yaw': float(paired_yaw)} if paired_yaw is not None else {})))
            starts[-1]['source_start_id'] = f'train-{source:03d}' if source is not None else None
            accepted += 1
        if accepted != count:
            raise ValueError(f'Cannot fit {count} {split} starts with these constraints (accepted {accepted})')
    for source in list(starts[:n_yaw]):
        starts.append(row('yaw', [source['x'], source['y']], source.pop('paired_yaw'), paired_id=source['id']))
    for index, (p, q) in enumerate(poses):
        starts.append(row('train', p[:2], face_yaw(q), orientation_wxyz=q.tolist(), source_episode=recorded.episode_ids[index]))
        starts[-1]['source_start_id'] = starts[-1]['id']
    data = dict(schema_version=1, reference_frame='robot base link (base)', units='m', angle_units='rad',
                dataset_path=str(Path(dataset).resolve()), seed=seed, git_commit=git_commit,
                created_at=datetime.now(timezone.utc).isoformat(), scene=scene,
                parameters=dict(seed=seed, n_id=n_id, n_ood=n_ood, n_yaw=n_yaw, min_dist_m=min_dist_m,
                                ood_margin_m=list(ood_margin_m), yaw_mode=yaw_mode, require_table_support=True,
                                **({'reach_radius_m': list(reach)} if reach else {})),
                training_region=dict(hull_xy=polygon.tolist(), low=recorded.low.tolist(), high=recorded.high.tolist(),
                                     pan_limits_rad=pan_limits, pan_centre_rad=pan_offset,
                                     pan_rule='cube-start azimuth relative to circular centre, expanded by 10 degrees',
                                     **({'reach_rule': 'id/ood starts lie between the closest and farthest recorded pick from the base'} if reach else {}),
                                     rel_yaw_distribution=dict(mean=float(relative_yaws.mean()), std=float(relative_yaws.std()),
                                                               min=float(relative_yaws.min()), max=float(relative_yaws.max())),
                                     yaw_rule='matched: bearing plus nearest recorded relative yaw; radial: bearing; random: uniform [0, pi/2)' ), starts=starts)
    return data


def _generate_task_distribution(dataset, scene, *, task, arm, seed, n_id, n_ood, n_yaw, git_commit, **options):
    from core.tasks.base import sample_start_in_base

    recorded = RecordedStarts(dataset, robot_states_root=dataset, arm=arm, task=task)
    primary = primary_object(task)
    xy = np.array([start[primary]['position'][:2] for start in recorded.starts])
    try:
        polygon = hull(xy)
    except ValueError:
        polygon = np.empty((0, 2))
    rng = np.random.default_rng(seed)
    mean_state = np.mean(recorded.robot_states, axis=0).tolist()
    starts = []

    def row(split, objects, state, **extra):
        p = objects[primary]['position']
        return dict(id=f'{split}-{sum(start["split"] == split for start in starts):03d}', split=split,
                    zone='OUT' if split == 'ood' else recorded.zone(p[:2]), objects=_json_objects(objects),
                    arm_start_state=state, **extra)

    for split, count in (('id', n_id), ('ood', n_ood)):
        accepted = 0
        for _ in range(500000):
            if accepted == count:
                break
            poses = sample_start_in_base(task, rng, scene)
            objects = poses_to_objects(poses)
            if primary not in objects:
                raise ValueError(f'Task sample_start omits primary object {primary!r}')
            if not task.start_is_valid(poses, scene):
                continue
            if len(polygon) and (inside_polygon(objects[primary]['position'][:2], polygon) != (split == 'id')):
                continue
            paired = None
            if split == 'id' and accepted < n_yaw:
                paired = copy_objects(objects)
                yaw = rng.uniform(0, np.pi / 2)
                paired[primary]['orientation_wxyz'] = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
                if not task.start_is_valid(objects_to_poses(paired), scene):
                    continue
            starts.append(row(split, objects, mean_state.copy()))
            if paired is not None:
                starts[-1]['_paired_objects'] = paired
            accepted += 1
        if accepted != count:
            raise ValueError(f'Cannot fit {count} {split} starts with the task distribution (accepted {accepted})')
    for source in list(starts[:n_yaw]):
        starts.append(row('yaw', source.pop('_paired_objects'), mean_state.copy(), paired_id=source['id']))
    for index, objects in enumerate(recorded.starts):
        starts.append(row('train', objects, recorded.robot_state(index).tolist(), source_episode=recorded.episode_ids[index]))
    return dict(schema_version=2, reference_frame=f'robot base link ({arm.links.base})', units='m', angle_units='rad',
                dataset_path=str(Path(dataset).resolve()), seed=seed, git_commit=git_commit,
                created_at=datetime.now(timezone.utc).isoformat(), scene=scene,
                parameters=dict(seed=seed, n_id=n_id, n_ood=n_ood, n_yaw=n_yaw, sampler='task_distribution'),
                training_region=dict(hull_xy=polygon.tolist(), low=recorded.low.tolist(), high=recorded.high.tolist()),
                starts=starts)


def generate_eval_set(dataset, scene, *, task=None, arm=None, seed=1984, n_id=60, n_ood=20, n_yaw=20,
                      min_dist_m=.02, ood_margin_m=(.02, .05), yaw_mode='matched', reach_envelope=False,
                      git_commit=None):
    """Sample and validate task starts; legacy datasets retain the original draw sequence."""
    from core.datasets.task_meta import SIDECAR_DIR

    task = task_instance(task)
    arm = arm or load_arm_spec()[0]
    if not isinstance(git_commit, str) or not git_commit.strip() or git_commit == 'unknown':
        raise ValueError('Eval set generation requires explicit source git provenance')
    if any(type(n) is not int or n < 0 for n in (n_id, n_ood, n_yaw)) or n_yaw > n_id:
        raise ValueError('Counts must be nonnegative integers, with n_yaw <= n_id')
    if (not math.isfinite(min_dist_m) or min_dist_m < 0 or len(ood_margin_m) != 2
            or not np.isfinite(ood_margin_m).all() or not 0 < ood_margin_m[0] < ood_margin_m[1]):
        raise ValueError('Require min_dist_m >= 0 and 0 < ood_margin_m[0] < ood_margin_m[1]')
    if yaw_mode not in ('matched', 'radial', 'random'):
        raise ValueError('yaw_mode must be matched, radial or random')
    scene = normalize_scene(scene, task)
    from core.datasets.sources import recorded_joint_mapping

    saved_mapping, _ = recorded_joint_mapping(dataset, arm)
    if saved_mapping is not None:
        selected_mapping = scene.get('joint_mapping')
        if selected_mapping is not None and mapping_for_arm(arm, selected_mapping) != mapping_for_arm(arm, saved_mapping):
            raise ValueError('Scene joint mapping differs from the recorded robot-state codec')
        scene['joint_mapping'] = saved_mapping
    options = dict(seed=seed, n_id=n_id, n_ood=n_ood, n_yaw=n_yaw, min_dist_m=min_dist_m,
                   ood_margin_m=ood_margin_m, yaw_mode=yaw_mode, reach_envelope=reach_envelope,
                   git_commit=git_commit, arm=arm, task=task)
    if (Path(dataset) / SIDECAR_DIR).is_dir():
        data = _generate_task_distribution(dataset, scene, **options)
    else:
        data = _generate_recorded_region(dataset, scene, **options)
        # This is freshly generated data, not a conversion of a frozen set.
        for start in data['starts']:
            start['objects'] = _json_objects(start_objects(start, task=task))
            for key in ('x', 'y', 'yaw', 'orientation_wxyz'):
                start.pop(key, None)
        data['schema_version'] = 2
    data.update(task=task.name, arm=arm.name, joint_names=list(arm.joint_names),
                reference_frame=f'robot base link ({arm.links.base})',
                joint_mapping=mapping_for_arm(arm, scene.get('joint_mapping', {})))
    data['eval_set_id'] = eval_set_digest(data)
    return validate_eval_set(data, arm=arm, task=task)


def main(argv=None):
    from core.paths import repo_root
    from core.utils.provenance import git_commit

    parser = argparse.ArgumentParser(description='Build a task benchmark from recorded first-frame starts')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--task', default=None)
    parser.add_argument('--arm', default=os.environ.get('S2R_ARM'))
    parser.add_argument('--seed', type=int, default=1984)
    for name, default in (('n_id', 60), ('n_ood', 20), ('n_yaw', 20)):
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--min_dist_m', type=float, default=.02)
    parser.add_argument('--ood_margin_m', type=float, nargs=2, default=[.02, .05])
    parser.add_argument('--yaw_mode', choices=['matched', 'radial', 'random'], default='matched')
    parser.add_argument('--reach_envelope', action='store_true')
    parser.add_argument('--scene_json', type=Path, required=True, help='Live named-object scene snapshot')
    parser.add_argument('--git-commit', default=os.environ.get('S2R_GIT_COMMIT'),
                        help='Explicit source revision when git is unavailable in the container')
    args = parser.parse_args(argv)
    arm, _ = load_arm_spec(args.arm)
    task = task_instance(args.task)
    commit = args.git_commit or git_commit(repo_root(), required=True)
    scene = json.loads(args.scene_json.read_text())
    data = generate_eval_set(args.dataset, scene, task=task, arm=arm, seed=args.seed,
        n_id=args.n_id, n_ood=args.n_ood, n_yaw=args.n_yaw, min_dist_m=args.min_dist_m,
        ood_margin_m=args.ood_margin_m, yaw_mode=args.yaw_mode, reach_envelope=args.reach_envelope,
        git_commit=commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(eval_set_id=data['eval_set_id'],
                         splits=dict(Counter(start['split'] for start in data['starts']))), indent=2))


if __name__ == '__main__':
    main()
