"""Recorded object starts and first-frame robot states, without simulator imports."""
from pathlib import Path
import json

import numpy as np

from utils.arms.spec import load_arm_spec
from .geometry import rotation_matrix


def task_instance(task=None):
    from core.tasks.loader import load_task

    return load_task(task) if task is None or isinstance(task, str) else task


def primary_object(task=None):
    task = task_instance(task)
    return task.spec.raw.get('start', {}).get('object') or next(iter(task.spec.tracked_objects))


def _directory(directory):
    from utils.datasets.task_meta import LEGACY_SIDECAR_DIR, SIDECAR_DIR

    path = Path(directory)
    if path.name not in (SIDECAR_DIR, LEGACY_SIDECAR_DIR):
        for name in (SIDECAR_DIR, LEGACY_SIDECAR_DIR):
            if (path / name).is_dir():
                return path / name
    return path


def start_episode_ids(directory):
    ids = []
    for path in sorted(_directory(directory).glob('episode_*.json')):
        try:
            ids.append(int(path.stem.split('_', 1)[1]))
        except ValueError as error:
            raise ValueError(f'Cannot read the episode index from {path.name}') from error
    return ids


def load_starts(directory, *, task=None, arm=None, legacy_object=None):
    """Read the first frame of each sidecar into named WXYZ object poses."""
    from utils.datasets.task_meta import read_task_meta

    if not directory or not _directory(directory).is_dir():
        raise ValueError(f'Recorded starts directory missing: {directory!r}; select a dataset or sidecar directory')
    paths = sorted(_directory(directory).glob('episode_*.json'))
    if not paths:
        raise ValueError(f'No episode_*.json files in recorded starts directory: {directory}')
    task = task_instance(task)
    arm = arm or load_arm_spec()[0]
    legacy_object = legacy_object or primary_object(task)
    starts = []
    for path in paths:
        try:
            data = read_task_meta(path, task=task.name, arm=arm.name, legacy_object=legacy_object)
            expected_frame = f'robot base link ({arm.links.base})'
            if data.get('reference_frame', expected_frame) != expected_frame:
                raise ValueError(f'reference_frame must be {expected_frame}')
            objects = {}
            for name, values in data['objects'].items():
                position = np.asarray(values['position'][0], dtype=float)
                orientation = np.asarray(values['orientation_wxyz'][0], dtype=float)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError('position frame 0 must contain three finite values')
                rotation_matrix(orientation)
                objects[name] = dict(position=position, orientation_wxyz=orientation / np.linalg.norm(orientation))
            starts.append(objects)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid recorded start {path}: {error}') from error
    names = set(starts[0])
    if any(set(start) != names for start in starts):
        raise ValueError('Recorded starts must contain the same named objects in every episode')
    return starts


def load_robot_starts(dataset_root, episode_ids, *, arm=None):
    """Read only frame zero of requested episodes, retaining requested episode order."""
    import pyarrow.parquet as pq

    arm = arm or load_arm_spec()[0]
    metadata = Path(dataset_root) / 'meta/info.json'
    if metadata.is_file():
        features = json.loads(metadata.read_text()).get('features', {})
        names = features.get('observation.state', {}).get('names')
        if names is not None and tuple(names) != arm.feature_names:
            raise ValueError(f'Recorded robot state joint order differs from arm {arm.name!r}')
    paths = sorted(Path(dataset_root).glob('data/*/*.parquet'))
    if not paths:
        raise ValueError(f'No data parquet files under {dataset_root}; cannot read recorded robot starts')
    first = {}
    for path in paths:
        table = pq.read_table(path, columns=['episode_index', 'observation.state'],
                              filters=[('frame_index', '=', 0), ('episode_index', 'in', list(episode_ids))])
        for episode, state in zip(table.column('episode_index').to_pylist(), table.column('observation.state').to_pylist()):
            first[int(episode)] = np.asarray(state, dtype=float)
    missing = [episode for episode in episode_ids if episode not in first]
    if missing:
        raise ValueError(f'No first-frame robot state in {dataset_root} for episodes {missing}')
    for episode in episode_ids:
        if first[episode].shape != (len(arm.joint_names),) or not np.isfinite(first[episode]).all():
            raise ValueError(f'Invalid first-frame robot state for episode {episode}')
    return [first[episode] for episode in episode_ids]


def copy_objects(objects):
    return {name: {key: np.asarray(value).copy() for key, value in pose.items()} for name, pose in objects.items()}


class RecordedStarts:
    def __init__(self, directory, mode='cycle', random_fraction=0.0, seed=None, robot_states_root=None, *, arm=None, task=None,
                 scene=None):
        if mode not in ('cycle', 'random'):
            raise ValueError('start mode must be cycle or random')
        if not 0 <= random_fraction <= 1:
            raise ValueError('random_fraction must be in [0, 1]')
        self.task = task_instance(task)
        self.scene = scene
        self.arm = arm or load_arm_spec()[0]
        self.primary_object = primary_object(self.task)
        self.starts = load_starts(directory, task=self.task, arm=self.arm)
        if self.primary_object not in self.starts[0]:
            raise ValueError(f'Recorded starts omit primary object {self.primary_object!r}')
        self.episode_ids = start_episode_ids(directory)
        self.robot_states = (load_robot_starts(robot_states_root, self.episode_ids, arm=self.arm)
                             if robot_states_root is not None else None)
        self.joint_mapping = None
        if robot_states_root is not None:
            from utils.datasets.sources import recorded_joint_mapping

            self.joint_mapping, _ = recorded_joint_mapping(robot_states_root, self.arm)
        xy = np.array([objects[self.primary_object]['position'][:2] for objects in self.starts])
        self.low, self.high = xy.min(axis=0), xy.max(axis=0)
        centre = (self.low + self.high) / 2
        self.away_axis = int(np.argmax(np.abs(centre)))
        self.away_sign = 1 if centre[self.away_axis] >= 0 else -1
        self.mode, self.random_fraction = mode, random_fraction
        self.rng = np.random.default_rng(seed)
        self.order = []
        self.current = None

    def zone(self, xy):
        span = self.high - self.low
        fraction = np.divide(np.asarray(xy) - self.low, span, out=np.full(2, 0.5), where=span != 0)
        axis = self.away_axis
        away = fraction[axis] if self.away_sign > 0 else 1 - fraction[axis]
        left_sign = self.away_sign * (1 if axis == 0 else -1)
        left = fraction[1 - axis] if left_sign > 0 else 1 - fraction[1 - axis]
        row = min(2, max(0, int(away * 3)))
        col = min(2, max(0, int(left * 3)))
        return 'NMF'[row] + 'RCL'[col]

    def sample(self, accepts_random=None):
        # Keep the legacy draw order, including the random-fraction draw at zero.
        if self.rng.random() < self.random_fraction:
            if accepts_random is None:
                raise ValueError('Random starts require a live task validity check')
            for _ in range(10000):
                if len(self.starts[0]) == 1:
                    position = np.r_[self.rng.uniform(self.low, self.high), 0.0]
                    yaw = self.rng.uniform(0, np.pi / 2)
                    objects = {self.primary_object: dict(position=position,
                        orientation_wxyz=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))}
                else:
                    from core.tasks.base import sample_start_in_base

                    objects = poses_to_objects(sample_start_in_base(self.task, self.rng, self.scene))
                if accepts_random(objects):
                    return -1, objects, self.zone(objects[self.primary_object]['position'][:2])
            raise ValueError('Cannot sample a valid object start after 10000 attempts')
        if self.mode == 'random':
            index = int(self.rng.integers(len(self.starts)))
        else:
            if not self.order:
                self.order = self.rng.permutation(len(self.starts)).tolist()
            index = self.order.pop()
        objects = copy_objects(self.starts[index])
        return index, objects, self.zone(objects[self.primary_object]['position'][:2])

    def robot_state(self, index):
        if self.robot_states is None:
            raise ValueError('Recorded robot starts were not loaded (robot_states_root is None)')
        if index == -1:
            return np.mean(self.robot_states, axis=0)
        return self.robot_states[index].copy()


def poses_to_objects(poses):
    """Convert Task's XYZW Pose contract into the WXYZ disk/reset contract."""
    objects = {}
    for name, pose in poses.items():
        x, y, z, w = pose.quat_xyzw
        objects[name] = dict(position=np.asarray(pose.position, dtype=float),
                             orientation_wxyz=np.array([w, x, y, z], dtype=float))
    return objects


def objects_to_poses(objects):
    from core.tasks.base import Pose

    result = {}
    for name, pose in objects.items():
        w, x, y, z = pose['orientation_wxyz']
        result[name] = Pose.from_lists(pose['position'], (x, y, z, w))
    return result
