"""Simulator-independent recorded starts, geometry and evaluation reports."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


def load_starts(directory):
    if not directory or not Path(directory).is_dir():
        raise ValueError(f"Recorded starts directory missing: {directory!r}; set PICK_PLACE_EVAL_STARTS or --cube_starts")
    paths = sorted(Path(directory).glob("episode_*.json"))
    if not paths:
        raise ValueError(f"No episode_*.json files in recorded starts directory: {directory}")
    starts = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
            if data['units'] != 'm':
                raise ValueError('units must be "m"')
            position = np.asarray(data['position'][0], dtype=float)
            orientation = np.asarray(data['orientation_wxyz'][0], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('position frame 0 must contain three finite values')
            rotation_matrix(orientation)
            if data.get('reference_frame', 'robot base link (base)') != 'robot base link (base)':
                raise ValueError('reference_frame must be robot base link (base)')
            starts.append((position, orientation / np.linalg.norm(orientation)))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid recorded start {path}: {exc}") from exc
    return starts


def start_episode_ids(directory):
    """Episode index of each recorded start, in the same order as load_starts()."""
    ids = []
    for path in sorted(Path(directory).glob("episode_*.json")):
        try:
            ids.append(int(path.stem.split("_", 1)[1]))
        except ValueError as exc:
            raise ValueError(f"Cannot read the episode index from {path.name}") from exc
    return ids


# LeRobot joint order and the USD joint limits (degrees) used by LeRobotSO101Interface.
SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SO101_USD_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SO101_USD_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-100.0, 100.0), "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0), "wrist_roll": (-160.0, 160.0), "gripper": (-10.0, 100.0),
}


def dataset_state_to_radians(state, joint_mapping=None):
    """Dataset units (observation.state) -> sim joint radians, in SO101_JOINTS order.

    Same maths as LeRobotSO101Interface.get_mapped_actions_vectorized: arm joints map
    -100..100 and the gripper 0..100 onto the calibrated mapping span (joint_mapping
    overrides the USD span per joint), then clamp to the USD limits.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError(f"Robot state must be six finite values, got {state!r}")
    mapping = {name: SO101_USD_LIMITS_DEG[name] for name in SO101_JOINTS}
    for name, span in (joint_mapping or {}).items():
        if name not in mapping:
            raise ValueError(f"Unknown joint mapping: {name}")
        mapping[name] = (float(span["joint_min"]), float(span["joint_max"]))
    radians = np.empty(6)
    for i, name in enumerate(SO101_JOINTS):
        normalized = state[i] / 100.0 if name == "gripper" else (state[i] + 100.0) / 200.0
        low, high = mapping[name]
        usd_low, usd_high = SO101_USD_LIMITS_DEG[name]
        degrees = np.clip(low + normalized * (high - low), usd_low, usd_high)
        radians[i] = np.deg2rad(degrees)
    return radians


def load_robot_starts(dataset_root, episode_ids):
    """First-frame observation.state of each episode, read from the dataset parquet files."""
    import pyarrow.parquet as pq

    paths = sorted(Path(dataset_root).glob("data/*/*.parquet"))
    if not paths:
        raise ValueError(f"No data parquet files under {dataset_root}; cannot read recorded robot starts")
    first = {}
    for path in paths:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"])
        episodes = table.column("episode_index").to_pylist()
        frames = table.column("frame_index").to_pylist()
        states = table.column("observation.state").to_pylist()
        for episode, frame, state in zip(episodes, frames, states):
            if frame == 0:
                first[int(episode)] = np.asarray(state, dtype=float)
    missing = [e for e in episode_ids if e not in first]
    if missing:
        raise ValueError(f"No first-frame robot state in {dataset_root} for episodes {missing}")
    for episode in episode_ids:
        if first[episode].shape != (6,) or not np.isfinite(first[episode]).all():
            raise ValueError(f"Invalid first-frame robot state for episode {episode}")
    return [first[e] for e in episode_ids]


class RecordedStarts:
    def __init__(self, directory, mode='cycle', random_fraction=0.0, seed=None, robot_states_root=None):
        if mode not in ('cycle', 'random'):
            raise ValueError('start mode must be cycle or random')
        if not 0 <= random_fraction <= 1:
            raise ValueError('random_fraction must be in [0, 1]')
        self.starts = load_starts(directory)
        self.episode_ids = start_episode_ids(directory)
        # Optional: the arm's recorded first-frame state, paired with each cube start.
        self.robot_states = (load_robot_starts(robot_states_root, self.episode_ids)
                             if robot_states_root is not None else None)
        xy = np.array([p[:2] for p, _ in self.starts])
        self.low, self.high = xy.min(axis=0), xy.max(axis=0)
        # The axis with the largest displacement of the bounding-box centre
        # from the base is the away axis. Ties choose X. Increasing signed
        # coordinate is farther; left is 90 degrees CCW from away (+Z up).
        centre = (self.low + self.high) / 2
        self.away_axis = int(np.argmax(np.abs(centre)))
        self.away_sign = 1 if centre[self.away_axis] >= 0 else -1
        self.mode, self.random_fraction = mode, random_fraction
        self.rng = np.random.default_rng(seed)
        self.order = []

    def zone(self, xy):
        span = self.high - self.low
        fraction = np.divide(np.asarray(xy) - self.low, span,
                             out=np.full(2, 0.5), where=span != 0)
        a = self.away_axis
        away = fraction[a] if self.away_sign > 0 else 1 - fraction[a]
        left_sign = self.away_sign * (1 if a == 0 else -1)
        left = fraction[1-a] if left_sign > 0 else 1 - fraction[1-a]
        row = min(2, max(0, int(away * 3)))
        col = min(2, max(0, int(left * 3)))
        return 'NMF'[row] + 'RCL'[col]

    def sample(self, accepts_random=None):
        if self.rng.random() < self.random_fraction:
            if accepts_random is None:
                raise ValueError('Random starts require a live box footprint check')
            for _ in range(10000):
                position = np.r_[self.rng.uniform(self.low, self.high), 0.0]
                yaw = self.rng.uniform(0, np.pi/2)
                orientation = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                if accepts_random(position, orientation):
                    return -1, position, orientation, self.zone(position[:2])
            raise ValueError('Cannot sample a non-overlapping cube footprint inside recorded bounds after 10000 attempts')
        if self.mode == 'random':
            index = int(self.rng.integers(len(self.starts)))
        else:
            if not self.order:
                self.order = self.rng.permutation(len(self.starts)).tolist()
            index = self.order.pop()
        position, orientation = self.starts[index]
        return index, position.copy(), orientation.copy(), self.zone(position[:2])

    def robot_state(self, index):
        """Recorded first-frame arm state (dataset units) for start `index`; -1 (random) -> mean."""
        if self.robot_states is None:
            raise ValueError('Recorded robot starts were not loaded (robot_states_root is None)')
        if index == -1:
            return np.mean(self.robot_states, axis=0)
        return self.robot_states[index].copy()


def upright_orientation(base_orientation, plane, yaw_orientation):
    """Yaw about the live table normal, expressed in the live base frame."""
    normal = np.array([-plane[0], -plane[1], 1.0])
    normal = rotation_matrix(base_orientation).T @ (normal / np.linalg.norm(normal))
    tilt = np.array([1 + normal[2], -normal[1], normal[0], 0.0])
    if np.linalg.norm(tilt) < 1e-8:
        tilt = np.array([0., 1., 0., 0.])
    tilt /= np.linalg.norm(tilt)
    w, x, y, z = tilt
    a, b, c, d = yaw_orientation
    return np.array([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                     w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])


def resting_pose(position, orientation, base_position, base_orientation, plane, cube_size):
    """Preserve base XY, ignore recorded Z and solve for tabletop support.

    plane is (a,b,c): world z=a*x+b*y+c. Same oriented support and
    0.2 mm clearance as reset_object_pose, solved along the live base Z axis.
    """
    base_rotation = rotation_matrix(base_orientation)
    rotation = base_rotation @ rotation_matrix(orientation)
    normal = np.array([-plane[0], -plane[1], 1.0])
    support = np.sum(np.abs(normal @ rotation) * np.asarray(cube_size)/2)
    world = np.asarray(base_position) + base_rotation @ np.r_[position[:2], 0.0]
    denominator = normal @ base_rotation[:, 2]
    if abs(denominator) < 1e-8:
        raise ValueError('Live base Z axis is parallel to the table')
    world += base_rotation[:, 2] * ((plane[2] + support + 0.0002 - normal @ world) / denominator)
    return world, rotation


def footprint(position, rotation, size, bottom_origin=False):
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                        for z in ((0., 1.) if bottom_origin else (-.5, .5))])
    return (corners * np.asarray(size)) @ np.asarray(rotation).T[:, :2] + np.asarray(position)[:2]


def footprints_overlap(first, second):
    """Separating axis test of convex projected cuboids, with no added margin."""
    # All point-pair normals include every convex-hull edge normal; extra
    # axes cannot separate overlapping convex hulls.
    for points in (first, second):
        for i, point in enumerate(points):
            for other in points[i+1:]:
                edge = other - point
                axis = np.array([-edge[1], edge[0]])
                if not np.any(axis):
                    continue
                a, b = first @ axis, second @ axis
                if a.max() <= b.min() or b.max() <= a.min():
                    return False
    return True


def select_instruction(color, default, by_color=None):
    if by_color is None:
        return default
    if not isinstance(by_color, dict) or color not in by_color or not isinstance(by_color[color], str) or not by_color[color].strip():
        raise ValueError(f'--lang_instruction_by_color requires a non-empty instruction for {color!r}')
    return by_color[color]


def results_report(episodes, task, checkpoint, seed, *, random_fraction=0.0,
                   episode_length_s=None, action_horizon=16, step_dt=None):
    """Add simulated success timing without changing existing schema-1 fields.

    Success steps count control transitions from the episode reset, including
    settling steps. Median time includes successful episodes only; the 15-second
    rate uses every completed episode as its denominator. Without step_dt, old
    callers retain their rates and receive null timing metrics.
    """
    def summarize(rows):
        successes = sum(bool(row['success']) for row in rows)
        success_times = ([row['success_step'] * step_dt for row in rows if row['success']]
                         if step_dt is not None else None)
        within_15 = (sum(time <= 15 or math.isclose(time, 15, rel_tol=0, abs_tol=1e-12)
                         for time in success_times) if success_times is not None else None)
        return {'episodes': len(rows), 'successes': successes,
                'success_rate': successes / len(rows) if rows else None,
                'successes_within_15s': within_15,
                'success_rate_within_15s': within_15 / len(rows) if rows and within_15 is not None else None,
                'median_time_to_success_s': median(success_times) if success_times else None}

    def grouped(key):
        return {value: summarize([row for row in episodes if row[key] == value])
                for value in sorted({row[key] for row in episodes})}

    return {'schema_version': 1, 'task': task, 'checkpoint': checkpoint, 'seed': seed,
            'random_fraction': random_fraction, 'episode_length_s': episode_length_s,
            'action_horizon': action_horizon, 'step_dt': step_dt,
            'time': datetime.now(timezone.utc).isoformat(), 'overall': summarize(episodes),
            'by_start': {kind: summarize([row for row in episodes if row['start_kind'] == kind])
                         for kind in ('recorded', 'random')},
            'by_zone': grouped('zone'), 'by_cube_color': grouped('cube_color'), 'episodes': episodes}
