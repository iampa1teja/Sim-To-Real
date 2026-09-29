"""Simulator-independent recorded starts, geometry and evaluation reports."""

from datetime import datetime, timezone
import json
from pathlib import Path

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


class RecordedStarts:
    def __init__(self, directory, mode='cycle', random_fraction=0.0, seed=None):
        if mode not in ('cycle', 'random'):
            raise ValueError('start mode must be cycle or random')
        if not 0 <= random_fraction <= 1:
            raise ValueError('random_fraction must be in [0, 1]')
        self.starts = load_starts(directory)
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


def results_report(episodes, task, checkpoint, seed):
    def summarize(rows):
        successes = sum(bool(row['success']) for row in rows)
        return {'episodes': len(rows), 'successes': successes,
                'success_rate': successes / len(rows) if rows else None}

    def grouped(key):
        return {value: summarize([row for row in episodes if row[key] == value])
                for value in sorted({row[key] for row in episodes})}

    return {'schema_version': 1, 'task': task, 'checkpoint': checkpoint, 'seed': seed,
            'time': datetime.now(timezone.utc).isoformat(), 'overall': summarize(episodes),
            'by_start': {kind: summarize([row for row in episodes if row['start_kind'] == kind])
                         for kind in ('recorded', 'random')},
            'by_zone': grouped('zone'), 'by_cube_color': grouped('cube_color'), 'episodes': episodes}
