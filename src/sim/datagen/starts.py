"""Deterministic grid coverage for scripted generation, independent of Isaac."""
from itertools import product
import math
import random

import numpy as np

from core.tasks.task import Task
from utils.geometry import Pose, xyzw_to_wxyz


class GridStarts:
    """Visit cell centres and yaw bins across all tracked-object combinations.

    A seeded, coprime traversal spreads capped runs through the Cartesian
    product without allocating that product. Invalid combinations are skipped;
    after the selected starts are exhausted, visits cycle in the same order.
    """

    def __init__(self, task, *, cells_per_axis=5, yaw_steps=4, max_starts=256, seed=1984):
        for name, value in (("cells_per_axis", cells_per_axis), ("yaw_steps", yaw_steps), ("max_starts", max_starts)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Keep SDK configuration traversal confined to data; modules may contain cycles.
        self.spec, self.scene = task.spec, None
        self.cells_per_axis, self.yaw_steps, self.max_starts = cells_per_axis, yaw_steps, max_starts
        self.names = tuple(task.spec.tracked_objects)
        if not self.names:
            raise ValueError("Grid starts require tracked task objects")
        self.choices = []
        self.visits = {}
        for name in self.names:
            region = {**task.spec.start, **task.spec.objects[name].params.get("start", {})}
            low, high = np.asarray(region["xy_min"], dtype=float), np.asarray(region["xy_max"], dtype=float)
            yaw = np.asarray(region.get("yaw_deg", (0., 90.)), dtype=float)
            if (low.shape != (2,) or high.shape != (2,) or yaw.shape != (2,) or
                    not np.isfinite(np.r_[low, high, yaw]).all() or np.any(low > high) or yaw[0] > yaw[1]):
                raise ValueError(f"{name}: grid starts need finite, ordered xy/yaw ranges")
            xs, ys = [low[i] + (np.arange(cells_per_axis) + .5) * (high[i] - low[i]) / cells_per_axis
                      if high[i] > low[i] else np.array([low[i]]) for i in range(2)]
            yaws = (yaw[0] + (np.arange(yaw_steps) + .5) * (yaw[1] - yaw[0]) / yaw_steps
                    if yaw[1] > yaw[0] else np.array([yaw[0]]))
            choices = []
            for ix, iy, iz in product(range(len(xs)), range(len(ys)), range(len(yaws))):
                angle = math.radians(float(yaws[iz])) / 2
                pose = Pose.from_lists((xs[ix], ys[iy], region.get("z", 0.)),
                                       (0., 0., math.sin(angle), math.cos(angle)))
                choices.append((pose, (ix, iy, iz)))
            self.choices.append(choices)
            self.visits[name] = np.zeros((len(xs), len(ys), len(yaws)), dtype=int)
        self.total = math.prod(map(len, self.choices))
        rng = random.Random(seed)
        self.offset = rng.randrange(self.total)
        self.stride = rng.randrange(1, self.total) if self.total > 1 else 1
        while math.gcd(self.stride, self.total) != 1:
            self.stride = (self.stride + 1) % self.total or 1
        self.examined, self.cycle = 0, 0
        self.starts = []

    def _candidate(self, index):
        rows = []
        for choices in reversed(self.choices):
            index, digit = divmod(index, len(choices))
            rows.append(choices[digit])
        return tuple(reversed(rows))

    def _valid(self, rows):
        minimum = self.spec.start.get("minimum_separation_m", 0.)
        radius = self.spec.start.get("minimum_base_radius_m", 0.)
        xy = [np.asarray(pose.position[:2]) for pose, _ in rows]
        return (all(np.linalg.norm(p) >= radius for p in xy) and
                all(np.linalg.norm(a - b) >= minimum for i, a in enumerate(xy) for b in xy[:i]))

    def _objects(self, rows):
        task = Task(self.spec)
        result = {}
        for name, (pose, _) in zip(self.names, rows):
            pose = task.start_pose_to_base(pose, self.scene)
            result[name] = {"position": np.asarray(pose.position),
                            "orientation_wxyz": xyzw_to_wxyz(pose.quat_xyzw)}
        return result

    def sample(self, accepts_random=None):
        while len(self.starts) < self.max_starts and self.examined < self.total:
            index = (self.offset + self.stride * self.examined) % self.total
            self.examined += 1
            rows = self._candidate(index)
            if not self._valid(rows):
                continue
            objects = self._objects(rows)
            if accepts_random is not None and not accepts_random(objects):
                continue
            self.starts.append(rows)
            return self._visit(rows, objects)
        for _ in self.starts:
            rows = self.starts[self.cycle % len(self.starts)]
            self.cycle += 1
            objects = self._objects(rows)
            if accepts_random is None or accepts_random(objects):
                return self._visit(rows, objects)
        raise ValueError(f"No valid grid starts for task {self.spec.name!r}")

    def _visit(self, rows, objects):
        for name, (_, cell) in zip(self.names, rows):
            self.visits[name][cell] += 1
        # Sampled starts retain the existing upright/table-placement path.
        return -1, objects, "grid"

    def coverage(self):
        return {"cells_per_axis": self.cells_per_axis, "yaw_steps": self.yaw_steps,
                "max_starts": self.max_starts, "candidate_combinations": self.total,
                "examined_combinations": self.examined, "unique_starts": len(self.starts),
                "draws": int(next(iter(self.visits.values())).sum()),
                "objects": {name: {"cell_visits": counts.tolist(), "xy_visits": counts.sum(axis=2).tolist()}
                            for name, counts in self.visits.items()}}
