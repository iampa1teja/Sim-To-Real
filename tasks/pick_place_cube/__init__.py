import numpy as np

from core.tasks.base import StepState, Task, load_task_spec

REACH_DISTANCE_M = 0.03 
STAGES   = ("reached", "grasped", "lifted", "over_box", "placed", "success")
FAILURES = ("never_reached", "missed_grasp", "dropped_before_box", "dropped_outside",
            "placed_not_confirmed", "timeout_holding")

class PickPlaceTask(Task):

    @property
    def stages(self):
        return STAGES

    def observe(self, state: StepState):
        cube = np.asarray(state.objects["blue_cube"].position)
        box = state.objects["white_box"]
        signals = state.signals
        holding = bool(signals.get("holding", False))
        return {
            "reached": float(np.linalg.norm(np.asarray((state.tcp or state.ee).position) - cube)) <= REACH_DISTANCE_M,
            "grasped": bool(signals.get("grasped", False)),
            "lifted": bool(signals.get("lifted", False)),
            "over_box": holding and self._over_box(cube, box),
            "placed": bool(signals.get("placed", False)),
            "success": bool(signals.get("success", False)),
        }

    def _over_box(self, cube, box):
        size = tuple(self.spec.objects["white_box"].params["size_outer"])
        qx, qy, qz, qw = box.quat_xyzw
        polygon = _footprint(box.position, _rotation_matrix((qw, qx, qy, qz)), size, bottom_origin=True)
        return _inside_convex(cube[:2], polygon)

def _rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _footprint(position, rotation, size, bottom_origin=False):
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                        for z in ((0., 1.) if bottom_origin else (-.5, .5))])
    return (corners * np.asarray(size)) @ np.asarray(rotation).T[:, :2] + np.asarray(position)[:2]


def _inside_convex(point, corners):
    centre = corners.mean(axis=0)
    ordered = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
    edges = np.roll(ordered, -1, axis=0) - ordered
    rel = point - ordered
    return bool(np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] >= 0))

TASK = PickPlaceTask(load_task_spec("pick_place_cube"))
