import math 
import numpy as np 

from core.tasks.base import Pose, StepState, Task, load_task_spec

REACH_DISTANCE_M = 0.03 
STAGES   = ("reached", "grasped", "lifted", "over_box", "placed", "success")
FAILURES = ("never_reached", "missed_grasp", "dropped_before_box", "dropped_outside",
            "placed_not_confirmed", "timeout_holding")

class PickPlaceTask(Task):

    @property
    def stages(self):
        return STAGES

    def sample_start(self, rng):
        start = self.spec.raw.get("start")
        if not start:
            raise ValueError("tasks/pick_place_cube/task.toml has no [task.start]; random starts need xy_min, xy_max, yaw_deg")
        x, y = rng.uniform(start["xy_min"], start["xy_max"])
        yaw = math.radians(rng.uniform(*start.get("yaw_deg", (0.0, 90.0))))
        return {"blue_cube": Pose.from_lists((x, y, 0.0), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))}

    def observe(self, state: StepState):
        cube = np.asarray(state.objects["blue_cube"].position)
        box = state.objects["white_box"]
        signals = state.signals
        holding = bool(signals.get("holding", False))
        return {
            "reached": float(np.linalg.norm(np.asarray(state.ee.position) - cube)) <= REACH_DISTANCE_M,
            "grasped": bool(signals.get("grasped", False)),
            "lifted": bool(signals.get("lifted", False)),
            "over_box": holding and self._over_box(cube, box),
            "placed": bool(signals.get("placed", False)),
            "success": bool(signals.get("success", False)),
        }

    def failure_mode(self, flags, holding, success):
        if success:
            return None
        if flags.get("placed"):
            return "placed_not_confirmed"
        if not flags.get("reached"):
            return "never_reached"
        if not flags.get("grasped"):
            return "missed_grasp"
        if holding:
            return "timeout_holding"
        if not flags.get("over_box"):
            return "dropped_before_box"
        return "dropped_outside"

    def _over_box(self, cube, box):
        from .eval import footprint, rotation_matrix 

        size = tuple(self.spec.objects["white_box"].params["size_outer"])
        qx, qy, qz, qw = box.quat_xyzw
        polygon = footprint(box.position, rotation_matrix((qw, qx, qy, qz)), size, bottom_origin=True)
        return _inside_convex(cube[:2], polygon)

def _inside_convex(point, corners):
    centre = corners.mean(axis=0)
    ordered = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
    edges = np.roll(ordered, -1, axis=0) - ordered
    rel = point - ordered
    return bool(np.all(edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0] >= 0))

TASK = PickPlaceTask(load_task_spec("pick_place_cube"))
