"""Pick and place through Cartesian waypoints; all motion is executed by live IK."""
import numpy as np

from core.benchmark.geometry import rotation_matrix
from core.tasks.base import Pose
from core.tasks.expert import Waypoint


class PickPlaceExpert:
    def __init__(self, task):
        self.task = task

    def plan(self, context):
        options = context.options
        cube = np.asarray(context.state.objects["blue_cube"].position, dtype=float)
        box = context.state.objects["white_box"]
        qx, qy, qz, qw = box.quat_xyzw
        rotation = rotation_matrix((qw, qx, qy, qz))
        normal = np.asarray(context.table_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        orientation = tuple(options.get("orientation_xyzw", context.control_pose.quat_xyzw))
        size = self.task.spec.objects["blue_cube"].size
        container = self.task.spec.objects["white_box"].params
        # Aim at the nearest clear interior point rather than requiring the box
        # centre to be reachable. The circular support bound admits any cube yaw.
        radius = np.linalg.norm(np.asarray(size[:2]) / 2)
        clear = np.asarray(container["inner_half_size"]) - radius - options.get("place_margin_m", .003)
        if np.any(clear <= 0):
            raise ValueError("Task object cannot fit the container with the expert clearance")
        source_local = rotation.T @ (cube - np.asarray(box.position))
        xy = np.clip(source_local[:2], -clear, clear)
        if "place_xy" in options:
            xy = np.asarray(options["place_xy"], dtype=float)
            if xy.shape != (2,) or np.any(np.abs(xy) > clear):
                raise ValueError("Expert place_xy must lie inside the clear container interior")
        local = np.r_[xy, container["floor_z"] + size[2] / 2 + .0002]
        placed = np.asarray(box.position) + rotation @ local
        approach = float(options.get("approach_height_m", .06))
        lift = float(options.get("lift_height_m", .09))
        duration = float(options.get("move_seconds", 2.0))
        opened = float(options.get("gripper_open_fraction", .45))
        closed = float(options.get("gripper_closed_fraction", 0.0))
        timeout = float(options.get("waypoint_timeout_seconds", max(8., duration * 3)))

        def point(name, position, opening, seconds=duration):
            return Waypoint(name, Pose.from_lists(position, orientation), opening, seconds,
                            float(options.get("position_tolerance_m", .004)), max(timeout, seconds))

        return (
            point("approach", cube + normal * approach, opened),
            point("descend", cube, opened),
            point("close", cube, closed, float(options.get("grasp_seconds", .8))),
            point("lift", cube + normal * lift, closed),
            point("transport", placed + normal * lift, closed),
            point("lower", placed, closed),
            point("release", placed, opened, float(options.get("release_seconds", .8))),
            point("retreat", placed + normal * approach, opened),
            point("settle", placed + normal * approach, opened, float(options.get("settle_seconds", 1.5))),
        )


def create(task):
    return PickPlaceExpert(task)
