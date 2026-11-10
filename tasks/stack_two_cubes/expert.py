"""Short, fixed-orientation pick-and-stack waypoints for two live cuboids."""
import math
import numpy as np

from core.benchmark.geometry import rotation_matrix
from core.tasks.base import Pose
from core.tasks.expert import Waypoint


class StackExpert:
    def __init__(self, task):
        self.task = task

    def plan(self, context):
        options = context.options
        if {"transport_roll_degrees", "transport_orientation_xyzw"} & options.keys():
            raise ValueError("Stack expert keeps the pickup orientation; transport roll overrides are unsupported")
        blue, red = context.state.objects["cube_a"], context.state.objects["cube_b"]
        orientation = tuple(options.get("orientation_xyzw", context.control_pose.quat_xyzw))
        normal = np.asarray(context.table_normal, dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
            raise ValueError("Stack expert requires a finite table normal")
        normal /= np.linalg.norm(normal)

        def rotation(pose):
            x, y, z, w = pose.quat_xyzw
            return rotation_matrix((w, x, y, z))

        red_rotation = rotation(red)
        blue_half = np.asarray(self.task.spec.objects["cube_a"].size) / 2
        red_half = np.asarray(self.task.spec.objects["cube_b"].size) / 2
        # Heights and support are expressed in the target cube's own frame.
        # For upright equal cubes this is 10 + 10 + 0.5 mm above its centre.
        blue_height = np.abs((red_rotation.T @ rotation(blue))[2]) @ blue_half
        clearance = float(options.get("stack_clearance_m", .0005))
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("stack_clearance_m must be finite and nonnegative")
        stacked = np.asarray(red.position) + red_rotation[:, 2] * (red_half[2] + blue_height + clearance)
        source = np.asarray(blue.position)
        approach = float(options.get("approach_height_m", .06))
        lift = float(options.get("lift_height_m", .05))
        if min(approach, lift) <= 0:
            raise ValueError("Approach and lift heights must be positive")
        duration = float(options.get("move_seconds", 1.5))
        transport = float(options.get("transport_seconds", .5))
        opened = float(options.get("gripper_open_fraction", .45))
        closed = float(options.get("gripper_closed_fraction", 0.0))
        timeout = float(options.get("waypoint_timeout_seconds", 8.0))
        tolerance = float(options.get("position_tolerance_m", .004))
        precise = min(tolerance, self.task.spec.raw["success"]["xy_tolerance"] / 4)
        fps = float(options.get("control_fps", 30.))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("control_fps must be positive")
        settle = max(float(options.get("settle_seconds", 1.2)),
                     (self.task.spec.raw["success"]["confirm_steps"] + 1) / fps)

        def point(name, position, opening, seconds=duration, tolerance_m=tolerance):
            return Waypoint(name, Pose.from_lists(position, orientation), opening, seconds,
                            tolerance_m, max(timeout, seconds))

        return (
            point("approach", source + normal * approach, opened),
            point("descend", source, opened),
            point("close", source, closed, float(options.get("grasp_seconds", .8))),
            point("lift", source + normal * lift, closed),
            point("transport", stacked + normal * lift, closed, transport),
            point("lower", stacked, closed, tolerance_m=precise),
            point("release", stacked, opened, float(options.get("release_seconds", .8)), precise),
            point("retreat", stacked + normal * lift, opened),
            point("settle", stacked + normal * lift, opened, settle),
        )


def create(task):
    return StackExpert(task)
