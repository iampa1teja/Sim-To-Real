"""Pick and place through Cartesian waypoints; all motion is executed by live IK."""
import math
import numpy as np

from core.eval.geometry import rotation_matrix
from core.tasks.base import Pose
from core.tasks.expert import Waypoint, gripper_commands, grasp_orientation


def _transport_arc(start, end, origin, normal, margin, *, interpolate_radius=False):
    """Keep interpolation chords outside both endpoints' minimum base radius."""
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("transport_arc_margin_m must be finite and positive")
    offsets = [np.asarray(point) - origin for point in (start, end)]
    heights = [np.dot(offset, normal) for offset in offsets]
    radial = [offset - height * normal for offset, height in zip(offsets, heights)]
    radii = [np.linalg.norm(value) for value in radial]
    if min(radii) < 1e-6:
        raise ValueError("Expert transport endpoints must be clear of the actual arm base")
    first, last = [value / radius for value, radius in zip(radial, radii)]
    angle = math.atan2(np.dot(normal, np.cross(first, last)), np.dot(first, last))
    if abs(angle) < 1e-6:
        return []
    if interpolate_radius:
        # Shrink a distant pickup's radius toward the box instead of carrying
        # its large radius all the way around the arm. Verify whole chords,
        # including their interiors, against the required inner clearance.
        tangent = np.cross(normal, first)
        count = max(2, math.ceil(abs(angle) / (math.pi / 12)))
        for _ in range(12):
            points = []
            for fraction in np.linspace(0., 1., count + 1):
                direction = first * math.cos(angle * fraction) + tangent * math.sin(angle * fraction)
                radius = radii[0] * (1 - fraction) + radii[1] * fraction + margin * math.sin(math.pi * fraction)
                height = heights[0] * (1 - fraction) + heights[1] * fraction
                points.append(origin + height * normal + radius * direction)
            safe = True
            for left, right in zip(points, points[1:]):
                a, b = left - origin, right - origin
                a -= np.dot(a, normal) * normal
                b -= np.dot(b, normal) * normal
                delta = b - a
                t = np.clip(-np.dot(a, delta) / max(np.dot(delta, delta), 1e-20), 0., 1.)
                if np.linalg.norm(a + t * delta) < min(radii) - 1e-9:
                    safe = False
                    break
            if safe:
                return points
            count *= 2
        raise ValueError("Cannot construct a transport path with the required base clearance")
    radius = max(radii) + margin
    # Bound chord sag as well as the angular step: straight interpolation
    # between adjacent arc points must not cut back through the inner workspace.
    max_angle = min(math.pi / 6, 2 * math.acos(min(radii) / radius))
    count = max(1, math.ceil(abs(angle) / max_angle))
    tangent = np.cross(normal, first)
    points = []
    for fraction in np.linspace(0., 1., count + 1):
        direction = first * math.cos(angle * fraction) + tangent * math.sin(angle * fraction)
        height = heights[0] * (1 - fraction) + heights[1] * fraction
        points.append(origin + height * normal + radius * direction)
    return points


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
        if {"transport_orientation_xyzw", "transport_roll_degrees"} & options.keys():
            raise ValueError("The expert keeps its pickup orientation during transport; remove transport roll overrides")
        size = self.task.spec.objects["blue_cube"].size
        container = self.task.spec.objects["white_box"].params
        # The circular support bound admits any cube yaw. Centre placement keeps
        # the fixed finger away from the container rim during the lower phase.
        radius = np.linalg.norm(np.asarray(size[:2]) / 2)
        clear = np.asarray(container["inner_half_size"]) - radius - options.get("place_margin_m", .003)
        if np.any(clear <= 0):
            raise ValueError("Task object cannot fit the container with the expert clearance")
        xy = np.zeros(2)
        if "place_xy" in options:
            xy = np.asarray(options["place_xy"], dtype=float)
            if xy.shape != (2,) or np.any(np.abs(xy) > clear):
                raise ValueError("Expert place_xy must lie inside the clear container interior")
        local = np.r_[xy, container["floor_z"] + size[2] / 2 + .0002]
        placed = np.asarray(box.position) + rotation @ local
        approach = float(options.get("approach_height_m", .06))
        lift = float(options.get("lift_height_m", .09))
        release_height = float(options.get("release_height_m", 0.))
        if not math.isfinite(release_height) or not 0 <= release_height <= lift:
            raise ValueError("release_height_m must be between zero and lift_height_m")
        duration = float(options.get("move_seconds", 2.0))
        transport_duration = float(options.get("transport_seconds", duration))
        opened, closed = gripper_commands(context, max(size[:2]))
        orientation = grasp_orientation(context, context.state.objects["blue_cube"], orientation)
        timeout = float(options.get("waypoint_timeout_seconds", max(8., duration * 3)))
        if context.arm_base is None:
            raise ValueError("Pick-place expert transport requires the actual arm_base world pose")
        source = cube + normal * lift
        destination = placed + normal * lift
        route = _transport_arc(source, destination, np.asarray(context.arm_base.position), normal,
                               float(options.get("transport_arc_margin_m", .01)),
                               interpolate_radius=bool(options.get("transport_interpolate_radius", False)))

        def point(name, position, opening, seconds=duration, *, object_goal=False):
            return Waypoint(name, Pose.from_lists(position, orientation), duration_s=seconds,
                            tolerance_m=float(options.get("position_tolerance_m", .004)),
                            timeout_s=max(timeout, seconds),
                            object_ref='blue_cube' if object_goal and options.get('object_feedback', False) else None, **opening)

        release_pose = placed + normal * release_height
        lower = (() if release_height == lift else
                 (point("lower", release_pose, closed, object_goal=True),))
        return (
            point("approach", cube + normal * approach, opened),
            point("descend", cube, opened),
            point("close", cube, closed, float(options.get("grasp_seconds", .8))),
            point("lift", source, closed),
            *(point(f"transport_arc_{index}", position, closed, transport_duration, object_goal=True)
              for index, position in enumerate(route)),
            point("transport", destination, closed, float(options.get("final_transport_seconds", transport_duration)), object_goal=True),
            *lower,
            point("release", release_pose, opened, float(options.get("release_seconds", .8)), object_goal=True),
            point("retreat", placed + normal * max(approach, release_height), opened, object_goal=True),
            point("settle", placed + normal * max(approach, release_height), opened,
                  float(options.get("settle_seconds", 1.5)), object_goal=True),
        )


def create(task):
    return PickPlaceExpert(task)
