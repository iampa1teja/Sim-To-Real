"""place(obj, target, relation): carry the held object to the target, release it and retreat.

``relation="inside"`` places the object centre in a container's clear interior (``inner_half_size``,
``floor_z``; centred unless ``place_xy`` is given). ``transport="arc"`` carries it at the lift height
around the actual arm base instead of cutting through the inner workspace; ``"direct"`` goes straight
to the point above the placement. The pickup orientation is kept throughout the transport.

``relation="on_top"`` (``transport="direct"`` only) stacks the object on the target: its centre goes
along the target's own up axis by the target's half height, the object's half extent along that axis
and ``clearance`` (option ``stack_clearance_m``). It goes to ``transport`` above that point, then
``lower`` and ``release`` there with the tighter tolerance ``precise`` (``yaw_precise`` when the arm
controls yaw), and ``retreat``/``settle`` back at the lift height. ``settle`` lasts at least
``settle_steps`` control steps (option ``control_fps``, 30). ``feedback="yaw"`` sets ``object_ref``
on these waypoints exactly when the arm controls yaw (default ``"option"``: the ``object_feedback`` option).
"""
import math

import numpy as np

from sim.actions.action import Move
from utils.geometry import rotation_matrix_wxyz


def transport_arc(start, end, origin, normal, margin, *, interpolate_radius=False):
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


def _controls_yaw(ctx):
    return ctx.arm is not None and "z" in ctx.arm.ik.orientation_axes


def _rotation(pose):
    qx, qy, qz, qw = pose.quat_xyzw
    return rotation_matrix_wxyz((qw, qx, qy, qz))


class Place(Move):
    def __init__(self, obj, target, relation="inside", *, transport="arc", lift=.09, retreat=.06,
                 clearance=.003, release_height=0., arc_margin=.01, release_seconds=.8, settle_seconds=1.5,
                 precise=None, yaw_precise=None, settle_steps=0, feedback="option"):
        if relation not in ("inside", "on_top"):
            raise ValueError(f"place relation must be 'inside' or 'on_top', got {relation!r}")
        if transport not in ("arc", "direct"):
            raise ValueError(f"place transport must be 'arc' or 'direct', got {transport!r}")
        if relation == "on_top" and (transport != "direct" or None in (precise, yaw_precise)):
            raise ValueError("place(relation='on_top') needs transport='direct', precise and yaw_precise")
        if feedback not in ("option", "yaw"):
            raise ValueError(f"place feedback must be 'option' or 'yaw', got {feedback!r}")
        super().__init__(obj)
        self.target, self.relation, self.transport = target, relation, transport
        self.lift, self.retreat, self.clearance = lift, retreat, clearance
        self.release_height, self.arc_margin = release_height, arc_margin
        self.release_seconds, self.settle_seconds = release_seconds, settle_seconds
        self.precise, self.yaw_precise, self.settle_steps = precise, yaw_precise, settle_steps
        self.feedback_rule = feedback

    def feedback(self, ctx):
        return _controls_yaw(ctx) if self.feedback_rule == "yaw" else super().feedback(ctx)

    def placement(self, ctx):
        """World position of the object centre when placed."""
        options = ctx.options
        box = ctx.state.objects[self.target]
        qx, qy, qz, qw = box.quat_xyzw
        rotation = rotation_matrix_wxyz((qw, qx, qy, qz))
        size = self.size(ctx)
        container = ctx.objects[self.target].params
        # The circular support bound admits any cube yaw. Centre placement keeps
        # the fixed finger away from the container rim during the lower phase.
        radius = np.linalg.norm(np.asarray(size[:2]) / 2)
        clear = np.asarray(container["inner_half_size"]) - radius - options.get("place_margin_m", self.clearance)
        if np.any(clear <= 0):
            raise ValueError("Task object cannot fit the container with the expert clearance")
        xy = np.zeros(2)
        if "place_xy" in options:
            xy = np.asarray(options["place_xy"], dtype=float)
            if xy.shape != (2,) or np.any(np.abs(xy) > clear):
                raise ValueError("Expert place_xy must lie inside the clear container interior")
        local = np.r_[xy, container["floor_z"] + size[2] / 2 + .0002]
        return np.asarray(box.position) + rotation @ local

    def validate(self, ctx):
        if self.relation == "on_top":
            return self._validate_on_top(ctx)
        options = ctx.options
        if {"transport_orientation_xyzw", "transport_roll_degrees"} & options.keys():
            raise ValueError("The expert keeps its pickup orientation during transport; remove transport roll overrides")
        self.placement(ctx)
        lift = float(options.get("lift_height_m", self.lift))
        release_height = float(options.get("release_height_m", self.release_height))
        if not math.isfinite(release_height) or not 0 <= release_height <= lift:
            raise ValueError("release_height_m must be between zero and lift_height_m")

    def plan(self, ctx):
        self.validate(ctx)
        if self.relation == "on_top":
            return self._plan_on_top(ctx)
        options = ctx.options
        normal = self.normal(ctx)
        placed = self.placement(ctx)
        approach = float(options.get("approach_height_m", self.retreat))
        lift = float(options.get("lift_height_m", self.lift))
        release_height = float(options.get("release_height_m", self.release_height))
        transport_duration = float(options.get("transport_seconds", self.duration(ctx)))
        opened, closed = self.gripper(ctx)
        source = self.position(ctx) + normal * lift
        destination = placed + normal * lift
        route = []
        if self.transport == "arc":
            if ctx.arm_base is None:
                raise ValueError("Pick-place expert transport requires the actual arm_base world pose")
            route = transport_arc(source, destination, np.asarray(ctx.arm_base.position), normal,
                                  float(options.get("transport_arc_margin_m", self.arc_margin)),
                                  interpolate_radius=bool(options.get("transport_interpolate_radius", False)))

        def point(name, position, opening, seconds=None):
            return self.waypoint(ctx, name, position, opening, seconds, object_goal=True)

        release_pose = placed + normal * release_height
        lower = () if release_height == lift else (point("lower", release_pose, closed),)
        retreat = placed + normal * max(approach, release_height)
        return (
            *(point(f"transport_arc_{index}", position, closed, transport_duration)
              for index, position in enumerate(route)),
            point("transport", destination, closed, float(options.get("final_transport_seconds", transport_duration))),
            *lower,
            point("release", release_pose, opened, float(options.get("release_seconds", self.release_seconds))),
            point("retreat", retreat, opened),
            point("settle", retreat, opened, float(options.get("settle_seconds", self.settle_seconds))),
        )

    # On top of another object ------------------------------------------------------------------------

    def _on_top_position(self, ctx):
        """World position of the object centre resting on the target, along the target's up axis."""
        support = ctx.state.objects[self.target]
        rotation = _rotation(support)
        half = np.asarray(self.size(ctx)) / 2
        support_half = np.asarray(self.size(ctx, self.target)) / 2
        height = np.abs((rotation.T @ _rotation(ctx.state.objects[self.obj]))[2]) @ half
        clearance = float(ctx.options.get("stack_clearance_m", self.clearance))
        return np.asarray(support.position) + rotation[:, 2] * (support_half[2] + height + clearance)

    def _validate_on_top(self, ctx):
        options = ctx.options
        if {"transport_roll_degrees", "transport_orientation_xyzw"} & options.keys():
            raise ValueError("Stack expert keeps the pickup orientation; transport roll overrides are unsupported")
        normal = np.asarray(ctx.table_normal, dtype=float)
        if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
            raise ValueError("Stack expert requires a finite table normal")
        clearance = float(options.get("stack_clearance_m", self.clearance))
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("stack_clearance_m must be finite and nonnegative")
        if min(float(options.get("approach_height_m", self.retreat)), float(options.get("lift_height_m", self.lift))) <= 0:
            raise ValueError("Approach and lift heights must be positive")
        self.gripper(ctx)
        self.orientation(ctx)
        fps = float(options.get("control_fps", 30.))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("control_fps must be positive")

    def _plan_on_top(self, ctx):
        options = ctx.options
        stacked = self._on_top_position(ctx)
        above = stacked + self.normal(ctx) * float(options.get("lift_height_m", self.lift))
        opened, closed = self.gripper(ctx)
        precise = min(float(options.get("position_tolerance_m", .004)), self.precise)
        if _controls_yaw(ctx):
            precise = min(precise, self.yaw_precise)
        settle = max(float(options.get("settle_seconds", self.settle_seconds)),
                     self.settle_steps / float(options.get("control_fps", 30.)))

        def point(name, position, opening, seconds=None, tolerance=None):
            return self.waypoint(ctx, name, position, opening, seconds, object_goal=True, tolerance=tolerance)

        return (
            point("transport", above, closed, float(options.get("transport_seconds", self.duration(ctx)))),
            point("lower", stacked, closed, tolerance=precise),
            point("release", stacked, opened, float(options.get("release_seconds", self.release_seconds)), precise),
            point("retreat", above, opened),
            point("settle", above, opened, settle),
        )


def place(obj, target, relation="inside", *, transport="arc", lift=.09, retreat=.06, clearance=.003,
          release_height=0., arc_margin=.01, release_seconds=.8, settle_seconds=1.5,
          precise=None, yaw_precise=None, settle_steps=0, feedback="option",
          rim=None, level=None, pitch=None, carry_height=None):
    if relation == "nested":
        from sim.actions.moves.rim import NestedPlace
        return NestedPlace(obj, target, level=level, pitch=pitch, clearance=clearance,
                           carry_height=carry_height, retreat=retreat, release_seconds=release_seconds,
                           settle_seconds=settle_seconds, rim=rim)
    return Place(obj, target, relation, transport=transport, lift=lift, retreat=retreat, clearance=clearance,
                 release_height=release_height, arc_margin=arc_margin, release_seconds=release_seconds,
                 settle_seconds=settle_seconds, precise=precise, yaw_precise=yaw_precise,
                 settle_steps=settle_steps, feedback=feedback)
