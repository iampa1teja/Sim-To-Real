"""Rim pickup geometry and nested placement, composed by approach/pick/place.

All dimensions, tolerances and timings come from the task's move declarations or expert options.
Imports remain simulator-free; scipy orientation math is loaded only while planning/controlling.
"""
from dataclasses import dataclass
import math

import numpy as np

from sim.actions.action import Move, Waypoint, gripper_commands
from utils.geometry import Pose, rotation_matrix_xyzw


@dataclass(frozen=True)
class Rim:
    radius: float
    height: float
    tilt_degrees: float
    correction_degrees: float
    width: float
    tolerance: float
    precise: float
    travel_tolerance: float
    close_error: str

    def frame(self, ctx, centre, side=1.):
        from scipy.spatial.transform import Rotation
        normal = Move.normal(ctx)
        radius = float(ctx.options.get('grasp_radius_m', self.radius))
        tilt = math.radians(float(ctx.options.get('tool_tilt_degrees', self.tilt_degrees)))
        fallback = tuple(ctx.options.get('orientation_xyzw', ctx.control_pose.quat_xyzw))
        jaw = ctx.arm is not None and ctx.arm.gripper.kind == 'jaw'
        if not jaw:
            axis = ctx.arm.gripper.closing_axis or (1., 0., 0.)
            direction = rotation_matrix_xyzw(fallback) @ axis
            direction -= normal * np.dot(direction, normal)
            direction /= np.linalg.norm(direction)
            return direction, fallback
        origin = np.asarray(ctx.arm_base.position)
        radial = np.asarray(centre) - origin
        radial -= normal * np.dot(radial, normal)
        distance = np.linalg.norm(radial)
        if distance <= radius:
            raise ValueError(self.close_error)
        radial /= distance
        tangent = np.cross(radial, normal)
        ratio = radius / distance
        closing = tangent * math.sqrt(1 - ratio * ratio) - radial * ratio * side
        tcp_radial = np.asarray(centre) + closing * radius * side - origin
        tcp_radial -= normal * np.dot(tcp_radial, normal)
        tcp_radial /= np.linalg.norm(tcp_radial)
        z = normal * math.cos(tilt) - tcp_radial * math.sin(tilt)
        y = np.cross(z, closing)
        return closing, tuple(Rotation.from_matrix(np.column_stack((closing, y, z))).as_quat())

    def pickup(self, ctx, obj):
        source = np.asarray(ctx.state.objects[obj].position)
        side = float(ctx.options.get('rim_side', 1.))
        direction, orientation = self.frame(ctx, source, side)
        rim = source + direction * float(ctx.options.get('grasp_radius_m', self.radius)) * side + Move.normal(ctx) * float(
            ctx.options.get('grasp_height_m', self.height))
        # Keep the original clearance read (and its refusal for an incomplete object state).
        min(np.linalg.norm((rim - np.asarray(pose.position))[:2])
            for name, pose in ctx.state.objects.items() if name != obj)
        return rim, orientation, side

    def point(self, ctx, obj, label, position, opening, seconds=None, *, orientation=None,
              object_goal=False, tolerance=None, side=1.):
        if orientation is None:
            _, orientation = self.frame(ctx, position, side)
        seconds = float(ctx.options['move_seconds']) if seconds is None else seconds
        timeout = float(ctx.options['waypoint_timeout_seconds'])
        return Waypoint(obj + '_' + label, Pose.from_lists(position, orientation), duration_s=seconds,
                        timeout_s=max(timeout, seconds), tolerance_m=self.tolerance if tolerance is None else tolerance,
                        object_ref=obj if object_goal else None, **opening)

    def control_orientation(self, waypoint, state, control_pose, nominal, ctx, fraction, cache):
        from scipy.spatial.transform import Rotation
        if not ctx.options.get('upright_object_feedback', True):
            return nominal
        name = waypoint.object_ref
        if name is None or waypoint.name.endswith('_lift'):
            return nominal
        if not state.signals.get('holding', False):
            return cache.get(name, nominal)
        normal = np.asarray(ctx.table_normal)
        normal /= np.linalg.norm(normal)
        up = rotation_matrix_xyzw(state.objects[name].quat_xyzw)[:, 2]
        axis = np.cross(up, normal)
        sine = np.linalg.norm(axis)
        angle = math.atan2(sine, float(np.dot(up, normal)))
        if sine < 1e-8:
            orientation = np.asarray(control_pose.quat_xyzw)
        else:
            correction = Rotation.from_rotvec(axis / sine * min(angle, math.radians(self.correction_degrees)) * fraction)
            orientation = (correction * Rotation.from_quat(control_pose.quat_xyzw)).as_quat()
        cache[name] = orientation
        return orientation


class RimMove(Move):
    def __init__(self, obj, rim):
        super().__init__(obj)
        self.group, self.rim = obj, rim

    def control_orientation(self, *args):
        return self.rim.control_orientation(*args)

    def gripper(self, ctx):
        return gripper_commands(ctx, self.rim.width)


class RimApproach(RimMove):
    replan = True

    def __init__(self, obj, height, rim):
        super().__init__(obj, rim)
        self.height = height

    def plan(self, ctx):
        point, orientation, _ = self.rim.pickup(ctx, self.obj)
        opened, _ = self.gripper(ctx)
        position = point + self.normal(ctx) * float(ctx.options.get('approach_height_m', self.height))
        return (self.rim.point(ctx, self.obj, 'approach', position, opened, orientation=orientation),)


class RimPick(RimMove):
    def __init__(self, obj, lift, grasp_seconds, rim):
        super().__init__(obj, rim)
        self.lift, self.grasp_seconds = lift, grasp_seconds

    def plan(self, ctx):
        point, orientation, side = self.rim.pickup(ctx, self.obj)
        opened, closed = self.gripper(ctx)
        lifted = self.position(ctx) + self.normal(ctx) * float(ctx.options.get('lift_height_m', self.lift))
        return (self.rim.point(ctx, self.obj, 'descend', point, opened, orientation=orientation),
                self.rim.point(ctx, self.obj, 'grasp', point, closed, self.grasp_seconds, orientation=orientation),
                self.rim.point(ctx, self.obj, 'lift', lifted, closed, object_goal=True, side=side))


class NestedPlace(RimMove):
    def __init__(self, obj, target, *, level, pitch, clearance, carry_height, retreat, release_seconds, settle_seconds, rim):
        super().__init__(obj, rim)
        self.target, self.level, self.pitch = target, level, pitch
        self.clearance, self.carry_height, self.retreat = clearance, carry_height, retreat
        self.release_seconds, self.settle_seconds = release_seconds, settle_seconds

    def plan(self, ctx):
        options, normal = ctx.options, self.normal(ctx)
        target = self.position(ctx, self.target)
        destination = target + normal * (self.level * float(options.get('nesting_pitch_m', self.pitch))
                                         + float(options.get('release_clearance_m', self.clearance)))
        carried = destination + normal * np.dot(target + normal * float(
            options.get('carry_height_above_support_m', self.carry_height)) - destination, normal)
        retreat = destination + normal * float(options.get('retreat_height_m', self.retreat))
        opened, closed = self.gripper(ctx)
        side = float(options.get('rim_side', 1.))
        def point(label, position, opening, seconds=None, tolerance=None):
            return self.rim.point(ctx, self.obj, label, position, opening, seconds, object_goal=True,
                                  tolerance=tolerance, side=side)
        return (point('transport', carried, closed, tolerance=self.rim.travel_tolerance),
                point('lower', destination, closed, tolerance=self.rim.precise),
                point('release', destination, opened, self.release_seconds, self.rim.precise),
                point('retreat', retreat, opened, tolerance=self.rim.travel_tolerance),
                point('settle', retreat, opened, self.settle_seconds, self.rim.travel_tolerance))
