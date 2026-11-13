"""Physical rim grasps and two placements using native arm IK and contact feedback."""
import math
import numpy as np
from core.tasks.base import Pose
from core.tasks.expert import Waypoint, gripper_commands
from core.scenes.anchors import quaternion_matrix


class BowlsExpert:
    def __init__(self, task):
        self.task = task

    def plan(self, context):
        self._waypoint_overrides = {}
        self._orientation_cache = {}
        return self._make_plan(context)

    def adapt_waypoint(self, waypoint, context):
        if waypoint.name.endswith('_approach'):
            name = waypoint.name.removesuffix('_approach')
            # A free support bowl can shift during the first placement. Freeze a
            # fresh placement plan before the next pickup instead of chasing it.
            self._waypoint_overrides.update({p.name:p for p in self._make_plan(context)
                                            if p.name.startswith(name+'_')})
        return self._waypoint_overrides.get(waypoint.name, waypoint)

    def control_orientation(self, waypoint, state, control_pose, nominal, context, fraction):
        from scipy.spatial.transform import Rotation
        if not context.options.get('upright_object_feedback', True):
            return nominal
        name = waypoint.object_ref
        if name is None or waypoint.name.endswith('_lift'):
            return nominal
        if not state.signals.get('holding', False):
            return self._orientation_cache.get(name, nominal)
        normal = np.asarray(context.table_normal)
        normal /= np.linalg.norm(normal)
        up = quaternion_matrix(state.objects[name].quat_xyzw)[:,2]
        axis = np.cross(up, normal)
        sine = np.linalg.norm(axis)
        angle = math.atan2(sine, float(np.dot(up, normal)))
        if sine < 1e-8:
            orientation = np.asarray(control_pose.quat_xyzw)
        else:
            correction = Rotation.from_rotvec(axis/sine*min(angle, math.radians(8.))*fraction)
            orientation = (correction*Rotation.from_quat(control_pose.quat_xyzw)).as_quat()
        self._orientation_cache[name] = orientation
        return orientation

    def _make_plan(self, context):
        from scipy.spatial.transform import Rotation
        options = context.options
        order = options.get('order', ['yellow_bowl', 'red_bowl', 'blue_bowl'])
        if len(order) != 3 or set(order) != set(self.task.spec.tracked_objects):
            raise ValueError('order must contain each of the three bowls exactly once')
        if context.arm_base is None:
            raise ValueError('Rim planning requires the actual robot base pose')
        normal = np.asarray(context.table_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        origin = np.asarray(context.arm_base.position)
        radius = float(options.get('grasp_radius_m', .058))
        rim_height = float(options.get('grasp_height_m', .012))
        lift = float(options.get('lift_height_m', .07))
        nesting = float(options.get('nesting_pitch_m', .022))
        duration = float(options.get('move_seconds', 2.2))
        timeout = float(options.get('waypoint_timeout_seconds', 7.))
        tilt = math.radians(float(options.get('tool_tilt_degrees', 21.)))
        opened, closed = gripper_commands(context, .004)
        target = np.asarray(context.state.objects[order[0]].position)
        jaw = context.arm is not None and context.arm.gripper.kind == 'jaw'
        fallback_orientation = tuple(options.get('orientation_xyzw', context.control_pose.quat_xyzw))

        def grasp_frame(centre, side=1.):
            if not jaw:
                axis = context.arm.gripper.closing_axis or (1.,0.,0.)
                direction = quaternion_matrix(fallback_orientation) @ axis
                direction -= normal*np.dot(direction,normal)
                direction /= np.linalg.norm(direction)
                return direction, fallback_orientation
            radial = np.asarray(centre)-origin
            radial -= normal*np.dot(radial, normal)
            distance = np.linalg.norm(radial)
            if distance <= radius:
                raise ValueError('Bowl rim is too close to the arm base')
            radial /= distance
            # SO101 closes tangentially to its pan direction. Solve for the rim
            # point whose normal is tangent at the TCP, rather than at the bowl
            # centre. This avoids pinching at an oblique angle on wider bowls.
            tangent = np.cross(radial, normal)
            ratio = radius/distance
            closing = tangent*math.sqrt(1-ratio*ratio)-radial*ratio*side
            tcp_radial = np.asarray(centre)+closing*radius*side-origin
            tcp_radial -= normal*np.dot(tcp_radial,normal)
            tcp_radial /= np.linalg.norm(tcp_radial)
            z = normal*math.cos(tilt)-tcp_radial*math.sin(tilt)
            y = np.cross(z, closing)
            orientation = tuple(Rotation.from_matrix(np.column_stack((closing,y,z))).as_quat())
            return closing, orientation

        def point(label, p, opening, seconds=duration, object_ref=None, orientation=None, precise=False, side=1., tolerance_m=None):
            if orientation is None:
                _, orientation = grasp_frame(p,side)
            return Waypoint(label, Pose.from_lists(p, orientation), duration_s=seconds,
                timeout_s=max(timeout, seconds), tolerance_m=tolerance_m if tolerance_m is not None else (.006 if precise else .005),
                object_ref=object_ref, **opening)

        plan = []
        for level, name in enumerate(order[1:], 1):
            source = np.asarray(context.state.objects[name].position)
            candidates = []
            for side in (float(options.get('rim_side', 1.)),):
                direction, orientation = grasp_frame(source, side)
                rim = source + direction*radius*side + normal*rim_height
                clearance = min(np.linalg.norm((rim-np.asarray(pose.position))[:2])
                                for other,pose in context.state.objects.items() if other != name)
                candidates.append((clearance,side,rim,orientation))
            _, side, rim, orientation = max(candidates, key=lambda x:x[0])
            destination = target + normal*(level*nesting+float(options.get('release_clearance_m', .012)))
            # Carry at the proven pickup lift height; adding stack height again
            # can push a five-axis arm past its wrist limit on the second bowl.
            carried = destination + normal*np.dot(target+normal*float(options.get('carry_height_above_support_m', .105))-destination, normal)
            plan += [point(name+'_approach', rim+normal*float(options.get('approach_height_m', .035)), opened, orientation=orientation),
                     point(name+'_descend', rim, opened, orientation=orientation),
                     point(name+'_grasp', rim, closed, .8, orientation=orientation),
                     point(name+'_lift', source+normal*lift, closed, object_ref=name, side=side),
                     point(name+'_transport', carried, closed, object_ref=name, side=side, tolerance_m=.012),
                     point(name+'_lower', destination, closed, object_ref=name, precise=True, side=side),
                     point(name+'_release', destination, opened, .8, object_ref=name, precise=True, side=side),
                     point(name+'_retreat', destination+normal*float(options.get('retreat_height_m', .04)), opened, object_ref=name, side=side, tolerance_m=.012),
                     point(name+'_settle', destination+normal*float(options.get('retreat_height_m', .04)), opened, 1.4, object_ref=name, side=side, tolerance_m=.012)]
        return tuple(plan)


def create(task):
    return BowlsExpert(task)
