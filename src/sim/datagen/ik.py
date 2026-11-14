"""Library IK (DifferentialIKController) for every arm, with Newton offline FK/Jacobians.

This module only expresses frames, applies the TCP offset, caps each step and
clamps to declared limits. All kinematics math comes from the libraries.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
import torch


@dataclass(frozen=True)
class IKOptions:
    damping: float = 0.0125  # the library consumes the unscaled task Jacobian
    orientation_weight: float = 0.05  # weight on each declared base-frame axis
    max_position_step_m: float = 0.01
    max_orientation_step_rad: float = math.pi  # let weighted axes correct tilt during large free yaw
    max_joint_step_rad: float = 0.08
    max_joint_step_m: float = 0.005
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = 0.03
    integral_gain: float = 1.0
    max_integral_bias_rad: float = 0.08
    max_integral_bias_m: float = 0.005

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for key, value in vars(self).items()
               if key != 'integral_gain') or not math.isfinite(self.integral_gain) or self.integral_gain < 0:
            raise ValueError('IK options must be finite and positive')


@dataclass(frozen=True)
class IKResult:
    joint_positions: torch.Tensor  # (N,J), full manifest action order, radians/metres
    position_error: torch.Tensor  # (N,), metres, before this command
    orientation_error: torch.Tensor  # (N,), radians, declared components only
    converged: torch.Tensor  # (N,), measured pose satisfies both declared tolerances
    limited: torch.Tensor  # (N,), joint-step or runtime limits bounded this command


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, 'torch') else torch.as_tensor(value)


def _arm_bounds(arm, options):
    """Manifest-order arm IDs and limits, in simulator radians/metres."""
    bounds = {key: [] for key in ('ids', 'step_caps', 'bias_caps', 'lower', 'upper')}
    for index, joint in enumerate(arm.action_joints):
        if joint.role != 'arm':
            continue
        revolute = joint.type == 'revolute'
        bounds['ids'].append(index)
        bounds['step_caps'].append(options.max_joint_step_rad if revolute else options.max_joint_step_m)
        bounds['bias_caps'].append(options.max_integral_bias_rad if revolute else options.max_integral_bias_m)
        bounds['lower'].append(math.radians(joint.limits[0]) if revolute else joint.limits[0])
        bounds['upper'].append(math.radians(joint.limits[1]) if revolute else joint.limits[1])
    if not bounds['ids']:
        raise ValueError('IK requires at least one joint with role arm')
    return bounds


def _controller_cfg(arm, options, *, use_orientation):
    from isaaclab.controllers import DifferentialIKControllerCfg

    return DifferentialIKControllerCfg(
        command_type='pose' if use_orientation and arm.ik.orientation_axes else 'position',
        use_relative_mode=False,
        ik_method=arm.ik.method,
        ik_params={'lambda_val' if arm.ik.method == 'dls' else 'lambda_min': options.damping},
        orientation_weight=tuple(options.orientation_weight if axis in arm.ik.orientation_axes else 0.
                                 for axis in ('x', 'y', 'z')),
        joint_limit_avoidance_gain=arm.ik.joint_limit_avoidance_gain,
        joint_limit_avoidance_margin=arm.ik.joint_limit_avoidance_margin,
    )


class _JointStep:
    """A bounded library IK step on poses/Jacobians expressed in the arm base."""

    def __init__(self, arm, options, num_envs, device):
        self.arm, self.options = arm, options
        self.num_envs, self.device = num_envs, device
        self.bounds = _arm_bounds(arm, options)
        self._controllers = {}

    def solve(self, q, ee_pos, ee_quat, jacobian, target_pos, target_quat=None):
        from isaaclab.controllers import DifferentialIKController
        from isaaclab.utils.math import compute_pose_error, quat_from_angle_axis, quat_mul

        q = _tensor(q)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if not q.is_floating_point():
            q = q.to(torch.float64)
        count = len(self.arm.action_joints)
        if q.ndim != 2 or q.shape[1] != count or not torch.isfinite(q).all():
            raise ValueError(f'IK joint_positions must have finite shape (N,{count})')
        if q.shape[0] != self.num_envs:
            raise ValueError(f'IK joint_positions must have finite shape (N,{count})')
        inputs = {}
        for label, value, width in (('position', ee_pos, 3), ('target_position', target_pos, 3),
                                    ('orientation', ee_quat, 4),
                                    *([('target_orientation', target_quat, 4)] if target_quat is not None else [])):
            value = torch.as_tensor(value, dtype=q.dtype, device=q.device)
            if value.shape == (width,):
                value = value.unsqueeze(0).expand(q.shape[0], -1)
            if value.shape != (q.shape[0], width) or not torch.isfinite(value).all():
                raise ValueError(f'{label} must contain {width} finite values per environment')
            if width == 4:
                norm = value.norm(dim=-1, keepdim=True)
                if (norm < 1e-12).any():
                    raise ValueError(f'{label} quaternion must be nonzero')
                value = value / norm
            inputs[label] = value
        position, orientation = inputs['position'], inputs['orientation']
        target_position = inputs['target_position']
        jacobian = torch.as_tensor(jacobian, dtype=q.dtype, device=q.device)
        if jacobian.ndim == 2:
            jacobian = jacobian.unsqueeze(0)
        if jacobian.shape != (q.shape[0], 6, count) or not torch.isfinite(jacobian).all():
            raise ValueError(f'IK jacobian must have finite shape (N,6,{count})')

        error = target_position - position
        position_error = error.norm(dim=-1)
        command_position = position + error * (
            self.options.max_position_step_m / position_error.clamp_min(self.options.max_position_step_m)
        ).unsqueeze(-1)
        axes = [index for index, axis in enumerate(('x', 'y', 'z')) if axis in self.arm.ik.orientation_axes]
        use_orientation = target_quat is not None and bool(axes)
        orientation_error = torch.zeros_like(position_error)
        if use_orientation:
            _, rotation_error = compute_pose_error(position, orientation, target_position,
                                                   inputs['target_orientation'], rot_error_type='axis_angle')
            orientation_error = rotation_error[:, axes].norm(dim=-1)
            angle = rotation_error.norm(dim=-1)
            increment = quat_from_angle_axis(angle.clamp_max(self.options.max_orientation_step_rad),
                                             rotation_error / angle.clamp_min(1e-12).unsqueeze(-1))
            command_orientation = quat_mul(increment, orientation)
            command = torch.cat((command_position, command_orientation), dim=-1)
        else:
            command = command_position
        if use_orientation not in self._controllers:
            controller = DifferentialIKController(_controller_cfg(self.arm, self.options,
                                                                  use_orientation=use_orientation),
                                                   self.num_envs, self.device)
            if self.arm.ik.joint_limit_avoidance_gain > 0:
                controller.set_joint_pos_limits(q.new_tensor(self.bounds['lower']), q.new_tensor(self.bounds['upper']))
            self._controllers[use_orientation] = controller
        controller = self._controllers[use_orientation]
        ids = self.bounds['ids']
        controller.set_command(command.float(), ee_quat=orientation.float())
        new = controller.compute(position.float(), orientation.float(), jacobian[:, :, ids].float(),
                                 q[:, ids].float()).to(q.dtype)
        delta = new - q[:, ids]
        caps = q.new_tensor(self.bounds['step_caps'])
        scale = (delta.abs() / caps).amax(dim=-1, keepdim=True).clamp_min(1.)
        proposed = q[:, ids] + delta / scale
        clipped = proposed.clamp(q.new_tensor(self.bounds['lower']), q.new_tensor(self.bounds['upper']))
        result = q.clone()
        result[:, ids] = clipped
        limited = (scale[:, 0] > 1.) | (clipped != proposed).any(dim=-1)
        return IKResult(result, position_error, orientation_error,
                        (position_error <= self.options.position_tolerance_m)
                        & (orientation_error <= self.options.orientation_tolerance_rad), limited)


class ArticulationIK:
    """Adapt live link-origin Jacobians to the base-frame library controller."""

    def __init__(self, robot, arm, *, position_offset_m=None, options=None, control_dt=1 / 30):
        self.robot, self.arm = robot, arm
        self.options = options or IKOptions()
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError('IK control_dt must be finite and positive')
        self.control_dt = control_dt
        self._integral_bias = None
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
                                            else position_offset_m, dtype=float)
        if self.position_offset_m.shape != (3,) or not np.isfinite(self.position_offset_m).all():
            raise ValueError('IK position_offset_m must contain three finite metre values')
        self.joint_ids, names = robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)
        if tuple(names) != arm.action_usd_joint_names:
            raise ValueError('IK articulation joints do not match the manifest order')
        ee, _ = robot.find_bodies(arm.links.ee)
        base, _ = robot.find_bodies(arm.links.base)
        if len(ee) != 1 or len(base) != 1:
            raise ValueError('IK requires exactly one declared ee and base body')
        self.ee_id, self.base_id = ee[0], base[0]
        self.jacobian_body_id = self.ee_id - (1 if robot.is_fixed_base else 0)
        if self.jacobian_body_id < 0:
            raise ValueError('IK ee must have a Jacobian row; it cannot be the fixed root')
        self._step = _JointStep(arm, self.options, _tensor(robot.data.joint_pos).shape[0], robot.device)

    def current_pose(self):
        from isaaclab.utils.math import quat_apply

        positions = _tensor(self.robot.data.body_link_pos_w)[:, self.ee_id]
        orientations = _tensor(self.robot.data.body_link_quat_w)[:, self.ee_id]
        offset = positions.new_tensor(self.position_offset_m).expand_as(positions)
        return positions + quat_apply(orientations, offset), orientations

    def reset(self):
        """Clear learned command bias after an episode reset/warmup."""
        self._integral_bias = None

    @property
    def integral_bias(self):
        return None if self._integral_bias is None else self._integral_bias.clone()

    def solve(self, position_world, orientation_xyzw=None, *, advance=True):
        """Return a bounded command; previews do not advance integral load bias."""
        from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = quat_apply(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_position = _tensor(self.robot.data.body_link_pos_w)[:, self.base_id]
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        targets = []
        for label, value, width in (('target_position', position_world, 3),
                                    *([('target_orientation', orientation_xyzw, 4)]
                                      if orientation_xyzw is not None else [])):
            target = torch.as_tensor(value, dtype=position.dtype, device=position.device)
            if target.shape == (width,):
                target = target.unsqueeze(0).expand(position.shape[0], -1)
            if target.shape != (position.shape[0], width) or not torch.isfinite(target).all():
                raise ValueError(f'{label} must contain {width} finite values per environment')
            if width == 4:
                norm = target.norm(dim=-1, keepdim=True)
                if (norm < 1e-12).any():
                    raise ValueError(f'{label} quaternion must be nonzero')
                target = target / norm
            targets.append(target)
        position_base, orientation_base = subtract_frame_transforms(base_position, base_orientation,
                                                                    position, orientation)
        target_base, target_orientation_base = subtract_frame_transforms(
            base_position, base_orientation, targets[0], targets[1] if len(targets) == 2 else None)
        rotation = base_orientation[:, None, :].expand(-1, len(columns), -1)
        jacobian[:, :3] = quat_apply_inverse(rotation, jacobian[:, :3].transpose(1, 2)).transpose(1, 2)
        jacobian[:, 3:] = quat_apply_inverse(rotation, jacobian[:, 3:].transpose(1, 2)).transpose(1, 2)
        solved = self._step.solve(joint_positions, position_base, orientation_base, jacobian,
                                   target_base, target_orientation_base if len(targets) == 2 else None)
        if self.options.integral_gain == 0:
            return solved
        return self._compensate_load(solved, joint_positions, advance=advance)

    def _compensate_load(self, solved, measured, *, advance):
        bounds = _arm_bounds(self.arm, self.options)
        ids = bounds['ids']
        delta = solved.joint_positions[:, ids] - measured[:, ids]
        previous = (torch.zeros_like(measured) if self._integral_bias is None
                    else self._integral_bias.to(measured))
        bias = previous[:, ids]
        step_caps = measured.new_tensor(bounds['step_caps'])
        bias_caps = measured.new_tensor(bounds['bias_caps'])
        lower = measured.new_tensor(bounds['lower'])
        upper = measured.new_tensor(bounds['upper'])
        if advance:
            proposed = (bias + self.options.integral_gain * self.control_dt * delta).clamp(-bias_caps, bias_caps)
            increment = proposed - bias
            total = delta + proposed
            # Freeze integration that would drive an already lead-limited command farther outward.
            blocked = ((total > step_caps) & (increment > 0)) | ((total < -step_caps) & (increment < 0))
            bias = torch.where(blocked, bias, proposed)
            # Back-project at hard runtime limits; retained bias must not wind up beyond a stop.
            bias = bias.clamp(lower - measured[:, ids] - delta, upper - measured[:, ids] - delta)
            stored = torch.zeros_like(measured)
            stored[:, ids] = bias
            self._integral_bias = stored.detach().clone()
        total = delta + bias
        bounded = total.clamp(-step_caps, step_caps)
        command = (measured[:, ids] + bounded).clamp(lower, upper)
        result = measured.clone()
        result[:, ids] = command
        limited = solved.limited | (bounded != total).any(dim=-1) | (command != measured[:, ids] + bounded).any(dim=-1)
        return IKResult(result, solved.position_error, solved.orientation_error, solved.converged, limited)


class ArmKinematics:
    """Newton FK and Jacobians for command poses, without simulator stepping.

    Coordinates and Jacobian columns retain manifest action order. Newton owns
    USD parsing and kinematics; this adapter only expresses the declared base
    frame, applies the TCP offset and bounds the final dispatched command.
    """

    @classmethod
    def from_arm(cls, arm, *, position_offset_m=None):
        import newton

        builder = newton.ModelBuilder()
        try:
            metadata = builder.add_usd(str(arm.usd_path), floating=False)
        except ValueError as error:
            if 'Reversed joints are not supported:' not in str(error):
                raise
            raise ValueError(
                f'{error} To import {arm.usd_path}, '
                'swap body0/body1 and the local frames in the USD.') from error
        model = builder.finalize(device='cpu')
        return cls(model, arm, position_offset_m=position_offset_m,
                   _meters_per_unit=metadata['linear_unit'])

    def __init__(self, model, arm, *, position_offset_m=None, _meters_per_unit=1.):
        import newton
        import warp as wp
        from isaaclab.utils.math import make_pose, matrix_from_quat

        self.arm, self._model = arm, model
        self._meters_per_unit = float(_meters_per_unit)
        if not math.isfinite(self._meters_per_unit) or self._meters_per_unit <= 0:
            raise ValueError('USD FK meters_per_unit must be finite and positive')
        # Newton 1.6.1 imports rigid assets in authored stage units. Convert only
        # at the library boundary; public coordinates and TCP offsets stay SI.
        self._coordinate_units = np.asarray([1. if joint.type == 'revolute' else self._meters_per_unit
                                             for joint in arm.action_joints])
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
                                            else position_offset_m, dtype=float)
        if self.position_offset_m.shape != (3,) or not np.isfinite(self.position_offset_m).all():
            raise ValueError('IK position_offset_m must contain three finite metre values')
        if model.articulation_count != 1:
            raise ValueError('USD FK requires exactly one articulation')

        def named_index(labels, name, label):
            matches = [index for index, value in enumerate(labels) if value.rsplit('/', 1)[-1] == name]
            if len(matches) != 1:
                raise ValueError(f'USD FK requires one declared {label} {name!r}')
            return matches[0]

        self.base_id = named_index(model.body_label, arm.links.base, 'base body')
        self.ee_id = named_index(model.body_label, arm.links.ee, 'ee body')
        joint_ids = [named_index(model.joint_label, name, 'joint') for name in arm.action_usd_joint_names]
        coordinate_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        joint_types = model.joint_type.numpy()
        for manifest_joint, joint_id in zip(arm.action_joints, joint_ids):
            expected_type = (newton.JointType.REVOLUTE if manifest_joint.type == 'revolute'
                             else newton.JointType.PRISMATIC)
            if (joint_types[joint_id] != expected_type
                    or coordinate_starts[joint_id + 1] - coordinate_starts[joint_id] != 1
                    or dof_starts[joint_id + 1] - dof_starts[joint_id] != 1):
                raise ValueError(f'USD FK joint {manifest_joint.usd!r} differs from its declared joint type')
        self._coordinate_ids = coordinate_starts[joint_ids].astype(int)
        self._dof_ids = dof_starts[joint_ids].astype(int)
        action_indices = {joint.name: index for index, joint in enumerate(arm.action_joints)}
        self._mimic_coordinates = []
        for joint in arm.joints:
            if joint.mimic is not None:
                index = named_index(model.joint_label, joint.usd, 'mimic joint')
                unit = 1. if joint.type == 'revolute' else self._meters_per_unit
                self._mimic_coordinates.append((int(coordinate_starts[index]),
                    action_indices[joint.mimic.joint], joint.mimic.multiplier, joint.mimic.offset, unit))

        # Newton computes exact structural dependencies here; no IK optimizer is
        # constructed or run. This remains reliable at singular configurations.
        dependencies = []
        for body_id in (self.base_id, self.ee_id):
            objective = newton.ik.IKObjectivePosition(
                body_id, wp.vec3(), wp.zeros(1, dtype=wp.vec3, device=model.device))
            objective.set_batch_layout(3, 0, 1)
            objective.bind_device(model.device)
            objective.init_buffers(model, newton.ik.IKJacobianType.ANALYTIC)
            dependencies.append(objective.affects_dof.numpy().astype(bool))
        base_dependencies, ee_dependencies = dependencies
        if np.any(base_dependencies & ~ee_dependencies):
            raise ValueError('USD FK requires the declared base to precede the ee')
        declared = np.zeros(model.joint_dof_count, dtype=bool)
        declared[self._dof_ids] = True
        missing_dofs = np.flatnonzero(ee_dependencies & ~base_dependencies & ~declared)
        if missing_dofs.size:
            joint_id = int(np.searchsorted(dof_starts, missing_dofs[0], side='right') - 1)
            name = model.joint_label[joint_id].rsplit('/', 1)[-1]
            raise ValueError(f'USD FK joint {name!r} must be fixed or declared in arm.action_joints')

        child_joints = np.flatnonzero(model.joint_child.numpy() == self.ee_id)
        if child_joints.size != 1:
            raise ValueError('USD FK ee must have exactly one incoming joint')
        articulation_start = int(model.articulation_start.numpy()[0])
        self._jacobian_row = 6 * (int(child_joints[0]) - articulation_start)
        self._state = model.state()
        self._joint_q = self._state.joint_q
        self._joint_qd = self._state.joint_qd
        self._default_q = model.joint_q.numpy().copy()

        # Model initialization poses are the authored USD frames returned by
        # Newton's importer, before evaluating the declared joints at zero.
        position, orientation = self._tcp_pose(model.body_q.numpy())
        expected = make_pose(position, matrix_from_quat(orientation)).numpy()
        if not np.allclose(self.fk(np.zeros(len(arm.action_joints))), expected, atol=2e-5, rtol=0):
            raise ValueError('USD FK zero pose disagrees with authored base/ee transforms')

    def _tcp_pose(self, body_poses):
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        poses = torch.as_tensor(body_poses, dtype=torch.float64)
        base, ee = poses[self.base_id], poses[self.ee_id]
        position, orientation = subtract_frame_transforms(
            base[:3] * self._meters_per_unit, base[3:], ee[:3] * self._meters_per_unit, ee[3:])
        offset = position.new_tensor(self.position_offset_m)
        return position + quat_apply(orientation, offset), orientation

    def _evaluate(self, joint_positions):
        import newton

        q = np.asarray(joint_positions, dtype=float)
        if q.shape != (len(self.arm.action_joints),) or not np.isfinite(q).all():
            raise ValueError(f'FK requires {len(self.arm.action_joints)} finite action-order radians/metres')
        coordinates = self._default_q.copy()
        coordinates[self._coordinate_ids] = q / self._coordinate_units
        for coordinate, source, multiplier, offset, unit in self._mimic_coordinates:
            coordinates[coordinate] = (q[source] * multiplier + offset) / unit
        # eval_jacobian reads state.joint_q, so FK and Jacobian share this buffer.
        self._joint_q.assign(coordinates)
        newton.eval_fk(self._model, self._joint_q, self._joint_qd, self._state)
        return self._state.body_q.numpy()

    def fk(self, joint_positions):
        from isaaclab.utils.math import make_pose, matrix_from_quat

        position, orientation = self._tcp_pose(self._evaluate(joint_positions))
        return make_pose(position, matrix_from_quat(orientation)).numpy()

    def jacobian(self, joint_positions):
        """Return the TCP Jacobian; rows are base-frame linear/angular velocity.

        Columns retain manifest order and use radians/metres. Gripper roles have
        zero columns because a Cartesian arm command must preserve their targets.
        """
        import newton
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        poses = torch.as_tensor(self._evaluate(joint_positions), dtype=torch.float64)
        ee, base = poses[self.ee_id], poses[self.base_id]
        jacobian = newton.eval_jacobian(self._model, self._state).numpy()
        rows = jacobian[0, self._jacobian_row:self._jacobian_row + 6]
        result = torch.as_tensor(rows[:, self._dof_ids], dtype=torch.float64).clone()
        result[:3] *= self._meters_per_unit
        com = ee.new_tensor(self._model.body_com.numpy()[self.ee_id]) * self._meters_per_unit
        offset_world = quat_apply(ee[3:], ee.new_tensor(self.position_offset_m) - com)
        result[:3] += torch.cross(result[3:].T, offset_world.expand(len(self.arm.action_joints), -1), dim=-1).T
        for half in (slice(0, 3), slice(3, 6)):
            result[half] = quat_apply_inverse(base[3:], result[half].T).T
        result /= result.new_tensor(self._coordinate_units)
        gripper_ids = [index for index, joint in enumerate(self.arm.action_joints) if joint.role != 'arm']
        result[:, gripper_ids] = 0
        return result.numpy()

    def inverse(self, target_pose_base, measured_joint_positions, *, options=None, max_iterations=12,
                position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4):
        """Invert a commanded TCP pose locally, without physics or integral feedback.

        The target is a rigid base-frame 4x4 transform. Iteration starts at the
        measured manifest-order radians/metres. The dispatched command respects
        the original measurement's joint-step window and declared limits, while
        gripper joints remain exactly measured. Returned errors describe that
        final bounded command, using only the declared orientation axes.
        """
        from isaaclab.utils.math import compute_pose_error, quat_from_matrix

        if (not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1
                or any(not math.isfinite(value) or value <= 0
                       for value in (position_tolerance_m, orientation_tolerance_rad))):
            raise ValueError('Inverse iterations and pose tolerances must be positive')
        target = np.asarray(target_pose_base, dtype=float)
        if (target.shape != (4, 4) or not np.isfinite(target).all()
                or not np.allclose(target[3], [0., 0., 0., 1.], atol=1e-8, rtol=0)
                or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(torch.det(torch.from_numpy(target[:3, :3])).item(), 1., atol=1e-6, rtol=0)):
            raise ValueError('Inverse target must be a finite rigid 4x4 base-frame transform')
        measured = np.asarray(measured_joint_positions, dtype=float).copy()
        self.fk(measured)
        options = options or IKOptions()
        local_options = replace(options, position_tolerance_m=position_tolerance_m,
                                orientation_tolerance_rad=orientation_tolerance_rad)
        bounds = _arm_bounds(self.arm, options)
        ids = bounds['ids']
        step_caps = np.asarray(bounds['step_caps'])
        dispatch_lower = np.maximum(bounds['lower'], measured[ids] - step_caps)
        dispatch_upper = np.minimum(bounds['upper'], measured[ids] + step_caps)
        if np.any(dispatch_lower > dispatch_upper):
            raise ValueError('Measured arm joint lies beyond the runtime limit and permitted command step')
        step = _JointStep(self.arm, local_options, num_envs=1, device='cpu')
        target_position = torch.from_numpy(target[None, :3, 3].copy())
        target_orientation = quat_from_matrix(torch.from_numpy(target[None, :3, :3].copy()))
        q, limited = measured.copy(), False
        for _ in range(max_iterations):
            current = self.fk(q)
            solved = step.solve(torch.from_numpy(q[None].copy()),
                                torch.from_numpy(current[None, :3, 3].copy()),
                                quat_from_matrix(torch.from_numpy(current[None, :3, :3].copy())),
                                torch.from_numpy(self.jacobian(q)[None]),
                                target_position, target_orientation)
            limited |= bool(solved.limited.item())
            if solved.converged.item():
                break
            q = solved.joint_positions[0].numpy().copy()
        bounded = q.copy()
        bounded[ids] = np.clip(q[ids], dispatch_lower, dispatch_upper)
        limited |= bool(np.any(bounded != q))
        current = torch.from_numpy(self.fk(bounded)[None])
        position_error, rotation_error = compute_pose_error(
            current[:, :3, 3], quat_from_matrix(current[:, :3, :3]),
            target_position, target_orientation, rot_error_type='axis_angle')
        position_error = position_error.norm(dim=-1)
        axes = [{'x': 0, 'y': 1, 'z': 2}[axis] for axis in self.arm.ik.orientation_axes]
        orientation_error = rotation_error[:, axes].norm(dim=-1)
        return IKResult(torch.from_numpy(bounded[None]), position_error, orientation_error,
                        (position_error <= position_tolerance_m)
                        & (orientation_error <= orientation_tolerance_rad), torch.tensor([limited]))




def make_ik(robot, arm, **kwargs):
    """Construct the optional arm-owned IK backend, or the library adapter."""
    from utils.arms.spec import load_arm_utils

    hook = getattr(load_arm_utils(arm), 'make_ik', None)
    return hook(robot, arm, **kwargs) if hook is not None else ArticulationIK(robot, arm, **kwargs)


def make_kinematics(arm, **kwargs):
    """Construct the optional arm-owned command kinematics, or Newton FK."""
    from utils.arms.spec import load_arm_utils

    hook = getattr(load_arm_utils(arm), 'make_kinematics', None)
    return hook(arm, **kwargs) if hook is not None else ArmKinematics.from_arm(arm, **kwargs)
