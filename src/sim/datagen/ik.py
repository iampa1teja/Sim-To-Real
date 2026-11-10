"""Bounded native-Jacobian IK and side-effect-free USD forward kinematics.

Targets use world metres/XYZW. Orientation constraints are rotation-error components
in the declared arm-base frame, so a low-DOF arm need not promise a full 6-D pose.
Each solve returns one limited joint target; callers step physics before solving again.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class IKOptions:
    damping: float = 0.001
    orientation_weight: float = 0.05  # metres per radian in the weighted least-squares objective
    max_position_step_m: float = 0.01
    max_orientation_step_rad: float = 0.1
    max_joint_step_rad: float = 0.08
    max_joint_step_m: float = 0.005
    position_tolerance_m: float = 0.001
    orientation_tolerance_rad: float = 0.03

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError('IK options must be finite and positive')


@dataclass(frozen=True)
class IKResult:
    joint_positions: torch.Tensor  # (N,J), full manifest action order, radians/metres
    position_error: torch.Tensor  # (N,), metres, before this command
    orientation_error: torch.Tensor  # (N,), radians, declared components only
    converged: torch.Tensor  # (N,), measured pose satisfies both declared tolerances
    limited: torch.Tensor  # (N,), joint-step or runtime joint limits bounded this command


def _tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, 'torch') else torch.as_tensor(value)


def _batch(value, width, reference, label):
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.shape == (width,):
        value = value.unsqueeze(0).expand(reference.shape[0], -1)
    if value.shape != (reference.shape[0], width) or not torch.isfinite(value).all():
        raise ValueError(f'{label} must contain {width} finite values per environment')
    return value


def _quaternion(value, reference, label):
    value = _batch(value, 4, reference, label)
    norm = value.norm(dim=-1, keepdim=True)
    if (norm < 1e-12).any():
        raise ValueError(f'{label} quaternion must be nonzero')
    return value / norm


def _rotate(quaternion, vector):
    while quaternion.ndim < vector.ndim:
        quaternion = quaternion.unsqueeze(-2)
    xyz, scalar = quaternion[..., :3], quaternion[..., 3:4]
    uv = torch.cross(xyz.expand_as(vector), vector, dim=-1)
    return vector + 2 * (scalar * uv + torch.cross(xyz.expand_as(vector), uv, dim=-1))


def _inverse_rotate(quaternion, vector):
    conjugate = torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)
    return _rotate(conjugate, vector)


def _rotation_error(target, current):
    # target * conjugate(current), in world coordinates; choose the shortest quaternion arc.
    a, b = target[..., :3], -current[..., :3]
    w_a, w_b = target[..., 3:4], current[..., 3:4]
    vector = w_a * b + w_b * a + torch.cross(a, b, dim=-1)
    scalar = w_a * w_b - (a * b).sum(dim=-1, keepdim=True)
    sign = torch.where(scalar < 0, -1., 1.)
    vector, scalar = vector * sign, scalar * sign
    norm = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, scalar.clamp_min(0))
    return vector * torch.where(norm > 1e-8, angle / norm.clamp_min(1e-12), 2.)


def _bounded_norm(value, maximum):
    return value * (maximum / value.norm(dim=-1, keepdim=True).clamp_min(maximum))


def solve_damped_step(arm, joint_positions, position, orientation, jacobian, target_position,
                      target_orientation=None, *, base_orientation=(0., 0., 0., 1.), options=None):
    """Pure batched DLS step; Jacobian rows are world link/TCP linear then angular velocity.

    Columns are in manifest action order. Revolute columns act on radians, prismatic
    columns on metres. Gripper columns never enter the solve and their values stay exact.
    """
    options = options or IKOptions()
    q = _tensor(joint_positions)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if not q.is_floating_point():
        q = q.to(torch.float64)
    count = len(arm.joints)
    if q.ndim != 2 or q.shape[1] != count or not torch.isfinite(q).all():
        raise ValueError(f'IK joint_positions must have finite shape (N,{count})')
    current_position = _batch(position, 3, q, 'position')
    target_position = _batch(target_position, 3, q, 'target_position')
    current_orientation = _quaternion(orientation, q, 'orientation')
    base_orientation = _quaternion(base_orientation, q, 'base_orientation')
    jacobian = torch.as_tensor(jacobian, dtype=q.dtype, device=q.device)
    if jacobian.ndim == 2:
        jacobian = jacobian.unsqueeze(0)
    if jacobian.shape != (q.shape[0], 6, count) or not torch.isfinite(jacobian).all():
        raise ValueError(f'IK jacobian must have finite shape (N,6,{count})')
    arm_ids = [index for index, joint in enumerate(arm.joints) if joint.role == 'arm']
    if not arm_ids:
        raise ValueError('IK requires at least one joint with role arm')
    error = target_position - current_position
    position_error = error.norm(dim=-1)
    task_jacobian = jacobian[:, :3, arm_ids]
    task_error = _bounded_norm(error, options.max_position_step_m)
    orientation_error = torch.zeros_like(position_error)
    axes = [{'x': 0, 'y': 1, 'z': 2}[axis] for axis in arm.ik.orientation_axes]
    if target_orientation is not None and axes:
        target_orientation = _quaternion(target_orientation, q, 'target_orientation')
        error = _inverse_rotate(base_orientation, _rotation_error(target_orientation, current_orientation))[:, axes]
        orientation_error = error.norm(dim=-1)
        angular = _inverse_rotate(base_orientation, jacobian[:, 3:, arm_ids].transpose(1, 2)).transpose(1, 2)
        task_jacobian = torch.cat((task_jacobian, angular[:, axes] * options.orientation_weight), dim=1)
        task_error = torch.cat((task_error, _bounded_norm(error, options.max_orientation_step_rad)
                                * options.orientation_weight), dim=-1)
    step_limits = q.new_tensor([options.max_joint_step_rad if arm.joints[index].type == 'revolute'
                               else options.max_joint_step_m for index in arm_ids])
    scaled = task_jacobian * step_limits
    identity = torch.eye(scaled.shape[1], dtype=q.dtype, device=q.device)
    system = scaled @ scaled.transpose(1, 2) + options.damping ** 2 * identity
    normalized_step = (scaled.transpose(1, 2) @ torch.linalg.solve(system, task_error.unsqueeze(-1))).squeeze(-1)
    scale = normalized_step.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    delta = normalized_step / scale * step_limits
    lower = q.new_tensor([math.radians(arm.joints[index].limits[0]) if arm.joints[index].type == 'revolute'
                          else arm.joints[index].limits[0] for index in arm_ids])
    upper = q.new_tensor([math.radians(arm.joints[index].limits[1]) if arm.joints[index].type == 'revolute'
                          else arm.joints[index].limits[1] for index in arm_ids])
    proposed = q[:, arm_ids] + delta
    clipped = proposed.clamp(lower, upper)
    result = q.clone()
    result[:, arm_ids] = clipped
    limited = (scale[:, 0] > 1) | (clipped != proposed).any(dim=-1)
    return IKResult(result, position_error, orientation_error,
                    (position_error <= options.position_tolerance_m)
                    & (orientation_error <= options.orientation_tolerance_rad), limited)


class ArticulationIK:
    """Read Newton/Isaac Lab's native link-origin Jacobian without changing simulation state."""

    def __init__(self, robot, arm, *, position_offset_m=None, options=None):
        self.robot, self.arm = robot, arm
        self.options = options or IKOptions()
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
                                            else position_offset_m, dtype=float)
        if self.position_offset_m.shape != (3,) or not np.isfinite(self.position_offset_m).all():
            raise ValueError('IK position_offset_m must contain three finite metre values')
        self.joint_ids, names = robot.find_joints(list(arm.usd_joint_names), preserve_order=True)
        if tuple(names) != arm.usd_joint_names:
            raise ValueError('IK articulation joints do not match the manifest order')
        ee, _ = robot.find_bodies(arm.links.ee)
        base, _ = robot.find_bodies(arm.links.base)
        if len(ee) != 1 or len(base) != 1:
            raise ValueError('IK requires exactly one declared ee and base body')
        self.ee_id, self.base_id = ee[0], base[0]
        self.jacobian_body_id = self.ee_id - (1 if robot.is_fixed_base else 0)
        if self.jacobian_body_id < 0:
            raise ValueError('IK ee must have a Jacobian row; it cannot be the fixed root')

    def current_pose(self):
        positions = _tensor(self.robot.data.body_link_pos_w)[:, self.ee_id]
        orientations = _tensor(self.robot.data.body_link_quat_w)[:, self.ee_id]
        offset = positions.new_tensor(self.position_offset_m).expand_as(positions)
        return positions + _rotate(orientations, offset), orientations

    def solve(self, position_world, orientation_xyzw=None):
        position, orientation = self.current_pose()
        joint_positions = _tensor(self.robot.data.joint_pos)[:, self.joint_ids]
        columns = [index + self.robot.num_base_dofs for index in self.joint_ids]
        jacobian = _tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body_id, :, columns].clone()
        offset = _rotate(orientation, position.new_tensor(self.position_offset_m).expand_as(position))
        jacobian[:, :3] += torch.cross(jacobian[:, 3:].transpose(1, 2),
                                      offset[:, None, :].expand(-1, len(columns), -1), dim=-1).transpose(1, 2)
        base_orientation = _tensor(self.robot.data.body_link_quat_w)[:, self.base_id]
        return solve_damped_step(self.arm, joint_positions, position, orientation, jacobian,
                                 position_world, orientation_xyzw, base_orientation=base_orientation,
                                 options=self.options)


class ArmKinematics:
    """Read a USD joint chain once for arbitrary-action FK, without a live simulator.

    Movable joints on the base-to-ee path must be declared revolute/prismatic joints;
    fixed joints are retained. Closed-loop mechanisms and scaled body transforms are
    rejected rather than guessed. Returned transforms use the declared base frame.
    """

    @classmethod
    def from_arm(cls, arm, *, position_offset_m=None):
        from pxr import Usd

        stage = Usd.Stage.Open(str(arm.usd_path))
        if stage is None:
            raise ValueError(f'Cannot open arm USD {arm.usd_path}')
        return cls(stage, arm, position_offset_m=position_offset_m)

    def __init__(self, stage, arm, *, position_offset_m=None):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.arm = arm
        self.position_offset_m = np.asarray(arm.ik.position_offset_m if position_offset_m is None
                                            else position_offset_m, dtype=float)
        if self.position_offset_m.shape != (3,) or not np.isfinite(self.position_offset_m).all():
            raise ValueError('IK position_offset_m must contain three finite metre values')
        root = stage.GetDefaultPrim()
        if not root.IsValid():
            raise ValueError('Arm USD needs a default prim')
        prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        bodies = {}
        for label, name in (('base', arm.links.base), ('ee', arm.links.ee)):
            matches = [prim for prim in prims if prim.GetName() == name]
            if len(matches) != 1:
                raise ValueError(f'USD FK requires one declared {label} body {name!r}')
            bodies[label] = matches[0]
        scale = UsdGeom.GetStageMetersPerUnit(stage)

        def frame(rotation, translation):
            value = np.eye(4)
            quaternion = np.r_[np.asarray(rotation.GetImaginary()), rotation.GetReal()]
            value[:3, :3] = _numpy_rotation(quaternion)
            value[:3, 3] = np.asarray(translation) * scale
            return value

        indices = {joint.usd: index for index, joint in enumerate(arm.joints)}
        graph = {}
        for prim in prims:
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            left, right = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(left) != 1 or len(right) != 1:
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                kind, axis = 'revolute', UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                kind, axis = 'prismatic', UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.FixedJoint):
                kind, axis = 'fixed', 'X'
            else:
                kind, axis = 'unsupported', 'X'
            edge = (frame(joint.GetLocalRot0Attr().Get(), joint.GetLocalPos0Attr().Get()),
                    frame(joint.GetLocalRot1Attr().Get(), joint.GetLocalPos1Attr().Get()),
                    kind, {'X': 0, 'Y': 1, 'Z': 2}[axis], indices.get(prim.GetName()), prim.GetName())
            graph.setdefault(str(left[0]), []).append((str(right[0]), edge, False))
            graph.setdefault(str(right[0]), []).append((str(left[0]), edge, True))
        start, goal = str(bodies['base'].GetPath()), str(bodies['ee'].GetPath())
        queue, found = [(start, [], {start})], []
        while queue:
            node, chain, visited = queue.pop(0)
            if node == goal:
                found.append(chain)
                continue
            for neighbor, edge, reverse in graph.get(node, []):
                if neighbor not in visited:
                    queue.append((neighbor, chain + [(edge, reverse)], visited | {neighbor}))
        if len(found) != 1:
            raise ValueError('USD FK requires one unambiguous serial base-to-ee joint chain')
        self.chain = found[0]
        for edge, _ in self.chain:
            _left, _right, kind, _axis, index, name = edge
            if kind not in ('fixed', 'revolute', 'prismatic') or (kind != 'fixed' and index is None):
                raise ValueError(f'USD FK joint {name!r} must be fixed or declared in arm.joints')
            if kind != 'fixed' and arm.joints[index].type != kind:
                raise ValueError(f'USD FK joint {name!r} differs from its declared joint type')
        cache = UsdGeom.XformCache()
        matrices = [np.asarray(cache.GetLocalToWorldTransform(bodies[key])).T for key in ('base', 'ee')]
        for matrix in matrices:
            if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
                raise ValueError('USD FK does not support scaled body transforms')
        expected = np.linalg.inv(matrices[0]) @ matrices[1]
        expected[:3, 3] *= scale
        expected[:3, 3] += expected[:3, :3] @ self.position_offset_m
        if not np.allclose(self.fk(np.zeros(len(arm.joints))), expected, atol=2e-5, rtol=0):
            raise ValueError('USD FK zero pose disagrees with authored base/ee transforms')

    def fk(self, joint_positions):
        q = np.asarray(joint_positions, dtype=float)
        if q.shape != (len(self.arm.joints),) or not np.isfinite(q).all():
            raise ValueError(f'FK requires {len(self.arm.joints)} finite action-order radians/metres')
        result = np.eye(4)
        for (left, right, kind, axis, index, _name), reverse in self.chain:
            motion = np.eye(4)
            if kind == 'revolute':
                vector = np.zeros(3)
                vector[axis] = math.sin(q[index] / 2)
                motion[:3, :3] = _numpy_rotation(np.r_[vector, math.cos(q[index] / 2)])
            elif kind == 'prismatic':
                motion[axis, 3] = q[index]
            transform = left @ motion @ np.linalg.inv(right)
            result = result @ (np.linalg.inv(transform) if reverse else transform)
        result[:3, 3] += result[:3, :3] @ self.position_offset_m
        return result


def _numpy_rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm((x, y, z, w))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('FK joint frame quaternion must be finite and nonzero')
    x, y, z, w = np.asarray((x, y, z, w)) / norm
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                     [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                     [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
