"""Isaac-free poses and scene frames; all quaternions here are XYZW.

Gf composes X*Y*Z rotations as extrinsic XYZ (Hamilton qz*qy*qx).
A surveyed mount's Gf yaw*tilt is Hamilton tilt*yaw. Retaining these orders
preserves the fitted room without depending on USD to read its coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def _position(values, label="position"):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite three-vector")
    return values


def _unit(values, label="vector"):
    values = _position(values, label)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"{label} must be a finite nonzero three-vector")
    return values / norm


def quaternion_multiply(a, b):
    """Hamilton product: apply rotation b, then rotation a."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    vector = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return tuple(float(value) for value in (*vector, a[3] * b[3] - np.dot(a[:3], b[:3])))


def euler_xyz_degrees_to_xyzw(angles):
    x, y, z = (math.radians(float(value)) / 2 for value in angles)
    sx, sy, sz, cx, cy, cz = math.sin(x), math.sin(y), math.sin(z), math.cos(x), math.cos(y), math.cos(z)
    return (sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(np.linalg.norm(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError("Quaternion must contain four finite nonzero XYZW components")
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _matrix_quaternion(matrix):
    """Stable active-rotation matrix -> XYZW without a SciPy dependency."""
    m = np.asarray(matrix, dtype=np.float64)
    if np.trace(m) > 0:
        scale = 2 * math.sqrt(1 + np.trace(m))
        q = ((m[2, 1]-m[1, 2])/scale, (m[0, 2]-m[2, 0])/scale,
             (m[1, 0]-m[0, 1])/scale, scale/4)
    else:
        index = int(np.argmax(np.diag(m)))
        other, last = (index+1) % 3, (index+2) % 3
        scale = 2 * math.sqrt(1 + m[index, index]-m[other, other]-m[last, last])
        q = np.zeros(4)
        q[index] = scale/4
        q[other] = (m[index, other]+m[other, index])/scale
        q[last] = (m[index, last]+m[last, index])/scale
        q[3] = (m[last, other]-m[other, last])/scale
    return tuple(float(value) for value in np.asarray(q) / np.linalg.norm(q))


def yaw_up_quaternion(yaw_degrees, up):
    """Match Gf.Rotation(Z,yaw) * Gf.Rotation(Z,up), including quaternion sign."""
    up = _unit(up, "up")
    cosine = float(up[2])
    if cosine < -1 + 1e-12:
        tilt = (1.0, 0.0, 0.0, 0.0)
    else:
        scale = math.sqrt(2 * (1 + cosine))
        tilt = (-float(up[1])/scale, float(up[0])/scale, 0.0, scale/2)
    half = math.radians(float(yaw_degrees))/2
    return quaternion_multiply(tilt, (0.0, 0.0, math.sin(half), math.cos(half)))


def anchor_basis(anchor):
    """A single rigid frame rotates both positions and orientations."""
    return quaternion_matrix(anchor_quaternion(anchor))


def anchor_position(anchor, offset=(0.0, 0.0, 0.0)):
    value = _position(anchor.position) + anchor_basis(anchor) @ _position(offset, "anchor offset")
    return tuple(float(component) for component in value)


def anchor_quaternion(anchor):
    if anchor.right is None:
        return yaw_up_quaternion(anchor.yaw_degrees, anchor.up)
    # Independently surveyed right/up vectors can contain rounding error.
    # Orthogonalize them once; the resulting quaternion defines the same rigid
    # rotation used by all position and orientation conversions.
    up = _unit(anchor.up, "up")
    inward = _unit(np.cross(up, _unit(anchor.right, "right")), "right/up cross product")
    right = _unit(np.cross(inward, up), "anchor right")
    quaternion = _matrix_quaternion(np.column_stack((right, inward, up)))
    if anchor.yaw_degrees:
        half = math.radians(anchor.yaw_degrees)/2
        quaternion = quaternion_multiply(quaternion, (0.0, 0.0, math.sin(half), math.cos(half)))
    return quaternion


def camera_quaternion(camera):
    """USD/OpenGL camera: -Z forward and +Y up; explicit Euler angles win."""
    if camera.euler_degrees is not None:
        return euler_xyz_degrees_to_xyzw(camera.euler_degrees)
    forward = _unit(np.asarray(camera.target)-np.asarray(camera.position), "camera target direction")
    up = _unit(camera.up or (0.0, 0.0, 1.0), "camera up")
    right = _unit(np.cross(forward, up), "camera direction/up cross product")
    up = np.cross(right, forward)
    return _matrix_quaternion(np.column_stack((right, up, -forward)))


def mount_pose(scene, mount, arm=None):
    anchor = scene.anchors.get(mount.anchor)
    if anchor is None:
        if mount.exists:
            raise ValueError(f"Mount anchor {mount.anchor!r} is missing from scene {scene.name!r}")
        if arm is None:
            return Pose((0.0, 0.0, 0.0))
        return Pose(arm.initial_state.position, euler_xyz_degrees_to_xyzw(arm.initial_state.rotation_euler_degrees))
    # Calibration corrections were measured in world metres, not along the
    # tilted mount axes. Adding them after transforming would change the fit.
    position = tuple(float(value + delta) for value, delta in zip(anchor.position, mount.translation_correction_m))
    orientation = anchor_quaternion(anchor)
    if arm is not None and arm.initial_state.rotation_relative_to_anchor:
        orientation = quaternion_multiply(orientation, euler_xyz_degrees_to_xyzw(arm.initial_state.rotation_euler_degrees))
    return Pose(position, orientation)


def mount_joint_positions(arm, mount):
    unknown = set(mount.start_joint_positions) - set(arm.joint_names)
    if unknown:
        raise ValueError(f"Mount start pose has unknown joints for {arm.name!r}: {sorted(unknown)}")
    positions = {joint.usd: mount.start_joint_positions.get(joint.name, joint.initial)
                 for joint in arm.action_joints}
    for joint in arm.joints:
        if joint.mimic is not None:
            source = next(source for source in arm.action_joints if source.name == joint.mimic.joint)
            positions[joint.usd] = positions[source.usd] * joint.mimic.multiplier + joint.mimic.offset
    return positions


def _frame(frame, scene, arm_pose):
    if frame == "world":
        return np.zeros(3), np.eye(3), (0.0, 0.0, 0.0, 1.0)
    if frame == "arm_base":
        if arm_pose is None:
            raise ValueError("arm_base coordinates require an explicit arm pose")
        return np.asarray(arm_pose.position), quaternion_matrix(arm_pose.orientation_xyzw), arm_pose.orientation_xyzw
    if frame not in scene.anchors:
        raise ValueError(f"Scene {scene.name!r} has no coordinate anchor {frame!r}")
    anchor = scene.anchors[frame]
    return np.asarray(anchor.position), anchor_basis(anchor), anchor_quaternion(anchor)


def frame_to_world(position, frame, scene, arm_pose=None):
    origin, basis, _ = _frame(frame, scene, arm_pose)
    value = origin + basis @ _position(position)
    return tuple(float(component) for component in value)


def world_to_frame(position, frame, scene, arm_pose=None):
    origin, basis, _ = _frame(frame, scene, arm_pose)
    value = np.linalg.solve(basis, _position(position)-origin)
    return tuple(float(component) for component in value)


def transform_pose(pose: Pose, frame, scene, arm_pose=None) -> Pose:
    _, _, quaternion = _frame(frame, scene, arm_pose)
    return Pose(frame_to_world(pose.position, frame, scene, arm_pose),
                quaternion_multiply(quaternion, pose.orientation_xyzw))
