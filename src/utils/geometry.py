"""Isaac-free poses, scene frames and convex footprint geometry.

Two quaternion orders meet here. Live poses and Isaac Lab tensors are XYZW; the on-disk
JSON (eval sets, captured scenes, dataset sidecars) is WXYZ. Every function that takes a
quaternion names its order in the function or parameter name; convert explicitly with
``xyzw_to_wxyz``/``wxyz_to_xyzw``. Scene-frame helpers below return XYZW.

Gf composes X*Y*Z rotations as extrinsic XYZ (Hamilton qz*qy*qx).
A surveyed mount's Gf yaw*tilt is Hamilton tilt*yaw. Retaining these orders
preserves the fitted room without depending on USD to read its coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np


@dataclass(frozen=True, slots=True)
class Pose:
    position: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_lists(cls, position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)) -> Pose:
        return cls(tuple(float(v) for v in position), tuple(float(v) for v in quat_xyzw))


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


def _is_tensor(q):
    torch = sys.modules.get("torch")  # a tensor implies torch is already imported
    return torch is not None and isinstance(q, torch.Tensor)


def xyzw_to_wxyz(q):
    """Reorder the last axis. Tensors stay tensors; anything else becomes a float64 array."""
    if _is_tensor(q):
        return sys.modules["torch"].cat((q[..., 3:4], q[..., :3]), dim=-1)
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q):
    """Reorder the last axis. Tensors stay tensors; anything else becomes a float64 array."""
    if _is_tensor(q):
        return sys.modules["torch"].cat((q[..., 1:4], q[..., 0:1]), dim=-1)
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quaternion_multiply(a_xyzw, b_xyzw):
    """Hamilton product of XYZW quaternions: apply rotation b, then rotation a."""
    a, b = np.asarray(a_xyzw, dtype=np.float64), np.asarray(b_xyzw, dtype=np.float64)
    vector = a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3])
    return tuple(float(value) for value in (*vector, a[3] * b[3] - np.dot(a[:3], b[:3])))


def euler_xyz_degrees_to_xyzw(angles):
    x, y, z = (math.radians(float(value)) / 2 for value in angles)
    sx, sy, sz, cx, cy, cz = math.sin(x), math.sin(y), math.sin(z), math.cos(x), math.cos(y), math.cos(z)
    return (sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)


def rotation_matrix_xyzw(quaternion):
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
    return rotation_matrix_xyzw(anchor_quaternion(anchor))


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
        return np.asarray(arm_pose.position), rotation_matrix_xyzw(arm_pose.quat_xyzw), arm_pose.quat_xyzw
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
                quaternion_multiply(quaternion, pose.quat_xyzw))


# Footprints and resting poses for eval-set starts; inputs follow the WXYZ disk convention.
def rotation_matrix_wxyz(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def upright_orientation_wxyz(base_orientation_wxyz, plane, yaw_orientation_wxyz):
    """Yaw about the live table normal, expressed in the live base frame."""
    normal = np.array([-plane[0], -plane[1], 1.0])
    normal = rotation_matrix_wxyz(base_orientation_wxyz).T @ (normal / np.linalg.norm(normal))
    tilt = np.array([1 + normal[2], -normal[1], normal[0], 0.0])
    if np.linalg.norm(tilt) < 1e-8:
        tilt = np.array([0., 1., 0., 0.])
    tilt /= np.linalg.norm(tilt)
    w, x, y, z = tilt
    a, b, c, d = yaw_orientation_wxyz
    return np.array([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c,
                     w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])


def resting_pose(position, orientation_wxyz, base_position, base_orientation_wxyz, plane, cube_size):
    """Preserve base XY, ignore recorded Z and solve for tabletop support.

    plane is (a,b,c): world z=a*x+b*y+c. Same oriented support and
    0.2 mm clearance as reset_object_pose, solved along the live base Z axis.
    """
    base_rotation = rotation_matrix_wxyz(base_orientation_wxyz)
    rotation = base_rotation @ rotation_matrix_wxyz(orientation_wxyz)
    normal = np.array([-plane[0], -plane[1], 1.0])
    support = np.sum(np.abs(normal @ rotation) * np.asarray(cube_size) / 2)
    world = np.asarray(base_position) + base_rotation @ np.r_[position[:2], 0.0]
    denominator = normal @ base_rotation[:, 2]
    if abs(denominator) < 1e-8:
        raise ValueError('Live base Z axis is parallel to the table')
    world += base_rotation[:, 2] * ((plane[2] + support + 0.0002 - normal @ world) / denominator)
    return world, rotation


def footprint(position, rotation, size, bottom_origin=False):
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5)
                        for z in ((0., 1.) if bottom_origin else (-.5, .5))])
    return (corners * np.asarray(size)) @ np.asarray(rotation).T[:, :2] + np.asarray(position)[:2]


def footprints_overlap(first, second):
    """Separating axis test of convex projected cuboids, with no added margin."""
    # All point-pair normals include every convex-hull edge normal; extra
    # axes cannot separate overlapping convex hulls.
    for points in (first, second):
        for i, point in enumerate(points):
            for other in points[i + 1:]:
                edge = other - point
                axis = np.array([-edge[1], edge[0]])
                if not np.any(axis):
                    continue
                a, b = first @ axis, second @ axis
                if a.max() <= b.min() or b.max() <= a.min():
                    return False
    return True


def hull(points):
    """CCW monotone-chain convex hull, without a SciPy dependency."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    halves = []
    for ordered in (points, points[::-1]):
        half = []
        for p in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.extend(half[:-1])
    if len(halves) < 3:
        raise ValueError('Training starts must cover a non-degenerate 2D convex hull')
    return np.array(halves)


def inside_polygon(point, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(point) - polygon
    return bool(np.all(edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0] >= -1e-12))


def hull_distance(point, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(np.sum((point - polygon) * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
    return float(np.linalg.norm(point - (polygon + t[:, None] * edges), axis=1).min())


def polygons_match(first, second, *, atol=1e-5):
    """Compare convex boundaries in metres, independent of vertex sampling/order."""
    polygons = []
    for points in (first, second):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            return False
        try:
            polygons.append(hull(points))
        except ValueError:
            return False
    first, second = polygons
    # Nearly collinear mesh vertices can enter/leave the hull after tiny float
    # changes. Compare each hull to the other's edges, not its vertex array.
    return (all(hull_distance(point, second) <= atol for point in first)
            and all(hull_distance(point, first) <= atol for point in second))


def face_yaw_wxyz(quaternion_wxyz):
    rotation = rotation_matrix_wxyz(quaternion_wxyz)
    axis = np.argmax(np.linalg.norm(rotation[:2], axis=0))
    return float(np.arctan2(rotation[1, axis], rotation[0, axis]) % (np.pi / 2))


def wrap_yaw(angle):
    """Square-symmetric relative yaw in [-pi/4, pi/4)."""
    return (np.asarray(angle) + np.pi / 4) % (np.pi / 2) - np.pi / 4


def sample_yaw(xy, training_xy, training_relative_yaws, mode, rng):
    """Return (absolute yaw modulo pi/2, relative yaw, copied source index)."""
    bearing = math.atan2(xy[1], xy[0])
    source = int(np.argmin(np.linalg.norm(training_xy - xy, axis=1))) if mode == 'matched' else None
    if mode == 'matched':
        yaw = (bearing + training_relative_yaws[source]) % (np.pi / 2)
    elif mode == 'radial':
        yaw = bearing % (np.pi / 2)
    elif mode == 'random':
        yaw = rng.uniform(0, np.pi / 2)
    else:
        raise ValueError('Unknown yaw mode')
    return float(yaw), float(wrap_yaw(yaw - bearing)), source


def circular_mean(angles, period):
    mean = np.mean(np.exp(2j * np.pi * np.asarray(angles) / period))
    if abs(mean) < 1e-6:
        raise ValueError('Recorded orientations have no stable circular alignment rule')
    return float(np.angle(mean) * period / (2 * np.pi))


def pose_for_xy(xy, yaw, scene, object_name=None):
    object_name = object_name or scene['start_object']
    size = scene['objects'][object_name]['size']
    q = upright_orientation_wxyz(scene['base_quaternion_wxyz'], scene['table_plane_world'],
                            [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    world, rotation = resting_pose(np.r_[xy, 0], q, scene['base_position_world'],
                                   scene['base_quaternion_wxyz'], scene['table_plane_world'], size)
    return q, world, rotation


def supported_on_table(xy, yaw, scene, object_name=None):
    object_name = object_name or scene['start_object']
    _, world, rotation = pose_for_xy(xy, yaw, scene, object_name)
    base_rotation = rotation_matrix_wxyz(scene['base_quaternion_wxyz'])
    centre = base_rotation.T @ (world - np.asarray(scene['base_position_world']))
    projected = footprint(centre, base_rotation.T @ rotation, scene['objects'][object_name]['size'])
    return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)

