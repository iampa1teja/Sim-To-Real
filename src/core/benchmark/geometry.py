"""Isaac-free pose and convex footprint geometry. Disk quaternions are WXYZ."""
import math
import numpy as np

def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def upright_orientation(base_orientation, plane, yaw_orientation):
    """Yaw about the live table normal, expressed in the live base frame."""
    normal = np.array([-plane[0], -plane[1], 1.0])
    normal = rotation_matrix(base_orientation).T @ (normal / np.linalg.norm(normal))
    tilt = np.array([1 + normal[2], -normal[1], normal[0], 0.0])
    if np.linalg.norm(tilt) < 1e-8:
        tilt = np.array([0., 1., 0., 0.])
    tilt /= np.linalg.norm(tilt)
    w, x, y, z = tilt
    a, b, c, d = yaw_orientation
    return np.array([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c,
                     w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])


def resting_pose(position, orientation, base_position, base_orientation, plane, cube_size):
    """Preserve base XY, ignore recorded Z and solve for tabletop support.

    plane is (a,b,c): world z=a*x+b*y+c. Same oriented support and
    0.2 mm clearance as reset_object_pose, solved along the live base Z axis.
    """
    base_rotation = rotation_matrix(base_orientation)
    rotation = base_rotation @ rotation_matrix(orientation)
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


def face_yaw(quaternion):
    rotation = rotation_matrix(quaternion)
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
    q = upright_orientation(scene['base_quaternion_wxyz'], scene['table_plane_world'],
                            [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    world, rotation = resting_pose(np.r_[xy, 0], q, scene['base_position_world'],
                                   scene['base_quaternion_wxyz'], scene['table_plane_world'], size)
    return q, world, rotation


def supported_on_table(xy, yaw, scene, object_name=None):
    object_name = object_name or scene['start_object']
    _, world, rotation = pose_for_xy(xy, yaw, scene, object_name)
    base_rotation = rotation_matrix(scene['base_quaternion_wxyz'])
    centre = base_rotation.T @ (world - np.asarray(scene['base_position_world']))
    projected = footprint(centre, base_rotation.T @ rotation, scene['objects'][object_name]['size'])
    return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)

