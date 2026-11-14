"""Renamed and moved quaternion helpers return bit-identical results to the code they replaced."""
import numpy as np
import pytest

from utils.geometry import (face_yaw_wxyz, quaternion_multiply, resting_pose, rotation_matrix_wxyz,
                            rotation_matrix_xyzw, upright_orientation_wxyz, wxyz_to_xyzw, xyzw_to_wxyz)


# Old core/eval/geometry.py and utils/geometry.py bodies, copied verbatim as the reference.
def old_rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("orientation_wxyz must contain four finite values and be nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def old_quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isfinite(np.linalg.norm(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError("Quaternion must contain four finite nonzero XYZW components")
    x, y, z, w = q / np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def old_upright_orientation(base_orientation, plane, yaw_orientation):
    normal = np.array([-plane[0], -plane[1], 1.0])
    normal = old_rotation_matrix(base_orientation).T @ (normal / np.linalg.norm(normal))
    tilt = np.array([1 + normal[2], -normal[1], normal[0], 0.0])
    if np.linalg.norm(tilt) < 1e-8:
        tilt = np.array([0., 1., 0., 0.])
    tilt /= np.linalg.norm(tilt)
    w, x, y, z = tilt
    a, b, c, d = yaw_orientation
    return np.array([w * a - x * b - y * c - z * d, w * b + x * a + y * d - z * c,
                     w * c - x * d + y * a + z * b, w * d + x * c - y * b + z * a])


def old_resting_pose(position, orientation, base_position, base_orientation, plane, cube_size):
    base_rotation = old_rotation_matrix(base_orientation)
    rotation = base_rotation @ old_rotation_matrix(orientation)
    normal = np.array([-plane[0], -plane[1], 1.0])
    support = np.sum(np.abs(normal @ rotation) * np.asarray(cube_size) / 2)
    world = np.asarray(base_position) + base_rotation @ np.r_[position[:2], 0.0]
    denominator = normal @ base_rotation[:, 2]
    if abs(denominator) < 1e-8:
        raise ValueError('Live base Z axis is parallel to the table')
    world += base_rotation[:, 2] * ((plane[2] + support + 0.0002 - normal @ world) / denominator)
    return world, rotation


def old_face_yaw(quaternion):
    rotation = old_rotation_matrix(quaternion)
    axis = np.argmax(np.linalg.norm(rotation[:2], axis=0))
    return float(np.arctan2(rotation[1, axis], rotation[0, axis]) % (np.pi / 2))


def old_wxyz(q):  # sim/workspace/envs/mdp/_common.py
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def old_xyzw(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def same(new, old):
    new, old = np.asarray(new), np.asarray(old)
    return new.dtype == old.dtype and new.shape == old.shape and np.array_equal(new, old)


@pytest.fixture
def quaternions():
    rng = np.random.default_rng(5)
    return [rng.normal(size=4) for _ in range(20)] + [np.array([0., 0., 0., 1.]), (1, 0, 0, 0)]


def test_matrices_and_moved_helpers_are_bit_identical(quaternions):
    rng = np.random.default_rng(7)
    for q in quaternions:
        assert same(rotation_matrix_wxyz(q), old_rotation_matrix(q))
        assert same(rotation_matrix_xyzw(q), old_quaternion_matrix(q))
        assert face_yaw_wxyz(q) == old_face_yaw(q)
        base, plane, size = rng.normal(size=4), (*rng.normal(scale=.02, size=2), .75), rng.uniform(.02, .06, 3)
        assert same(upright_orientation_wxyz(base, plane, q), old_upright_orientation(base, plane, q))
        position, base_position = rng.normal(size=3), rng.normal(size=3)
        for new, old in zip(resting_pose(position, q, base_position, base, plane, size),
                            old_resting_pose(position, q, base_position, base, plane, size)):
            assert same(new, old)
    with pytest.raises(ValueError, match="orientation_wxyz must contain"):
        rotation_matrix_wxyz((0, 0, 0, 0))
    with pytest.raises(ValueError, match="XYZW components"):
        rotation_matrix_xyzw((0, 0, 0, 0))


def test_reorders_match_the_replaced_expressions(quaternions):
    rng = np.random.default_rng(11)
    batch = rng.normal(size=(3, 5, 4)).astype(np.float32)
    for q in [*quaternions, batch, list(batch[0, 0]), tuple(float(v) for v in batch[1, 1])]:
        assert same(xyzw_to_wxyz(q), old_wxyz(q))
        assert same(wxyz_to_xyzw(q), old_xyzw(q))
    for q in quaternions:
        q32 = np.asarray(q, dtype=np.float32)
        # scripted.py/mimic.py: rotation_matrix((q[3], q[0], q[1], q[2])) on a float32 live quaternion
        assert same(rotation_matrix_wxyz(xyzw_to_wxyz(q32)), old_rotation_matrix((q32[3], q32[0], q32[1], q32[2])))
        x, y, z, w = (float(v) for v in q)
        # core/eval/starts.py poses_to_objects and rollout.py pose_matrix/control_pose
        assert same(xyzw_to_wxyz((x, y, z, w)), np.array([w, x, y, z], dtype=float))
        assert same(rotation_matrix_wxyz(xyzw_to_wxyz((x, y, z, w))), old_rotation_matrix((w, x, y, z)))
        # core/tasks/base.py start_is_valid: the tuple (w, x, y, z) fed to upright_orientation
        base, plane = rng.normal(size=4), (.01, -.02, .75)
        assert same(upright_orientation_wxyz(base, plane, xyzw_to_wxyz((x, y, z, w))),
                    old_upright_orientation(base, plane, (w, x, y, z)))
        # base.py start_pose_to_base and eval_set.py legacy conversion feed quaternion_multiply
        assert quaternion_multiply(q, wxyz_to_xyzw(q)) == quaternion_multiply(q, (q[1], q[2], q[3], q[0]))
        relative = quaternion_multiply(q, q)
        assert xyzw_to_wxyz(relative).tolist() == [relative[3], *relative[:3]]
        # clearance.py: [*xyzw[3:], *xyzw[:3]]
        assert xyzw_to_wxyz((x, y, z, w)).tolist() == [w, x, y, z]


def test_scene_start_helpers_are_bit_identical():
    import math
    from utils.geometry import footprint, inside_polygon, pose_for_xy, supported_on_table

    def old_pose_for_xy(xy, yaw, scene, object_name=None):
        object_name = object_name or scene['start_object']
        size = scene['objects'][object_name]['size']
        q = old_upright_orientation(scene['base_quaternion_wxyz'], scene['table_plane_world'],
                                    [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        world, rotation = old_resting_pose(np.r_[xy, 0], q, scene['base_position_world'],
                                           scene['base_quaternion_wxyz'], scene['table_plane_world'], size)
        return q, world, rotation

    def old_supported_on_table(xy, yaw, scene, object_name=None):
        object_name = object_name or scene['start_object']
        _, world, rotation = old_pose_for_xy(xy, yaw, scene, object_name)
        base_rotation = old_rotation_matrix(scene['base_quaternion_wxyz'])
        centre = base_rotation.T @ (world - np.asarray(scene['base_position_world']))
        projected = footprint(centre, base_rotation.T @ rotation, scene['objects'][object_name]['size'])
        return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)

    rng = np.random.default_rng(3)
    scene = dict(start_object='cube', objects={'cube': {'size': [.03, .03, .03]}},
                 base_quaternion_wxyz=[.999, .01, -.02, .03], base_position_world=[.1, -.2, .75],
                 table_plane_world=[.004, -.003, .74], table_outline_base_xy=[[-.1, -.4], [.4, -.4], [.4, .3], [-.1, .3]])
    for _ in range(50):
        xy, yaw = rng.uniform(-.3, .4, 2), rng.uniform(0, np.pi / 2)
        for new, old in zip(pose_for_xy(xy, yaw, scene), old_pose_for_xy(xy, yaw, scene)):
            assert same(new, old)
        assert supported_on_table(xy, yaw, scene) == old_supported_on_table(xy, yaw, scene)


def test_tensor_reorders_match_the_replaced_torch_cat():
    torch = pytest.importorskip("torch")
    quaternion = torch.randn(6, 4, dtype=torch.float32)
    position = torch.randn(6, 3, dtype=torch.float32)
    pose = torch.randn(4, 7, dtype=torch.float64)
    # teleop/loop.py object_pose_in_base
    new = torch.cat((position, xyzw_to_wxyz(quaternion)), dim=-1)
    old = torch.cat((position, quaternion[:, 3:4], quaternion[:, :3]), dim=-1)
    assert new.dtype == old.dtype and torch.equal(new, old)
    # recording/replay_scene.py
    new, old = wxyz_to_xyzw(pose[:, 3:7]), torch.cat((pose[:, 4:7], pose[:, 3:4]), dim=-1)
    assert new.dtype == old.dtype and torch.equal(new, old)
    assert torch.equal(wxyz_to_xyzw(xyzw_to_wxyz(quaternion)), quaternion)
