"""World/table/arm-base frames preserve the measured mount and placement."""

import numpy as np
import pytest

from utils.arms.spec import load_arm_spec
from utils.geometry import Pose, anchor_basis, anchor_position, camera_quaternion, euler_xyz_degrees_to_xyzw, frame_to_world, mount_joint_positions, mount_pose, quaternion_matrix, transform_pose, world_to_frame
from utils.scene import AnchorSpec, MountSpec, load_mount_spec, load_scene_spec


def test_meeting_mount_correction_is_added_in_world_metres():
    scene = load_scene_spec("rpd_meeting_room")
    arm, _ = load_arm_spec("so101")
    mount = load_mount_spec(scene, arm)
    pose = mount_pose(scene, mount, arm)
    np.testing.assert_array_equal(pose.position, [0.696053387506893, -0.6128356057211214, 0.7717429149057083])
    # Captured from Gf yaw*tilt before migration. Configures exact float32 targets.
    expected = [-0.0009490871253245302, -0.009222361160756174, 0.9991571638848088, -0.039987612405775586]
    np.testing.assert_array_equal(np.asarray(pose.orientation_xyzw, dtype=np.float32), np.asarray(expected, dtype=np.float32))
    np.testing.assert_allclose(pose.orientation_xyzw, expected, atol=1e-14, rtol=0)
    assert mount_joint_positions(arm, mount)["Pitch"] == -1.57


def test_table_coordinates_preserve_old_plane_equation_box_position():
    scene = load_scene_spec("rpd_meeting_room")
    local = (-0.247171785265209, -0.042505977873273204, 0.000999999999999829)
    expected = [0.9440419640831723, -0.5939824362907724, 0.7951637516104304]
    position = anchor_position(scene.anchors["table"], local)
    np.testing.assert_allclose(position, expected, rtol=0, atol=5e-16)
    np.testing.assert_array_equal(np.asarray(position, dtype=np.float32), np.asarray(expected, dtype=np.float32))
    np.testing.assert_allclose(world_to_frame(position, "table", scene), local, atol=1e-15, rtol=0)


def test_fixed_camera_euler_order_matches_refined_gf_rotation():
    scene = load_scene_spec("rpd_meeting_room")
    # Independently captured with Gf X*Y*Z from the Nov 11 camera refinement.
    expected = [0.026340993876749055, -0.3223150580700269, -0.9448851899562549, 0.051099248372479186]
    quaternion = camera_quaternion(scene.cameras["realsense_rgb"])
    # Gf may choose the equivalent negative quaternion. Compare active rotations.
    np.testing.assert_allclose(quaternion_matrix(quaternion), quaternion_matrix(expected), atol=1e-14, rtol=0)


def test_missing_mount_uses_anchor_and_manifest_joint_defaults(synthetic_arm):
    scene = load_scene_spec("blank_table")
    mount = load_mount_spec(scene, synthetic_arm)
    assert not mount.exists
    pose = mount_pose(scene, mount, synthetic_arm)
    assert pose.position == scene.anchors["mount"].position
    assert mount_joint_positions(synthetic_arm, mount) == synthetic_arm.initial_joint_positions
    with pytest.raises(ValueError, match="unknown joints"):
        mount_joint_positions(synthetic_arm, MountSpec(start_joint_positions={"bad": 0.0}))


def test_world_and_arm_frame_round_trip_and_pose_rotation():
    scene = load_scene_spec("blank_table")
    arm = Pose((1.0, 2.0, 3.0), euler_xyz_degrees_to_xyzw((0, 0, 90)))
    position = (0.4, -0.2, 0.1)
    np.testing.assert_allclose(frame_to_world(position, "world", scene), position, atol=0)
    world = frame_to_world(position, "arm_base", scene, arm)
    np.testing.assert_allclose(world, [1.2, 2.4, 3.1], atol=1e-15)
    np.testing.assert_allclose(world_to_frame(world, "arm_base", scene, arm), position, atol=1e-15)
    transformed = transform_pose(Pose(position), "arm_base", scene, arm)
    np.testing.assert_allclose(transformed.position, world, atol=0)
    np.testing.assert_allclose(quaternion_matrix(transformed.orientation_xyzw), quaternion_matrix(arm.orientation_xyzw), atol=0)
    with pytest.raises(ValueError, match="explicit arm pose"):
        frame_to_world(position, "arm_base", scene)


def test_surveyed_right_is_orthogonalized_into_one_rigid_pose_frame():
    from types import SimpleNamespace
    anchor = AnchorSpec(position=(0.3, -0.4, 0.8), yaw_degrees=15,
                        right=(1.0, 0.1, 0.01), up=(0.001, 0.2, 1.0))
    scene = SimpleNamespace(name="surveyed", anchors={"table": anchor})
    basis = anchor_basis(anchor)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-15)
    assert np.linalg.det(basis) == pytest.approx(1)
    local = Pose((0.2, -0.1, 0.05), euler_xyz_degrees_to_xyzw((10, 20, 30)))
    world = transform_pose(local, "table", scene)
    np.testing.assert_allclose(world.position, np.asarray(anchor.position) + basis @ local.position, atol=0)
    np.testing.assert_allclose(quaternion_matrix(world.orientation_xyzw),
                               basis @ quaternion_matrix(local.orientation_xyzw), atol=1e-15)


@pytest.mark.parametrize("value", [0, [1, 2], [0, float("nan"), 1]])
def test_frames_reject_malformed_or_nonfinite_positions(value):
    scene = load_scene_spec("blank_table")
    with pytest.raises(ValueError, match="finite three-vector"):
        frame_to_world(value, "world", scene)
    with pytest.raises(ValueError, match="finite three-vector"):
        world_to_frame(value, "world", scene)
