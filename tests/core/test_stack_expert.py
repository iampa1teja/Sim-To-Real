"""Stack waypoints preserve orientation and target the red cube's actual top."""
import numpy as np
import pytest

from core.scenes import quaternion_matrix
from core.tasks.base import Pose, StepState
from core.tasks.expert import ExpertContext
from core.tasks.loader import load_task


def context(options=None, rotation=(0., 0., 0., 1.)):
    poses = {"cube_a": Pose((.183, -.085, .76), rotation), "cube_b": Pose((.235, -.125, .76), rotation)}
    return ExpertContext(StepState(poses, Pose((.1, -.1, .9)), 0.), Pose((.1, -.1, .9)),
                         tuple(quaternion_matrix(rotation)[:, 2]), options or {}, Pose((0., 0., .75)))


def test_stack_target_open_close_and_fast_transport_are_task_owned():
    task = load_task("stack_two_cubes")
    current = context()
    points = task.expert().plan(current)
    assert [point.name for point in points] == ["approach", "descend", "close", "lift", "transport", "lower", "release", "retreat", "settle"]
    named = {point.name: point for point in points}
    np.testing.assert_allclose(named["lower"].pose.position, [.235, -.125, .7805], atol=1e-12)
    np.testing.assert_allclose(named["transport"].pose.position, [.235, -.125, .8305], atol=1e-12)
    assert named["transport"].duration_s == .5
    assert named["lower"].tolerance_m <= .001
    assert named["close"].gripper_open_fraction == named["lower"].gripper_open_fraction == 0.
    assert named["release"].gripper_open_fraction > 0
    assert all(point.pose.quat_xyzw == current.control_pose.quat_xyzw for point in points)
    assert named["settle"].duration_s >= 25 / 30


def test_lower_height_uses_the_red_cube_orientation_basis():
    quaternion = (np.sin(.15), 0., 0., np.cos(.15))
    current = context(rotation=quaternion)
    points = load_task("stack_two_cubes").expert().plan(current)
    lower = next(point for point in points if point.name == "lower")
    expected = np.asarray(current.state.objects["cube_b"].position) + quaternion_matrix(quaternion)[:, 2] * .0205
    np.testing.assert_allclose(lower.pose.position, expected, atol=1e-12)


def test_settle_cannot_be_shorter_than_confirmation_and_no_carry_roll():
    task = load_task("stack_two_cubes")
    points = task.expert().plan(context({"settle_seconds": .01}))
    assert points[-1].duration_s >= 26 / 30
    with pytest.raises(ValueError, match="pickup orientation"):
        task.expert().plan(context({"transport_roll_degrees": 90.}))
