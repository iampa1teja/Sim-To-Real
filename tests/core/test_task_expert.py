"""Optional expert loading preserves errors and produces physical waypoints only."""
from dataclasses import replace
import numpy as np
import pytest

from core.tasks.base import Pose, StepState
from core.tasks.expert import ExpertContext
from core.tasks.loader import load_task


def test_optional_expert_missing_is_clear(tmp_path):
    task = load_task("pick_place_cube")
    task = type(task)(replace(task.spec, name="missing_expert", task_dir=tmp_path))
    with pytest.raises(NotImplementedError, match="no expert.py"):
        task.expert()


def test_internal_expert_import_failure_is_not_masked(tmp_path, monkeypatch):
    from core.tasks import loader
    task = load_task("pick_place_cube")
    (tmp_path / "expert.py").write_text("import unavailable_internal_dependency\n")
    task = type(task)(replace(task.spec, task_dir=tmp_path))
    def broken(*args):
        raise ModuleNotFoundError("unavailable_internal_dependency")
    monkeypatch.setattr(loader, "load_task_module", broken)
    with pytest.raises(ModuleNotFoundError, match="unavailable_internal_dependency"):
        task.expert()


def test_task_waypoints_use_world_geometry_and_open_close_release():
    task = load_task("pick_place_cube")
    state = StepState({"blue_cube": Pose((.1, -.1, .76)), "white_box": Pose((-.1, -.1, .75))},
                      Pose((0, 0, 1)), 0.)
    points = task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.), arm_base=Pose((0., 0., .75))))
    assert [point.name for point in points[:4]] == ["approach", "descend", "close", "lift"]
    assert [point.name for point in points[-5:]] == ["transport", "lower", "release", "retreat", "settle"]
    assert all(point.name.startswith("transport_arc_") for point in points[4:-5])
    np.testing.assert_allclose(points[1].pose.position, state.objects["blue_cube"].position)
    assert points[2].gripper_open_fraction == 0 and points[-3].gripper_open_fraction > 0
    assert points[3].pose.position[2] > points[1].pose.position[2] + .01
    assert all(point.pose.quat_xyzw == state.ee.quat_xyzw for point in points)


def test_transport_arc_preserves_clearance_and_targets_container_center():
    task = load_task("pick_place_cube")
    base = Pose((.7, -.6, .77))
    state = StepState({"blue_cube": Pose((.48, -.51, .80)), "white_box": Pose((.94, -.59, .795))},
                      Pose((.55, -.55, .95)), 0.)
    points = task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.), arm_base=base))
    lift = next(point for point in points if point.name == "lift")
    transport = [point for point in points if point.name.startswith("transport")]
    lower = next(point for point in points if point.name == "lower")
    floor = task.spec.objects["white_box"].params["floor_z"]
    np.testing.assert_allclose(lower.pose.position, [.94, -.59, .795 + floor + .0102], atol=1e-12)
    np.testing.assert_allclose(transport[-1].pose.position, np.asarray(lower.pose.position) + [0., 0., .09])
    path = [np.asarray(lift.pose.position)] + [np.asarray(point.pose.position) for point in transport]
    endpoint_radius = min(np.linalg.norm((point - base.position)[:2]) for point in (path[0], path[-1]))
    for left, right in zip(path, path[1:]):
        for fraction in np.linspace(0., 1., 11):
            point = left * (1 - fraction) + right * fraction
            assert np.linalg.norm((point - base.position)[:2]) >= endpoint_radius - 1e-12
    assert len(transport) > 2
    assert all(point.gripper_open_fraction == 0 for point in transport)
    # An explicit clear interior placement remains available.
    custom = task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.), {"place_xy": [.01, .01]}, base))
    placed = next(point for point in custom if point.name == "lower")
    np.testing.assert_allclose(np.asarray(placed.pose.position) - lower.pose.position, [.01, .01, 0.], atol=1e-12)


def test_transport_arc_uses_rigid_world_geometry_on_a_tilted_table():
    from core.scenes.anchors import quaternion_matrix

    task = load_task("pick_place_cube")
    base = Pose((0., 0., .75))
    state = StepState({"blue_cube": Pose((.1, -.1, .76)), "white_box": Pose((-.1, -.1, .75))},
                      Pose((0., 0., 1.)), 0.)
    original = task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.), arm_base=base))
    quaternion = (0., np.sin(.2), 0., np.cos(.2))
    rotation, shift = quaternion_matrix(quaternion), np.array([2., -3., 1.])
    def transformed(pose):
        return Pose.from_lists(rotation @ pose.position + shift, quaternion)
    moved = StepState({name: transformed(pose) for name, pose in state.objects.items()}, transformed(state.ee), 0.)
    planned = task.expert().plan(ExpertContext(moved, moved.ee, tuple(rotation @ [0., 0., 1.]), arm_base=transformed(base)))
    assert [point.name for point in planned] == [point.name for point in original]
    for before, after in zip(original, planned):
        np.testing.assert_allclose(after.pose.position, rotation @ before.pose.position + shift, atol=1e-12)


def test_transport_requires_actual_base_pose():
    task = load_task("pick_place_cube")
    state = StepState({"blue_cube": Pose((.1, -.1, .76)), "white_box": Pose((-.1, -.1, .75))},
                      Pose((0., 0., 1.)), 0.)
    with pytest.raises(ValueError, match="actual arm_base"):
        task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.)))


@pytest.mark.parametrize("old_option", ["transport_roll_degrees", "transport_orientation_xyzw"])
def test_post_pickup_roll_overrides_are_refused(old_option):
    task = load_task("pick_place_cube")
    state = StepState({"blue_cube": Pose((.1, -.1, .76)), "white_box": Pose((-.1, -.1, .75))},
                      Pose((0., 0., 1.)), 0.)
    context = ExpertContext(state, state.ee, (0., 0., 1.), {old_option: 90.}, Pose((0., 0., .75)))
    with pytest.raises(ValueError, match="keeps its pickup orientation"):
        task.expert().plan(context)


def test_raised_release_opens_without_extra_closed_hold_over_container():
    task = load_task("pick_place_cube")
    state = StepState({"blue_cube": Pose((.1, -.1, .76)), "white_box": Pose((-.1, -.1, .75))},
                      Pose((0., 0., 1.)), 0.)
    options = {"release_height_m": .09, "transport_seconds": .5}
    context = ExpertContext(state, state.ee, (0., 0., 1.), options, Pose((0., 0., .75)))
    points = task.expert().plan(context)
    names = [point.name for point in points]
    assert "orient_place" not in names and "lower" not in names
    assert all(point.pose.quat_xyzw == state.ee.quat_xyzw for point in points)
    assert all(point.duration_s == .5 for point in points if point.name.startswith("transport"))
    assert points[0].duration_s == 2.
    transport, release = points[names.index("transport"):names.index("transport") + 2]
    assert release.name == "release" and release.gripper_open_fraction > 0
    assert release.pose == transport.pose and transport.duration_s == .5
    assert points[-2].pose.position[2] >= release.pose.position[2]
    with pytest.raises(ValueError, match="release_height_m"):
        task.expert().plan(replace(context, options={"release_height_m": -.01}))
