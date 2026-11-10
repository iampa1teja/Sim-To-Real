"""Optional expert loading preserves errors and produces physical waypoints only."""
from dataclasses import replace
import numpy as np
import pytest

from core.tasks.base import Pose, StepState
from core.tasks.expert import ExpertContext
from core.tasks.loader import load_task


def test_optional_expert_missing_is_clear():
    with pytest.raises(NotImplementedError, match="no expert.py"):
        load_task("stack_two_cubes").expert()


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
    points = task.expert().plan(ExpertContext(state, state.ee, (0., 0., 1.)))
    assert [point.name for point in points] == ["approach", "descend", "close", "lift", "transport", "lower", "release", "retreat", "settle"]
    np.testing.assert_allclose(points[1].pose.position, state.objects["blue_cube"].position)
    assert points[2].gripper_open_fraction == 0 and points[6].gripper_open_fraction > 0
    assert points[3].pose.position[2] > points[1].pose.position[2] + .01
    assert all(point.pose.quat_xyzw == state.ee.quat_xyzw for point in points)
