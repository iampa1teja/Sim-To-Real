"""Task plugins own stage semantics while one tracer latches episode milestones."""
from dataclasses import replace
import numpy as np
import pytest

from core.tasks.base import Pose, StepState, Task, sample_start_in_base
from core.tasks.loader import load_task
from core.tasks.stages import StageTrace


def state(**signals):
    return StepState(objects={"blue_cube": Pose((0, 0, 0)), "white_box": Pose((1, 0, 0))},
                     ee=Pose((.1, 0, 0)), gripper=0., signals=signals)


@pytest.mark.parametrize("signals,expected", [
    ({}, "never_reached"),
    ({"reach_distance": .03}, "missed_grasp"),
    ({"reach_distance": .03, "grasped": True}, "dropped_before_box"),
    ({"reach_distance": .03, "grasped": True, "holding": True}, "timeout_holding"),
    ({"placed": True}, "placed_not_confirmed"),
    ({"success": True}, None),
])
def test_pick_place_failure_results_preserve_the_previous_trace(signals, expected):
    trace = StageTrace(load_task("pick_place_cube"))
    trace.observe(state(**signals))
    assert trace.finish()["failure_mode"] == expected


def test_stage_latching_and_live_holding_preserve_dropped_outside_result():
    task = load_task("pick_place_cube")
    trace = StageTrace(task)
    held = replace(state(reach_distance=0., grasped=True, lifted=True, holding=True),
                   objects={"blue_cube": Pose((1, 0, .1)), "white_box": Pose((1, 0, 0))})
    trace.observe(held)
    assert trace.progress_label() == "stage=over box, held=yes"
    trace.observe(state())
    assert trace.progress_label() == "stage=over box, held=no"
    assert trace.finish(False) == {"stages": {"reached": True, "grasped": True, "lifted": True,
                                            "over_box": True, "placed": False, "success": False},
                                   "failure_mode": "dropped_outside"}
    assert trace.finish(True)["failure_mode"] is None


def test_second_task_has_two_sampled_objects_and_distinct_stage_names():
    task = load_task("stack_two_cubes")
    sampled = task.sample_start(np.random.default_rng(7))
    assert set(sampled) == set(task.spec.tracked_objects) == {"cube_a", "cube_b"}
    assert np.linalg.norm(np.array(sampled["cube_a"].position) - sampled["cube_b"].position) >= .06
    step = StepState(sampled, Pose((1, 0, 0)), 0., {"stacked": True})
    trace = StageTrace(task)
    trace.observe(step)
    assert "over_target" in trace.stages and "over_box" not in trace.stages
    assert trace.finish(False)["failure_mode"] == "stacked_not_confirmed"


def test_start_validation_rejects_table_overhang_and_object_overlap():
    task = load_task("stack_two_cubes")
    scene = {"base_position_world": [0, 0, 0], "base_quaternion_wxyz": [1, 0, 0, 0],
             "table_plane_world": [0, 0, 0], "table_outline_base_xy": [[-.3, -.3], [.3, -.3], [.3, .3], [-.3, .3]],
             "objects": {name: {"size": [.02] * 3} for name in task.spec.objects}}
    assert task.start_is_valid({"cube_a": Pose((.1, 0, 0)), "cube_b": Pose((-.1, 0, 0))}, scene)
    assert not task.start_is_valid({"cube_a": Pose((.299, 0, 0))}, scene)
    assert not task.start_is_valid({"cube_a": Pose((.1, 0, 0)), "cube_b": Pose((.105, 0, 0))}, scene)


def test_sampling_boundary_converts_declared_frame_without_changing_plugin_signature():
    task = load_task("pick_place_cube")
    spec = replace(task.spec, start={"frame": "table", "object": "blue_cube", "xy_min": [.1, .2],
                                   "xy_max": [.1, .2], "yaw_deg": [0., 0.]})
    task = type(task)(spec)
    scene = {"base_position_world": [1, 2, 0], "base_quaternion_wxyz": [1, 0, 0, 0],
             "frames": {"table": {"position_world": [3, 4, .75], "orientation_world_wxyz": [1, 0, 0, 0]}}}
    assert task.sample_start(np.random.default_rng(0))["blue_cube"].position == (.1, .2, 0.)
    converted = sample_start_in_base(task, np.random.default_rng(0), scene)["blue_cube"]
    np.testing.assert_allclose(converted.position, [2.1, 2.2, .75])
    assert converted.quat_xyzw == (0., 0., 0., 1.)


def test_incomplete_task_plugins_fail_instead_of_reporting_all_false():
    with pytest.raises(TypeError, match="abstract"):
        Task(load_task("pick_place_cube").spec)
