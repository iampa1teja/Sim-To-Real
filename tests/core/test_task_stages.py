"""Task plugins own stage semantics while one tracer latches episode milestones."""
from dataclasses import replace
import numpy as np
import pytest

from core.tasks.task import Pose, StepState, Task, sample_start_in_base
from core.tasks.task import load_task
from core.tasks.stages import StageTrace, failure_mode


def state(**signals):
    return StepState(objects={"blue_cube": Pose((0, 0, 0)), "white_box": Pose((1, 0, 0))},
                     ee=Pose((.1, 0, 0)), gripper=0., signals=signals)


@pytest.mark.parametrize("signals,expected", [
    ({}, "failed_at_reached"),
    ({"reach_distance": .03}, "failed_at_grasped"),
    ({"reach_distance": .03, "grasped": True}, "failed_at_lifted"),
    ({"reach_distance": .03, "grasped": True, "holding": True}, "failed_at_lifted"),
    ({"placed": True}, "failed_at_success"),
    ({"success": True}, None),
])
def test_pick_place_failure_names_follow_the_furthest_stage(signals, expected):
    trace = StageTrace(load_task("pick_place_cube"))
    trace.observe(state(**signals))
    result = trace.finish()
    assert result["failure_mode"] == expected
    assert result["timeout_holding"] == bool(expected is not None and signals.get("holding", False))


def test_stage_latching_and_live_holding_report_the_next_stage():
    task = load_task("pick_place_cube")
    trace = StageTrace(task)
    held = replace(state(reach_distance=0., grasped=True, lifted=True, holding=True),
                   objects={"blue_cube": Pose((1, 0, .1)), "white_box": Pose((1, 0, 0))})
    trace.observe(held)
    assert trace.finish()["failure_mode"] == "failed_at_placed"
    assert trace.finish()["timeout_holding"] is True
    assert trace.progress_label() == "stage=over box, held=yes"
    trace.observe(state())
    assert trace.progress_label() == "stage=over box, held=no"
    assert trace.finish(False) == {"stages": {"reached": True, "grasped": True, "lifted": True,
                                            "over_box": True, "placed": False, "success": False},
                                   "failure_mode": "failed_at_placed", "timeout_holding": False}
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
    assert trace.finish(False)["failure_mode"] == "failed_at_success"


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


def test_task_stage_declarations_must_match_the_manifest():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="task.toml stages"):
        Task(load_task("pick_place_cube").spec, SimpleNamespace(stages={}))


@pytest.mark.parametrize("flags,expected", [
    ((False, False, False), "failed_at_a"),
    ((True, False, False), "failed_at_b"),
    ((False, True, False), "failed_at_done"),
    ((True, True, False), "failed_at_done"),
    ((False, False, True), None),
    ((True, False, True), None),
    ((False, True, True), None),
    ((True, True, True), None),
])
@pytest.mark.parametrize("holding", [False, True])
def test_failure_rule_handles_stage_gaps_and_separate_holding(flags, expected, holding):
    stages = ("a", "b", "done")
    result = failure_mode(stages, dict(zip(stages, flags)), holding)
    assert result == {"failure_mode": expected, "timeout_holding": holding and expected is not None}


@pytest.mark.parametrize("name", ["pick_place_cube", "stack_two_cubes", "stack_bowls"])
def test_failure_rules_use_each_tasks_declared_stage_order(name):
    task = load_task(name)
    flags = {}
    for stage in task.stages:
        assert task.failure_mode(flags, state(holding=True)) == "failed_at_" + stage
        flags[stage] = True
    assert task.failure_mode(flags, state(holding=True)) is None
    assert not hasattr(task.module, "failure_mode")
