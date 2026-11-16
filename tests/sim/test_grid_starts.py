"""Grid coverage, constraints and source selection stay independent of the SDK."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.tasks.task import load_task
from sim.datagen.starts import GridStarts


def serialized(sampler):
    return {name: tuple(np.r_[row["position"], row["orientation_wxyz"]])
            for name, row in sampler.sample()[1].items()}


def test_single_object_visits_every_cell_and_yaw_once_then_cycles():
    sampler = GridStarts(load_task("pick_place_cube"), cells_per_axis=2, yaw_steps=3, max_starts=12, seed=7)
    starts = [serialized(sampler) for _ in range(12)]
    assert len({tuple(row["blue_cube"]) for row in starts}) == 12
    assert sampler.coverage()["objects"]["blue_cube"]["cell_visits"] == [[[1, 1, 1]] * 2] * 2
    assert serialized(sampler) == starts[0]
    assert sampler.coverage()["draws"] == 13


def test_seed_and_cap_are_repeatable_without_advancing_global_randomness():
    task = load_task("pick_place_cube")
    before = np.random.get_state()
    a, b = [GridStarts(task, cells_per_axis=4, yaw_steps=2, max_starts=3, seed=91) for _ in range(2)]
    first = [serialized(a) for _ in range(9)]
    assert first == [serialized(b) for _ in range(9)]
    assert first[:3] == first[3:6] == first[6:]
    assert a.coverage()["unique_starts"] == 3
    after = np.random.get_state()
    assert all(np.array_equal(x, y) for x, y in zip(before, after))


@pytest.mark.parametrize("name", ["stack_two_cubes", "stack_bowls"])
def test_multi_object_combinations_respect_task_separation_and_radius(name):
    task = load_task(name)
    sampler = GridStarts(task, cells_per_axis=5, yaw_steps=1, max_starts=8, seed=13)
    rows = [serialized(sampler) for _ in range(8)]
    assert len({tuple(row.values()) for row in rows}) == 8
    for row in rows:
        assert set(row) == set(task.spec.tracked_objects)
        xy = [np.array(pose[:2]) for pose in row.values()]
        assert all(np.linalg.norm(p) >= task.spec.start.get("minimum_base_radius_m", 0.) for p in xy)
        assert all(np.linalg.norm(p - q) >= task.spec.start["minimum_separation_m"]
                   for i, p in enumerate(xy) for q in xy[:i])
    assert all(sum(np.array(value["cell_visits"]).flat) == 8
               for value in sampler.coverage()["objects"].values())


def test_live_geometry_filter_and_declared_frame_conversion_apply_before_visiting():
    task = load_task("pick_place_cube")
    task = type(task)(replace(task.spec, start={**task.spec.start, "frame": "table"}))
    sampler = GridStarts(task, cells_per_axis=2, yaw_steps=1)
    sampler.scene = {"base_position_world": [1, 0, 0], "base_quaternion_wxyz": [1, 0, 0, 0],
                     "frames": {"table": {"position_world": [2, 0, .75],
                                          "orientation_world_wxyz": [1, 0, 0, 0]}}}
    _, objects, zone = sampler.sample(lambda objects: objects["blue_cube"]["position"][0] > 1.15)
    assert zone == "grid" and objects["blue_cube"]["position"][0] > 1.15
    assert objects["blue_cube"]["position"][2] == .75
    assert sampler.coverage()["draws"] == 1
    with pytest.raises(ValueError, match="No valid grid"):
        GridStarts(load_task("pick_place_cube"), cells_per_axis=1, yaw_steps=1).sample(lambda _: False)


@pytest.mark.parametrize("options", [{"cells_per_axis": 0}, {"yaw_steps": -1}, {"max_starts": 0}])
def test_invalid_grid_parameters_fail_before_launch(options):
    with pytest.raises(ValueError, match="positive integer"):
        GridStarts(load_task("pick_place_cube"), **options)


def test_scripted_default_is_grid_but_reference_yaml_and_mimic_keep_random(tmp_path, monkeypatch):
    from sim.datagen import mimic, scripted

    monkeypatch.setattr(scripted, "add_generation_launch_arguments",
                        lambda parser: parser.add_argument("--headless", action="store_true"))
    assert scripted.parse_args(["--destination", str(tmp_path)]).start_mode == "grid"
    path = Path(__file__).resolve().parents[2] / "workspace/configs/datagen/scripted.yaml"
    assert scripted.parse_args(["--config_path", str(path)]).start_mode == "random"
    source = tmp_path / "source"
    (source / "datagen_meta").mkdir(parents=True)
    args = ["--destination", str(tmp_path / "out"), "--source_dataset", str(source)]
    assert mimic.parse_args(args).start_mode == "random"
    with pytest.raises(SystemExit):
        mimic.parse_args(args + ["--start_mode", "grid"])
