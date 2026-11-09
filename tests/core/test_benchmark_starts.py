"""Frozen schedules and first-frame dataset reads survive Phase 0 optimizations."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.tasks.loader import load_task_module
from core.utils.hashing import json_sha256


SETS = Path(__file__).resolve().parents[2] / "tasks/pick_place_cube/eval_sets"


@pytest.mark.parametrize("filename,expected_hash", [
    ("pick_place_v1_75ep_reach150_seed1984.json", "3cf60eb861dd632f9d2b6eba222de6dd38f938aa1913c0f4bc89915ba7ca1ae5"),
    ("pick_place_v1_75ep_seed1984.json", "a6281b1fc90b3e9f313de87652a401b3a4fcdfc1757afb1c02e8720038f9bcfb"),
    ("pick_place_v1_seed1984.json", "e0615113a8a5392b11b76239407b3ce7a5215cc2c383b3e932834beebef2521c"),
])
def test_frozen_benchmark_schedule_and_seeds(filename, expected_hash):
    benchmark = load_task_module("benchmark", "pick_place_cube")
    # Hashes captured from the original implementation, including all starts and repeat seeds.
    starts = benchmark.EvalSetStarts(SETS / filename, repeats=2, seed=1984)
    assert json_sha256(starts.schedule) == expected_hash
    selected = benchmark.EvalSetStarts(SETS / filename, splits=["id"], repeats=2, seed=1984)
    assert selected.schedule == [row for row in starts.schedule if row["split"] == "id"]


def test_smoke_selection_uses_the_shared_selector():
    benchmark = load_task_module("benchmark", "pick_place_cube")
    starts = benchmark.EvalSetStarts(SETS / "pick_place_v1_seed1984.json", selection="one_per_zone")
    assert len(starts.schedule) == len({row["zone"] for row in starts.schedule})
    assert starts.schedule


def test_first_frame_states_follow_requested_episode_order(tmp_path):
    evaluation = load_task_module("eval", "pick_place_cube")
    chunk = tmp_path / "data/chunk-000"
    chunk.mkdir(parents=True)
    first, second = np.arange(6, dtype=float), np.arange(6, dtype=float) + 10
    table = pa.table({
        "episode_index": [2, 2, 7, 7, 99],
        "frame_index": [0, 1, 1, 0, 1],
        "observation.state": [first, [999.0] * 6, [999.0] * 6, second, [999.0] * 6],
    })
    pq.write_table(table, chunk / "episodes.parquet")
    actual = evaluation.load_robot_starts(tmp_path, [7, 2, 7])
    np.testing.assert_equal(actual, [second, first, second])
    with pytest.raises(ValueError, match="episodes \\[99\\]"):
        evaluation.load_robot_starts(tmp_path, [99])
