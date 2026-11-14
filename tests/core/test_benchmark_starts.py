"""Frozen schedules and first-frame dataset reads survive Phase 0 optimizations."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.benchmark import eval_set as benchmark, starts as evaluation
from core.utils.hashing import json_sha256


SETS = Path(__file__).resolve().parents[2] / "workspace/tasks/pick_place_cube/eval_sets"


@pytest.mark.parametrize("filename,expected_hash", [
    ("pick_place_v1_75ep_reach150_seed1984.json", "76880c20c565ec753fc696ee7d29aed6217b15082d223ed995b281bad5152a07"),
    ("pick_place_v1_75ep_seed1984.json", "d80b82182dc965ae5c7659c2a8220deb668fcbdd0804f4d793e3cd0822589ba9"),
    ("pick_place_v1_seed1984.json", "12ed6cbdfa028fcfa2ba5a3ccdb235d84530b30b1c7536b22640ff01afb1fa78"),
])
def test_frozen_benchmark_schedule_and_seeds(filename, expected_hash):
    # Original schema-1 schedules captured before conversion, with poses normalized
    # to named objects. All starts, raw robot states and repeat seeds are covered.
    starts = benchmark.EvalSetStarts(SETS / filename, repeats=2, seed=1984)
    assert json_sha256(starts.schedule) == expected_hash
    selected = benchmark.EvalSetStarts(SETS / filename, splits=["id"], repeats=2, seed=1984)
    assert selected.schedule == [row for row in starts.schedule if row["split"] == "id"]


def test_smoke_selection_uses_the_shared_selector():
    starts = benchmark.EvalSetStarts(SETS / "pick_place_v1_seed1984.json", selection="one_per_zone")
    assert len(starts.schedule) == len({row["zone"] for row in starts.schedule})
    assert starts.schedule


def test_first_frame_states_follow_requested_episode_order(tmp_path):
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
