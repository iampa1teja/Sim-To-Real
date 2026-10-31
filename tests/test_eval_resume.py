"""Continuation preserves results and rejects incompatible benchmark histories."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from sim_to_real_so101.utils.eval_resume import find_report, load_episodes, next_video_attempt, write_report
from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts


SCHEDULE = [dict(id=f"s{i}", repeat=0, policy_seed=100 + i, split="id") for i in range(3)]
IDENTITY = dict(task="pick_place", checkpoint="so101", config_fingerprint="same", action_horizon=30)


def report(count=1):
    return dict(IDENTITY, complete=count == len(SCHEDULE), episodes=[
        dict(episode=i, start_id=row["id"], repeat=0, policy_seed=row["policy_seed"], split="id",
             steps=20, success=True, success_step=20) for i, row in enumerate(SCHEDULE[:count])])


def test_failed_checkpoint_write_keeps_previous_json_and_removes_temporary_file(tmp_path):
    path = tmp_path / "results.json"
    write_report(path, report())
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_report(path, {"bad": float("nan")})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    write_report(path, report(2))
    assert len(json.loads(path.read_text())["episodes"]) == 2


def test_latest_matching_results_ignores_newer_other_configuration(tmp_path):
    for index, name in enumerate(("old", "latest", "other", "corrupt")):
        path = tmp_path / f"so101_{name}.json"
        data = report(3 if name == "latest" else 1)
        if name == "other":
            data["config_fingerprint"] = "different"
        path.write_text("bad JSON" if name == "corrupt" else json.dumps(data))
        os.utime(path, ns=(1000 + index, 1000 + index))
    expected = tmp_path / "so101_latest.json"
    assert find_report(tmp_path, "so101_*.json", "same", "so101", "pick_place") == expected
    assert find_report(expected, "unused", "same", "so101", "pick_place") == expected
    with pytest.raises(ValueError, match="No matching"):
        find_report(tmp_path, "so101_*.json", "missing", "so101", "pick_place")


@pytest.mark.parametrize("change", ["horizon", "seed", "repeat", "start", "duplicate", "partial", "too_many"])
def test_resume_rejects_incompatible_or_incomplete_rows(tmp_path, change):
    data = report(2)
    if change == "horizon":
        data["action_horizon"] = 8
    elif change == "seed":
        data["episodes"][0]["policy_seed"] += 1
    elif change == "repeat":
        data["episodes"][0]["repeat"] = 1
    elif change == "start":
        data["episodes"][0]["start_id"] = "wrong"
    elif change == "duplicate":
        data["episodes"][1] = copy.deepcopy(data["episodes"][0])
    elif change == "partial":
        data["episodes"][0]["steps"] = 0
    else:
        data["episodes"] *= 2
    path = tmp_path / "report.json"
    write_report(path, data)
    with pytest.raises(ValueError, match="Cannot resume"):
        load_episodes(path, IDENTITY, SCHEDULE)


def test_empty_first_episode_checkpoint_and_legacy_reports_can_resume(tmp_path):
    path = tmp_path / "report.json"
    for count in (0, 1, 3):
        data = report(count)
        data.pop("config_fingerprint")
        write_report(path, data)
        assert load_episodes(path, IDENTITY, SCHEDULE)["episodes"] == data["episodes"]


def test_video_restart_skips_all_existing_attempts_including_partial_files(tmp_path):
    assert next_video_attempt(tmp_path) == 0
    for name in ("episode_000000_room.mp4", "episode_000003_wrist.mp4", "unrelated.mp4"):
        (tmp_path / name).write_bytes(b"retain")
    assert next_video_attempt(tmp_path) == 4
    assert all(path.read_bytes() == b"retain" for path in tmp_path.iterdir())


def test_benchmark_offset_preserves_original_start_and_policy_seed():
    path = Path(__file__).resolve().parents[1] / "eval_sets/pick_place_v1_seed1984.json"
    original = EvalSetStarts(path, repeats=2, seed=17)
    resumed = EvalSetStarts(path, repeats=2, seed=17, start_offset=152)
    index, *_ = resumed.sample()
    assert index == 152
    assert resumed.current == original.schedule[152]
    assert resumed.robot_state(index).tolist() == original.robot_state(index).tolist()
