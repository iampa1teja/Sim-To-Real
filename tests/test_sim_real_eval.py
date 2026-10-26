"""Paired sim/real evaluation: stage ranks, trial records, summaries, report and the session state machine. CPU only."""
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

from sim_to_real_so101.utils.pick_place_benchmark import STAGES  # noqa: E402
from sim_to_real_so101.utils.sim_real_eval import (
    STAGE_ORDER,
    SimRealSession,
    Trial,
    append_trial,
    furthest_stage,
    load_trials,
    stage_rank,
    summarize_trials,
    write_trials_report,
)

START = {"start_id": "id-000", "split": "id", "x": 0.1, "y": -0.1, "yaw": 0.0,
         "arm_start_state": [0, 0, 0, 0, 0, 0]}


def make_trial(trial_id, sim_stage, real_stage=None):
    real = None if real_stage is None else {"stage": real_stage, "note": ""}
    return Trial(trial_id=trial_id, start_id="id-000", split="id", x=0.1, y=-0.1, yaw=0.0,
                 sim={"stage": sim_stage}, real=real, created="2026-01-01T00:00:00")


def make_session(tmp_path, calls, sim_result=None, real_result=None, sim_error=None, **kwargs):
    sim_result = sim_result if sim_result is not None else {
        "stage": "grasped", "success": False, "steps": 5, "seconds": 0.5, "failure_mode": "timeout_holding"}
    real_result = real_result if real_result is not None else {
        "status": "completed", "steps": 3, "seconds": 0.1, "detail": ""}

    def spawner(start):
        calls.append(("spawn", dict(start)))

    def sim_runner(start):
        calls.append(("sim", dict(start)))
        if sim_error is not None:
            raise sim_error
        return dict(sim_result)

    def real_runner(start, should_stop):
        calls.append(("real", dict(start)))
        return dict(real_result)

    return SimRealSession(spawner, sim_runner, real_runner, tmp_path / "out", **kwargs)


def run_to_real_done(session):
    session.spawn(START)
    session.mark_placed()
    session.run_sim()
    session.run_real(lambda: False)


def test_stage_rank_and_furthest_stage():
    assert stage_rank("none") == 0
    assert stage_rank("success") == 6
    with pytest.raises(ValueError):
        stage_rank("flying")
    stages = {"reached": True, "grasped": True, "lifted": False, "over_box": False, "placed": False,
              "success": False}
    assert furthest_stage(stages) == "grasped"
    assert furthest_stage({name: False for name in STAGES}) == "none"


def test_summary_rates_confusion_and_pairs():
    trials = [
        make_trial(0, "success", "success"),
        make_trial(1, "success", "grasped"),
        make_trial(2, "reached", "none"),
        make_trial(3, "placed", "placed"),
    ]
    summary = summarize_trials(trials)
    assert summary["n"] == 4

    reached = summary["stage_rates"]["reached"]
    assert reached["sim"] == pytest.approx(1.0)
    assert reached["real"] == pytest.approx(0.75)
    assert reached["gap"] == pytest.approx(0.25)

    success = summary["stage_rates"]["success"]
    assert success["sim"] == pytest.approx(0.5)
    assert success["real"] == pytest.approx(0.25)
    assert success["gap"] == pytest.approx(0.25)

    confusion = summary["confusion"]
    assert confusion["success"]["grasped"] == 1
    assert confusion["reached"]["none"] == 1
    for sim_stage in STAGE_ORDER:
        for real_stage in STAGE_ORDER:
            assert real_stage in confusion[sim_stage]

    paired = summary["paired_success"]
    assert paired["both"] == 1
    assert paired["sim_only"] == 1
    assert paired["real_only"] == 0
    assert paired["neither"] == 2
    assert "p_value" in paired["mcnemar"]


def test_summary_empty():
    summary = summarize_trials([])
    assert summary["n"] == 0
    for stage in STAGES:
        rates = summary["stage_rates"][stage]
        assert rates == {"sim": None, "real": None, "gap": None}


def test_summary_ignores_unlabelled():
    trials = [make_trial(0, "success", "success"), make_trial(1, "success", None)]
    assert summarize_trials(trials)["n"] == 1


def test_jsonl_round_trip_and_override(tmp_path):
    path = tmp_path / "trials.jsonl"
    assert load_trials(path) == []

    append_trial(path, make_trial(0, "reached", "none"))
    append_trial(path, make_trial(1, "reached", "none"))
    append_trial(path, make_trial(1, "reached", "placed"))

    loaded = load_trials(path)
    assert len(loaded) == 2
    by_id = {t.trial_id: t for t in loaded}
    assert by_id[1].real["stage"] == "placed"
    assert by_id[0].real["stage"] == "none"
    assert by_id[1].sim == {"stage": "reached"}


def test_report_schema(tmp_path):
    trials = [make_trial(0, "success", "success"), make_trial(1, "grasped", "none")]
    header = {"checkpoint": "ckpt-1", "set_hash": "abc"}
    path = tmp_path / "report.json"

    write_trials_report(path, header, trials)

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["schema"] == "so101_sim_real_eval_v1"
    assert report["header"] == header
    assert len(report["trials"]) == len(trials)
    assert "summary" in report
    assert report["summary"]["n"] == 2
    assert not (tmp_path / "report.json.tmp").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_session_full_trial(tmp_path):
    calls = []
    session = make_session(tmp_path, calls)
    session.spawn(START)
    session.mark_placed()
    session.run_sim()
    session.run_real(lambda: False)
    session.label_real("reached", "slipped")
    trial = session.save()

    lines = (tmp_path / "out" / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert (tmp_path / "out" / "report.json").exists()
    snap = session.snapshot()
    assert snap["state"] == "idle"
    assert trial.real["stage"] == "reached"
    assert trial.real["note"] == "slipped"
    assert [name for name, _ in calls] == ["spawn", "sim", "real"]


def test_run_real_requires_sim_and_placed(tmp_path):
    calls = []
    session = make_session(tmp_path, calls)
    session.spawn(START)
    with pytest.raises(RuntimeError):
        session.run_real(lambda: False)

    session.run_sim()
    with pytest.raises(RuntimeError):
        session.run_real(lambda: False)
    assert [name for name, _ in calls] == ["spawn", "sim"]


def test_run_sim_requires_spawn(tmp_path):
    session = make_session(tmp_path, [])
    with pytest.raises(RuntimeError):
        session.run_sim()


def test_label_validation(tmp_path):
    session = make_session(tmp_path, [])
    run_to_real_done(session)
    with pytest.raises((ValueError, RuntimeError)):
        session.label_real("flying")
    with pytest.raises(RuntimeError):
        session.save()
    assert not (tmp_path / "out" / "trials.jsonl").exists()


def test_no_concurrent_runs(tmp_path):
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def spawner(start):
        calls.append("spawn")

    def blocking_sim(start):
        entered.set()
        release.wait(5)
        return {"stage": "none", "success": False, "steps": 1, "seconds": 0.1, "failure_mode": ""}

    def real_runner(start, should_stop):
        return {"status": "completed", "steps": 1, "seconds": 0.1, "detail": ""}

    session = SimRealSession(spawner, blocking_sim, real_runner, tmp_path / "out")
    session.spawn(START)
    worker = threading.Thread(target=session.run_sim)
    worker.start()
    try:
        deadline = time.monotonic() + 2.0
        while session.snapshot()["state"] != "sim_running":
            if time.monotonic() > deadline:
                pytest.fail("sim run did not reach sim_running within 2 s")
            time.sleep(0.01)
        assert entered.wait(2.0)

        with pytest.raises(RuntimeError):
            session.run_sim()
        with pytest.raises(RuntimeError):
            session.run_real(lambda: False)
        with pytest.raises(RuntimeError):
            session.spawn(START)
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert session.snapshot()["state"] == "sim_done"


def test_rerun_sim_clears_real(tmp_path):
    calls = []
    session = make_session(tmp_path, calls)
    run_to_real_done(session)
    session.label_real("placed")

    session.run_sim()

    snap = session.snapshot()
    assert snap["real"] is None
    assert snap["label"] is None
    assert snap["state"] == "sim_done"
    assert [name for name, _ in calls] == ["spawn", "sim", "real", "sim"]


def test_discard_writes_nothing(tmp_path):
    session = make_session(tmp_path, [])
    session.spawn(START)
    session.run_sim()
    session.discard()

    assert not (tmp_path / "out" / "trials.jsonl").exists()
    assert session.snapshot()["state"] == "idle"


def test_runner_exception_restores_state(tmp_path):
    session = make_session(tmp_path, [], sim_error=ValueError("sim exploded"))
    session.spawn(START)
    with pytest.raises(ValueError):
        session.run_sim()
    assert session.snapshot()["state"] == "spawned"


def test_resume_next_id(tmp_path):
    session = make_session(tmp_path, [])
    run_to_real_done(session)
    session.label_real("none")
    session.save()

    resumed = SimRealSession(lambda s: None, lambda s: {"stage": "none"}, lambda s, stop: {"stage": "none"},
                             tmp_path / "out", trials=load_trials(tmp_path / "out" / "trials.jsonl"))
    assert resumed.snapshot()["next_trial_id"] == 1
    assert resumed.snapshot()["trial_count"] == 1
