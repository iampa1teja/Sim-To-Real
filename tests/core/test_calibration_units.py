"""Mixed-joint telemetry keeps metre fields and thresholds separate from degrees."""
import json

import pytest

from core.calibration.gap import CalibrationGapLogger, Thresholds, summarize
from core.calibration.report import generate_report


def write_log(path, *, units, real, sim, commanded=None):
    with CalibrationGapLogger(path, joint_order=[f"j{i}" for i in range(len(units))],
                              runtime_limits=[[-1000, 1000]] * len(units),
                              provenance=dict(joint_units=units, gripper_joints=["j1"])) as log:
        for index in range(5):
            row = log.record(real=real, sim=sim, commanded=real if commanded is None else commanded,
                             sim_target=real, timestamp_ns=index * 100_000_000)
    return row, log.summary


def test_mixed_physical_schema_has_no_degree_labelled_metre_fields(tmp_path):
    path = tmp_path / "mixed.jsonl"
    row, report = write_log(path, units=["deg", "m"], real=[0.0, 0.0], sim=[0.5, 0.002])
    assert "real" in row and "real_deg" not in row
    assert "sim_minus_real" in row and "sim_minus_real_deg" not in row
    assert report["schema_version"] == 3
    rotary = report["joints"]["j0"]
    slide = report["grippers"]["j1"]
    assert rotary["sim_tracking_verdict"] == "PASS"
    assert slide["sim_tracking_verdict"] == "WARN"
    stats = slide["static_not_at_limit"]["sim_minus_sim_target"]
    assert stats["unit"] == "m" and stats["mean_abs"] == pytest.approx(0.002)
    assert "mean_abs_deg" not in stats
    # Real report generation consumes the generic keys and preserves unit-labelled rows.
    generate_report(path, tmp_path / "report")
    assert "m" in (tmp_path / "report/report.md").read_text()


def test_revolute_only_log_keeps_legacy_degree_fields(tmp_path):
    row, report = write_log(tmp_path / "rotary.jsonl", units=["deg"], real=[0.0], sim=[0.5])
    assert row["real_deg"] == [0.0]
    assert row["sim_minus_real_deg"] == [0.5]
    assert report["schema_version"] == 2
    assert report["joints"]["j0"]["static_not_at_limit"]["sim_minus_real_deg"]["mean_abs_deg"] == 0.5


def test_metre_stillness_and_branch_detection_are_not_degree_thresholds(tmp_path):
    path = tmp_path / "moving.jsonl"
    with CalibrationGapLogger(path, joint_order=["slide"], runtime_limits=[[-1000, 1000]],
                              provenance=dict(joint_units=["m"])) as log:
        for index in range(5):
            # 0.01 m/s exceeds the default 0.001 m/s stillness gate.
            row = log.record(real=[index * 0.001], sim=[200.0], sim_target=[0.0],
                             timestamp_ns=index * 100_000_000)
        assert not row["static"][0]
        assert not row["suspected_branch_error"][0]
    assert log.summary["joints"]["slide"]["sim_tracking_verdict"] == "NO_DATA"


def test_explicit_metre_thresholds_change_only_metric_verdict(tmp_path):
    path = tmp_path / "configured.jsonl"
    write_log(path, units=["deg", "m"], real=[0.0, 0.0], sim=[0.5, 0.002])
    result = summarize(path, Thresholds(pass_m=0.003, warn_m=0.005), write=False)
    assert result["joints"]["j0"]["sim_tracking_verdict"] == "PASS"
    assert result["grippers"]["j1"]["sim_tracking_verdict"] == "PASS"
