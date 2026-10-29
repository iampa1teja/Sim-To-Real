"""Pose rule boundaries, synthetic report actions and lock mutation guards."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.calibration_check import CalibrationGapLogger
from sim_to_real_so101.utils.calibration_poses import (
    POSE_SET_PATH, PoseThresholds, box_vertices, check_clearance, load_pose_set, summarize_poses,
)
from sim_to_real_so101.utils.calibration_report import generate_report, lock_report, recommended_actions
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS, SETUP_PATH, file_sha256, resolve_calibration


def measurements(offsets):
    return [dict(type="pose", accepted=True, target_deg=[i * 10.] * 6, real_minus_target_deg=[offset] * 5 + [99.],
                 at_limit={}) for i, offset in enumerate(offsets)]


@pytest.mark.parametrize("values, verdict, reason", [
    ([1.] * 5, "PASS", "WITHIN_PASS"), ([-1.] * 5, "PASS", "WITHIN_PASS"),
    ([1.00001] * 5, "WARN", "WITHIN_WARN"), ([2.] * 5, "WARN", "WITHIN_WARN"),
    ([-2.] * 5, "WARN", "WITHIN_WARN"), ([2.00001] * 5, "FAIL", "OFFSET"),
    ([0, .25, .5, .75, 1], "PASS", "WITHIN_PASS"),
    ([0, .25, .5, .75, 1.00001], "FAIL", "SPREAD"), ([0.] * 4, "FAIL", "INSUFFICIENT"),
    ([], "FAIL", "INSUFFICIENT"),
])
def test_pose_lock_rule(values, verdict, reason):
    result = summarize_poses(measurements(values))
    assert all(j["verdict"] == verdict and j["reason"] == reason for j in result["joints"].values())
    assert result["decision"] == ("NOT_LOCKABLE" if verdict == "FAIL" else "LOCKABLE")
    assert result["gripper"]["verdict"] == "FAIL"


def test_pose_threshold_overrides_rejections_limits_and_duplicates():
    rows = measurements([2.5] * 6)
    result = summarize_poses(rows, PoseThresholds(warn_deg=3, min_poses=6))
    assert result["decision"] == "LOCKABLE" and result["thresholds"]["warn_deg"] == 3
    rows[0]["accepted"] = False
    rows[1]["at_limit"] = {"shoulder_pan": True}
    rows[2]["target_deg"] = rows[3]["target_deg"]
    result = summarize_poses(rows)
    assert result["joints"]["shoulder_pan"]["accepted_poses"] == 3
    assert result["decision"] == "NOT_LOCKABLE"


def test_committed_pose_coverage_and_collision_margin(tmp_path):
    setup = json.loads(SETUP_PATH.read_text())
    bounds = [[setup["runtime_joint_limits"][n][k] for k in ("joint_min", "joint_max")] for n in FOLLOWER_JOINTS]
    poses, identity = load_pose_set(FOLLOWER_JOINTS, bounds)
    assert 7 <= len(poses) <= 9 and np.all(poses[0][1] == 0)
    assert min(identity["coverage_fraction"].values()) >= .6
    altered = tmp_path / "bad.toml"
    altered.write_text(POSE_SET_PATH.read_text().replace("margin_m = 0.02", "margin_m = 0.019"))
    with pytest.raises(ValueError, match="2 cm"):
        load_pose_set(FOLLOWER_JOINTS, bounds, altered)
    box = box_vertices([0, 0, .021], [.1, .1, .1])
    result = check_clearance({"a": (box, np.eye(3)), "b": (box + [.13, 0, 0], np.eye(3))}, set(), np.zeros(3), [0, 0, 1])
    assert result["status"] == "OK"
    result = check_clearance({"a": (box - [0, 0, .002], np.eye(3))}, set(), np.zeros(3), [0, 0, 1])
    assert result["status"] == "REJECT"


def synthetic_logs(tmp_path, offsets=None):
    follower, leader = tmp_path / "f.json", tmp_path / "l.json"
    for path in (follower, leader):
        path.write_bytes((ROOT / "tests/fixtures/rpd_robot.json").read_bytes())
    selection = resolve_calibration(environ={"SO101_FOLLOWER_CALIBRATION": str(follower), "SO101_LEADER_CALIBRATION": str(leader)})
    header = dict(selection.provenance, type="provenance", mode="operator_poses", git_commit="synthetic", git_dirty=False,
                  joint_order=list(FOLLOWER_JOINTS), pose_set={"path": str(POSE_SET_PATH), "sha256": file_sha256(POSE_SET_PATH)},
                  pose_thresholds=asdict(PoseThresholds()))
    pose = tmp_path / "poses.jsonl"
    rows = measurements(offsets if offsets is not None else [.5] * 6)
    pose.write_text("\n".join(json.dumps(r) for r in [header, *rows]) + "\n")
    gap = tmp_path / "gap.jsonl"
    with CalibrationGapLogger(gap, joint_order=FOLLOWER_JOINTS, runtime_limits=[[-100, 100]] * 6, provenance=header) as log:
        for i in range(100):
            angle = i * .01
            real = np.array([angle] * 5 + [100.])
            sim = real + .5
            log.record(commanded_deg=real - 4, real_deg=real, sim_deg=sim, sim_target_deg=sim - (2 + angle),
                       timestamp_ns=1_000_000_000 + i * 100_000_000)
    return pose, gap


def test_report_from_synthetic_logs_and_plots(tmp_path):
    pose, gap = synthetic_logs(tmp_path, [3.] * 6)
    before = [file_sha256(p) for p in (pose, gap)]
    report = generate_report(pose, gap, tmp_path / "report")
    assert report["calibration"]["decision"] == "NOT_LOCKABLE"
    rules = {(a["joint"], a["rule"]) for a in report["actions"]}
    for name in FOLLOWER_JOINTS[:-1]:
        assert all((name, rule) in rules for rule in ("pose_fail", "real_servo", "sim_stiffness"))
    assert ("gripper", "limit_mismatch") in rules
    stats = report["tracking"]["joints"]["elbow_flex"]["static_frames"]
    assert stats["real_minus_commanded_deg"]["mean_signed_deg"] == pytest.approx(4)
    assert stats["sim_minus_sim_target_deg"]["correlation"] == pytest.approx(1)
    text = (tmp_path / "report/report.md").read_text()
    assert "NOT a calibration verdict" in text and "Mean ± std" in text
    assert before == [file_sha256(p) for p in (pose, gap)]
    for name in report["plots"]["files"]:
        assert (tmp_path / "report" / name).read_bytes().startswith(b"\x89PNG")


def test_strict_action_boundaries(tmp_path):
    pose, gap = synthetic_logs(tmp_path)
    report = generate_report(pose, gap, tmp_path / "report")
    tracking = report["tracking"]
    tracking["gripper"] = None
    for joint in tracking["joints"].values():
        joint["static_frames"]["real_minus_commanded_deg"]["mean_abs_deg"] = 3
        joint["static_frames"]["sim_minus_sim_target_deg"].update(mean_signed_deg=1, correlation=.8)
        joint["counts"].update(total=100, excluded_at_limit=5)
    assert recommended_actions(report["calibration"], tracking) == []


@pytest.mark.parametrize("mutation", ["verdict", "hash", "joint", "setup"])
def test_lock_refuses_without_mutation(tmp_path, mutation):
    pose, gap = synthetic_logs(tmp_path)
    report = generate_report(pose, gap, tmp_path / "report")
    setup = tmp_path / "source/sim_to_real_so101/assets/real_setup.json"
    setup.parent.mkdir(parents=True)
    setup.write_bytes(SETUP_PATH.read_bytes())
    if mutation == "verdict":
        report["calibration"]["decision"] = "NOT_LOCKABLE"
    elif mutation == "hash":
        Path(report["identity"]["follower_calibration"]["path"]).write_text("changed")
    elif mutation == "joint":
        report["calibration"]["joints"]["shoulder_pan"]["accepted_poses"] = 0
    else:
        setup.write_text("changed")
    path = tmp_path / "report/report.json"
    path.write_text(json.dumps(report))

    def forbid(*args, **kwargs):
        raise AssertionError("Updater must not run")
    with pytest.raises(ValueError, match="Refusing lock"):
        lock_report(path, root=tmp_path, runner=forbid)
    assert not (tmp_path / "docs").exists()


def test_lock_calls_updater_and_writes_audit_without_git_mutations(tmp_path, capsys):
    pose, gap = synthetic_logs(tmp_path)
    report = generate_report(pose, gap, tmp_path / "report")
    setup = tmp_path / "source/sim_to_real_so101/assets/real_setup.json"
    setup.parent.mkdir(parents=True)
    setup.write_bytes(SETUP_PATH.read_bytes())
    calls = []

    def update(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": True}
        setup.write_text(setup.read_text() + "\n")
    path = tmp_path / "report/report.json"
    audit = lock_report(path, root=tmp_path, now=datetime(2026, 10, 29), runner=update)
    value = json.loads(audit.read_text())
    assert value["report"]["sha256"] == file_sha256(path)
    assert value["pose_set"]["sha256"] == file_sha256(POSE_SET_PATH)
    assert len(calls) == 1 and calls[0][1].endswith("scripts/update_sim_calibration.py")
    assert "git tag calib-20261029" in capsys.readouterr().out
    assert file_sha256(report["identity"]["follower_calibration"]["path"]) == report["identity"]["follower_calibration"]["sha256"]


def test_gripper_motion_never_rejects_body_pose():
    from sim_to_real_so101.utils.calibration_check import pose_measurement

    mapping = resolve_calibration(environ={}).mapping
    samples = np.zeros((30, 6))
    samples[:, -1] = np.arange(30)
    pose = pose_measurement("gripper-moving", [0] * 6, {"observations": samples.tolist()},
                            mapping, FOLLOWER_JOINTS, [[-500, 500]] * 6)
    assert pose["accepted"] and not pose["joint_accepted"]["gripper"]
