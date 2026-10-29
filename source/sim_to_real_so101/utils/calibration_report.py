"""Combined offline evidence and an explicit, hash-guarded simulation lock."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np

from .calibration_check import GAPS, gap_data, summarize
from .calibration_poses import BODY_JOINTS, PoseThresholds, read_pose_log
from .joint_calibration import calibration_identity, file_sha256

ROOT = Path(__file__).resolve().parents[3]
ACTION_THRESHOLDS = {"real_tracking_deg": 3., "sim_mean_deg": 1., "sim_angle_correlation": .8,
                     "limit_fraction": .05}
ACTUATORS = dict(zip(BODY_JOINTS, ("rotation", "pitch", "elbow", "wrist_pitch", "wrist_roll")), gripper="gripper")


def recommended_actions(calibration, tracking):
    """Apply the operator's four rules; tracking never changes the pose verdict."""
    actions = []
    for name, joint in calibration["joints"].items():
        if joint["verdict"] == "FAIL":
            actions.append(dict(joint=name, rule="pose_fail", action=f"Recalibrate {name}; pose check: {joint['reason']}.",
                                evidence=joint))
    joints = dict(tracking["joints"])
    if tracking.get("gripper"):
        joints["gripper"] = tracking["gripper"]
    for name, joint in joints.items():
        real, sim = (joint["static_frames"][key] for key in GAPS[1:])
        if real["mean_abs_deg"] is not None and real["mean_abs_deg"] > ACTION_THRESHOLDS["real_tracking_deg"]:
            actions.append(dict(joint=name, rule="real_servo", action=f"Real servo problem on {name}: inspect gains, load or blockage.",
                                evidence=real))
        if (sim["correlation"] is not None and abs(sim["correlation"]) > ACTION_THRESHOLDS["sim_angle_correlation"]
                and abs(sim["mean_signed_deg"]) > ACTION_THRESHOLDS["sim_mean_deg"]):
            actions.append(dict(joint=name, rule="sim_stiffness",
                                action=f"Sim stiffness is too low for {name}: inspect assets/so101.py actuator group {ACTUATORS[name]}.",
                                evidence=sim))
        counts = joint["counts"]
        fraction = counts["excluded_at_limit"] / counts["total"] if counts["total"] else 0
        if fraction > ACTION_THRESHOLDS["limit_fraction"]:
            actions.append(dict(joint=name, rule="limit_mismatch", action=f"Limit mismatch on {name}: inspect runtime limits.",
                                evidence={"fraction_at_limit": fraction}))
    return actions


def git_identity(root=ROOT):
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    try:
        return {"git_commit": subprocess.check_output(command + ["rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip(),
                "git_dirty": bool(subprocess.check_output(command + ["status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def combined_identity(pose, gap):
    """Refuse to combine sessions that used different calibrations or mappings."""
    for key in ("follower_calibration", "real_setup"):
        if not pose.get(key, {}).get("sha256") or pose[key]["sha256"] != (gap.get(key) or {}).get("sha256"):
            raise ValueError(f"Pose/gap {key} identity is missing or differs; capture both sessions with the same calibration")
    if pose.get("mapping") != gap.get("mapping") or pose.get("mapping_parameters") != gap.get("mapping_parameters"):
        raise ValueError("Pose/gap mappings differ")
    leader = gap.get("leader_calibration") or {}
    if not leader.get("path") or not leader.get("sha256"):
        raise ValueError("Gap log must identify the connected leader calibration")
    pose_leader = pose.get("leader_calibration") or {}
    if pose_leader.get("sha256") and pose_leader["sha256"] != leader["sha256"]:
        raise ValueError("Pose/gap leader calibration identities differ")
    follower = pose["follower_calibration"]
    return dict(follower_calibration=follower, leader_calibration=leader, real_setup=pose["real_setup"],
                follower_id=pose.get("follower_id") or Path(follower["path"]).stem,
                matches_embedded_calibration=follower["sha256"] == pose.get("embedded_follower_sha256"),
                embedded_follower_sha256=pose.get("embedded_follower_sha256"),
                git_commit=pose.get("git_commit"), git_dirty=pose.get("git_dirty"),
                gap_git_commit=gap.get("git_commit"), gap_git_dirty=gap.get("git_dirty"),
                pose_set=pose["pose_set"], mapping=pose["mapping"], mapping_parameters=pose["mapping_parameters"])


def make_plots(output, measurements, calibration, gap_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return {"status": "skipped", "reason": "matplotlib is not installed", "files": []}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for i, (name, ax) in enumerate(zip((*BODY_JOINTS, "gripper"), axes.flat)):
        for width, color in ((2, "#fff0c2"), (1, "#d4efdf")):
            ax.axhspan(-width, width, color=color)
        for index, row in enumerate(measurements):
            accepted = row.get("joint_accepted", {}).get(name, row["accepted"]) if name == "gripper" else row["accepted"]
            accepted = accepted and not row.get("at_limit", {}).get(name, False)
            ax.scatter(index + 1, row["real_minus_target_deg"][i], marker="o" if accepted else "x",
                       color="#1261a0" if accepted else "#a32121")
        ax.set(title=name + (" (nonblocking)" if name == "gripper" else ""), xlabel="Reference pose", ylabel="Real − target (°)")
        ax.axhline(0, color="gray", linewidth=.6)
    fig.savefig(output / "pose-offsets.png", dpi=150)
    plt.close(fig)
    _, _, order, _, still, _, gaps = gap_data(gap_path)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for i, (name, ax) in enumerate(zip(order, axes.flat)):
        for key in GAPS:
            values = gaps[key][still[:, i], i]
            values = values[np.isfinite(values)]
            if len(values):
                ax.hist(values, bins=20, histtype="step", label=key.removesuffix("_deg"))
        ax.set(title=name, xlabel="Gap on still frames (°)", ylabel="Frames")
        if ax.patches:
            ax.legend(fontsize=7)
    fig.savefig(output / "tracking-gaps.png", dpi=150)
    plt.close(fig)
    return {"status": "written", "files": ["pose-offsets.png", "tracking-gaps.png"],
            "pose_bands_deg": [1, 2], "tracking_scope": "all still frames, including runtime limits"}


def report_markdown(report):
    def n(value):
        return "—" if value is None else f"{value:.3f}"

    calibration, tracking = report["calibration"], report["tracking"]
    lines = ["# Calibration report", "", f"Calibration decision: **{calibration['decision']}**", "",
             "## Calibration identity", "", "```json", json.dumps(report["identity"], indent=2), "```", "",
             "## Independent pose check", "", f"Thresholds used: `{json.dumps(calibration['thresholds'])}`.", "",
             "Offsets are real − target. Spread is max − min across distinct accepted joint targets.", "",
             "| Joint | Median ° | Spread ° | Accepted poses | Verdict | Reason |",
             "| --- | ---: | ---: | ---: | --- | --- |"]
    for name, joint in {**calibration["joints"], "gripper (never blocks)": calibration["gripper"]}.items():
        lines.append(f"| {name} | {n(joint['median_offset_deg'])} | {n(joint['spread_deg'])} | "
                     f"{joint['accepted_poses']} | {joint['verdict']} | {joint['reason']} |")
    lines += ["", "## Tracking — NOT a calibration verdict", "",
              "Statistics use all still frames (including limits). Correlation is with real joint angle. "
              "Real-tracking and sim-tracking verdicts use mean absolute gap on still, unsaturated frames.", "",
              f"Tracking thresholds: `{json.dumps(tracking['thresholds'])}`.", "",
              "| Joint | Real-tracking | Sim-tracking | Frames at limit |", "| --- | --- | --- | ---: |"]
    joints = {**tracking["joints"], "gripper": tracking["gripper"]}
    for name, joint in joints.items():
        counts = joint["counts"]
        fraction = counts["excluded_at_limit"] / counts["total"] if counts["total"] else 0
        lines.append(f"| {name} | {joint['real_tracking_verdict']} | {joint['sim_tracking_verdict']} | {fraction:.1%} |")
    lines += ["", "| Joint | Gap | n | Mean ± std ° | Angle correlation r |", "| --- | --- | ---: | --- | ---: |"]
    for name, joint in joints.items():
        for key, stats in joint["static_frames"].items():
            lines.append(f"| {name} | {key.removesuffix('_deg')} | {stats['count']} | "
                         f"{n(stats['mean_signed_deg'])} ± {n(stats['std_deg'])} | {n(stats['correlation'])} |")
    lines += ["", "## Recommended actions", "", f"Rule thresholds: `{json.dumps(report['action_thresholds'])}`.", ""]
    lines += [f"- {action['action']}" for action in report["actions"]] or ["No action rule was triggered."]
    lines += ["", "## Plots", ""]
    if report["plots"]["status"] == "written":
        lines += [f"![{name}]({name})" for name in report["plots"]["files"]]
    else:
        lines.append(f"Skipped: {report['plots']['reason']}.")
    lines += ["", "## Evidence", "", "```json", json.dumps(report["inputs"], indent=2), "```", "",
              "Operator visual matching has limited angular accuracy. Tracking is not an independent physical zero measurement.", ""]
    return "\n".join(lines)


def generate_report(pose_path, gap_path, output, thresholds=None):
    pose_header, measurements, calibration = read_pose_log(pose_path, thresholds)
    tracking = summarize(gap_path, write=False)
    if tracking["provenance"]["joint_order"] != [*BODY_JOINTS, "gripper"]:
        raise ValueError("Gap log must contain all six joints in follower order")
    identity = combined_identity(pose_header, tracking["provenance"])
    output = Path(output)
    for name in ("report.json", "report.md", "pose-offsets.png", "tracking-gaps.png"):
        if (output / name).resolve() in {Path(pose_path).resolve(), Path(gap_path).resolve()}:
            raise ValueError("Report output must not replace input logs")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
                  generator=git_identity(), calibration=calibration, tracking=tracking,
                  inputs={"poses": calibration_identity(pose_path), "gap": calibration_identity(gap_path)},
                  action_thresholds=ACTION_THRESHOLDS, actions=recommended_actions(calibration, tracking))
    report["plots"] = make_plots(output, measurements, calibration, gap_path)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report_markdown(report))
    return report


def lock_report(path, *, root=ROOT, now=None, runner=subprocess.run):
    """Update only simulation calibration and write an audit; never commit or tag."""
    path, root = Path(path).resolve(), Path(root)
    report = json.loads(path.read_text())
    calibration = report.get("calibration", {})
    if calibration.get("decision") != "LOCKABLE":
        raise ValueError("Refusing lock: report is not LOCKABLE")
    thresholds = PoseThresholds(**calibration["thresholds"])
    for name in BODY_JOINTS:
        joint = calibration.get("joints", {}).get(name, {})
        if (joint.get("verdict") not in {"PASS", "WARN"} or joint.get("accepted_poses", 0) < thresholds.min_poses
                or not np.isfinite([joint.get("median_offset_deg", np.nan), joint.get("spread_deg", np.nan)]).all()
                or abs(joint["median_offset_deg"]) > thresholds.warn_deg or not 0 <= joint["spread_deg"] <= thresholds.spread_deg):
            raise ValueError(f"Refusing lock: {name} does not satisfy LOCKABLE thresholds")
    identity = report["identity"]
    follower = identity["follower_calibration"]
    if file_sha256(follower["path"]) != follower["sha256"]:
        raise ValueError("Refusing lock: follower file hash mismatch")
    setup = root / "source/sim_to_real_so101/assets/real_setup.json"
    if file_sha256(setup) != identity["real_setup"]["sha256"]:
        raise ValueError("Refusing lock: real_setup.json changed since the pose check")
    if not identity.get("pose_set", {}).get("sha256"):
        raise ValueError("Refusing lock: missing pose-set hash")
    identifier = identity["follower_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", identifier):
        raise ValueError("Invalid follower ID in report")
    date = (now or datetime.now()).strftime("%Y%m%d")
    audit_path = root / f"docs/reports/calibration-locks/{date}-{identifier}.json"
    if audit_path.exists():
        raise FileExistsError(f"Lock record already exists: {audit_path}")
    report_hash = file_sha256(path)
    runner([sys.executable, str(root / "scripts/update_sim_calibration.py"), "--calibration", follower["path"],
            "--follower-id", identifier], check=True)
    if file_sha256(follower["path"]) != follower["sha256"]:
        raise ValueError("Follower changed during lock; review simulation output")
    audit = dict(schema_version=1, **git_identity(root), follower_calibration=follower,
                 leader_calibration=identity["leader_calibration"], real_setup_before=identity["real_setup"],
                 real_setup_after=calibration_identity(setup), report={"path": str(path), "sha256": report_hash},
                 pose_set=identity["pose_set"], thresholds=calibration["thresholds"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x") as stream:
        stream.write(json.dumps(audit, indent=2) + "\n")
    commands = [shlex.join(["git", "add", str(setup.relative_to(root)), str(audit_path.relative_to(root))]),
                shlex.join(["git", "commit", "-m", f"Lock {identifier} calibration {date}"]),
                shlex.join(["git", "tag", f"calib-{date}"])]
    print("Lock written. Review and run these commands yourself:\n" + "\n".join(commands))
    return audit_path
