"""`s2r calibration-report --gap <telemetry.jsonl>`: tracking report from a recording joint-angle log.

This report is tracking evidence only (sim vs real vs commanded on still frames) and never a
calibration verdict. Writes report.json, report.md and tracking-gaps.png into --output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from utils.arms.calibration import calibration_identity
from utils.provenance import git_identity
from utils.calibration.gap import GAPS, gap_data, summarize

ACTION_THRESHOLDS = {"real_tracking_deg": 3., "sim_mean_deg": 1., "sim_angle_correlation": .8,
                     "limit_fraction": .05, "real_tracking_m": .003, "sim_mean_m": .001}


def _all_joints(tracking):
    joints = {**tracking["joints"], **tracking.get("grippers", {})}
    if not tracking.get("grippers") and tracking.get("gripper"):
        names = tracking.get("provenance", {}).get("gripper_joints", ["legacy_gripper"])
        joints[names[0]] = tracking["gripper"]
    return joints


def _value(stats, key):
    return stats.get(key, stats.get(key + "_deg"))


def recommended_actions(tracking):
    """Per-joint rules use its own physical unit; errors are never pooled."""
    actions = []
    gaps = tracking.get("gap_fields", GAPS)
    for name, joint in _all_joints(tracking).items():
        unit = joint.get("unit", "deg")
        real, sim = (joint["static_frames"][key] for key in gaps[1:])
        mean_abs = _value(real, "mean_abs")
        if mean_abs is not None and mean_abs > ACTION_THRESHOLDS["real_tracking_" + unit]:
            actions.append(dict(joint=name, unit=unit, rule="real_servo",
                                action=f"Real servo problem on {name}: inspect gains, load or blockage.", evidence=real))
        mean_signed = _value(sim, "mean_signed")
        if (sim["correlation"] is not None and abs(sim["correlation"]) > ACTION_THRESHOLDS["sim_angle_correlation"]
                and mean_signed is not None and abs(mean_signed) > ACTION_THRESHOLDS["sim_mean_" + unit]):
            actions.append(dict(joint=name, unit=unit, rule="sim_stiffness",
                                action=f"Sim stiffness is too low for {name}: inspect actuator gains in arm.toml.", evidence=sim))
        counts = joint["counts"]
        fraction = counts["excluded_at_limit"] / counts["total"] if counts["total"] else 0
        if fraction > ACTION_THRESHOLDS["limit_fraction"]:
            actions.append(dict(joint=name, unit=unit, rule="limit_mismatch",
                                action=f"Limit mismatch on {name}: inspect runtime limits.", evidence={"fraction_at_limit": fraction}))
    return actions


def make_plots(output, gap_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        return {"status": "skipped", "reason": "matplotlib is not installed", "files": []}
    header, _, order, _, still, _, gaps = gap_data(gap_path)
    rows = (len(order) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), constrained_layout=True)
    for i, (name, ax) in enumerate(zip(order, np.ravel(axes))):
        unit = header.get("joint_units", ["deg"] * len(order))[i]
        for key in gaps:
            values = gaps[key][still[:, i], i]
            values = values[np.isfinite(values)]
            if len(values):
                # Keep constant-gap bins resolvable in the joint's native unit.
                low, high = float(values.min()), float(values.max())
                width = 0.1 if unit == "deg" else 0.0001
                if high - low < width:
                    low, high = (low + high) / 2 - width / 2, (low + high) / 2 + width / 2
                ax.hist(values, bins=20, range=(low, high), histtype="step", label=key.removesuffix("_deg"))
        unit = header.get("joint_units", ["deg"] * len(order))[i]
        ax.set(title=name, xlabel=f"Gap on still frames ({unit})", ylabel="Frames")
        if ax.patches:
            ax.legend(fontsize=7)
    fig.savefig(output / "tracking-gaps.png", dpi=150)
    plt.close(fig)
    return {"status": "written", "files": ["tracking-gaps.png"], "tracking_scope": "all still frames, including runtime limits"}


def report_markdown(report):
    def n(value):
        return "—" if value is None else f"{value:.3f}"

    tracking = report["tracking"]
    lines = ["# Calibration tracking report — NOT a calibration verdict", "",
             "## Identity", "", "```json", json.dumps(report["identity"], indent=2), "```", "",
             "Statistics use all still frames (including limits). Correlation is with real joint angle. "
             "Real-tracking and sim-tracking verdicts use mean absolute gap on still, unsaturated frames.", "",
             f"Tracking thresholds: `{json.dumps(tracking['thresholds'])}`.", "",
             "| Joint | Real-tracking | Sim-tracking | Frames at limit |", "| --- | --- | --- | ---: |"]
    joints = _all_joints(tracking)
    for name, joint in joints.items():
        counts = joint["counts"]
        fraction = counts["excluded_at_limit"] / counts["total"] if counts["total"] else 0
        lines.append(f"| {name} | {joint['real_tracking_verdict']} | {joint['sim_tracking_verdict']} | {fraction:.1%} |")
    lines += ["", "| Joint | Unit | Gap | n | Mean ± std | Position correlation r |", "| --- | --- | --- | ---: | --- | ---: |"]
    for name, joint in joints.items():
        for key, stats in joint["static_frames"].items():
            lines.append(f"| {name} | {joint.get('unit', 'deg')} | {key.removesuffix('_deg')} | {stats['count']} | "
                         f"{n(_value(stats, 'mean_signed'))} ± {n(_value(stats, 'std'))} | {n(stats['correlation'])} |")
    lines += ["", "## Recommended actions", "", f"Rule thresholds: `{json.dumps(report['action_thresholds'])}`.", ""]
    lines += [f"- {action['action']}" for action in report["actions"]] or ["No action rule was triggered."]
    lines += ["", "## Plots", ""]
    if report["plots"]["status"] == "written":
        lines += [f"![{name}]({name})" for name in report["plots"]["files"]]
    else:
        lines.append(f"Skipped: {report['plots']['reason']}.")
    lines += ["", "## Evidence", "", "```json", json.dumps(report["inputs"], indent=2), "```", "",
              "Tracking is not an independent physical zero measurement.", ""]
    return "\n".join(lines)


def generate_report(gap_path, output):
    tracking = summarize(gap_path, write=False)
    provenance = tracking["provenance"]
    output = Path(output)
    for name in ("report.json", "report.md", "tracking-gaps.png"):
        if (output / name).resolve() == Path(gap_path).resolve():
            raise ValueError("Report output must not replace the input log")
    output.mkdir(parents=True, exist_ok=True)
    identity = {key: provenance.get(key) for key in ("follower_id", "leader_id", "follower_calibration",
                                                     "leader_calibration", "arm", "unit", "joint_units", "mapping",
                                                     "mapping_parameters")}
    report = dict(schema_version=2, created_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
                  generator=git_identity(Path(__file__).resolve().parents[3]), tracking=tracking,
                  inputs={"gap": calibration_identity(gap_path)}, action_thresholds=ACTION_THRESHOLDS,
                  actions=recommended_actions(tracking))
    report["plots"] = make_plots(output, gap_path)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "report.md").write_text(report_markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gap", required=True, help="telemetry.joint_angle_log JSONL written by s2r record")
    parser.add_argument("--output", default="", help="report folder (default: <gap>.report/)")
    args = parser.parse_args(argv)
    output = args.output or f"{args.gap}.report"
    report = generate_report(args.gap, output)
    print(f"[calibration_report] {len(report['actions'])} recommended action(s); report: {Path(output) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
