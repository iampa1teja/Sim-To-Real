"""Passive angle comparisons and reproducible reports; numpy and stdlib only.

Angles and differences are linear degrees. No value is wrapped modulo a turn.
The caller owns every device, command, mapping and PhysX read-back.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

LIMITATION = (
    "Real angles are computed from follower encoders through the same calibration used by the sim. "
    "Encoder-based checks cannot detect a zero-pose error made during calibration, including holding "
    "the arm away from the USD zero pose during 'move to middle of range'. An independent reference "
    "(operator matching a rendered pose, inclinometer or camera) is required."
)
INTERPRETATION = (
    "Sim minus real is an encoder/mapping/tracking comparison. Real minus commanded measures real "
    "tracking; sim minus sim target measures sim tracking. Clamping, sign, scale, wrist-frame offsets "
    "and lag can affect these gaps. No physical zero offset is established or applied."
)
GAPS = ("sim_minus_real_deg", "real_minus_commanded_deg", "sim_minus_sim_target_deg")


@dataclass(frozen=True)
class Thresholds:
    """Verdict and per-joint stillness gates, with explicit physical units."""

    pass_deg: float = 1.0
    warn_deg: float = 3.0
    static_velocity_deg_s: float = 1.0
    static_window_s: float = 0.25
    limit_tolerance_deg: float = 0.05

    def __post_init__(self):
        if not all(np.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Thresholds must be finite and nonnegative")
        if self.warn_deg < self.pass_deg or self.static_window_s <= 0:
            raise ValueError("Require warn_deg >= pass_deg and a positive static window")


def _thresholds(value):
    return value if isinstance(value, Thresholds) else Thresholds(**(value or {}))


def _vector(values, size):
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite angles")
    return result


def _bounds(values, size):
    bounds = np.asarray(values, dtype=float)
    if bounds.shape != (size, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Runtime limits must be finite increasing pairs in joint_order")
    return bounds


def units_to_degrees(values, mapping, joint_order):
    """Decode follower-normalized units without the sim's runtime clamp."""
    values = _vector(values, len(joint_order))
    fraction = np.asarray([(v / 100 if name == "gripper" else (v + 100) / 200)
                           for name, v in zip(joint_order, values)])
    low = np.asarray([mapping[name]["joint_min"] for name in joint_order])
    high = np.asarray([mapping[name]["joint_max"] for name in joint_order])
    return low + fraction * (high - low)


def _still(history, thresholds, size):
    if len(history) < 3 or (history[-1][0] - history[0][0]) / 1e9 < thresholds.static_window_s:
        return np.zeros(size, dtype=bool)
    stamps = np.asarray([item[0] - history[0][0] for item in history], dtype=float) / 1e9
    real = np.asarray([item[1] for item in history])
    sim = np.asarray([item[2] for item in history])
    velocity = np.maximum(abs(np.diff(real, axis=0)), abs(np.diff(sim, axis=0))) / np.diff(stamps)[:, None]
    return np.all(velocity < thresholds.static_velocity_deg_s, axis=0)


def _advance(history, stamp, real, sim, thresholds):
    if history and stamp <= history[-1][0]:
        raise ValueError("Frame timestamps must strictly increase")
    history.append((stamp, real, sim))
    # Retain one point just before the window boundary, so the full window is measured.
    cutoff = stamp - thresholds.static_window_s * 1e9
    while len(history) > 2 and history[1][0] <= cutoff:
        history.popleft()
    return _still(history, thresholds, len(real))


class CalibrationGapLogger:
    """Write an exclusive, line-buffered trace and summarize it on every exit."""

    def __init__(self, path, *, joint_order, runtime_limits, provenance, thresholds=None):
        self.path = Path(path)
        self.joint_order = tuple(joint_order)
        if not self.joint_order or len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("joint_order must contain distinct joint names")
        self.limits = _bounds(runtime_limits, len(self.joint_order))
        self.thresholds = _thresholds(thresholds)
        self.header = dict(provenance, type="provenance", schema_version=1,
                           joint_order=list(self.joint_order), runtime_limits=self.limits.tolist(),
                           thresholds=asdict(self.thresholds), interpretation=INTERPRETATION, limitation=LIMITATION)
        for key in ("follower_calibration", "leader_calibration", "real_setup", "mapping", "git_commit"):
            self.header.setdefault(key, None)
        encoded = json.dumps(self.header, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.stream.write(encoded + "\n")
        self.frames = 0
        self.history = deque()
        self.summary = None

    def record(self, *, commanded_deg, real_deg, sim_deg, sim_target_deg, timestamp_ns=None, extra=None):
        """Observe the completed tick; unknown startup commands/targets remain null."""
        size = len(self.joint_order)
        real, sim = _vector(real_deg, size), _vector(sim_deg, size)
        command = None if commanded_deg is None else _vector(commanded_deg, size)
        target = None if sim_target_deg is None else _vector(sim_target_deg, size)
        stamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        static = _advance(self.history, stamp, real, sim, self.thresholds)
        gaps = {GAPS[0]: sim - real, GAPS[1]: None if command is None else real - command,
                GAPS[2]: None if target is None else sim - target}
        at_limit = np.zeros(size, dtype=bool)
        for values in (real, sim, command, target):
            if values is not None:
                at_limit |= ((values <= self.limits[:, 0] + self.thresholds.limit_tolerance_deg)
                             | (values >= self.limits[:, 1] - self.thresholds.limit_tolerance_deg))
        branch = {key: (np.zeros(size, dtype=bool) if gap is None else abs(gap) > 180)
                  for key, gap in gaps.items()}
        row = dict(type="frame", frame=self.frames, timestamp_unix_ns=stamp,
                   commanded_deg=None if command is None else command.tolist(), real_deg=real.tolist(),
                   sim_deg=sim.tolist(), sim_target_deg=None if target is None else target.tolist(),
                   **{key: None if gap is None else gap.tolist() for key, gap in gaps.items()},
                   at_limit=at_limit.tolist(), static=static.tolist(),
                   suspected_branch_error=np.any(list(branch.values()), axis=0).tolist(),
                   branch_error_gaps={key: value.tolist() for key, value in branch.items()})
        if extra:
            if row.keys() & extra.keys():
                raise ValueError("Extra telemetry cannot replace calibration-check fields")
            row.update(extra)
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.frames += 1
        return row

    def close(self):
        """Finalize even a header-only run; the offline empty-log API still raises."""
        if not self.stream.closed:
            self.stream.close()
            self.summary = summarize(self.path, self.thresholds, allow_empty=True)
        return self.summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _stats(gap, real):
    valid = np.isfinite(gap)
    gap, real = gap[valid], real[valid]
    result = dict(count=len(gap), mean_signed_deg=None, mean_abs_deg=None, median_deg=None, std_deg=None,
                  p95_abs_deg=None, max_abs_deg=None, slope=None, correlation=None)
    if not len(gap):
        return result
    result.update(mean_signed_deg=float(gap.mean()), mean_abs_deg=float(abs(gap).mean()),
                  median_deg=float(np.median(gap)), std_deg=float(gap.std()),
                  p95_abs_deg=float(np.percentile(abs(gap), 95)), max_abs_deg=float(abs(gap).max()))
    centered_real, centered_gap = real - real.mean(), gap - gap.mean()
    variance = float(centered_real @ centered_real)
    if variance > 1e-12:
        result["slope"] = float(centered_real @ centered_gap / variance)
        gap_variance = float(centered_gap @ centered_gap)
        if gap_variance > 1e-12:
            result["correlation"] = float(centered_real @ centered_gap / np.sqrt(variance * gap_variance))
    return result


def summarize(path, thresholds=None, *, allow_empty=False):
    """Recompute stillness and all three linear gaps, and write both summary files."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or records[0].get("type") != "provenance":
        raise ValueError("Expected a calibration-check provenance header")
    header, rows = records[0], records[1:]
    if not rows and not allow_empty:
        raise ValueError("No calibration-check frames were captured")
    thresholds = _thresholds(header.get("thresholds") if thresholds is None else thresholds)
    order = header["joint_order"]
    bounds = _bounds(header["runtime_limits"], len(order))
    history, still, at_limit = deque(), [], []
    reals, gap_rows = [], {key: [] for key in GAPS}
    for row in rows:
        real, sim = _vector(row["real_deg"], len(order)), _vector(row["sim_deg"], len(order))
        command = None if row["commanded_deg"] is None else _vector(row["commanded_deg"], len(order))
        target = None if row["sim_target_deg"] is None else _vector(row["sim_target_deg"], len(order))
        still.append(_advance(history, row["timestamp_unix_ns"], real, sim, thresholds))
        limited = np.zeros(len(order), dtype=bool)
        for values in (real, sim, command, target):
            if values is not None:
                limited |= ((values <= bounds[:, 0] + thresholds.limit_tolerance_deg)
                            | (values >= bounds[:, 1] - thresholds.limit_tolerance_deg))
        at_limit.append(limited)
        reals.append(real)
        gap_rows[GAPS[0]].append(sim - real)
        gap_rows[GAPS[1]].append(np.full(len(order), np.nan) if command is None else real - command)
        gap_rows[GAPS[2]].append(np.full(len(order), np.nan) if target is None else sim - target)
    shape = (len(rows), len(order))
    real_values = np.asarray(reals).reshape(shape)
    still, at_limit = np.asarray(still, dtype=bool).reshape(shape), np.asarray(at_limit, dtype=bool).reshape(shape)
    gaps = {key: np.asarray(value).reshape(shape) for key, value in gap_rows.items()}
    joints = {}
    for i, name in enumerate(order):
        mask = still[:, i] & ~at_limit[:, i]
        filtered = {key: _stats(value[mask, i], real_values[mask, i]) for key, value in gaps.items()}
        mean_abs = filtered[GAPS[0]]["mean_abs_deg"]
        verdict = ("NO_DATA" if mean_abs is None else "PASS" if mean_abs <= thresholds.pass_deg
                   else "WARN" if mean_abs <= thresholds.warn_deg else "FAIL")
        joints[name] = dict(all_frames={key: _stats(value[:, i], real_values[:, i]) for key, value in gaps.items()},
                            static_not_at_limit=filtered, verdict=verdict,
                            counts=dict(total=len(rows), used=int(mask.sum()), excluded_at_limit=int(at_limit[:, i].sum()),
                                        excluded_moving=int((~still[:, i]).sum()),
                                        excluded_either=int((~mask).sum()),
                                        suspected_branch_error=int(np.any([abs(v[:, i]) > 180 for v in gaps.values()], axis=0).sum())))
    result = dict(schema_version=1, frames=len(rows), provenance=header, thresholds=asdict(thresholds),
                  joints={key: value for key, value in joints.items() if key != "gripper"},
                  gripper=joints.get("gripper"), limitation=LIMITATION, interpretation=INTERPRETATION,
                  notes=["Exclusion counts overlap; startup window frames count as moving/unknown.",
                         "Slope near 0 is consistent with a constant gap; near -2 suggests sign inversion; "
                         "other slopes suggest span/scale mismatch. Insufficient angle coverage gives null slope.",
                         "Gripper is separate: its closed anchor is unresolved in real_setup.json."])
    Path(str(path) + ".summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(result, Path(str(path) + ".summary.md"))
    return result


def write_markdown(summary, path):
    """Write every gap statistic for all and filtered frames in separate tables."""
    def number(value):
        return "—" if value is None else f"{value:.3f}"

    text = ["# Sim-vs-real calibration check", "", f"Frames: {summary['frames']}", "",
            summary["interpretation"], "", summary["limitation"], "",
            "Verdicts use mean absolute sim−real gap on static, not-at-limit frames. "
            f"PASS ≤ {summary['thresholds']['pass_deg']:g}°; WARN ≤ {summary['thresholds']['warn_deg']:g}°; "
            "FAIL above WARN; NO_DATA means no qualifying frames.", ""]
    groups = (("Body joints", summary["joints"]),
              ("Gripper (closed anchor unresolved)", {"gripper": summary["gripper"]} if summary["gripper"] else {}))
    for label, joints in groups:
        text.extend([f"## {label}", "", "| Joint | Verdict | Used | At limit | Moving/unknown | >180° |",
                     "| --- | --- | ---: | ---: | ---: | ---: |"])
        for name, result in joints.items():
            count = result["counts"]
            text.append(f"| {name} | {result['verdict']} | {count['used']} | {count['excluded_at_limit']} | "
                        f"{count['excluded_moving']} | {count['suspected_branch_error']} |")
        for subset in ("all_frames", "static_not_at_limit"):
            text.extend(["", f"### {subset.replace('_', ' ')}", "",
                         "| Joint | Gap | n | Mean signed ° | Mean abs ° | Median ° | Std ° | p95 abs ° | Max abs ° | Slope | Corr |",
                         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for name, result in joints.items():
                for gap, stats in result[subset].items():
                    columns = " | ".join(number(stats[key]) for key in (
                        "mean_signed_deg", "mean_abs_deg", "median_deg", "std_deg", "p95_abs_deg", "max_abs_deg", "slope", "correlation"))
                    text.append(f"| {name} | {gap.removesuffix('_deg')} | {stats['count']} | {columns} |")
        text.append("")
    text.extend(summary["notes"])
    text.extend(["", "## Provenance", "", "```json", json.dumps(summary["provenance"], indent=2), "```", ""])
    Path(path).write_text("\n".join(text), encoding="utf-8")


def reference_poses(joint_order, runtime_limits):
    """Return zero plus five distinct targets spanning the interior of each range."""
    bounds = _bounds(runtime_limits, len(joint_order))
    if np.any(bounds[:, 0] >= 0) or np.any(bounds[:, 1] <= 0):
        raise ValueError("All-zero reference must be inside every runtime limit")
    poses = [("all-zero", np.zeros(len(joint_order)))]
    for index, fraction in enumerate(np.linspace(0.15, 0.85, 5)):
        pose = bounds[:, 0] + fraction * (bounds[:, 1] - bounds[:, 0])
        poses.append((f"interior-{index + 1}", pose))
    return poses


def pose_measurement(name, target_deg, samples, mapping, joint_order, runtime_limits):
    """Return a measurement and an additive candidate; never apply an offset."""
    observations = np.asarray(samples["observations"], dtype=float)
    if observations.shape != (30, len(joint_order)) or not np.isfinite(observations).all():
        raise ValueError("Each pose needs 30 finite observations")
    target = _vector(target_deg, len(joint_order))
    real = units_to_degrees(np.median(observations, axis=0), mapping, joint_order)
    bounds = _bounds(runtime_limits, len(joint_order))
    std = observations.std(axis=0, ddof=1)
    gap = real - target
    return dict(name=name, accepted=bool(np.all(std <= 0.3)), samples=samples,
                target_deg=target.tolist(), real_deg=real.tolist(), real_minus_target_deg=gap.tolist(),
                suspected_branch_error=(abs(gap) > 180).tolist(),
                offsets_degrees={joint: float(-gap[i]) for i, joint in enumerate(joint_order) if joint != "gripper"},
                at_limit={joint: bool(min(real[i], target[i]) <= bounds[i, 0] + .05
                                      or max(real[i], target[i]) >= bounds[i, 1] - .05)
                          for i, joint in enumerate(joint_order)}, applied=False)
