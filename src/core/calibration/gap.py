"""Passive comparisons in per-joint degrees/metres; no simulator or device imports.

Revolute-only logs retain the established degree schema. Mixed-joint logs use
physical-value fields with explicit joint_units; metre and degree errors are
never pooled. Verdicts use thresholds for each joint's own physical unit.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

LIMITATION = (
    "Real positions are computed from encoders through the selected calibration. "
    "Encoder-based checks cannot detect a zero-pose error made during calibration. "
    "An independent reference (rendered pose, inclinometer or camera) is required."
)
INTERPRETATION = (
    "Sim minus real compares encoder mapping and tracking. Real minus commanded measures real tracking; "
    "sim minus sim target measures sim tracking. No independent physical zero offset is established or applied."
)
GAPS = ("sim_minus_real_deg", "real_minus_commanded_deg", "sim_minus_sim_target_deg")
PHYSICAL_GAPS = tuple(key.removesuffix("_deg") for key in GAPS)


@dataclass(frozen=True)
class Thresholds:
    """Separate native-unit gates: 1/3 degrees or 1/3 mm pass/warn by default."""
    pass_deg: float = 1.0
    warn_deg: float = 3.0
    static_velocity_deg_s: float = 1.0
    static_window_s: float = 0.25
    limit_tolerance_deg: float = 0.05
    pass_m: float = 0.001
    warn_m: float = 0.003
    static_velocity_m_s: float = 0.001
    limit_tolerance_m: float = 0.00005

    def __post_init__(self):
        if not all(np.isfinite(value) and value >= 0 for value in asdict(self).values()):
            raise ValueError("Thresholds must be finite and nonnegative")
        if self.warn_deg < self.pass_deg or self.warn_m < self.pass_m or self.static_window_s <= 0:
            raise ValueError("Require warn >= pass for each unit and a positive static window")


def _thresholds(value):
    return value if isinstance(value, Thresholds) else Thresholds(**(value or {}))


def _vector(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {size} finite joint positions")
    return values


def _bounds(values, size):
    values = np.asarray(values, dtype=float)
    if values.shape != (size, 2) or not np.isfinite(values).all() or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError("Runtime limits must be finite increasing pairs in joint_order")
    return values


def _units(header):
    units = header.get("joint_units", ["deg"] * len(header["joint_order"]))
    if len(units) != len(header["joint_order"]) or any(unit not in ("deg", "m") for unit in units):
        raise ValueError("joint_units must specify deg or m for every joint")
    return tuple(units)


def _fields(header):
    legacy = all(unit == "deg" for unit in _units(header))
    suffix = "_deg" if legacy else ""
    return tuple(name + suffix for name in ("commanded", "real", "sim", "sim_target")), GAPS if legacy else PHYSICAL_GAPS


def _gates(units, thresholds, name):
    return np.asarray([getattr(thresholds, name + ("_deg" if unit == "deg" else "_m")) for unit in units])


def _advance(history, stamp, real, sim, thresholds, units):
    if history and stamp <= history[-1][0]:
        raise ValueError("Frame timestamps must strictly increase")
    history.append((stamp, real, sim))
    cutoff = stamp - thresholds.static_window_s * 1e9
    while len(history) > 2 and history[1][0] <= cutoff:
        history.popleft()
    if len(history) < 3 or (history[-1][0] - history[0][0]) / 1e9 < thresholds.static_window_s:
        return np.zeros(len(real), dtype=bool)
    stamps = np.asarray([item[0] - history[0][0] for item in history], dtype=float) / 1e9
    reals, sims = np.asarray([item[1] for item in history]), np.asarray([item[2] for item in history])
    velocity = np.maximum(abs(np.diff(reals, axis=0)), abs(np.diff(sims, axis=0))) / np.diff(stamps)[:, None]
    velocity_limits = np.asarray([thresholds.static_velocity_deg_s if unit == "deg" else thresholds.static_velocity_m_s
                                  for unit in units])
    return np.all(velocity < velocity_limits, axis=0)


def _at_limit(values, bounds, tolerance):
    result = np.zeros(len(bounds), dtype=bool)
    for value in values:
        if value is not None:
            result |= (value <= bounds[:, 0] + tolerance) | (value >= bounds[:, 1] - tolerance)
    return result


class CalibrationGapLogger:
    def __init__(self, path, *, joint_order, runtime_limits, provenance, thresholds=None):
        self.path = Path(path)
        self.joint_order = tuple(joint_order)
        if not self.joint_order or len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("joint_order must contain distinct names")
        self.limits = _bounds(runtime_limits, len(self.joint_order))
        self.thresholds = _thresholds(thresholds)
        self.header = dict(provenance, type="provenance", joint_order=list(self.joint_order),
                           runtime_limits=self.limits.tolist(), thresholds=asdict(self.thresholds),
                           interpretation=INTERPRETATION, limitation=LIMITATION)
        self.units = _units(self.header)
        self.fields, self.gaps = _fields(self.header)
        self.header["schema_version"] = 1 if self.gaps == GAPS else 3
        self.header["joint_units"] = list(self.units)
        for key in ("follower_calibration", "leader_calibration", "mapping", "git_commit"):
            self.header.setdefault(key, None)
        encoded = json.dumps(self.header, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8", buffering=1)
        self.stream.write(encoded + "\n")
        self.frames, self.history, self.summary = 0, deque(), None

    def record(self, *, commanded=None, real=None, sim=None, sim_target=None, timestamp_ns=None, extra=None, **legacy):
        if legacy:
            if self.gaps != GAPS or set(legacy) - set(self.fields):
                raise ValueError("Degree-labelled arguments are only valid for revolute-only logs")
            commanded, real, sim, sim_target = [legacy.get(key, value) for key, value in
                                               zip(self.fields, (commanded, real, sim, sim_target))]
        size = len(self.joint_order)
        real, sim = _vector(real, size), _vector(sim, size)
        command = None if commanded is None else _vector(commanded, size)
        target = None if sim_target is None else _vector(sim_target, size)
        stamp = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        static = _advance(self.history, stamp, real, sim, self.thresholds, self.units)
        gaps = dict(zip(self.gaps, (sim - real, None if command is None else real - command,
                                   None if target is None else sim - target)))
        at_limit = _at_limit((real, sim, command, target), self.limits,
                             _gates(self.units, self.thresholds, "limit_tolerance"))
        revolute = np.asarray([unit == "deg" for unit in self.units])
        branch = {key: np.zeros(size, dtype=bool) if gap is None else (abs(gap) > 180) & revolute
                  for key, gap in gaps.items()}
        values = dict(zip(self.fields, (command, real, sim, target)))
        row = dict(type="frame", frame=self.frames, timestamp_unix_ns=stamp,
                   **{key: None if value is None else value.tolist() for key, value in values.items()},
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
        if not self.stream.closed:
            self.stream.close()
            self.summary = summarize(self.path, self.thresholds, allow_empty=True)
        return self.summary

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


def _stats(gap, real, *, unit="deg", legacy=True):
    valid = np.isfinite(gap)
    gap, real = gap[valid], real[valid]
    keys = ("mean_signed", "mean_abs", "median", "std", "p95_abs", "max_abs")
    suffix = "_deg" if legacy else ""
    result = dict(count=len(gap), **{key + suffix: None for key in keys}, slope=None, correlation=None)
    if not legacy:
        result["unit"] = unit
    if not len(gap):
        return result
    result.update(zip((key + suffix for key in keys), map(float,
                      (gap.mean(), abs(gap).mean(), np.median(gap), gap.std(), np.percentile(abs(gap), 95), abs(gap).max()))))
    centred_real, centred_gap = real - real.mean(), gap - gap.mean()
    variance = float(centred_real @ centred_real)
    if variance > 1e-12:
        result["slope"] = float(centred_real @ centred_gap / variance)
        gap_variance = float(centred_gap @ centred_gap)
        if gap_variance > 1e-12:
            result["correlation"] = float(centred_real @ centred_gap / np.sqrt(variance * gap_variance))
    return result


def gap_data(path, thresholds=None, *, allow_empty=False):
    with Path(path).open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or records[0].get("type") != "provenance":
        raise ValueError("Expected a calibration-check provenance header")
    header, rows = records[0], records[1:]
    if not rows and not allow_empty:
        raise ValueError("No calibration-check frames were captured")
    thresholds = _thresholds(header.get("thresholds") if thresholds is None else thresholds)
    order, units = header["joint_order"], _units(header)
    bounds = _bounds(header["runtime_limits"], len(order))
    fields, gap_keys = _fields(header)
    history, still, limits, reals = deque(), [], [], []
    gap_rows = {key: [] for key in gap_keys}
    for row in rows:
        command_key, real_key, sim_key, target_key = fields
        real, sim = _vector(row[real_key], len(order)), _vector(row[sim_key], len(order))
        command = None if row[command_key] is None else _vector(row[command_key], len(order))
        target = None if row[target_key] is None else _vector(row[target_key], len(order))
        still.append(_advance(history, row["timestamp_unix_ns"], real, sim, thresholds, units))
        limits.append(_at_limit((real, sim, command, target), bounds, _gates(units, thresholds, "limit_tolerance")))
        reals.append(real)
        for key, gap in zip(gap_keys, (sim - real, np.full(len(order), np.nan) if command is None else real - command,
                                      np.full(len(order), np.nan) if target is None else sim - target)):
            gap_rows[key].append(gap)
    shape = (len(rows), len(order))
    return (header, thresholds, order, np.asarray(reals).reshape(shape),
            np.asarray(still, dtype=bool).reshape(shape), np.asarray(limits, dtype=bool).reshape(shape),
            {key: np.asarray(value).reshape(shape) for key, value in gap_rows.items()})


def tracking_verdict(stats, thresholds, *, unit="deg"):
    value = stats.get("mean_abs", stats.get("mean_abs_deg"))
    passed, warned = getattr(thresholds, "pass_" + unit), getattr(thresholds, "warn_" + unit)
    return "NO_DATA" if value is None else "PASS" if value <= passed + 1e-12 else "WARN" if value <= warned + 1e-12 else "FAIL"


def summarize(path, thresholds=None, *, allow_empty=False, write=True):
    header, thresholds, order, reals, still, at_limit, gaps = gap_data(path, thresholds, allow_empty=allow_empty)
    units = _units(header)
    _, gap_keys = _fields(header)
    legacy = gap_keys == GAPS
    joints = {}
    for index, name in enumerate(order):
        mask = still[:, index] & ~at_limit[:, index]
        def stats(values, rows):
            return _stats(values[rows, index], reals[rows, index], unit=units[index], legacy=legacy)
        filtered = {key: stats(values, mask) for key, values in gaps.items()}
        branch = np.any([abs(values[:, index]) > 180 for values in gaps.values()], axis=0) if units[index] == "deg" else np.zeros(len(reals), dtype=bool)
        joints[name] = dict(unit=units[index],
            all_frames={key: stats(values, slice(None)) for key, values in gaps.items()},
            static_not_at_limit=filtered,
            static_frames={key: stats(values, still[:, index]) for key, values in gaps.items()},
            real_tracking_verdict=tracking_verdict(filtered[gap_keys[1]], thresholds, unit=units[index]),
            sim_tracking_verdict=tracking_verdict(filtered[gap_keys[2]], thresholds, unit=units[index]),
            counts=dict(total=len(reals), used=int(mask.sum()), excluded_at_limit=int(at_limit[:, index].sum()),
                        excluded_moving=int((~still[:, index]).sum()), excluded_either=int((~mask).sum()),
                        suspected_branch_error=int(branch.sum())))
    gripper_names = set(header.get("gripper_joints", []))
    result = dict(schema_version=2 if legacy else 3, frames=len(reals), provenance=header, thresholds=asdict(thresholds),
                  gap_fields=list(gap_keys), joints={name: data for name, data in joints.items() if name not in gripper_names},
                  grippers={name: data for name, data in joints.items() if name in gripper_names},
                  limitation=LIMITATION, interpretation=INTERPRETATION,
                  notes=["Errors remain in each joint's native unit; no mixed degree/metre mean is computed.",
                         "Exclusion counts overlap; startup stillness windows are moving/unknown."])
    # The legacy singular field remains a report-format convenience for one gripper,
    # independent of its actual declared motor name.
    result["gripper"] = next(iter(result["grippers"].values())) if legacy and len(result["grippers"]) == 1 else None
    if write:
        Path(str(path) + ".summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_markdown(result, Path(str(path) + ".summary.md"))
    return result


def write_markdown(summary, path):
    def number(value):
        return "—" if value is None else f"{value:.6g}"
    lines = ["# Sim-vs-real tracking check", "", f"Frames: {summary['frames']}", "",
             summary["interpretation"], "", summary["limitation"], "",
             "Thresholds use degrees for rotary joints and metres for slides.", "",
             "| Joint | Unit | Real tracking | Sim tracking | Used |",
             "| --- | --- | --- | --- | ---: |"]
    joints = {**summary["joints"], **summary.get("grippers", {})}
    for name, data in joints.items():
        lines.append(f"| {name} | {data['unit']} | {data['real_tracking_verdict']} | {data['sim_tracking_verdict']} | {data['counts']['used']} |")
    lines += ["", "| Joint | Unit | Gap | n | Mean | Mean abs | p95 abs |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for name, data in joints.items():
        for key, stats in data["static_not_at_limit"].items():
            values = [number(stats.get(field, stats.get(field + "_deg"))) for field in ("mean_signed", "mean_abs", "p95_abs")]
            lines.append(f"| {name} | {data['unit']} | {key.removesuffix('_deg')} | {stats['count']} | {' | '.join(values)} |")
    lines += ["", "## Provenance", "", "```json", json.dumps(summary["provenance"], indent=2), "```", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
