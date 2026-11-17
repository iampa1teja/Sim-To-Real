"""Canonical content identities for scenarios, trajectories and appearance."""

import hashlib as _hashlib
import math as _math
import re as _re
from collections.abc import Mapping as _Mapping

import numpy as _np

from core.utils.hashing import canonical_json as _canonical_json, json_sha256 as _json_sha256


def quantize(x, step):
    if isinstance(x, (bool, _np.bool_)) or isinstance(step, (bool, _np.bool_)):
        raise TypeError("Booleans cannot be quantized")
    step = float(step)
    if not _math.isfinite(step) or step <= 0:
        raise ValueError("Quantization step must be finite and positive")
    if isinstance(x, _np.ndarray):
        values = _np.asarray(x, dtype=_np.float64)
        if not _np.all(_np.isfinite(values)):
            raise ValueError("Cannot quantize non-finite values")
        return _np.floor(values / step + 0.5).astype(_np.int64)
    value = float(x)
    if not _math.isfinite(value):
        raise ValueError("Cannot quantize a non-finite value")
    return _math.floor(value / step + 0.5)


def _quaternion(values):
    x, y, z, w = (float(v) for v in values)
    first = next((v for v in (x, y, z) if v != 0), 0)
    return [-x, -y, -z, -w] if w < 0 or (w == 0 and first < 0) else [x, y, z, w]


def _quantized(value, step, *, poses=False):
    if isinstance(value, (bool, _np.bool_)):
        return bool(value)
    if isinstance(value, (int, _np.integer)):
        return int(value)
    if isinstance(value, (float, _np.floating)):
        return quantize(value, step)
    if isinstance(value, _Mapping):
        values = dict(value)
        if poses and values.keys() == {"position", "quat_xyzw"}:
            values["quat_xyzw"] = _quaternion(values["quat_xyzw"])
        return {key: _quantized(item, step, poses=poses) for key, item in values.items()}
    if isinstance(value, (list, tuple, _np.ndarray)):
        return [_quantized(item, step, poses=poses) for item in value]
    return value


def scenario_key(spec) -> str:
    objects = []
    for name, state in spec.objects.items():
        if not state.active:
            continue
        roles = sorted(role for role, target in spec.roles.items() if target == name)
        x, y, z, w = _quaternion(state.pose.quat_xyzw)
        yaw = _math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        objects.append({"kind": _re.sub(r"_\d+$", "", name), "role": roles[0] if roles else None,
                        "color": state.color, "p": [quantize(float(c), 1e-3) for c in state.pose.position],
                        "yaw": quantize(yaw, _math.pi/180)})
    payload = {"v": 1, "task": spec.task, "arm": spec.arm, "scene": spec.scene,
               "objects": sorted(objects, key=_canonical_json), "goal": _quantized(spec.goal, 1e-3),
               "q0": [quantize(float(q), _math.pi/360) for q in spec.initial_q], "held": spec.held_object is not None}
    return _json_sha256(payload)


def spec_fp(spec) -> str:
    payload = spec.to_json()
    del payload["attempt_index"], payload["seed"]
    return _json_sha256({"v": 1, "spec": _quantized(payload, 1e-4, poses=True)})


def traj_fp(action_raw, state_raw, objects0) -> str:
    action, state = _np.asarray(action_raw, dtype=_np.float64), _np.asarray(state_raw, dtype=_np.float64)
    if action.ndim != 2 or state.shape != action.shape or not _np.all(_np.isfinite(action)) or not _np.all(_np.isfinite(state)):
        raise ValueError("Trajectories must have matching finite (N,J) arrays")
    def digest(array):
        values = _np.asarray(array, _np.float32).astype(_np.float64)
        return _hashlib.sha256(_np.floor(values * 100 + 0.5).astype("<i4").tobytes(order="C")).hexdigest()
    positions = {}
    for name, value in objects0.items():
        position = _np.asarray(value.position if hasattr(value, "position") else value, dtype=_np.float64)
        if position.shape != (3,) or not _np.all(_np.isfinite(position)):
            raise ValueError("objects0 must hold finite 3-D positions")
        positions[name] = [quantize(float(c), 1e-4) for c in position]
    return _json_sha256({"v": 1, "shape": [int(v) for v in action.shape], "action": digest(action),
                         "state": digest(state), "objects0": positions})


def appearance_fp(colors, instruction, light_exposure, robot_color) -> str:
    return _json_sha256({"v": 1, "colors": dict(colors), "instruction": instruction,
                         "light": None if light_exposure is None else quantize(float(light_exposure), 1e-3),
                         "robot_color": robot_color})


def episode_fp(traj_fp, appearance_fp) -> str:
    for value in (traj_fp, appearance_fp):
        if not isinstance(value, str) or _re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("Episode identity requires two SHA-256 hex digests")
    return _hashlib.sha256((traj_fp + appearance_fp).encode("ascii")).hexdigest()
