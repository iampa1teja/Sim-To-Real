"""Strict v2 episode sidecars and compatible v1 trajectory reading."""

import copy as _copy
import json as _json
import math as _math
import numbers as _numbers
import os as _os
import re as _re
import tempfile as _tempfile
from pathlib import Path as _Path

from core.generate.errors import SchemaError as _SchemaError
from core.generate.protocols import Kind as _Kind, SegmentLabel as _SegmentLabel
from core.generate.protocols import NoiseSpec as _NoiseSpec, PerturbationSpec as _PerturbationSpec, RecoveryInfo as _RecoveryInfo

SIDECAR_V2_DIR = "task_meta"
LABEL_FEATURES = {
    "label.stage": {"dtype": "int64", "shape": (1,), "names": ["stage"]},
    "label.imitate": {"dtype": "int64", "shape": (1,), "names": ["imitate"]},
    "label.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
}

_TOP_KEYS = set("schema_version episode_index num_frames fps task arm scene reference_frame units angle_units joint_names objects ee_pose instruction roles kind success failure_mode furthest_stage stages stage imitate phases segments perturbation recovery noise attempt fingerprints appearance source generator legacy_v1".split())
_FINGERPRINTS = set("scenario_key spec_fp traj_fp appearance_fp episode_fp".split())


def _require(condition, message):
    if not condition:
        raise _SchemaError(message)


def _keys(value, expected, optional=()):
    required = set(expected)
    _require(type(value) is dict and required <= value.keys() <= required.union(optional),
             f"Expected object keys {sorted(required)}")


def _integer(value, minimum=None, maximum=None):
    _require(type(value) is int, "Expected a JSON integer")
    _require((minimum is None or value >= minimum) and (maximum is None or value <= maximum), "Integer outside allowed range")


def _number(value):
    _require(type(value) in (int, float) and _math.isfinite(value), "Expected a finite JSON number")


def _string(value, nullable=False):
    _require(type(value) is str or (nullable and value is None), "Expected a string")


def _boolean(value, nullable=False):
    _require(type(value) is bool or (nullable and value is None), "Expected a boolean")


def _list(value, length=None):
    _require(type(value) is list and (length is None or len(value) == length), "Invalid JSON array or length")


def _json_tree(value):
    if type(value) is dict:
        for key, item in value.items():
            _string(key)
            _json_tree(item)
    elif type(value) is list:
        for item in value:
            _json_tree(item)
    elif type(value) is float:
        _number(value)
    else:
        _require(type(value) in (str, int, bool, type(None)), "Not a JSON value")


def _vector(value, width):
    _list(value, width)
    for component in value:
        _number(component)


def _tracks(value, n):
    for key, width in (("position", 3), ("orientation_wxyz", 4)):
        _list(value[key], n)
        for row in value[key]:
            _vector(row, width)
            if width == 4:
                _require(abs(_math.sqrt(sum(v*v for v in row)) - 1) <= 1e-3, "Quaternion must have unit norm")


def _digest(value, nullable=False):
    if nullable and value is None:
        return
    _require(type(value) is str and _re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA-256 fingerprint")


def _coverage(runs, n, *, phases=False):
    _list(runs)
    if phases and not runs:
        return
    _require(bool(runs), "Segments cannot be empty")
    previous = 0
    for run in runs:
        _keys(run, ("name", "start", "end") if phases else ("label", "start", "end", "phase"))
        _integer(run["start"], 0)
        _integer(run["end"], 1, n)
        _require(run["start"] == previous and run["end"] > run["start"], "Runs must be contiguous, ordered and non-empty")
        previous = run["end"]
        if phases:
            _string(run["name"])
        else:
            _SegmentLabel(run["label"])
            _string(run["phase"], True)
    _require(previous == n, "Runs must cover every frame")


def validate_sidecar(payload, *, num_frames=None, fps=None) -> None:
    """Validate exact schema keys, types, timelines and cross-field invariants."""
    try:
        _keys(payload, _TOP_KEYS)
        _json_tree(payload)
        _integer(payload["schema_version"], 2, 2)
        _integer(payload["episode_index"], 0)
        _integer(payload["num_frames"], 1)
        _integer(payload["fps"], 1)
        n = payload["num_frames"]
        _boolean(payload["legacy_v1"])
        legacy = payload["legacy_v1"]
        for key in ("task", "arm", "scene"):
            _string(payload[key], legacy)
        _string(payload["reference_frame"])
        _require(_re.fullmatch(r"robot base link \((.+)\)", payload["reference_frame"]) is not None, "Invalid reference frame")
        _require(payload["units"] == "m" and payload["angle_units"] == "rad", "Invalid units")
        for key in ("joint_names", "stages"):
            value = payload[key]
            if key == "joint_names" and value is None:
                continue
            _list(value)
            for item in value:
                _string(item)
        for key in ("instruction", "failure_mode", "furthest_stage"):
            _string(payload[key], True)
        _boolean(payload["success"], True)
        _Kind(payload["kind"])
        _require(type(payload["objects"]) is dict and type(payload["roles"]) is dict, "objects and roles must be mappings")
        for name, obj in payload["objects"].items():
            _string(name)
            _keys(obj, ("set", "size_m", "color", "active", "role", "position", "orientation_wxyz"))
            _require(obj["active"] is True, "Sidecars contain only active objects")
            for key in ("set", "color", "role"):
                _string(obj[key], True)
            _vector(obj["size_m"], 3)
            _tracks(obj, n)
        for key, value in payload["roles"].items():
            _string(key)
            _string(value)
        if payload["ee_pose"] is not None:
            _keys(payload["ee_pose"], ("position", "orientation_wxyz"))
            _tracks(payload["ee_pose"], n)
        if payload["stage"] is not None:
            _list(payload["stage"], n)
            for value in payload["stage"]:
                _integer(value, -1, len(payload["stages"]) - 1)
        _list(payload["imitate"], n)
        for value in payload["imitate"]:
            _boolean(value)
        _coverage(payload["phases"], n, phases=True)
        _coverage(payload["segments"], n)
        if payload["perturbation"] is not None:
            perturbation = payload["perturbation"]
            _keys(perturbation, ("type", "group", "onset_stage", "delay_steps", "params", "onset_frame", "applied"))
            _PerturbationSpec.from_json({key: value for key, value in perturbation.items() if key not in ("onset_frame", "applied")})
            if perturbation["onset_frame"] is not None:
                _integer(perturbation["onset_frame"], 0, n - 1)
            _boolean(perturbation["applied"])
        for key, cls in (("recovery", _RecoveryInfo), ("noise", _NoiseSpec)):
            if payload[key] is not None:
                cls.from_json(payload[key])
        if payload["attempt"] is not None:
            attempt = payload["attempt"]
            _keys(attempt, ("index", "seed", "shard", "spec_fp"), ("physics", "expert_params"))
            _integer(attempt["index"], 0)
            _integer(attempt["seed"], 0, 2**64 - 1)
            _list(attempt["shard"], 2)
            _integer(attempt["shard"][1], 1)
            _integer(attempt["shard"][0], 0, attempt["shard"][1] - 1)
            _digest(attempt["spec_fp"])
            if "physics" in attempt:
                physics = attempt["physics"]
                _require(type(physics) is dict and bool(physics), "Expected non-empty physics mapping")
                for name, entry in physics.items():
                    _require(name in payload["objects"] and payload["objects"][name]["active"],
                             "Physics must reference an active object")
                    _keys(entry, (), ("mass_kg", "static_friction", "dynamic_friction"))
                    _require(bool(entry), "Expected non-empty object physics mapping")
                    for value in entry.values():
                        _number(value)
            if "expert_params" in attempt:
                params = attempt["expert_params"]
                _require(type(params) is dict and bool(params), "Expected non-empty expert_params mapping")
                _json_tree(params)
        fingerprints = payload["fingerprints"]
        if fingerprints is None:
            _require(legacy, "Non-legacy sidecars require fingerprints")
        else:
            _keys(fingerprints, _FINGERPRINTS)
            for value in fingerprints.values():
                _digest(value, legacy)
        _keys(payload["appearance"], ("light_exposure", "robot_color"), ("camera_offsets",))
        if payload["appearance"]["light_exposure"] is not None:
            _number(payload["appearance"]["light_exposure"])
        _string(payload["appearance"]["robot_color"], True)
        if "camera_offsets" in payload["appearance"]:
            offsets = payload["appearance"]["camera_offsets"]
            _require(type(offsets) is dict and bool(offsets), "Expected non-empty camera_offsets mapping")
            for name, offset in offsets.items():
                _string(name)
                _keys(offset, ("position_m", "rotation_deg"))
                _vector(offset["position_m"], 3)
                _vector(offset["rotation_deg"], 3)
        if payload["source"] is not None:
            source = payload["source"]
            _keys(source, ("repo_id", "episode_index", "traj_fp", "episode_fp"))
            _string(source["repo_id"])
            _integer(source["episode_index"], 0)
            for key in ("traj_fp", "episode_fp"):
                _digest(source[key], True)
        _keys(payload["generator"], ("name", "git_commit"))
        _require(payload["generator"]["name"] in ("scripted_expert", "teleop", "dr_replay"), "Unknown generator")
        _string(payload["generator"]["git_commit"], True)
        if payload["kind"] == "success":
            _require(payload["perturbation"] is None, "Success sidecar cannot have a perturbation")
        if payload["kind"] == "recovery":
            _require(payload["recovery"] is not None, "Recovery sidecar requires recovery information")
        if payload["success"] is not None:
            _require(payload["success"] == (payload["failure_mode"] is None), "success and failure_mode disagree")
        _require(num_frames is None or num_frames == n, "num_frames does not match the dataset")
        _require(fps is None or fps == payload["fps"], "fps does not match the dataset")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        if isinstance(exc, _SchemaError):
            raise
        raise _SchemaError(f"Invalid episode sidecar: {exc}") from exc


def sidecar_path(root, index) -> _Path:
    if isinstance(index, bool) or not isinstance(index, _numbers.Integral):
        raise TypeError("Episode index must be an integer")
    if index < 0:
        raise ValueError("Episode index cannot be negative")
    return _Path(root) / SIDECAR_V2_DIR / f"episode_{int(index):06d}.json"


def _write_text_atomic(path, text):
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with _tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=".", suffix=".tmp", delete=False) as stream:
            temporary = _Path(stream.name)
            stream.write(text)
            stream.flush()
            _os.fsync(stream.fileno())
        _os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_sidecar(root, payload) -> _Path:
    validate_sidecar(payload)
    stored = _copy.deepcopy(payload)
    tracks = list(stored["objects"].values())
    if stored["ee_pose"] is not None:
        tracks.append(stored["ee_pose"])
    for track in tracks:
        for key in ("position", "orientation_wxyz"):
            track[key] = [[round(value, 6) for value in row] for row in track[key]]
    path = sidecar_path(root, stored["episode_index"])
    _write_text_atomic(path, _json.dumps(stored, sort_keys=True, allow_nan=False))
    return path


def _read_json(path):
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise _SchemaError(f"Invalid JSON in {path}") from exc


def read_sidecar(root, index) -> dict:
    payload = _read_json(sidecar_path(root, index))
    validate_sidecar(payload)
    _require(payload["episode_index"] == index, "Sidecar index differs from its filename")
    return payload


def load_episode(root, index, *, task="pick_place_cube", arm=None, scene=None) -> dict:
    path = sidecar_path(root, index)
    if path.exists():
        return read_sidecar(root, index)
    v1 = _read_json(_Path(root) / "pick_place_meta" / path.name)
    _require(type(v1) is dict and v1.get("episode_index") == index, "v1 index differs from its filename")
    return v1_to_v2(v1, task=task, arm=arm, scene=scene)


def v1_to_v2(v1, *, task, arm, scene) -> dict:
    _require(type(v1) is dict and set("episode_index reference_frame units fps num_frames cube_size_m position orientation_wxyz".split()) <= v1.keys(), "Missing v1 fields")
    _require(v1["reference_frame"] == "robot base link (base)" and v1["units"] == "m", "Invalid v1 frame or units")
    _integer(v1["num_frames"], 1)
    n = v1["num_frames"]
    result = {
        "schema_version": 2, "episode_index": v1["episode_index"], "num_frames": n, "fps": v1["fps"],
        "task": task, "arm": arm, "scene": scene, "reference_frame": v1["reference_frame"], "units": "m", "angle_units": "rad",
        "joint_names": None,
        "objects": {"blue_cube": {"set": None, "size_m": _copy.deepcopy(v1["cube_size_m"]), "color": "blue", "active": True,
                                  "role": "target", "position": _copy.deepcopy(v1["position"]),
                                  "orientation_wxyz": _copy.deepcopy(v1["orientation_wxyz"])}},
        "ee_pose": None, "instruction": None, "roles": {"target": "blue_cube", "container": "white_box"},
        "kind": "success", "success": None, "failure_mode": None, "furthest_stage": None, "stages": [], "stage": None,
        "imitate": [True] * n, "phases": [], "segments": [{"label": "nominal", "start": 0, "end": n, "phase": None}],
        "perturbation": None, "recovery": None, "noise": None, "attempt": None, "fingerprints": None,
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "teleop", "git_commit": None}, "legacy_v1": True,
    }
    validate_sidecar(result)
    return result


def list_sidecars(root) -> list[int]:
    directory = _Path(root) / SIDECAR_V2_DIR
    if not directory.is_dir():
        return []
    result = []
    for path in directory.iterdir():
        match = _re.fullmatch(r"episode_([0-9]+)\.json", path.name)
        if match is not None and path.is_file():
            index = int(match.group(1))
            if path.name == f"episode_{index:06d}.json":
                result.append(index)
    return sorted(result)
