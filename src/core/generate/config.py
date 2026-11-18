"""Strict YAML generation configuration with canonical resolved identities."""

import math as _math
import string as _string
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass, asdict as _asdict
from pathlib import Path as _Path
from types import MappingProxyType as _MappingProxyType

from core.generate.errors import ConfigError as _ConfigError
from core.utils.hashing import json_sha256 as _json_sha256


def _require(condition, message):
    if not condition:
        raise _ConfigError(message)


def _int(value, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), "Invalid integer setting")


def _float(value, minimum=0, maximum=None):
    _require(type(value) in (int, float) and _math.isfinite(value) and value >= minimum
             and (maximum is None or value <= maximum), "Invalid numeric setting")
    return float(value)


def _text(value):
    _require(type(value) is str and bool(value), "Expected a non-empty string")


def _table(value, allowed, required=()):
    _require(isinstance(value, _Mapping) and not value.keys() - set(allowed) and set(required) <= value.keys(), "Unknown or missing configuration keys")


def _validate_camera_jitter(value, *, max_position=None, max_rotation=None):
    _table(value, ("position_m", "rotation_deg"), ("position_m", "rotation_deg"))
    return {"position_m": _float(value["position_m"], maximum=max_position),
            "rotation_deg": _float(value["rotation_deg"], maximum=max_rotation)}


@_dataclass(frozen=True)
class Quotas:
    success: int
    failure: int
    recovery: int

    def __post_init__(self):
        for value in (self.success, self.failure, self.recovery):
            _int(value)


@_dataclass(frozen=True)
class NoiseConfig:
    fraction: float = 0.5
    theta: float = 0.15
    sigma_rad: float = 0.01
    max_abs_rad: float = 0.02

    def __post_init__(self):
        for key in ("fraction", "theta", "sigma_rad", "max_abs_rad"):
            object.__setattr__(self, key, _float(getattr(self, key), maximum=1 if key in ("fraction", "theta") else None))


@_dataclass(frozen=True)
class DRConfig:
    copies: int = 2
    terms: tuple[str, ...] = ("light", "robot_color")
    recolor: bool = True
    seed: int = 0
    camera_jitter: _Mapping[str, float] | None = None

    def __post_init__(self):
        _int(self.copies)
        _int(self.seed, maximum=2**64 - 1)
        _require(type(self.recolor) is bool, "dr.recolor must be boolean")
        _require(isinstance(self.terms, (tuple, list)) and all(type(v) is str for v in self.terms)
                 and len(set(self.terms)) == len(self.terms) and set(self.terms) <= {"light", "robot_color"}, "Invalid DR terms")
        object.__setattr__(self, "terms", tuple(self.terms))
        if self.camera_jitter is not None:
            jitter = _validate_camera_jitter(self.camera_jitter, max_position=0.05, max_rotation=10)
            object.__setattr__(self, "camera_jitter", _MappingProxyType(jitter))


@_dataclass(frozen=True)
class GenerateConfig:
    arm: str
    scene: str
    tasks: tuple[str, ...]
    master_seed: int
    quotas: _Mapping[str, Quotas]
    shard: tuple[int, int]
    dst_root: _Path
    repo_id: str
    fps: int
    max_attempt_factor: float
    max_consecutive_skips: int
    scenario_unique_across_kinds: bool
    noise: NoiseConfig
    dr: DRConfig
    perturbation_weights: _Mapping[str, float] | None
    cameras: bool
    sha256: str

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "quotas", _MappingProxyType(dict(self.quotas)))
        object.__setattr__(self, "shard", tuple(self.shard))
        object.__setattr__(self, "dst_root", _Path(self.dst_root))
        object.__setattr__(self, "max_attempt_factor", float(self.max_attempt_factor))
        if self.perturbation_weights is not None:
            object.__setattr__(self, "perturbation_weights", _MappingProxyType(dict(self.perturbation_weights)))
        payload = self.to_json()
        del payload["shard"], payload["dst_root"]
        object.__setattr__(self, "sha256", _json_sha256(payload))

    def to_json(self) -> dict:
        dr = dict(copies=self.dr.copies, terms=list(self.dr.terms), recolor=self.dr.recolor, seed=self.dr.seed)
        if self.dr.camera_jitter is not None:
            dr["camera_jitter"] = dict(self.dr.camera_jitter)
        return {"arm": self.arm, "scene": self.scene, "tasks": list(self.tasks), "master_seed": self.master_seed,
                "quotas": {task: _asdict(quota) for task, quota in self.quotas.items()}, "shard": list(self.shard),
                "dst_root": str(self.dst_root), "repo_id": self.repo_id, "fps": self.fps,
                "max_attempt_factor": self.max_attempt_factor, "max_consecutive_skips": self.max_consecutive_skips,
                "scenario_unique_across_kinds": self.scenario_unique_across_kinds, "noise": _asdict(self.noise),
                "dr": dr,
                "perturbation_weights": None if self.perturbation_weights is None else dict(self.perturbation_weights),
                "cameras": self.cameras}

    def dataset_root(self, task) -> _Path:
        if task not in self.tasks:
            raise ValueError(f"Task is not configured: {task}")
        return self.dst_root / task

    def dataset_repo_id(self, task) -> str:
        if task not in self.tasks:
            raise ValueError(f"Task is not configured: {task}")
        return self.repo_id.format(task=task)


_REQUIRED = {"arm", "scene", "tasks", "master_seed", "quotas", "dst_root", "repo_id"}
_OPTIONAL = {"per_task_quotas", "shard", "fps", "max_attempt_factor", "max_consecutive_skips",
             "scenario_unique_across_kinds", "noise", "dr", "perturbation_weights", "cameras"}
_QUOTA_KEYS = {"success", "failure", "recovery"}


def load_generate_config(path, overrides=None) -> GenerateConfig:
    import yaml as _yaml

    try:
        raw = _yaml.safe_load(_Path(path).read_text())
        _table(raw, _REQUIRED | _OPTIONAL)
        raw = dict(raw)
        if overrides is not None:
            _table(overrides, _REQUIRED | _OPTIONAL)
            raw.update({key: value for key, value in overrides.items() if value is not None})
        _table(raw, _REQUIRED | _OPTIONAL, _REQUIRED)
        for key in ("arm", "scene", "repo_id"):
            _text(raw[key])
        tasks = raw["tasks"]
        _require(isinstance(tasks, list) and bool(tasks), "tasks must be a non-empty list")
        for task in tasks:
            _text(task)
        _require(len(set(tasks)) == len(tasks), "tasks must be distinct")
        placeholders = []
        for _, name, fmt, conversion in _string.Formatter().parse(raw["repo_id"]):
            if name is not None:
                _require(name == "task" and not fmt and conversion is None, "Only {task} may occur in repo_id")
                placeholders.append(name)
        _require(len(tasks) == 1 or "task" in placeholders, "Multiple tasks require {task} in repo_id")
        _int(raw["master_seed"], maximum=2**64 - 1)
        _table(raw["quotas"], _QUOTA_KEYS, _QUOTA_KEYS)
        Quotas(**raw["quotas"])
        per_task = raw.get("per_task_quotas", {})
        _table(per_task, tasks)
        for value in per_task.values():
            _table(value, _QUOTA_KEYS)
        quotas = {task: Quotas(**{**raw["quotas"], **per_task.get(task, {})}) for task in tasks}
        shard = raw.get("shard", "0/1")
        if isinstance(shard, str):
            pieces = shard.split("/")
            _require(len(pieces) == 2 and all(piece.isdecimal() for piece in pieces), "Invalid shard string")
            shard = [int(piece) for piece in pieces]
        _require(isinstance(shard, list) and len(shard) == 2, "shard must be i/n or a two-list")
        _int(shard[1], 1)
        _int(shard[0], maximum=shard[1] - 1)
        _require(isinstance(raw["dst_root"], (str, _Path)), "dst_root must be a path")
        fps, skips = raw.get("fps", 30), raw.get("max_consecutive_skips", 1000)
        _int(fps, 1)
        _int(skips, 1)
        factor = _float(raw.get("max_attempt_factor", 3.0), 1)
        unique, cameras = raw.get("scenario_unique_across_kinds", True), raw.get("cameras", True)
        _require(type(unique) is bool and type(cameras) is bool, "unique and cameras must be boolean")
        noise, dr = raw.get("noise", {}), raw.get("dr", {})
        _table(noise, ("fraction", "theta", "sigma_rad", "max_abs_rad"))
        _table(dr, ("copies", "terms", "recolor", "seed", "camera_jitter"))
        if "terms" in dr:
            _require(isinstance(dr["terms"], list), "dr.terms must be a list")
        weights = raw.get("perturbation_weights")
        if weights is not None:
            _require(isinstance(weights, _Mapping) and bool(weights), "perturbation_weights must be a non-empty mapping")
            for name in weights:
                _text(name)
            weights = {name: _float(value) for name, value in weights.items()}
        return GenerateConfig(raw["arm"], raw["scene"], tuple(tasks), raw["master_seed"], quotas, tuple(shard),
                              _Path(raw["dst_root"]), raw["repo_id"], fps, factor, skips, unique,
                              NoiseConfig(**noise), DRConfig(**dr), weights, cameras, "")
    except (_yaml.YAMLError, ValueError, TypeError, KeyError, OverflowError) as exc:
        if isinstance(exc, _ConfigError):
            raise
        raise _ConfigError(f"{path}: {exc}") from exc
