"""Append-only generation records, crash reconciliation and run provenance."""

import json as _json
import math as _math
import os as _os
import re as _re
import tempfile as _tempfile
import warnings as _warnings
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass, fields as _fields
from pathlib import Path as _Path

from core.generate.errors import ResumeError as _ResumeError, SchemaError as _SchemaError
from core.generate.plan import KINDS as _KINDS
from core.generate.protocols import Kind as _Kind, AttemptStatus as _AttemptStatus
from core.generate.protocols import RejectionReason as _RejectionReason, SkipReason as _SkipReason
from core.utils.hashing import canonical_json as _canonical_json

_FP_NAMES = ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp")


def _require(condition, message):
    if not condition:
        raise _SchemaError(message)


def _integer(value, minimum=0, maximum=None):
    _require(type(value) is int and value >= minimum and (maximum is None or value <= maximum), "Invalid JSON integer")


def _text(value, *, nullable=False, nonempty=False):
    _require((nullable and value is None) or (type(value) is str and (bool(value) or not nonempty)), "Invalid string")


def _digest(value, nullable=False):
    _require((nullable and value is None) or (type(value) is str and _re.fullmatch(r"[0-9a-f]{64}", value) is not None), "Invalid fingerprint")


def _shard(value):
    _require(isinstance(value, (list, tuple)) and len(value) == 2, "Invalid shard pair")
    _integer(value[1], 1)
    _integer(value[0], maximum=value[1] - 1)


def _keys(value, expected):
    _require(isinstance(value, _Mapping) and value.keys() == set(expected), "Unknown or missing schema keys")


@_dataclass(frozen=True)
class ManifestLine:
    task: str
    kind: _Kind
    attempt_index: int
    seed: int
    shard: tuple[int, int]
    status: _AttemptStatus
    reason: _SkipReason | _RejectionReason | None
    episode_index: int | None
    scenario_key: str | None
    spec_fp: str | None
    traj_fp: str | None
    appearance_fp: str | None
    episode_fp: str | None
    success: bool | None
    failure_mode: str | None
    furthest_stage: str | None
    perturbation: str | None
    num_frames: int | None
    wall_s: float = 0.0

    def __post_init__(self):
        try:
            object.__setattr__(self, "kind", _Kind(self.kind))
            object.__setattr__(self, "status", _AttemptStatus(self.status))
            if self.reason is not None:
                try:
                    reason = _SkipReason(self.reason)
                except ValueError:
                    reason = _RejectionReason(self.reason)
                object.__setattr__(self, "reason", reason)
            _shard(self.shard)
            object.__setattr__(self, "shard", tuple(self.shard))
            _text(self.task, nonempty=True)
            _integer(self.attempt_index)
            _integer(self.seed, maximum=2**64 - 1)
            for name in _FP_NAMES:
                _digest(getattr(self, name), True)
            _require(self.success is None or type(self.success) is bool, "Invalid success flag")
            for name in ("failure_mode", "furthest_stage", "perturbation"):
                _text(getattr(self, name), nullable=True)
            for name in ("episode_index", "num_frames"):
                if getattr(self, name) is not None:
                    _integer(getattr(self, name))
            _require(type(self.wall_s) in (int, float) and _math.isfinite(self.wall_s), "wall_s must be finite")
            object.__setattr__(self, "wall_s", float(self.wall_s))
            if self.status is _AttemptStatus.ACCEPTED:
                _require(self.reason is None and self.episode_index is not None and self.num_frames is not None
                         and all(getattr(self, name) is not None for name in _FP_NAMES), "Incomplete accepted line")
            else:
                _require(self.episode_index is None, "Non-accepted attempt cannot have an episode index")
            if self.status is _AttemptStatus.REJECTED:
                _require(isinstance(self.reason, _RejectionReason), "Rejected attempt needs a rejection reason")
            if self.status is _AttemptStatus.SKIPPED:
                _require(isinstance(self.reason, _SkipReason), "Skipped attempt needs a skip reason")
                _require(all(getattr(self, name) is None for name in _FP_NAMES[2:]), "Skipped attempt cannot have trajectory fingerprints")
                if self.reason is _SkipReason.SAMPLING_FAILED:
                    _require(self.scenario_key is None and self.spec_fp is None, "Sampling failure has no spec")
                else:
                    _require(self.scenario_key is not None and self.spec_fp is not None, "Seen scenario requires spec fingerprints")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            if isinstance(exc, _SchemaError):
                raise
            raise _SchemaError(f"Invalid manifest line: {exc}") from exc

    def to_json(self) -> dict:
        result = {"v": 1}
        for field in _fields(self):
            value = getattr(self, field.name)
            if field.name in ("kind", "status", "reason") and value is not None:
                value = value.value
            elif field.name == "shard":
                value = list(value)
            result[field.name] = value
        return result

    @classmethod
    def from_json(cls, d):
        _keys(d, {"v", *(field.name for field in _fields(cls))})
        _require(type(d["v"]) is int and d["v"] == 1, "Expected manifest v=1")
        return cls(**{key: value for key, value in d.items() if key != "v"})

    @classmethod
    def from_result(cls, result, *, shard, perturbation=None, wall_s=0.0):
        fp = result.fingerprints
        return cls(result.task, result.kind, result.attempt_index, result.seed, shard, result.status, result.reason,
                   result.episode_index, *(getattr(fp, name) for name in _FP_NAMES), result.success,
                   result.failure_mode, result.furthest_stage, perturbation,
                   None if result.status is _AttemptStatus.SKIPPED else result.num_frames, wall_s)


def _key(line):
    return line.task, line.kind, line.attempt_index


class Manifest:
    def __init__(self, path):
        self.path = _Path(path)

    def _scan(self):
        if not self.path.exists():
            return [], 0
        data = self.path.read_bytes()
        complete = data.split(b"\n")[:-1]
        result, seen, offset = [], set(), 0
        for i, raw in enumerate(complete):
            try:
                payload = _json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                if i == len(complete) - 1:
                    break
                raise _SchemaError("Invalid non-final manifest line") from exc
            line = ManifestLine.from_json(payload)
            _require(_key(line) not in seen, "Duplicate attempt in manifest")
            seen.add(_key(line))
            result.append(line)
            offset += len(raw) + 1
        return result, offset

    def lines(self) -> list[ManifestLine]:
        return self._scan()[0]

    def _truncate_tail(self, offset):
        if self.path.exists() and self.path.stat().st_size != offset:
            with self.path.open("r+b") as stream:
                stream.truncate(offset)
                _os.fsync(stream.fileno())

    def append(self, line):
        line = ManifestLine.from_json(line.to_json() if isinstance(line, ManifestLine) else line)
        lines, offset = self._scan()
        _require(all(_key(line) != _key(old) for old in lines), "Duplicate attempt in manifest")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._truncate_tail(offset)
        with self.path.open("ab") as stream:
            stream.write((_canonical_json(line.to_json()) + "\n").encode("utf-8"))
            stream.flush()
            _os.fsync(stream.fileno())

    def accepted(self) -> list[ManifestLine]:
        return [line for line in self.lines() if line.status is _AttemptStatus.ACCEPTED]

    def last_index(self, kind):
        return max((line.attempt_index for line in self.lines() if line.kind is _Kind(kind)), default=None)


@_dataclass(frozen=True)
class ReconcileResult:
    rebuilt_lines: int
    deleted_sidecars: list[int]
    next_position: tuple[int, int]


def reconcile(root, writer, manifest) -> ReconcileResult:
    lines, offset = manifest._scan()
    accepted = [line for line in lines if line.status is _AttemptStatus.ACCEPTED]
    count, recorded = writer.episode_count(), len(accepted)
    if [line.episode_index for line in accepted] != list(range(recorded)) or not recorded <= count <= recorded + 1:
        raise _ResumeError("Manifest and committed episode counts disagree")
    directory = _Path(root) / "task_meta"
    sidecars = {int(path.stem.split("_")[1]): path for path in directory.glob("episode_*.json")
                if path.is_file() and _re.fullmatch(r"episode_[0-9]{6}\.json", path.name)}
    if any(index not in sidecars for index in range(count)):
        raise _ResumeError("A committed episode has no sidecar")
    rebuilt = None
    if count == recorded + 1:
        try:
            payload = _json.loads(sidecars[count - 1].read_text())
            if type(payload["episode_index"]) is not int or payload["episode_index"] != count - 1:
                raise ValueError("Rebuild sidecar has the wrong index")
            attempt, fp = payload["attempt"], payload["fingerprints"]
            rebuilt = ManifestLine(payload["task"], payload["kind"], attempt["index"], attempt["seed"], attempt["shard"],
                                    _AttemptStatus.ACCEPTED, None, count - 1, *(fp[name] for name in _FP_NAMES),
                                    payload["success"], payload["failure_mode"], payload["furthest_stage"],
                                    payload["perturbation"]["type"] if payload["perturbation"] is not None else None,
                                    payload["num_frames"], 0.0)
            if any(_key(line) == _key(rebuilt) for line in lines):
                raise ValueError("Rebuilt attempt is already recorded")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise _ResumeError(f"Cannot rebuild manifest line: {exc}") from exc
    deleted = sorted(index for index in sidecars if index >= count)
    if rebuilt is not None:
        manifest.append(rebuilt)
        lines.append(rebuilt)
    else:
        manifest._truncate_tail(offset)
    for index in deleted:
        sidecars[index].unlink()
    position = max(((line.attempt_index, _KINDS.index(line.kind)) for line in lines), default=None)
    next_position = (0, 0) if position is None else ((position[0], position[1] + 1) if position[1] < 2 else (position[0] + 1, 0))
    return ReconcileResult(int(rebuilt is not None), deleted, next_position)


_PLAN_KEYS = set("v config_sha256 config master_seed git_commit shard task task_toml_sha256 arm arm_files_sha256 scene quotas holdout created_at".split())


def _plain(value):
    if isinstance(value, _Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _validate_plan(plan):
    try:
        _keys(plan, _PLAN_KEYS)
        _require(type(plan["v"]) is int and plan["v"] == 1, "Expected plan v=1")
        for key in ("config_sha256", "task_toml_sha256"):
            _digest(plan[key])
        _require(isinstance(plan["config"], _Mapping), "config must be a mapping")
        _canonical_json(_plain(plan["config"]))
        _integer(plan["master_seed"], maximum=2**64 - 1)
        for key in ("git_commit", "task", "arm", "scene", "created_at"):
            _text(plan[key], nonempty=True)
        _shard(plan["shard"])
        _require(isinstance(plan["arm_files_sha256"], _Mapping), "arm files must be a mapping")
        for key, value in plan["arm_files_sha256"].items():
            _text(key)
            _digest(value)
        _keys(plan["quotas"], ("success", "failure", "recovery"))
        for value in plan["quotas"].values():
            _integer(value)
        _keys(plan["holdout"], ("colors", "counts"))
        for key in ("colors", "counts"):
            _require(type(plan["holdout"][key]) is list, "Holdout values must be lists")
        for value in plan["holdout"]["colors"]:
            _text(value)
        for value in plan["holdout"]["counts"]:
            _require(type(value) is int, "Holdout count must be an integer")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        if isinstance(exc, _SchemaError):
            raise
        raise _SchemaError(f"Invalid generation plan: {exc}") from exc


def _plan_path(root):
    return _Path(root) / "meta" / "generation_plan.json"


def read_plan_json(root) -> dict:
    path = _plan_path(root)
    try:
        plan = _json.loads(path.read_text())
    except (ValueError, UnicodeError) as exc:
        raise _SchemaError(f"Invalid JSON in {path}") from exc
    _validate_plan(plan)
    return plan


def write_plan_json(root, plan) -> dict:
    _validate_plan(plan)
    plain = _json.loads(_canonical_json(_plain(plan)))
    path = _plan_path(root)
    if path.exists():
        previous = read_plan_json(root)
        for key in ("config_sha256", "task_toml_sha256", "arm_files_sha256", "master_seed", "shard"):
            if previous[key] != plain[key]:
                raise _ResumeError(f"Generation plan differs in {key}")
        if previous["git_commit"] != plain["git_commit"]:
            _warnings.warn("Generation git_commit changed on resume", RuntimeWarning)
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with _tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = _Path(stream.name)
            _json.dump(plain, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            _os.fsync(stream.fileno())
        _os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return plain
