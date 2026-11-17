"""Immutable generation values and structural interfaces (v4 contracts §3)."""

from __future__ import annotations

import math as _math
import types as _types
from collections.abc import Mapping as _Mapping, Sequence as _Sequence
from dataclasses import dataclass as _dataclass, field as _field, fields as _fields
from enum import Enum as _Enum
from pathlib import Path as _Path
from typing import Any as _Any, Literal as _Literal, Protocol as _Protocol
from typing import Self as _Self, Union as _Union, get_args as _get_args
from typing import get_origin as _get_origin, get_type_hints as _get_type_hints
from typing import runtime_checkable as _runtime_checkable

import numpy as _np

from core.generate.errors import SchemaError as _SchemaError
from core.tasks.base import Pose as _Pose, StepState as _StepState


class Kind(str, _Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RECOVERY = "recovery"


class SegmentLabel(str, _Enum):
    NOMINAL = "nominal"
    PERTURBATION = "perturbation"
    FAILURE = "failure"
    RECOVERY = "recovery"


class AttemptStatus(str, _Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class SkipReason(str, _Enum):
    SCENARIO_SEEN = "scenario_seen"
    SAMPLING_FAILED = "sampling_failed"


class RejectionReason(str, _Enum):
    EXPERT_FAILED = "expert_failed"
    UNPLANNED_REGRESSION = "unplanned_regression"
    PERTURBATION_NOT_APPLIED = "perturbation_not_applied"
    PERTURBATION_INEFFECTIVE = "perturbation_ineffective"
    NO_REGRESSION = "no_regression"
    RECOVERY_FAILED = "recovery_failed"
    DUPLICATE_TRAJECTORY = "duplicate_trajectory"
    INVALID_STATE = "invalid_state"
    ENV_ERROR = "env_error"


class PerturbationGroup(str, _Enum):
    EXPERT = "expert"
    WORLD = "world"


def _freeze(value):
    """Copy JSON containers recursively into immutable equivalents."""
    if isinstance(value, _Mapping):
        return _types.MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _mapping(value):
    return _types.MappingProxyType(dict(value))


def _empty_mapping():
    return _types.MappingProxyType({})


def _array(value, shape=None):
    result = _np.array(value, dtype=_np.float64, order="C", copy=True)
    if result.ndim != 1 or (shape is not None and result.shape != shape):
        raise ValueError(f"Expected shape {shape if shape is not None else '(J,)'}; got {result.shape}")
    result.flags.writeable = False
    return result


def _check_seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be a Python integer in [0, 2**64)")


def _encode(value):
    if isinstance(value, _Enum):
        return value.value
    if isinstance(value, _Pose):
        return {"position": [float(v) for v in value.position],
                "quat_xyzw": [float(v) for v in value.quat_xyzw]}
    if isinstance(value, _Json):
        return value.to_json()
    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, _Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, _np.generic):
        return _encode(value.item())
    return value


def _validate_json(value):
    """Validate the whole tree, including arbitrary goal and parameter values."""
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise _SchemaError("JSON object keys must be strings")
            _validate_json(item)
    elif type(value) is list:
        for item in value:
            _validate_json(item)
    elif type(value) is float:
        if not _math.isfinite(value):
            raise _SchemaError("Non-finite JSON number")
    elif type(value) not in (str, int, bool, type(None)):
        raise _SchemaError(f"Not a JSON value: {type(value).__name__}")


def _decode(annotation, value):
    origin, args = _get_origin(annotation), _get_args(annotation)
    if annotation is _Any:
        return value
    if origin in (_types.UnionType, _Union):
        for option in args:
            try:
                return _decode(option, value)
            except (ValueError, TypeError, KeyError):
                pass
        raise _SchemaError(f"Value does not match {annotation}")
    if origin is _Mapping:
        if type(value) is not dict:
            raise _SchemaError("Expected a JSON object")
        return {_decode(args[0], key): _decode(args[1], item) for key, item in value.items()}
    if origin is tuple:
        if type(value) is not list:
            raise _SchemaError("Expected a JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item) for item in value)
        if len(value) != len(args):
            raise _SchemaError("Wrong tuple length")
        return tuple(_decode(kind, item) for kind, item in zip(args, value))
    if annotation is _np.ndarray:
        if type(value) is not list:
            raise _SchemaError("Expected a JSON array")
        return [_decode(float, item) for item in value]
    if annotation is _Pose or (isinstance(annotation, type) and issubclass(annotation, _Json)):
        return _from_dict(annotation, value)
    if isinstance(annotation, type) and issubclass(annotation, _Enum):
        if type(value) is not str:
            raise _SchemaError("Expected an enum value string")
        return annotation(value)
    if annotation is float:
        if type(value) not in (int, float):
            raise _SchemaError("Expected a JSON number")
        result = float(value)
        if not _math.isfinite(result):
            raise _SchemaError("Non-finite number")
        return result
    if type(value) is not annotation:
        raise _SchemaError(f"Expected {annotation}, got {type(value).__name__}")
    return value


def _from_dict(cls, payload):
    if type(payload) is not dict:
        raise _SchemaError("Expected a JSON object")
    schema = getattr(cls, "_schema", None)
    expected = {field.name for field in _fields(cls)}
    if schema is not None:
        expected.add("schema")
        if payload.get("schema") != schema:
            raise _SchemaError(f"Expected schema {schema}")
    if payload.keys() != expected:
        raise _SchemaError(f"Wrong keys for {cls.__name__}: expected {sorted(expected)}")
    hints = _get_type_hints(cls)
    values = {}
    for field in _fields(cls):
        value = payload[field.name]
        # §15 permits arbitrary nested JSON in params, beyond the table's type shorthand.
        if cls is PerturbationSpec and field.name == "params":
            if type(value) is not dict:
                raise _SchemaError("params must be a JSON object")
            values[field.name] = value
        else:
            values[field.name] = _decode(hints[field.name], value)
    return cls(**values)


class _Json:
    _schema = None

    def to_json(self) -> dict:
        result = {field.name: _encode(getattr(self, field.name)) for field in _fields(self)}
        if self._schema is not None:
            result["schema"] = self._schema
        return result

    @classmethod
    def from_json(cls, d) -> _Self:
        try:
            _validate_json(d)
            return _from_dict(cls, d)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _SchemaError(f"Invalid {cls.__name__}: {exc}") from exc


class _ArrayJson(_Json):
    __hash__ = None

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        for field in _fields(self):
            left, right = getattr(self, field.name), getattr(other, field.name)
            if isinstance(left, _np.ndarray):
                if left.shape != right.shape or not _np.array_equal(left, right):
                    return False
            elif left != right:
                return False
        return True


@_dataclass(frozen=True)
class ObjectState(_Json):
    pose: _Pose
    active: bool = True
    color: str | None = None
    __hash__ = None


@_dataclass(frozen=True)
class PerturbationSpec(_Json):
    type: str
    group: PerturbationGroup
    onset_stage: str | None
    delay_steps: int = 0
    params: _Mapping[str, float | int | str | tuple] = _field(default_factory=_empty_mapping)
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "group", PerturbationGroup(self.group))
        object.__setattr__(self, "params", _freeze(self.params))


@_dataclass(frozen=True)
class NoiseSpec(_Json):
    theta: float
    sigma_rad: float
    max_abs_rad: float
    __hash__ = None


@_dataclass(frozen=True)
class EpisodeSpec(_Json):
    task: str
    arm: str
    scene: str
    kind: Kind
    attempt_index: int
    seed: int
    objects: _Mapping[str, ObjectState]
    roles: _Mapping[str, str]
    goal: _Mapping[str, _Any]
    instruction: str
    initial_q: tuple[float, ...]
    held_object: str | None = None
    perturbation: PerturbationSpec | None = None
    noise: NoiseSpec | None = None
    max_steps: int = 600
    fps: int = 30
    _schema = "episode_spec/1"
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "kind", Kind(self.kind))
        object.__setattr__(self, "objects", _mapping(self.objects))
        object.__setattr__(self, "roles", _mapping(self.roles))
        object.__setattr__(self, "goal", _freeze(self.goal))
        object.__setattr__(self, "initial_q", tuple(self.initial_q))
        _check_seed(self.seed)
        if (self.kind is Kind.SUCCESS) != (self.perturbation is None):
            raise ValueError("Only success episodes have no perturbation")
        if self.noise is not None and self.kind is not Kind.SUCCESS:
            raise ValueError("Noise is allowed only on success episodes")
        for name in self.roles.values():
            if name not in self.objects or not self.objects[name].active:
                raise ValueError(f"Role names an inactive or missing object: {name}")

    @property
    def expert_mode(self) -> _Literal["closed_loop", "blind"]:
        return "blind" if self.kind is Kind.FAILURE else "closed_loop"


@_dataclass(frozen=True)
class StageProgress(_Json):
    frame: int
    stage_index: int
    furthest_index: int
    satisfied: _Mapping[str, bool]
    regressed_stage: str | None
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "satisfied", _mapping(self.satisfied))


@_dataclass(frozen=True, eq=False)
class ExpertObs(_ArrayJson):
    frame: int
    q: _np.ndarray
    ee: _Pose
    tcp: _Pose
    gripper_gap_m: float
    objects: _Mapping[str, ObjectState]
    signals: _Mapping[str, bool] = _field(default_factory=_empty_mapping)
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "q", _array(self.q))
        object.__setattr__(self, "objects", _mapping(self.objects))
        object.__setattr__(self, "signals", _mapping(self.signals))

    def step_state(self, joint_names: _Sequence[str] | None = None) -> _StepState:
        if joint_names is not None and len(joint_names) != len(self.q):
            raise ValueError("joint_names length must equal q length")
        values = dict(objects={name: state.pose for name, state in self.objects.items() if state.active},
                      ee=self.ee, gripper=self.gripper_gap_m, signals=dict(self.signals))
        fields = {field.name for field in _fields(_StepState)}
        if "tcp" in fields:
            values["tcp"] = self.tcp
        if "joints" in fields and joint_names is not None:
            values["joints"] = {name: float(q) for name, q in zip(joint_names, self.q)}
        return _StepState(**values)


@_dataclass(frozen=True, eq=False)
class ExpertAction(_ArrayJson):
    q_target: _np.ndarray
    phase: str
    done: bool
    injected: bool
    reentered: bool
    resumed_stage: str | None
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "q_target", _array(self.q_target))


@_dataclass(frozen=True, eq=False)
class FrameLabels(_ArrayJson):
    stage: int
    imitate: bool
    ee_pose_wxyz: _np.ndarray
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "ee_pose_wxyz", _array(self.ee_pose_wxyz, (7,)))


@_dataclass(frozen=True, eq=False)
class StepRecord(_ArrayJson):
    frame: int
    state_raw: _np.ndarray
    action_raw: _np.ndarray
    labels: FrameLabels
    phase: str
    injected: bool
    reentered: bool
    objects: _Mapping[str, _Pose]
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "state_raw", _array(self.state_raw))
        object.__setattr__(self, "action_raw", _array(self.action_raw))
        object.__setattr__(self, "objects", _mapping(self.objects))


@_dataclass(frozen=True)
class Segment(_Json):
    label: SegmentLabel
    start: int
    end: int
    phase: str | None = None
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "label", SegmentLabel(self.label))


@_dataclass(frozen=True)
class RecoveryInfo(_Json):
    detected_frame: int
    resumed_stage: str
    retries: int
    back_on_plan_frame: int
    __hash__ = None


@_dataclass(frozen=True)
class Fingerprints(_Json):
    scenario_key: str
    spec_fp: str
    traj_fp: str | None
    appearance_fp: str | None
    episode_fp: str | None
    __hash__ = None


@_dataclass(frozen=True)
class EpisodeResult(_Json):
    task: str
    kind: Kind
    attempt_index: int
    seed: int
    status: AttemptStatus
    reason: SkipReason | RejectionReason | None
    episode_index: int | None
    num_frames: int
    success: bool | None
    failure_mode: str | None
    furthest_stage: str | None
    stage: tuple[int, ...]
    imitate: tuple[bool, ...]
    segments: tuple[Segment, ...]
    perturbation_onset_frame: int | None
    perturbation_applied: bool
    recovery: RecoveryInfo | None
    fingerprints: Fingerprints
    _schema = "episode_result/1"
    __hash__ = None

    def __post_init__(self):
        object.__setattr__(self, "kind", Kind(self.kind))
        object.__setattr__(self, "status", AttemptStatus(self.status))
        if self.reason is not None:
            try:
                reason = SkipReason(self.reason)
            except ValueError:
                reason = RejectionReason(self.reason)
            object.__setattr__(self, "reason", reason)
        for name in ("stage", "imitate", "segments"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        _check_seed(self.seed)


@_runtime_checkable
class GenEnv(_Protocol):
    fps: int
    joint_names: tuple[str, ...]
    camera_names: tuple[str, ...]

    def reset(self, spec: EpisodeSpec) -> ExpertObs: ...
    def step(self, q_target: _np.ndarray) -> ExpertObs: ...
    def observe(self) -> ExpertObs: ...
    def set_object_pose(self, name: str, pose: _Pose) -> None: ...
    def render(self) -> dict[str, _np.ndarray]: ...
    def close(self) -> None: ...


@_runtime_checkable
class EpisodeWriter(_Protocol):
    root: _Path

    def episode_count(self) -> int: ...
    def episode_lengths(self) -> list[int]: ...
    def add_frame(self, action_raw: _np.ndarray, state_raw: _np.ndarray,
                  images: _Mapping[str, _np.ndarray], task: str, labels: FrameLabels) -> None: ...
    def discard_episode(self) -> None: ...
    def save_episode(self) -> int: ...
    def finalize(self) -> None: ...


@_runtime_checkable
class Expert(_Protocol):
    def reset(self, spec: EpisodeSpec, obs: ExpertObs) -> None: ...
    def act(self, obs: ExpertObs, progress: StageProgress) -> ExpertAction: ...

    @property
    def phase(self) -> str: ...

    @property
    def done(self) -> bool: ...
