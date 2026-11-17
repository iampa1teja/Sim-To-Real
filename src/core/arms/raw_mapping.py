"""Numpy-only raw dataset-unit and joint-radian conversion."""

from dataclasses import dataclass as _dataclass
from collections.abc import Mapping as _Mapping

import numpy as _np

from core.arms import spec as _spec
from core.scenes import load_setup as _load_setup


def _readonly(value, dtype=_np.float64):
    result = _np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _block(setup, arm):
    arms = setup.get("arms", {})
    return arms.get(arm or _spec.ARM_NAME, setup) if isinstance(arms, _Mapping) else setup


@_dataclass(frozen=True, eq=False)
class RawMapping:
    names: tuple[str, ...]
    mins_deg: _np.ndarray
    maxs_deg: _np.ndarray
    clamp_mins_deg: _np.ndarray
    clamp_maxs_deg: _np.ndarray
    gripper_mask: _np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        for key in ("mins_deg", "maxs_deg", "clamp_mins_deg", "clamp_maxs_deg", "gripper_mask"):
            value = _readonly(getattr(self, key), bool if key == "gripper_mask" else _np.float64)
            if value.shape != (len(self.names),):
                raise ValueError("Mapping bounds must match names")
            object.__setattr__(self, key, value)
        for low, high in ((self.mins_deg, self.maxs_deg), (self.clamp_mins_deg, self.clamp_maxs_deg)):
            if not (_np.all(_np.isfinite(low)) and _np.all(_np.isfinite(high)) and _np.all(high > low)):
                raise ValueError("Mapping bounds must be finite and increasing")

    @classmethod
    def from_tables(cls, names, mapping, runtime, gripper_joint):
        names = tuple(names)
        if set(runtime) != set(names) or set(mapping) - set(names) or gripper_joint not in names:
            raise ValueError("Mapping/runtime names must match the arm")
        effective = {**runtime, **mapping}
        try:
            def values(table, key):
                result = [table[name][key] for name in names]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in result):
                    raise ValueError("Joint bounds must be numbers")
                return result
            return cls(names, values(effective, "joint_min"), values(effective, "joint_max"),
                       values(runtime, "joint_min"), values(runtime, "joint_max"), [name == gripper_joint for name in names])
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed joint bounds") from exc

    @classmethod
    def from_setup(cls, setup=None, *, arm=None):
        setup = _load_setup() if setup is None else setup
        spec, _ = _spec.load_arm_spec(arm)
        if spec["lerobot"]["use_degrees"]:
            raise ValueError("RawMapping requires use_degrees=false")
        block = _block(setup, arm)
        return cls.from_tables(spec["lerobot"]["names"], block.get("sim_joint_mapping", {}),
                               block["runtime_joint_limits"], spec["lerobot"]["gripper_joint"])

    def _array(self, value):
        result = _np.asarray(value, dtype=_np.float64)
        if result.ndim < 1 or result.shape[-1] != len(self.names):
            raise ValueError("Last dimension must match joint names")
        return result

    def to_radians(self, raw) -> _np.ndarray:
        raw = self._array(raw)
        normalized = _np.where(self.gripper_mask, raw / 100, (raw + 100) / 200)
        degrees = _np.clip(self.mins_deg + normalized * (self.maxs_deg - self.mins_deg),
                           self.clamp_mins_deg, self.clamp_maxs_deg)
        return _np.deg2rad(degrees)

    def to_raw(self, rad) -> _np.ndarray:
        normalized = (_np.rad2deg(self._array(rad)) - self.mins_deg) / (self.maxs_deg - self.mins_deg)
        return _np.where(self.gripper_mask, normalized * 100, normalized * 200 - 100)

    def clamp_radians(self, rad) -> _np.ndarray:
        return _np.clip(self._array(rad), self.lower_rad, self.upper_rad)

    @property
    def lower_rad(self) -> _np.ndarray:
        return _readonly(_np.deg2rad(self.clamp_mins_deg))

    @property
    def upper_rad(self) -> _np.ndarray:
        return _readonly(_np.deg2rad(self.clamp_maxs_deg))
