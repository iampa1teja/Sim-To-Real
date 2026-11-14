from dataclasses import dataclass

import numpy as np
import torch

from utils.arms.spec import ArmSpec
from utils.arms.arms import load_arm_spec
from utils.arms.calibration import raw_endpoints

def _raw_endpoints(mapping, joint_order, gripper_joints):
    low, high = raw_endpoints(mapping, joint_order, gripper_joints)
    return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)


def mapping_for_arm(arm: ArmSpec, mapping):
    """Validate encoded-device metadata without guessing physical DEGREES zeroes."""
    mapping = getattr(mapping, "mapping", mapping)
    unknown = set(mapping) - set(arm.joint_names)
    if unknown:
        raise ValueError(f"Unknown joint mapping names: {sorted(unknown)}")
    spans = {name: dict(span) for name, span in arm.runtime_joint_limits.items()}
    spans.update({name: dict(span) for name, span in mapping.items()})
    if arm.lerobot.use_degrees:
        missing = [joint.name for joint in arm.action_joints if joint.role == "arm"
                   and ("raw_min" not in spans[joint.name] or "raw_max" not in spans[joint.name])]
        if missing:
            raise ValueError(f"Degree mode requires an explicit device codec from calibration for joints {missing}")
    _raw_endpoints(spans, arm.joint_names, arm.gripper_joint_names)
    for joint in arm.action_joints:
        # A stored explicit default codec is equivalent to the omitted default.
        # Degree-mode body endpoints remain explicit because their calibrated
        # zero/span cannot be inferred from the conventional normalized range.
        span = spans[joint.name]
        default_low = 0.0 if joint.role == "gripper" else -100.0
        if (not (arm.lerobot.use_degrees and joint.role == "arm")
                and span.get("raw_min", default_low) == default_low and span.get("raw_max", 100.0) == 100.0):
            span.pop("raw_min", None)
            span.pop("raw_max", None)
    return spans


def raw_joint_limits(arm: ArmSpec, mapping):
    """Return ordered raw-device bounds, accounting for codec direction and span."""
    spans = mapping_for_arm(arm, mapping)
    low, high = _raw_endpoints(spans, arm.joint_names, arm.gripper_joint_names)
    return {name: (float(min(a, b)), float(max(a, b))) for name, a, b in zip(arm.joint_names, low, high)}


def _conversion_arrays(values, mapping, joint_order, gripper_joints):
    """Resolve explicit physical and encoded-device bounds without scene state."""
    if isinstance(mapping, JointMapping):
        low, high, gripper = mapping.mins, mapping.maxs, mapping.gripper_mask
        raw_low, raw_high = mapping.raw_mins, mapping.raw_maxs
        if raw_low is None:
            raw_low = torch.where(gripper, 0.0, -100.0)
            raw_high = torch.full_like(low, 100.0)
        if isinstance(values, torch.Tensor):
            values = values.to(dtype=low.dtype, device=low.device)
        else:
            values = np.asarray(values, dtype=np.float64)
            low, high, raw_low, raw_high = (x.detach().cpu().double().numpy()
                                            for x in (low, high, raw_low, raw_high))
    else:
        joint_order = tuple(mapping) if joint_order is None else tuple(joint_order)
        low = np.asarray([mapping[name]["joint_min"] for name in joint_order], dtype=np.float64)
        high = np.asarray([mapping[name]["joint_max"] for name in joint_order], dtype=np.float64)
        if not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError("Mapping bounds must be finite and increasing")
        raw_low, raw_high = _raw_endpoints(mapping, joint_order, gripper_joints)
        if isinstance(values, torch.Tensor):
            if not values.is_floating_point():
                values = values.to(dtype=torch.float64)
            low, high, raw_low, raw_high = (torch.as_tensor(value, dtype=values.dtype, device=values.device)
                                            for value in (low, high, raw_low, raw_high))
        else:
            values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != len(low):
        raise ValueError(f"Expected {len(low)} joint values on the last axis")
    return values, low, high, raw_high - raw_low, -raw_low


def units_to_physical(values, mapping, joint_order=None, *, gripper_joints=()):
    """Decode normalized units with explicit bounds, without a runtime clamp.

    ``mapping`` is either a name-to-span dictionary or a ``JointMapping``. Dictionary
    callers supply gripper joint names explicitly; body joints span -100..100 and
    gripper joints 0..100. Optional raw_min/raw_max encode calibrated driver
    endpoint conventions, including reversed motors and LeRobot DEGREES zeroes.
    NumPy arrays and Torch tensors retain their backend.
    """
    values, low, high, scale, offset = _conversion_arrays(values, mapping, joint_order, gripper_joints)
    return low + (values + offset) / scale * (high - low)


def physical_to_units(values, mapping, joint_order=None, *, gripper_joints=()):
    """Encode physical degrees/metres into normalized units, without clamping or wrapping."""
    values, low, high, scale, offset = _conversion_arrays(values, mapping, joint_order, gripper_joints)
    return (values - low) / (high - low) * scale - offset

@dataclass(frozen=True)
class JointMapping:
    names: tuple[str, ...]
    mins: torch.Tensor 
    maxs: torch.Tensor
    clamp_mins: torch.Tensor
    clamp_maxs: torch.Tensor
    gripper_mask: torch.Tensor
    revolute_mask: torch.Tensor | None = None
    use_degrees: bool = False
    raw_mins: torch.Tensor | None = None
    raw_maxs: torch.Tensor | None = None

    @classmethod
    def from_arm(cls, arm: ArmSpec, mapping=None, *, device="cpu"):
        runtime = arm.runtime_joint_limits
        calibrated = mapping_for_arm(arm, runtime if mapping is None else mapping)
        names = arm.joint_names

        def bounds(source, key):
            return torch.tensor([float(source[name][key]) for name in names], dtype=torch.float32, device=device)

        mins, maxs = bounds(calibrated, "joint_min"), bounds(calibrated, "joint_max")
        clamp_mins, clamp_maxs = bounds(runtime, "joint_min"), bounds(runtime, "joint_max")
        for label, low, high in (("calibrated", mins, maxs), ("runtime", clamp_mins, clamp_maxs)):
            if not (torch.isfinite(low).all() and torch.isfinite(high).all() and (high > low).all()):
                raise ValueError(f"{label.capitalize()} joint bounds must be finite and increasing")
        return cls(names, mins, maxs, clamp_mins, clamp_maxs,
                   torch.tensor([joint.role == "gripper" for joint in arm.action_joints], device=device),
                   torch.tensor([joint.type == "revolute" for joint in arm.action_joints], device=device),
                   arm.lerobot.use_degrees,
                   *(torch.tensor(value, dtype=torch.float32, device=device) for value in
                     _raw_endpoints(calibrated, names, arm.gripper_joint_names)))

    def to_physical(self, raw) -> torch.Tensor:
        """Decode raw device values to degrees/metres without a runtime clamp."""
        raw = torch.as_tensor(raw, dtype=self.mins.dtype, device=self.mins.device)
        return units_to_physical(raw, self)

    def sim_to_physical(self, values) -> torch.Tensor:
        """Convert articulation radians/metres into calibration degrees/metres."""
        values = torch.as_tensor(values, dtype=self.mins.dtype, device=self.mins.device)
        if values.ndim == 0 or values.shape[-1] != len(self.names):
            raise ValueError(f"Expected {len(self.names)} joint values on the last axis")
        if self.revolute_mask is None:
            return torch.rad2deg(values)
        return torch.where(self.revolute_mask, torch.rad2deg(values), values)

    def to_sim(self, raw) -> torch.Tensor:
        """Map LeRobot actions to clamped articulation targets in radians/metres."""
        physical = torch.clamp(self.to_physical(raw), min=self.clamp_mins, max=self.clamp_maxs)
        if self.revolute_mask is None:
            return torch.deg2rad(physical)
        return torch.where(self.revolute_mask, torch.deg2rad(physical), physical)

    def to_raw(self, sim_values) -> torch.Tensor:
        physical = self.sim_to_physical(sim_values)
        return physical_to_units(physical, self)

    def raw_from_actions(self, action: dict) -> torch.Tensor:
        return torch.tensor([action[f"{name}.pos"] for name in self.names], dtype=self.mins.dtype,
                            device=self.mins.device)

    def to_sim_np(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        physical = units_to_physical(raw, self)
        clamp_mins, clamp_maxs = (t.detach().cpu().double().numpy() for t in (self.clamp_mins, self.clamp_maxs))
        physical = np.clip(physical, clamp_mins, clamp_maxs)
        if self.revolute_mask is None:
            return np.deg2rad(physical)
        return np.where(self.revolute_mask.detach().cpu().numpy(), np.deg2rad(physical), physical)



def usd_limits_deg(arm=None):
    """Declared physical limits: degrees for revolute joints, metres for prismatic."""
    arm = arm or load_arm_spec()[0]
    return {joint.name: tuple(joint.limits) for joint in arm.action_joints}


def dataset_state_to_radians(state, joint_mapping=None, *, arm=None):
    arm = arm or load_arm_spec()[0]
    limits = usd_limits_deg(arm)
    names = arm.joint_names
    state = np.asarray(state, dtype=float)
    if state.shape != (len(names),) or not np.isfinite(state).all():
        raise ValueError(f"Robot state must be {len(names)} finite values, got {state!r}")
    spans = arm.runtime_joint_limits
    for name, span in (joint_mapping or {}).items():
        if name not in spans:
            raise ValueError(f"Unknown joint mapping: {name}")
        spans[name] = dict(span)
    from utils.arms.mapping import mapping_for_arm, units_to_physical

    spans = mapping_for_arm(arm, spans)
    physical = units_to_physical(state, spans, names, gripper_joints=arm.gripper_joint_names)
    low, high = np.asarray([limits[name] for name in names]).T
    physical = np.clip(physical, low, high)
    revolute = np.asarray([joint.type == "revolute" for joint in arm.action_joints])
    return np.where(revolute, np.deg2rad(physical), physical)
