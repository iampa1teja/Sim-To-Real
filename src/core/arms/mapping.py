from dataclasses import dataclass

import numpy as np
import torch

from core.arms.joint_calibration import FOLLOWER_JOINTS, GRIPPER_JOINT
from core.arms.joint_limits import runtime_joint_limits
from core.arms.spec import load_arm_spec

ARM_SCALE, GRIPPER_SCALE = 200.0, 100.0


def _conversion_arrays(values, mapping, joint_order, gripper_joints):
    """Resolve explicit calibrated bounds without importing scene configuration."""
    if isinstance(mapping, JointMapping):
        low, high, gripper = mapping.mins, mapping.maxs, mapping.gripper_mask
        if isinstance(values, torch.Tensor):
            values = values.to(dtype=low.dtype, device=low.device)
        else:
            values = np.asarray(values, dtype=np.float64)
            low, high, gripper = (x.detach().cpu().numpy() for x in (low, high, gripper))
            low, high = low.astype(np.float64), high.astype(np.float64)
    else:
        joint_order = tuple(mapping) if joint_order is None else tuple(joint_order)
        low = np.asarray([mapping[name]["joint_min"] for name in joint_order], dtype=np.float64)
        high = np.asarray([mapping[name]["joint_max"] for name in joint_order], dtype=np.float64)
        if not np.isfinite([low, high]).all() or np.any(high <= low):
            raise ValueError("Mapping bounds must be finite and increasing")
        gripper = np.asarray([name in gripper_joints for name in joint_order])
        if isinstance(values, torch.Tensor):
            if not values.is_floating_point():
                values = values.to(dtype=torch.float64)
            low, high = (torch.as_tensor(x, dtype=values.dtype, device=values.device) for x in (low, high))
            gripper = torch.as_tensor(gripper, device=values.device)
        else:
            values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != len(low):
        raise ValueError(f"Expected {len(low)} joint values on the last axis")
    if isinstance(values, torch.Tensor):
        scale = torch.where(gripper, GRIPPER_SCALE, ARM_SCALE)
        offset = torch.where(gripper, 0.0, 100.0)
    else:
        scale = np.where(gripper, GRIPPER_SCALE, ARM_SCALE)
        offset = np.where(gripper, 0.0, 100.0)
    return values, low, high, scale, offset


def units_to_degrees(values, mapping, joint_order=None, *, gripper_joints=()):
    """Decode normalized units with explicit bounds, without a runtime clamp.

    ``mapping`` is either a name-to-span dictionary or a ``JointMapping``. Dictionary
    callers supply gripper joint names explicitly; body joints span -100..100 and
    gripper joints 0..100. NumPy arrays and Torch tensors retain their backend.
    """
    values, low, high, scale, offset = _conversion_arrays(values, mapping, joint_order, gripper_joints)
    return low + (values + offset) / scale * (high - low)


def degrees_to_units(values, mapping, joint_order=None, *, gripper_joints=()):
    """Encode physical angles into normalized units, without clamping or wrapping."""
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

    @classmethod
    def from_setup(cls, selection=None, device="cpu"):
        if load_arm_spec()[0]["lerobot"]["use_degrees"]:
            raise ValueError("JointMapping needs use_degrees = false in arm.toml [arm.lerobot]")
        
        runtime = runtime_joint_limits()
        mapping = dict(runtime)
        if selection is not None:
            mapping.update(selection.mapping or {})
        unknown = mapping.keys() - runtime.keys()

        if unknown:
            raise ValueError(f"Unknown joint mapping names: {sorted(unknown)}")
        names = FOLLOWER_JOINTS

        def bounds(source, key):
            return [float(source[name][key]) for name in names]

        mins       = torch.tensor(bounds(mapping, "joint_min"), dtype=torch.float32, device=device,)
        maxs       = torch.tensor(bounds(mapping, "joint_max"), dtype=torch.float32, device=device,)
        clamp_mins = torch.tensor(bounds(runtime, "joint_min"), dtype=torch.float32, device=device,)
        clamp_maxs = torch.tensor(bounds(runtime, "joint_max"), dtype=torch.float32, device=device,)

        for label, low, high in (("calibrated", mins, maxs), ("runtime", clamp_mins, clamp_maxs)):
            if not (torch.isfinite(low).all() and torch.isfinite(high).all() and (high > low).all()):
                raise ValueError(f"{label.capitalize()} joint bounds must be finite and increasing")

        gripper_mask = torch.tensor([name == GRIPPER_JOINT for name in names], device=device)

        return cls(
            names=names,
            mins=mins,
            maxs=maxs,
            clamp_mins=clamp_mins,
            clamp_maxs=clamp_maxs,
            gripper_mask=gripper_mask,
        )

    def to_radians(self, raw) -> torch.Tensor:
        raw = torch.as_tensor(raw, dtype=self.mins.dtype, device=self.mins.device)
        mapped_deg = units_to_degrees(raw, self)
        return torch.deg2rad(torch.clamp(mapped_deg, min=self.clamp_mins, max=self.clamp_maxs))

    def to_raw(self, radians) -> torch.Tensor:
        radians = torch.as_tensor(radians, dtype=self.mins.dtype, device=self.mins.device)
        return degrees_to_units(torch.rad2deg(radians), self)

    def raw_from_actions(self, action: dict) -> torch.Tensor: 
        return torch.tensor([action[f"{name}.pos"] for name in self.names], dtype=self.mins.dtype, device=self.mins.device)

    def to_radians_np(self, raw: np.ndarray) -> np.ndarray:
        mapped_degrees = units_to_degrees(np.asarray(raw, dtype=np.float64), self)
        clamp_mins, clamp_maxs = (t.detach().cpu().double().numpy() for t in (self.clamp_mins, self.clamp_maxs))
        return np.deg2rad(np.clip(mapped_degrees, clamp_mins, clamp_maxs))
