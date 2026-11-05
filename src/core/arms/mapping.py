from dataclasses import dataclass

import numpy as np
import torch

from core.arms.joint_calibration import FOLLOWER_JOINTS, GRIPPER_JOINT
from core.arms.joint_limits import runtime_joint_limits
from core.arms.spec import load_arm_spec

ARM_SCALE, GRIPPER_SCALE = 200.0, 100.0  

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
        arm, grip = ~self.gripper_mask, self.gripper_mask
        normalized = torch.empty_like(raw)
        normalized[..., arm] = (raw[..., arm] + 100.0) / ARM_SCALE
        normalized[..., grip] = raw[..., grip] / GRIPPER_SCALE
        mapped_deg = self.mins + normalized * (self.maxs - self.mins)
        return torch.deg2rad(torch.clamp(mapped_deg, min=self.clamp_mins, max=self.clamp_maxs))

    def to_raw(self, radians) -> torch.Tensor:
        radians = torch.as_tensor(radians, dtype=self.mins.dtype, device=self.mins.device)
        normalized = (torch.rad2deg(radians) - self.mins) / (self.maxs - self.mins)
        raw = torch.empty_like(normalized)
        raw[..., ~self.gripper_mask] = normalized[..., ~self.gripper_mask] * ARM_SCALE - 100.0
        raw[..., self.gripper_mask] = normalized[..., self.gripper_mask] * GRIPPER_SCALE
        return raw

    def raw_from_actions(self, action: dict) -> torch.Tensor: 
        return torch.tensor([action[f"{name}.pos"] for name in self.names], dtype=self.mins.dtype, device=self.mins.device)

    def to_radians_np(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        mins, maxs, clamp_mins, clamp_maxs = (t.detach().cpu().double().numpy() for t in (self.mins, self.maxs, self.clamp_mins, self.clamp_maxs))
        gripper_mask = self.gripper_mask.cpu().numpy()

        normalized = np.empty_like(raw)
        normalized[..., ~gripper_mask] = (raw[..., ~gripper_mask] + 100) / 200
        normalized[..., gripper_mask] = (raw[..., gripper_mask] / 100)

        mapped_degs = mins + normalized * (maxs - mins)
        mapped_degs = np.clip(mapped_degs, clamp_mins, clamp_maxs)

        return np.deg2rad(mapped_degs)
    