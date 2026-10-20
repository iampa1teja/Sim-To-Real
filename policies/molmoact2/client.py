"""Client side of the MolmoAct2 policy (runs in the teleop / real-robot containers, no torch).

Convert calibrated LeRobot joint values into the released SO-100/101 model frame
before inference, and invert that conversion for every returned action.
The simulator supplies its degree-per-range-unit scales and USD zero offsets.
Its range midpoints are not the original model's calibrated joint zeros. Direct clients
default to LeRobot degrees (gripper remains RANGE_0_100).
Wire format:
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)} model-frame absolute joint targets

Client option include_original_room=true appends that image after room/wrist;
the default remains the manifest's two cameras.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim_to_real_so101.policy_client.contract import ClientPolicy
from sim_to_real_so101.policy_client.discovery import PolicyManifest


def _include_original_room(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("include_original_room must be true or false")


class MolmoAct2ClientPolicy(ClientPolicy):
    def __init__(self, cameras: tuple[str, ...], language: bool,
                 input_joint_units: str = "lerobot_degrees", arm_degrees_per_unit: str = "",
                 include_original_room: bool = False, arm_zero_degrees: str = "") -> None:
        if not isinstance(include_original_room, bool):
            raise ValueError("include_original_room must be true or false")
        self.cameras = tuple(cameras)
        if include_original_room and "original_room" not in self.cameras:
            self.cameras += ("original_room",)
        self.language = language
        if input_joint_units not in ("lerobot_degrees", "lerobot_normalized", "usd_normalized", "model"):
            raise ValueError("input_joint_units must be lerobot_degrees, lerobot_normalized, usd_normalized, or model")
        # Official LeRobot joint-frame correction for the released checkpoint:
        # https://huggingface.co/docs/lerobot/main/en/molmoact2#joint-frame-transform-so-100101-zero-shot
        self.scale = np.asarray([1, -1, 1, 1, 1, 1], dtype=np.float32)
        self.offset = np.asarray([0, 90, 90, 0, 0, 0], dtype=np.float32)
        if input_joint_units in ("lerobot_normalized", "usd_normalized"):
            try:
                degrees = np.asarray([float(v) for v in arm_degrees_per_unit.split(",")], dtype=np.float32)
            except ValueError as exc:
                raise ValueError("arm_degrees_per_unit needs five finite nonzero scales") from exc
            if degrees.shape != (5,) or not np.isfinite(degrees).all() or np.any(degrees == 0):
                raise ValueError("arm_degrees_per_unit needs five finite nonzero scales")
            if input_joint_units == "usd_normalized":
                try:
                    zeros = np.asarray([float(v) for v in arm_zero_degrees.split(",")], dtype=np.float32)
                except ValueError as exc:
                    raise ValueError("arm_zero_degrees needs five finite USD joint midpoints") from exc
                if zeros.shape != (5,) or not np.isfinite(zeros).all():
                    raise ValueError("arm_zero_degrees needs five finite USD joint midpoints")
                # Reconstruct the actual USD angles before the model's frame
                # correction. Old LeRobot degrees use a fixed calibrated zero,
                # rather than the midpoint of each normalized simulator range.
                self.offset[:5] += self.scale[:5] * zeros
            self.scale[:5] *= degrees
        elif input_joint_units == "model":
            self.scale[:] = 1
            self.offset[:] = 0
        self.joint_transform = {
            "input_joint_units": input_joint_units,
            "model_joint_units": "so100_so101_molmoact2",
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
        }

    @classmethod
    def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> "MolmoAct2ClientPolicy":
        return cls(manifest.cameras, manifest.language,
                   options.get("input_joint_units", "lerobot_degrees"),
                   options.get("arm_degrees_per_unit", ""),
                   _include_original_room(options.get("include_original_room", False)),
                   options.get("arm_zero_degrees", ""))

    def build_request(
        self, joint_positions: np.ndarray, images: dict[str, np.ndarray], instruction: str
    ) -> dict[str, Any]:
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise KeyError(f"missing cameras {missing}; got {sorted(images)}")
        state = np.asarray(joint_positions, dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("joint state must contain six finite joint values")
        return {
            "state": state * self.scale + self.offset,
            "images": {c: np.asarray(images[c], dtype=np.uint8) for c in self.cameras},
            "instruction": instruction if self.language else "",
        }

    def parse_response(self, response: dict[str, Any], joint_order: list[str]) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(joint_order):
            raise ValueError(f"expected (T, {len(joint_order)}) actions, got {actions.shape}")
        if actions.shape[1] != 6 or not np.isfinite(actions).all():
            raise ValueError("actions must contain six finite joint values per step")
        return (actions - self.offset) / self.scale

    def reset(self) -> None:
        """No per-episode state."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return MolmoAct2ClientPolicy.from_manifest(manifest, options)
