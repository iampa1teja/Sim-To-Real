"""Client side of the MolmoAct2 policy (runs in the teleop / real-robot containers, no torch).

The server (server.py) owns the model's input format, so this side only picks the manifest cameras and
passes the joint state through. Wire format:
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)} absolute joint targets in joint_order
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim_to_real_so101.policy_client.contract import ClientPolicy
from sim_to_real_so101.policy_client.discovery import PolicyManifest


class MolmoAct2ClientPolicy(ClientPolicy):
    def __init__(self, cameras: tuple[str, ...], language: bool) -> None:
        self.cameras = cameras
        self.language = language

    @classmethod
    def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> "MolmoAct2ClientPolicy":
        return cls(manifest.cameras, manifest.language)

    def build_request(
        self, joint_positions: np.ndarray, images: dict[str, np.ndarray], instruction: str
    ) -> dict[str, Any]:
        missing = [c for c in self.cameras if c not in images]
        if missing:
            raise KeyError(f"missing cameras {missing}; got {sorted(images)}")
        return {
            "state": np.asarray(joint_positions, dtype=np.float32),
            "images": {c: np.asarray(images[c], dtype=np.uint8) for c in self.cameras},
            "instruction": instruction if self.language else "",
        }

    def parse_response(self, response: dict[str, Any], joint_order: list[str]) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(joint_order):
            raise ValueError(f"expected (T, {len(joint_order)}) actions, got {actions.shape}")
        return actions

    def reset(self) -> None:
        """No per-episode state."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return MolmoAct2ClientPolicy.from_manifest(manifest, options)
