"""Client side of the policy plugin (runs in the teleop container, no torch allowed).

Wire format agreed with server.py in this folder:
  request  = {"state": float32 (J,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (horizon, J)}   absolute joint targets, in joint_order
Change both files together if your model needs something else.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sim_to_real_so101.policy_client.contract import ClientPolicy
from sim_to_real_so101.policy_client.discovery import PolicyManifest


class TemplateClientPolicy(ClientPolicy):
    def __init__(self, cameras: tuple[str, ...], language: bool, options: dict[str, str]) -> None:
        self.cameras = cameras
        self.language = language
        self.options = options

    @classmethod
    def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> "TemplateClientPolicy":
        return cls(manifest.cameras, manifest.language, options)

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
            raise ValueError(f"expected (H, {len(joint_order)}) actions, got {actions.shape}")
        return actions

    def reset(self) -> None:
        """No per-episode state in the template."""


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return TemplateClientPolicy.from_manifest(manifest, options)
