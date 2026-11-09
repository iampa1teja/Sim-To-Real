"""Standard NumPy request/response format shared by policy plugins (no torch)."""

from typing import Any

import numpy as np

from .contract import ClientPolicy
from .discovery import PolicyManifest


class DefaultClientPolicy(ClientPolicy):
    """Send selected cameras and raw joint state; receive absolute joint targets."""

    def __init__(self, cameras: tuple[str, ...], language: bool, options: dict[str, str] | None = None) -> None:
        self.cameras = cameras
        self.language = language
        self.options = {} if options is None else options

    @classmethod
    def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> "DefaultClientPolicy":
        return cls(manifest.cameras, manifest.language, options)

    def build_request(
        self, joint_positions: np.ndarray, images: dict[str, np.ndarray], instruction: str
    ) -> dict[str, Any]:
        missing = [camera for camera in self.cameras if camera not in images]
        if missing:
            raise KeyError(f"missing cameras {missing}; got {sorted(images)}")
        return {
            "state": np.asarray(joint_positions, dtype=np.float32),
            "images": {camera: np.asarray(images[camera], dtype=np.uint8) for camera in self.cameras},
            "instruction": instruction if self.language else "",
        }

    def parse_response(self, response: dict[str, Any], joint_order: list[str]) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != len(joint_order):
            raise ValueError(f"expected (H, {len(joint_order)}) actions, got {actions.shape}")
        return actions

    def reset(self) -> None:
        """The standard wire format carries no per-episode state."""
