"""Server side of the policy plugin (runs in the policy container, torch allowed here).

Imported by path from policies/<name>/server.py, so keep heavy imports inside load().
"""
from __future__ import annotations

from server_core.contract import ServerPolicy


class TemplateServerPolicy(ServerPolicy):
    """Wraps one model behind the four-method server contract."""

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        """Load weights / build the model. Called once before serving, with the launcher's --checkpoint, --device and --option values."""
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> dict:
        """Clear per-episode state (history, caches), apply the seed if given, return a small ack dict."""
        raise NotImplementedError

    def infer(self, request: dict) -> dict:
        """request is {"state": (J,), "images": {camera: (H, W, 3) uint8}, "instruction": str}.

        Return {"actions": float32 array (horizon, J)} of absolute joint targets.
        """
        raise NotImplementedError

    def info(self) -> dict:
        """Return small metadata (name, horizon, joint order) for describe."""
        raise NotImplementedError


def create() -> ServerPolicy:
    """Factory called by load_server_policy."""
    raise NotImplementedError
