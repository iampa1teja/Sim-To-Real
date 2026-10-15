"""Dummy server policy: repeats the current joint state for the whole horizon.

Needs no model and no GPU, so it is the quickest way to test the launcher, the transport and
RemotePolicy end to end.
"""
from __future__ import annotations

import numpy as np

from server_core.contract import ServerPolicy

HORIZON = 8


class HoldPositionServerPolicy(ServerPolicy):
    def __init__(self) -> None:
        self.horizon = HORIZON
        self.seed: int | None = None
        self.calls = 0

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        self.horizon = int(options.get("horizon", HORIZON))

    def reset(self, seed: int | None = None) -> dict:
        self.seed = seed
        self.calls = 0
        return {"seed": seed}

    def infer(self, request: dict) -> dict:
        state = np.asarray(request["state"], dtype=np.float32)
        self.calls += 1
        return {"actions": np.tile(state, (self.horizon, 1))}

    def info(self) -> dict:
        return {"name": "hold_position", "horizon": self.horizon, "calls": self.calls}


def create() -> ServerPolicy:
    return HoldPositionServerPolicy()
