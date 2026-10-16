"""GR00T N1.7 server policy: wraps the official Gr00tPolicy behind the plugin contract.

Wire format (see client.py):
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)}  absolute joint targets, full model chunk
GR00T's own input format (batch and time dims, split arm/gripper state) is built here, so the
simulator and the real robot clients stay model-agnostic.
"""
from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from server_core.contract import ServerPolicy

DEFAULT_EMBODIMENT_TAG = "NEW_EMBODIMENT"
LANGUAGE_KEY = "annotation.human.task_description"


def build_observation(request: dict[str, Any]) -> dict[str, Any]:
    """Turn a plugin request into the (B=1, T=1, ...) observation Gr00tPolicy expects."""
    state = np.asarray(request["state"], dtype=np.float32)
    if state.shape != (6,):
        raise ValueError(f"state must have shape (6,), got {state.shape}")
    return {
        "video": {name: np.asarray(image)[None, None, ...] for name, image in request["images"].items()},
        "state": {"single_arm": state[None, None, :5], "gripper": state[None, None, 5:6]},
        "language": {LANGUAGE_KEY: [[request.get("instruction", "")]]},
    }


def decode_action(action: dict[str, Any]) -> np.ndarray:
    """Concatenate GR00T's (1, T, 5) arm and (1, T, 1) gripper outputs into a (T, 6) array."""
    arm = np.asarray(action["single_arm"])[0]
    gripper = np.asarray(action["gripper"])[0]
    return np.concatenate([arm, gripper], axis=-1).astype(np.float32)


def seed_everything(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32 integer")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Gr00tN17ServerPolicy(ServerPolicy):
    def __init__(self) -> None:
        self.policy = None
        self.embodiment_tag = DEFAULT_EMBODIMENT_TAG
        self.checkpoint = ""
        self._stack = contextlib.ExitStack()

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        if not checkpoint:
            raise ValueError("--checkpoint is required for groot_n17")
        if checkpoint.startswith("/") and not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Model path {checkpoint} does not exist")
        self.embodiment_tag = options.get("embodiment_tag", DEFAULT_EMBODIMENT_TAG)
        self.checkpoint = checkpoint

        from offline_cosmos import offline_cosmos_processor, transformers_hub_cache

        os.environ["TRANSFORMERS_CACHE"] = str(transformers_hub_cache())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        for name in ("GROOT_SKIP_HF_MODEL_WEIGHTS", "GROOT_HF_LOCAL_FIRST", "PYTEST_CURRENT_TEST"):
            os.environ.pop(name, None)

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        # Keep the offline processor patch for the whole server lifetime: the processor may be
        # rebuilt lazily, as it was when the official server ran inside this context.
        self._stack.enter_context(offline_cosmos_processor())
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(self.embodiment_tag),
            model_path=checkpoint,
            device=device,
            strict=True,
        )

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            seed_everything(seed)
        if self.policy is not None:
            self.policy.reset()
        return {"seed": seed}

    def infer(self, request: dict) -> dict:
        if self.policy is None:
            raise RuntimeError("policy not loaded")
        action, _info = self.policy.get_action(build_observation(request))
        return {"actions": decode_action(action)}

    def info(self) -> dict:
        return {
            "name": "groot_n17",
            "embodiment_tag": self.embodiment_tag,
            "checkpoint": Path(self.checkpoint).name if self.checkpoint else "",
        }


def create() -> ServerPolicy:
    return Gr00tN17ServerPolicy()
