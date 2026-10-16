"""MolmoAct2 server policy: wraps the official allenai/MolmoAct2 model behind the plugin contract.

Wire format (see client.py):
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)}  absolute joint targets, full model chunk
The model is loaded with trust_remote_code and queried through its own predict_action().

Options (--option KEY=VALUE):
  norm_tag    normalization statistics inside the checkpoint's norm_stats.json (default so100_so101_molmoact2)
  dtype       bfloat16 (default, under 16 GB) or float32 (about 26 GB)
  num_steps   flow-matching solver steps (default 10)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from server_core.contract import ServerPolicy

DEFAULT_NORM_TAG = "so100_so101_molmoact2"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_NUM_STEPS = 10
NUM_JOINTS = 6


def decode_actions(actions: Any) -> np.ndarray:
    """Return the model's action chunk as float32 (T, 6), dropping a leading batch dimension."""
    if hasattr(actions, "detach"):
        actions = actions.detach().float().cpu().numpy()
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != NUM_JOINTS:
        raise ValueError(f"expected (T, {NUM_JOINTS}) actions, got {actions.shape}")
    return actions


def seed_everything(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32 integer")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MolmoAct2ServerPolicy(ServerPolicy):
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.checkpoint = ""
        self.norm_tag = DEFAULT_NORM_TAG
        self.dtype_name = DEFAULT_DTYPE
        self.num_steps = DEFAULT_NUM_STEPS
        self.device = "cuda"

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        if not checkpoint:
            raise ValueError("--checkpoint is required for molmoact2 (local folder or Hugging Face id)")
        if checkpoint.startswith("/") and not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"Model path {checkpoint} does not exist")
        self.checkpoint = checkpoint
        self.device = device
        self.norm_tag = options.get("norm_tag", DEFAULT_NORM_TAG)
        self.dtype_name = options.get("dtype", DEFAULT_DTYPE)
        self.num_steps = int(options.get("num_steps", DEFAULT_NUM_STEPS))
        if self.dtype_name not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, self.dtype_name)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(checkpoint, trust_remote_code=True, dtype=dtype)
            .to(device)
            .eval()
        )

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            seed_everything(seed)
        return {"seed": seed}

    def infer(self, request: dict) -> dict:
        if self.model is None:
            raise RuntimeError("policy not loaded")
        import torch
        from PIL import Image

        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (NUM_JOINTS,):
            raise ValueError(f"state must have shape ({NUM_JOINTS},), got {state.shape}")
        # Camera order does not matter to the model, but keep the request's order for reproducibility.
        images = [Image.fromarray(np.asarray(image, dtype=np.uint8)) for image in request["images"].values()]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dtype_name == "bfloat16"):
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=request.get("instruction", ""),
                state=state,
                norm_tag=self.norm_tag,
                inference_action_mode="continuous",
                num_steps=self.num_steps,
                enable_depth_reasoning=False,
            )
        return {"actions": decode_actions(out.actions)}

    def info(self) -> dict:
        return {
            "name": "molmoact2",
            "norm_tag": self.norm_tag,
            "dtype": self.dtype_name,
            "checkpoint": Path(self.checkpoint).name if self.checkpoint else "",
        }


def create() -> ServerPolicy:
    return MolmoAct2ServerPolicy()
