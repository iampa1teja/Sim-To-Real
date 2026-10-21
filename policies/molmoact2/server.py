"""MolmoAct2 server policy: wraps the official allenai/MolmoAct2 model behind the plugin contract.

Wire format (see client.py):
  request  = {"state": float32 (6,), "images": {camera: uint8 (H, W, 3)}, "instruction": str}
  response = {"actions": float32 (T, 6)}  absolute joint targets, full model chunk
The model is loaded with trust_remote_code and queried through its own predict_action().
Action denormalization uses float32 even when model storage is bfloat16.

Options (--option KEY=VALUE):
  norm_tag    normalization statistics inside the checkpoint's norm_stats.json (default so100_so101_molmoact2)
  dtype       bfloat16 (default, under 16 GB) or float32 (about 26 GB)
  action_dtype defaults to dtype; float32 keeps the action expert/norms/RoPE in
               original checkpoint precision with a bfloat16 backbone (~1.1 GB extra)
  num_steps   flow-matching solver steps (default 10)
  action_range_diagnostics false (default); true exposes the last pre-clipping
               normalized action range through describe, without changing actions
"""
from __future__ import annotations

import json
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


def load_processor(checkpoint: str):
    """Load list-form checkpoint special tokens with Transformers 4.57."""
    from transformers import AutoProcessor
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    tokenizer_config = get_tokenizer_config(checkpoint)
    extra_tokens = tokenizer_config.get("extra_special_tokens")
    options = {}
    if isinstance(extra_tokens, list):
        # Newer checkpoint exports use a list; 4.57 expects a named-token dict.
        # Keep those tokens special without changing their checkpoint token IDs.
        options["extra_special_tokens"] = {}
        options["additional_special_tokens"] = (
            tokenizer_config.get("additional_special_tokens") or []
        ) + extra_tokens
    return AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, use_fast=True, **options)


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


def preserve_action_denormalization_precision(model: Any) -> None:
    """Keep the loaded model's physical joint targets out of bfloat16.

    The released normalizer returns the input tensor's dtype. In bf16 mode,
    predict_action therefore rounds absolute angles near 180 degrees to whole
    degrees before its final float32 cast. Promote only the normalized actions
    entering this model instance's cached statistics; weights and flow inference
    retain their configured dtype, and the checkpoint's statistics stay intact.
    """
    import torch

    stats = model._get_robot_stats()
    if getattr(stats, "_so101_float32_action_denormalization", False):
        return
    original = stats.unnormalize_action

    def unnormalize_action(action: Any, norm_tag: str) -> Any:
        if torch.is_tensor(action):
            action = action.float()
        return original(action, norm_tag)

    stats.unnormalize_action = unnormalize_action
    stats._so101_float32_action_denormalization = True


def _is_text_rotary(module: Any) -> bool:
    return type(module).__name__ == "MolmoAct2RotaryEmbedding"


def restore_float32_action_modules(model: Any, checkpoint: str) -> None:
    """Restore the official low-memory FP32 module layout from original weights.

    Loading everything as bf16 and merely widening these modules would retain
    rounded weights. Read only promoted tensors from safetensors on CPU before
    moving the model to its execution device. This avoids a full-FP32 backbone.
    """
    import torch
    from safetensors import safe_open

    expert = getattr(getattr(model, "model", None), "action_expert", None)
    if expert is None:
        raise ValueError("action_dtype=float32 requires a MolmoAct2 action expert")
    expert.float()
    gate = getattr(model.model, "action_expert_depth_gate", None)
    if gate is not None:
        gate.float()
    for module in model.modules():
        if (isinstance(module, torch.nn.LayerNorm)
                or type(module).__name__ in {"ActionExpertRMSNorm", "MolmoAct2RMSNorm"}
                or _is_text_rotary(module)):
            module.float()

    # The optional mixed profile applies only to this model's dtype tree.
    targets = {name: value for name, value in model.state_dict(keep_vars=True).items()
               if value.is_floating_point() and value.dtype == torch.float32}
    root = Path(checkpoint)
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(checkpoint, allow_patterns=["*.safetensors", "model.safetensors.index.json"]))
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as weights:
            weight_map = dict.fromkeys(weights.keys(), "model.safetensors")
    missing = targets.keys() - weight_map.keys()
    if missing:
        raise ValueError(f"FP32 module weights missing from checkpoint: {sorted(missing)}")
    by_shard: dict[str, list[str]] = {}
    for name in targets:
        by_shard.setdefault(weight_map[name], []).append(name)
    with torch.no_grad():
        for shard, names in by_shard.items():
            with safe_open(str(root / shard), framework="pt", device="cpu") as weights:
                for name in names:
                    original = weights.get_tensor(name)
                    if original.shape != targets[name].shape:
                        raise ValueError(f"FP32 module shape mismatch for {name}")
                    targets[name].copy_(original)


def reset_float32_rotary_caches(model: Any) -> None:
    """Refresh the non-buffer RoPE alias after dtype/device moves."""
    import torch

    for module in model.modules():
        if _is_text_rotary(module):
            module.original_inv_freq = module.inv_freq
            module._pos_sin_cache = torch.empty(0, device=module.inv_freq.device, dtype=torch.float32)
            module._pos_cos_cache = torch.empty(0, device=module.inv_freq.device, dtype=torch.float32)


def _strict_bool_option(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name} must be true or false")


def capture_action_range_diagnostics(model: Any, record: Any) -> None:
    """Observe this instance's trajectory before the unchanged normalizer clips it."""
    import torch

    stats = model._get_robot_stats()
    original = stats.unnormalize_action

    def unnormalize_action(action: Any, norm_tag: str) -> Any:
        values = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        shape = list(values.shape)
        values = np.atleast_1d(values)
        features = values.shape[-1]
        flat = values.reshape(-1, features) if features else np.empty((0, 0))
        minimum = flat.min(axis=0) if flat.shape[0] else np.full(features, np.nan)
        maximum = flat.max(axis=0) if flat.shape[0] else np.full(features, np.nan)
        record({
            "norm_tag": norm_tag,
            "shape": shape,
            "min": [float(value) if np.isfinite(value) else None for value in minimum],
            "max": [float(value) if np.isfinite(value) else None for value in maximum],
            "below_minus_one": (flat < -1).sum(axis=0).tolist(),
            "above_one": (flat > 1).sum(axis=0).tolist(),
            "nonfinite": (~np.isfinite(flat)).sum(axis=0).tolist(),
        })
        # Delegate the original tensor unchanged. Existing FP32 denormalization
        # remains inside this wrapper, and all released clipping/statistics apply.
        return original(action, norm_tag)

    stats.unnormalize_action = unnormalize_action


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
        self.action_dtype_name = DEFAULT_DTYPE
        self.num_steps = DEFAULT_NUM_STEPS
        self.device = "cuda"
        self.action_range_diagnostics = False
        self._last_raw_action_range = None

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        if not checkpoint:
            raise ValueError("--checkpoint is required for molmoact2 (local folder or Hugging Face id)")
        if checkpoint.startswith("/") and not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"Model path {checkpoint} does not exist")
        self.checkpoint = checkpoint
        self.device = device
        self.norm_tag = options.get("norm_tag", DEFAULT_NORM_TAG)
        self.dtype_name = options.get("dtype", DEFAULT_DTYPE)
        self.action_dtype_name = options.get("action_dtype", self.dtype_name)
        self.num_steps = int(options.get("num_steps", DEFAULT_NUM_STEPS))
        self.action_range_diagnostics = _strict_bool_option(
            options.get("action_range_diagnostics", False), "action_range_diagnostics")
        self._last_raw_action_range = None
        if self.dtype_name not in ("bfloat16", "float32"):
            raise ValueError("dtype must be bfloat16 or float32")
        if self.action_dtype_name not in ("bfloat16", "float32"):
            raise ValueError("action_dtype must be bfloat16 or float32")
        if self.dtype_name == "float32" and self.action_dtype_name != "float32":
            raise ValueError("action_dtype=bfloat16 requires dtype=bfloat16")

        import torch
        from transformers import AutoModelForImageTextToText

        dtype = getattr(torch, self.dtype_name)
        self.processor = load_processor(checkpoint)
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, trust_remote_code=True, dtype=dtype)
        mixed_action_precision = self.dtype_name == "bfloat16" and self.action_dtype_name == "float32"
        if mixed_action_precision:
            restore_float32_action_modules(self.model, checkpoint)
        self.model.to(device).eval()
        if mixed_action_precision:
            reset_float32_rotary_caches(self.model)
        if self.dtype_name == "bfloat16":
            preserve_action_denormalization_precision(self.model)
        if self.action_range_diagnostics:
            capture_action_range_diagnostics(
                self.model, lambda value: setattr(self, "_last_raw_action_range", value))

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
                normalize_language=True,
                enable_cuda_graph=True,
            )
        return {"actions": decode_actions(out.actions)}

    def info(self) -> dict:
        info = {
            "name": "molmoact2",
            "norm_tag": self.norm_tag,
            "dtype": self.dtype_name,
            "action_dtype": self.action_dtype_name,
            "action_denormalization_dtype": "float32",
            "checkpoint": Path(self.checkpoint).name if self.checkpoint else "",
        }
        if self.action_range_diagnostics:
            info["action_range_diagnostics"] = True
            info["last_raw_action_range"] = self._last_raw_action_range
        return info


def create() -> ServerPolicy:
    return MolmoAct2ServerPolicy()
