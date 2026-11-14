"""Shared adapters for Isaac Lab tensor and ProxyArray data."""

import re
import numpy as np
import torch


def as_torch(x):
    """Return a torch view of 3.0 data (a ProxyArray with a zero-copy ``.torch``); plain tensors pass through."""
    return x if isinstance(x, torch.Tensor) else x.torch


def _arm_spec(env):
    from utils.arms.spec import load_arm_spec

    return getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]


def env_indices(env, env_ids):
    """Any env_ids an event can receive (None, slice, list, tensor) as a 1-D index tensor.

    Isaac Lab's EventManager passes startup, reset and interval terms different shapes
    (and different versions differ), so every event that indexes by environment
    normalizes through this instead of assuming a tensor.
    """
    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def _wxyz(q):
    """XYZW (Isaac Lab 3.0) -> WXYZ (v1 on-disk / numpy geometry helpers)."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def _xyzw(q):
    """WXYZ -> XYZW."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _env_prim_path(env, asset, index):
    """Concrete prim path of ``asset`` in environment ``index``."""
    path = asset.cfg.prim_path
    env_paths = getattr(env.scene, "env_prim_paths", None)
    # Asset configs may retain the namespace token or contain either version's
    # expanded regex. USD lookups need the actual scene namespace in all cases.
    namespace_pattern = r"(?:^|/)env_(?:\.\*|\[\^/\]\+)(?=/|$)"
    if env_paths is not None:
        env_path = env_paths[index]
        path = path.replace("{ENV_REGEX_NS}", env_path)
        match = re.search(namespace_pattern, path)
        return env_path + path[match.end():] if match else path
    return re.sub(
        namespace_pattern,
        lambda match: ("/" if match.group().startswith("/") else "") + f"env_{index}",
        path,
    )


def _ee_body(env, robot):
    """(ee body index, ee prim path in env 0) for the arm's end-effector link from arm.toml."""
    ee_link = _arm_spec(env).links.ee
    ids, _ = robot.find_bodies(ee_link)
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one robot body named {ee_link!r}")
    return ids[0], _env_prim_path(env, robot, 0) + "/" + ee_link
