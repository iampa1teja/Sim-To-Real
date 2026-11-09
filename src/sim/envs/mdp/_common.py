"""Shared adapters for Isaac Lab tensor and ProxyArray data."""

import torch


def as_torch(x):
    """Return a torch view of 3.0 data (a ProxyArray with a zero-copy ``.torch``); plain tensors pass through."""
    return x if isinstance(x, torch.Tensor) else x.torch
