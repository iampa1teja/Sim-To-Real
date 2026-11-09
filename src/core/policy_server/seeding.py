"""Seed policy RNGs without importing torch until a seed is requested."""

import random

import numpy as np


def seed_everything(seed: int, *, deterministic_cudnn: bool = False) -> None:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a uint32 integer")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
