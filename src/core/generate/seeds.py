"""Deterministic, independent generation random streams."""

import hashlib as _hashlib

import numpy as _np


def derive_seed(*parts) -> int:
    """Hash colon-separated parts into an unsigned 64-bit Python integer."""
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(_hashlib.sha256(payload).digest()[:8], "big")


def rng_for(seed: int, purpose: str) -> _np.random.Generator:
    """Create a fresh generator for one purpose within an attempt."""
    return _np.random.default_rng(derive_seed(seed, purpose))
