"""Temporal camera frames and joint values for simulation rollout playback."""
from numbers import Real
from typing import Any

import numpy as np
import rerun as rr


def log_rerun_data(
    observation: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> None:
    """Log compressed RGB frames on the current timeline, alongside scalars."""
    for prefix, values in (("observation", observation), ("action", action)):
        for name, value in (values or {}).items():
            if value is None:
                continue
            key = str(name)
            if not key.startswith(prefix + "."):
                key = f"{prefix}.{key}"
            if isinstance(value, Real):
                rr.log(key, rr.Scalars(float(value)))
            elif isinstance(value, np.ndarray):
                arr = value
                if prefix == "observation" and arr.ndim == 3:
                    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                        arr = arr.transpose(1, 2, 0)
                    # Omit static=True: every image belongs to the current
                    # rollout time and must remain available during playback.
                    rr.log(key, rr.Image(arr).compress(jpeg_quality=90))
                else:
                    for index, scalar in enumerate(arr.flat):
                        rr.log(f"{key}_{index}", rr.Scalars(float(scalar)))
