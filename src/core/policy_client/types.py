from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ActionChunk:
    actions: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError(f"actions must be a 2d array got shape {self.actions.shape}")
        if self.actions.shape[0] == 0:
            raise ValueError("actions must have atleast one step")
