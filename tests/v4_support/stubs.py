"""In-memory test doubles for the WP0 protocols (contracts §3.3). numpy + standard library only."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


class ListWriter:
    """In-memory ``EpisodeWriter``: frames go to an open buffer; ``save_episode`` commits it to ``episodes``.

    ``episodes[i]`` is a list of dicts ``{"action_raw", "state_raw", "images", "task", "labels"}``
    (arrays copied). ``save_episode`` returns the index it committed (== ``episode_count()`` before the
    call) and raises ``RuntimeError`` on an empty buffer; ``discard_episode`` empties the buffer (no-op
    when empty); ``finalize`` sets ``finalized``. ``calls`` records method names in order, for
    call-order assertions. It does not persist anything (WP7's ``MemoryWriter`` does).
    """

    def __init__(self, root: str | Path = Path("memory")):
        self.root = Path(root)
        self.episodes: list[list[dict]] = []
        self.buffer: list[dict] = []
        self.finalized = False
        self.calls: list[str] = []

    def episode_count(self) -> int:
        self.calls.append("episode_count")
        return len(self.episodes)

    def episode_lengths(self) -> list[int]:
        self.calls.append("episode_lengths")
        return [len(e) for e in self.episodes]

    def add_frame(self, action_raw: np.ndarray, state_raw: np.ndarray, images: Mapping[str, np.ndarray],
                  task: str, labels) -> None:
        self.calls.append("add_frame")
        self.buffer.append({"action_raw": np.array(action_raw, copy=True), "state_raw": np.array(state_raw, copy=True),
                            "images": {k: np.array(v, copy=True) for k, v in images.items()}, "task": task,
                            "labels": labels})

    def discard_episode(self) -> None:
        self.calls.append("discard_episode")
        self.buffer = []

    def save_episode(self) -> int:
        self.calls.append("save_episode")
        if not self.buffer:
            raise RuntimeError("save_episode with an empty buffer")
        self.episodes.append(self.buffer)
        self.buffer = []
        return len(self.episodes) - 1

    def finalize(self) -> None:
        self.calls.append("finalize")
        self.finalized = True
