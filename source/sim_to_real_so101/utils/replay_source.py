"""Read recorded pick-place episodes and their per-frame cube sidecars for replay.

Pure Python (numpy); LeRobot is imported lazily only when a dataset must be opened.
"""

from dataclasses import dataclass
import json
import os

import numpy as np

SIDECAR_FRAME = "robot base link (base)"


@dataclass
class SourceEpisode:
    index: int
    start: int
    length: int
    actions: np.ndarray        # (N, 6) dataset units, copied unchanged into the new dataset
    states: np.ndarray         # (N, 6) dataset units, copied unchanged; also drives the replayed joints
    task: str
    poses_base: np.ndarray     # (N, 7) x y z qw qx qy qz, robot base-link frame
    cube_size: tuple = (0.02, 0.02, 0.02)


def _sidecar_path(root, index):
    return os.path.join(root, "pick_place_meta", f"episode_{index:06d}.json")


def _read_sidecar(root, index) -> tuple[dict, str]:
    path = _sidecar_path(root, index)
    if not os.path.exists(path):
        raise ValueError(f"Missing sidecar: {path}")
    with open(path, encoding="utf-8") as file:
        return json.load(file), path


def load_sidecar(root, index) -> np.ndarray:
    """Validated (N, 7) cube poses for one episode."""
    data, path = _read_sidecar(root, index)
    if data.get("units") != "m":
        raise ValueError(f"Invalid units in sidecar {path}")
    if data.get("reference_frame") != SIDECAR_FRAME:
        raise ValueError(f"Invalid reference frame in sidecar {path}")
    if data.get("episode_index") != index:
        raise ValueError(f"Episode index mismatch in sidecar {path}")
    positions = np.asarray(data.get("position"), dtype=np.float64)
    orientations = np.asarray(data.get("orientation_wxyz"), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Invalid position data in sidecar {path}")
    if orientations.ndim != 2 or orientations.shape[1] != 4:
        raise ValueError(f"Invalid orientation data in sidecar {path}")
    if data.get("num_frames") != len(positions) or len(orientations) != len(positions):
        raise ValueError(f"Frame count mismatch in sidecar {path}")
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise ValueError(f"Sidecar {path} contains NaN/Inf")
    if not np.allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=1e-3):
        raise ValueError(f"Non-unit quaternion in sidecar {path}")
    return np.concatenate((positions, orientations), axis=1)


def sidecar_cube_size(root, index) -> tuple:
    data, path = _read_sidecar(root, index)
    size = data.get("cube_size_m")
    if not size or len(size) != 3:
        raise ValueError(f"Missing cube_size_m in sidecar {path}")
    return tuple(float(v) for v in size)


def _episode_columns(hf_dataset, start, end):
    """(actions, states) for one episode without materialising whole columns when possible."""
    if hasattr(hf_dataset, "select"):  # Hugging Face Dataset (real LeRobotDataset)
        part = hf_dataset.select(range(start, end)).with_format("numpy")
        return np.asarray(part["action"]), np.asarray(part["observation.state"])
    return (np.asarray(hf_dataset["action"][start:end]),
            np.asarray(hf_dataset["observation.state"][start:end]))


def load_source(root, repo_id, exclude: set[int], dataset=None) -> list[SourceEpisode]:
    """All kept episodes with actions, states, task text and cube poses.

    `dataset` is injectable for tests; otherwise a stock LeRobotDataset is opened.
    """
    if dataset is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
    rows = list(dataset.meta.episodes)
    episode_indices = {int(row["episode_index"]) for row in rows}
    unknown_excluded = set(exclude) - episode_indices
    if unknown_excluded:
        raise ValueError(f"Unknown excluded episode(s): {sorted(unknown_excluded)}")
    episodes = []
    for row in rows:
        index = int(row["episode_index"])
        if index in exclude:
            continue
        start, end = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        length = end - start
        if length <= 0:
            raise ValueError(f"Episode {index} has invalid length: {length}")
        actions, states = _episode_columns(dataset.hf_dataset, start, end)
        for name, values in (("action", actions), ("state", states)):
            if values.shape != (length, 6):
                raise ValueError(f"Episode {index}: {name} shape {values.shape} != ({length}, 6)")
            if not np.isfinite(values).all():
                raise ValueError(f"Episode {index}: {name} contains NaN/Inf")
        poses_base = load_sidecar(root, index)
        if len(poses_base) != length:
            raise ValueError(f"Episode {index}: sidecar {_sidecar_path(root, index)} length "
                             f"{len(poses_base)} != episode length {length}")
        tasks = row["tasks"]
        task = tasks[0] if isinstance(tasks, (list, tuple)) else str(tasks)
        episodes.append(SourceEpisode(
            index=index, start=start, length=length,
            actions=actions.astype(np.float32, copy=False), states=states.astype(np.float32, copy=False),
            task=task, poses_base=poses_base, cube_size=sidecar_cube_size(root, index),
        ))
    return episodes
