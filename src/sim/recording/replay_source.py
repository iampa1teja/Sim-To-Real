# sim/recording/replay_source.py  (ported from v1 utils/replay_source.py)
"""Read recorded episodes and their per-frame object sidecars for replay.

Pure Python (numpy); LeRobot is imported lazily only when a dataset must be opened.
"""
# Changes from v1: the joint count comes from arm.toml (FOLLOWER_JOINTS) instead of a hard-coded 6, and the
# sidecar layout constants come from sim.recording.episode_metadata.

from dataclasses import dataclass
import json
import os

import numpy as np

from core.arms.joint_calibration import FOLLOWER_JOINTS
from core.datasets.task_meta import load_episode, validate_sidecar
from sim.recording.episode_metadata import SIDECAR_DIR, SIDECAR_FRAME


@dataclass
class SourceEpisode:
    index: int
    start: int
    length: int
    actions: np.ndarray        # (N, J) dataset units, copied unchanged into the new dataset
    states: np.ndarray         # (N, J) dataset units, copied unchanged; also drives the replayed joints
    task: str
    poses_base: np.ndarray     # (N, 7) x y z qw qx qy qz, robot base-link frame
    cube_size: tuple = (0.02, 0.02, 0.02)
    objects: dict[str, np.ndarray] | None = None
    colors: dict[str, str | None] | None = None
    roles: dict[str, str] | None = None
    meta: dict | None = None


def _sidecar_path(root, index):
    return os.path.join(root, SIDECAR_DIR, f"episode_{index:06d}.json")


def _read_sidecar(root, index) -> tuple[dict, str]:
    path = _sidecar_path(root, index)
    if not os.path.exists(path):
        raise ValueError(f"Missing sidecar: {path}")
    with open(path, encoding="utf-8") as file:
        return json.load(file), path


def load_sidecar(root, index) -> np.ndarray:
    """Validated (N, 7) object poses for one episode."""
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


def load_source(root, repo_id, exclude: set[int], dataset=None, *, task="pick_place_cube", arm=None,
                scene=None) -> list[SourceEpisode]:
    """All kept episodes with actions, states, task text and object poses.

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
        try:
            meta = load_episode(root, index, task=task, arm=arm, scene=scene)
        except FileNotFoundError as exc:
            raise ValueError(f"Missing sidecar for episode {index}") from exc
        validate_sidecar(meta, num_frames=length)
        joints = len(meta["joint_names"] or FOLLOWER_JOINTS)
        actions, states = _episode_columns(dataset.hf_dataset, start, end)
        for name, values in (("action", actions), ("state", states)):
            if values.shape != (length, joints):
                raise ValueError(f"Episode {index}: {name} shape {values.shape} != ({length}, {joints})")
            if not np.isfinite(values).all():
                raise ValueError(f"Episode {index}: {name} contains NaN/Inf")
        objects = {
            name: np.concatenate((np.asarray(obj["position"], dtype=np.float64),
                                  np.asarray(obj["orientation_wxyz"], dtype=np.float64)), axis=1)
            for name, obj in meta["objects"].items()
        }
        target = meta["roles"].get("target")
        poses_base = objects[target] if target in objects else np.empty((0, 7), dtype=np.float64)
        cube_size = tuple(meta["objects"][target]["size_m"]) if target in objects else ()
        tasks = row["tasks"]
        task_text = tasks[0] if isinstance(tasks, (list, tuple)) else str(tasks)
        episodes.append(SourceEpisode(
            index=index, start=start, length=length,
            actions=actions.astype(np.float32, copy=False), states=states.astype(np.float32, copy=False),
            task=task_text, poses_base=poses_base, cube_size=cube_size, objects=objects,
            colors={name: obj["color"] for name, obj in meta["objects"].items()},
            roles=dict(meta["roles"]), meta=meta,
        ))
    return episodes
