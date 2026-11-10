# sim/recording/replay_source.py  (ported from v1 utils/replay_source.py)
"""Read recorded episodes and their per-frame object sidecars for replay.

Pure Python (numpy); LeRobot is imported lazily only when a dataset must be opened.
"""
# Changes from v1: joint count and action order come from the selected ArmSpec, and the
# sidecar layout constants come from sim.recording.episode_metadata.

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np

from core.arms.spec import load_arm_spec
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


def _mapping_bounds(mapping, arm):
    """Keep complete physical spans and their device codecs when comparing datasets."""
    from core.arms.calibration import validate_mapping
    from core.arms.mapping import mapping_for_arm

    return mapping_for_arm(arm, validate_mapping(arm, mapping))


def recorded_joint_mapping(root, arm):
    """Recover dataset interpretation when its writer saved the calibrated spans."""
    if root is None:
        return None, None
    for relative in ("meta/recording_alignment.json", "meta/dr_replay_manifest.json", "meta/calibration_replay.json"):
        path = Path(root) / relative
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if data.get("arm") is not None and data["arm"] != arm.name:
            raise ValueError(f"Dataset {path} belongs to arm {data['arm']!r}, selected {arm.name!r}")
        if data.get("joint_names") is not None and tuple(data["joint_names"]) != arm.joint_names:
            raise ValueError(f"Dataset {path} joint order differs from arm {arm.name!r}")
        for key, expected in (("joint_units", [joint.unit for joint in arm.joints]),
                              ("joint_roles", [joint.role for joint in arm.joints]),
                              ("use_degrees", arm.lerobot.use_degrees)):
            if key in data and data[key] != expected:
                raise ValueError(f"Dataset {path} {key} differs from arm {arm.name!r}")
        mapping = data.get("joint_mapping")
        if mapping is not None:
            if not isinstance(mapping, dict):
                raise ValueError(f"Dataset {path} joint_mapping must be a table of calibrated spans")
            return _mapping_bounds(mapping, arm), str(path)
    return None, None


def replay_joint_mapping(root, arm, unit=None):
    """Resolve a same-trajectory replay's mapping; never reinterpret saved units."""
    from core.arms.calibration import resolve_calibration

    saved, path = recorded_joint_mapping(root, arm)
    selected = resolve_calibration(arm, unit=unit) if unit is not None or saved is None else None
    if saved is not None:
        if selected is not None and _mapping_bounds(selected.mapping, arm) != saved:
            raise ValueError(f"Selected unit {unit!r} differs from the dataset mapping in {path}; "
                             "DR replay copies raw actions unchanged and cannot silently remap them")
        return saved, {"mode": "recorded", "path": path, "unit": unit}
    return _mapping_bounds(selected.mapping, arm), {
        "mode": arm.lerobot.calibration,
        "unit": selected.unit.unit_id if selected.unit is not None else None,
        "source": "explicit unit or factory calibration; dataset has no saved joint mapping",
    }


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


def load_source(root, repo_id, exclude: set[int], dataset=None, *, arm=None) -> list[SourceEpisode]:
    """All kept episodes with actions, states, task text and object poses.

    `dataset` is injectable for tests; otherwise a stock LeRobotDataset is opened.
    """
    if dataset is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
    arm = arm or load_arm_spec()[0]
    joints = len(arm.joint_names)
    for name in ("action", "observation.state"):
        names = getattr(dataset, "features", {}).get(name, {}).get("names")
        if names is not None and tuple(names) != arm.feature_names:
            raise ValueError(f"Dataset {name} order {names!r} differs from arm {arm.name!r}: {arm.feature_names}")
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
            if values.shape != (length, joints):
                raise ValueError(f"Episode {index}: {name} shape {values.shape} != ({length}, {joints})")
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
