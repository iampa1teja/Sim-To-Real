"""Read recorded episodes and their per-frame object sidecars for replay.

Pure Python (numpy); LeRobot is imported lazily only when a dataset must be opened.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from core.arms.spec import load_arm_spec
from core.datasets.task_meta import load_task_meta, load_object_poses


@dataclass
class SourceEpisode:
    index: int
    start: int
    length: int
    actions: np.ndarray        # (N, J) dataset units, copied unchanged into the new dataset
    states: np.ndarray         # (N, J) dataset units, copied unchanged; also drives the replayed joints
    task: str
    objects_base: dict[str, np.ndarray]  # name -> (N, 7), x y z qw qx qy qz
    object_sizes: dict[str, tuple[float, float, float]]
    task_name: str | None
    arm_name: str
    reference_frame: str
    fps: float | None


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
        for key, expected in (("joint_units", [joint.unit for joint in arm.action_joints]),
                              ("joint_roles", [joint.role for joint in arm.action_joints]),
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


def load_sidecar(root, index, object_name=None, **kwargs) -> np.ndarray:
    """Read an explicitly named object, or the sole object in a legacy episode."""
    return load_object_poses(root, index, object_name, **kwargs)


def _episode_columns(hf_dataset, start, end):
    """(actions, states) for one episode without materialising whole columns when possible."""
    if hasattr(hf_dataset, "select"):  # Hugging Face Dataset (real LeRobotDataset)
        part = hf_dataset.select(range(start, end)).with_format("numpy")
        return np.asarray(part["action"]), np.asarray(part["observation.state"])
    return (np.asarray(hf_dataset["action"][start:end]),
            np.asarray(hf_dataset["observation.state"][start:end]))


def load_source(root, repo_id, exclude: set[int], dataset=None, *, arm=None, task=None, legacy_object=None) -> list[SourceEpisode]:
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
        tracked = getattr(getattr(task, "spec", task), "tracked_objects", None)
        metadata = load_task_meta(root, index, task=task, arm=arm, legacy_object=legacy_object,
                                  reference_frame=f"robot base link ({arm.links.base})", num_frames=length,
                                  tracked_objects=tracked)
        objects = metadata["objects"]
        objects_base = {
            name: np.concatenate((np.asarray(obj["position"], dtype=np.float64),
                                  np.asarray(obj["orientation_wxyz"], dtype=np.float64)), axis=1)
            for name, obj in objects.items()
        }
        tasks = row["tasks"]
        instruction = tasks[0] if isinstance(tasks, (list, tuple)) else str(tasks)
        episodes.append(SourceEpisode(
            index=index, start=start, length=length,
            actions=actions.astype(np.float32, copy=False), states=states.astype(np.float32, copy=False),
            task=instruction, objects_base=objects_base,
            object_sizes={name: tuple(obj["size"]) for name, obj in objects.items()},
            task_name=metadata["task"], arm_name=arm.name,
            reference_frame=metadata["reference_frame"], fps=metadata["fps"],
        ))
    return episodes
