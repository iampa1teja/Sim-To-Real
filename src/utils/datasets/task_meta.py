"""Isaac-free per-frame object trajectories, including legacy single-object datasets."""

import argparse
import json
import math
from pathlib import Path

from utils.io import atomic_write_text

SIDECAR_DIR = "task_meta"
LEGACY_SIDECAR_DIR = "pick_place_meta"
SIDECAR_FRAME = "robot base link (base)"


def _directories(root):
    root = Path(root)
    if root.name in (SIDECAR_DIR, LEGACY_SIDECAR_DIR):
        root = root.parent
    return root / SIDECAR_DIR, root / LEGACY_SIDECAR_DIR


def sidecar_path(root, index):
    """Prefer schema 2 when a dataset contains both layouts."""
    name = f"episode_{index:06d}.json"
    for directory in _directories(root):
        path = directory / name
        if path.is_file():
            return path
    raise ValueError(f"Missing task sidecar for episode {index}: {Path(root)}")


def _identity(value):
    return getattr(value, "name", value)


def _rows(values, width, length, label):
    if not isinstance(values, list) or len(values) != length or any(
        not isinstance(row, list) or len(row) != width or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row
        ) for row in values
    ):
        raise ValueError(f"Invalid {label} trajectory")


def validate_task_meta(data, *, task=None, arm=None, tracked_objects=None, reference_frame=None,
                       episode_index=None, num_frames=None, fps=None):
    """Validate a normalized schema-2 payload and optional dataset expectations."""
    if data.get("schema_version") != 2 or data.get("units") != "m":
        raise ValueError("Task sidecar requires schema_version 2 and metre units")
    if not isinstance(data.get("reference_frame"), str) or not data["reference_frame"]:
        raise ValueError("Task sidecar requires a reference_frame")
    for key, expected in (("task", _identity(task)), ("arm", _identity(arm)),
                          ("reference_frame", reference_frame), ("episode_index", episode_index),
                          ("num_frames", num_frames), ("fps", fps)):
        if expected is not None and data.get(key) != expected:
            raise ValueError(f"Task sidecar {key} mismatch: {data.get(key)!r} != {expected!r}")
    length = data.get("num_frames")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("Task sidecar num_frames must be a positive integer")
    index = data.get("episode_index")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
        raise ValueError("Task sidecar episode_index must be a nonnegative integer")
    rate = data.get("fps")
    if rate is not None and (isinstance(rate, bool) or not isinstance(rate, (int, float))
                             or not math.isfinite(rate) or rate <= 0):
        raise ValueError("Task sidecar fps must be positive")
    objects = data.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("Task sidecar objects must be a nonempty map")
    if tracked_objects is not None and set(objects) != set(tracked_objects):
        raise ValueError(f"Task sidecar objects {sorted(objects)} != tracked objects {sorted(tracked_objects)}")
    for name, obj in objects.items():
        if not isinstance(name, str) or not name or not isinstance(obj, dict):
            raise ValueError("Invalid task sidecar object")
        size = obj.get("size")
        _rows([size], 3, 1, f"{name}.size")
        if any(value <= 0 for value in size):
            raise ValueError(f"Invalid {name}.size: dimensions must be positive")
        _rows(obj.get("position"), 3, length, f"{name}.position")
        quaternions = obj.get("orientation_wxyz")
        _rows(quaternions, 4, length, f"{name}.orientation_wxyz")
        if any(abs(math.sqrt(sum(v * v for v in row)) - 1) > 1e-3 for row in quaternions):
            raise ValueError(f"Non-unit quaternion for {name}")
    return data


def read_task_meta(path, *, task=None, arm=None, legacy_object=None, **expected):
    """Read either schema, returning objects keyed by their task names.

    Schema 1 has no task/arm identity. Its caller may provide those explicitly; otherwise
    they remain unknown. Only this compatibility adapter supplies the historical object name.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Task sidecar must be a JSON object: {path}")
    schema = data.get("schema_version", 1)
    if schema == 1:
        tracked = getattr(getattr(task, "spec", task), "tracked_objects", ())
        if legacy_object is None:
            if len(tracked) > 1:
                raise ValueError("A legacy single-object sidecar needs an explicit legacy_object for this task")
            legacy_object = tracked[0] if tracked else "blue_cube"
        data = {
            "schema_version": 2, "task": _identity(task), "arm": _identity(arm),
            "episode_index": data.get("episode_index"), "num_frames": data.get("num_frames"),
            "reference_frame": data.get("reference_frame"), "units": data.get("units"), "fps": data.get("fps"),
            "objects": {legacy_object: {"size": data.get("cube_size_m"),
                                        "position": data.get("position"),
                                        "orientation_wxyz": data.get("orientation_wxyz")}},
        }
    elif schema != 2:
        raise ValueError(f"Unsupported task sidecar schema {schema!r}: {path}")
    else:
        for key in ("task", "arm"):
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError(f"Schema-2 task sidecar requires {key}: {path}")
        if data.get("fps") is None:
            raise ValueError(f"Schema-2 task sidecar requires fps: {path}")
        if "num_frames" not in data:
            objects = data.get("objects", {})
            first = next(iter(objects.values()), {}) if isinstance(objects, dict) else {}
            data["num_frames"] = len(first.get("position", [])) if isinstance(first, dict) else 0
        if "episode_index" not in data and path.stem.removeprefix("episode_").isdigit():
            data["episode_index"] = int(path.stem.removeprefix("episode_"))
    try:
        return validate_task_meta(data, task=task, arm=arm, **expected)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_task_meta(root, index, **kwargs):
    return read_task_meta(sidecar_path(root, index), episode_index=index, **kwargs)


def load_object_poses(root, index, object_name=None, **kwargs):
    """Return an N×7 WXYZ pose array; omission is valid only for a single object."""
    import numpy as np

    objects = load_task_meta(root, index, **kwargs)["objects"]
    if object_name is None:
        if len(objects) != 1:
            raise ValueError("An object name is required for a multi-object sidecar")
        object_name = next(iter(objects))
    if object_name not in objects:
        raise ValueError(f"Object {object_name!r} is absent from episode {index}")
    obj = objects[object_name]
    return np.concatenate((np.asarray(obj["position"], dtype=np.float64),
                           np.asarray(obj["orientation_wxyz"], dtype=np.float64)), axis=1)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, allow_nan=False), fsync=True, encoding="utf-8", newline="\n")


def save_task_trajectory(root, episode_index, poses, object_sizes, fps, *, task, arm,
                         reference_frame=SIDECAR_FRAME):
    """Atomically write all tracked objects with one pose per committed dataset frame."""
    if set(poses) != set(object_sizes):
        raise ValueError("Object trajectories and sizes must have the same names")
    objects = {}
    for name, values in poses.items():
        values = values.tolist() if hasattr(values, "tolist") else values
        if any(len(row) != 7 for row in values):
            raise ValueError(f"Invalid N×7 trajectory for {name}")
        objects[name] = {"size": [float(v) for v in object_sizes[name]],
                         "position": [[float(v) for v in row[:3]] for row in values],
                         "orientation_wxyz": [[float(v) for v in row[3:]] for row in values]}
    payload = {"schema_version": 2, "task": _identity(task), "arm": _identity(arm),
               "episode_index": episode_index, "num_frames": len(next(iter(poses.values()), [])),
               "reference_frame": reference_frame, "units": "m", "fps": fps, "objects": objects}
    for key in ("task", "arm"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"Writing a task sidecar requires {key}")
    validate_task_meta(payload)
    path = _directories(root)[0] / f"episode_{episode_index:06d}.json"
    _write(path, payload)
    return path


def validate_task_trajectories(root, episode_lengths, fps, **expected):
    """Refuse resume when either sidecar layout disagrees with committed episodes."""
    names = {f"episode_{index:06d}.json" for index in range(len(episode_lengths))}
    actual = {path.name for directory in _directories(root) for path in directory.glob("episode_*.json")}
    if actual != names:
        raise ValueError(f"Task sidecars at {root} do not match committed episodes")
    for index, length in enumerate(episode_lengths):
        load_task_meta(root, index, num_frames=length, fps=fps, **expected)


def replace_episode_meta(root, source_index, target_index):
    """Move an appended replacement into a saved slot, superseding either prior layout."""
    if source_index == target_index:
        raise ValueError("Source and target episode indices must differ")
    source = sidecar_path(root, source_index)
    read_task_meta(source, episode_index=source_index)
    data = json.loads(source.read_text())
    data["episode_index"] = target_index
    target = source.parent / f"episode_{target_index:06d}.json"
    _write(target, data)
    for directory in _directories(root):
        (directory / f"episode_{source_index:06d}.json").unlink(missing_ok=True)
        obsolete = directory / f"episode_{target_index:06d}.json"
        if obsolete != target:
            obsolete.unlink(missing_ok=True)


def migrate_dataset(root, *, task=None, arm=None, legacy_object=None):
    """Write schema 2 in place after validating every input; retain legacy files as backups."""
    root = Path(root)
    marker = root / "meta/recording_alignment.json"
    identity = json.loads(marker.read_text()) if marker.is_file() else {}
    task, arm = _identity(task) or identity.get("task"), _identity(arm) or identity.get("arm")
    if not task or not arm:
        raise ValueError("Migration requires task and arm identities (arguments or recording metadata)")
    paths = {path.name: path for directory in reversed(_directories(root))
             for path in directory.glob("episode_*.json")}
    payloads = []
    for name, path in sorted(paths.items()):
        index = int(path.stem.removeprefix("episode_"))
        data = read_task_meta(path, task=task, arm=arm, legacy_object=legacy_object, episode_index=index)
        if data.get("fps") is None:
            info = root / "meta/info.json"
            data["fps"] = json.loads(info.read_text()).get("fps") if info.is_file() else None
            if data["fps"] is None:
                raise ValueError(f"Migration requires fps in the sidecar or {info}")
        validate_task_meta(data)
        payloads.append((_directories(root)[0] / name, data))
    for path, data in payloads:
        _write(path, data)
    return len(payloads)


def main():
    parser = argparse.ArgumentParser(description="Migrate object sidecars to task_meta schema 2")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--arm")
    parser.add_argument("--legacy-object")
    args = parser.parse_args()
    print(f"Migrated {migrate_dataset(args.dataset, task=args.task, arm=args.arm, legacy_object=args.legacy_object)} episodes")


if __name__ == "__main__":
    main()
