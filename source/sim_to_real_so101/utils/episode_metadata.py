"""Atomic cube trajectory sidecars and checks before resuming recording."""

import json
import math
import os
from pathlib import Path
import tempfile


def save_cube_trajectory(root, episode_index, poses, cube_size, fps):
    """Write one pose per dataset frame, using Isaac's wxyz quaternion order."""
    directory = Path(root) / "pick_place_meta"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_index": episode_index,
        "reference_frame": "robot base link (base)",
        "units": "m",
        "fps": fps,
        "num_frames": len(poses),
        "cube_size_m": list(cube_size),
        "position": [[round(v, 6) for v in pose[:3]] for pose in poses],
        "orientation_wxyz": [[round(v, 6) for v in pose[3:]] for pose in poses],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=directory,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / f"episode_{episode_index:06d}.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_cube_trajectories(root, episode_lengths, fps):
    """Refuse resume if sidecars and committed dataset episodes disagree."""
    directory = Path(root) / "pick_place_meta"
    expected = {f"episode_{index:06d}.json" for index in range(len(episode_lengths))}
    actual = {path.name for path in directory.glob("episode_*.json")}
    if actual != expected:
        raise ValueError(f"Cube sidecars at {directory} do not match committed episodes.")
    for index, length in enumerate(episode_lengths):
        path = directory / f"episode_{index:06d}.json"
        payload = json.loads(path.read_text())
        if (payload.get("episode_index") != index or payload.get("num_frames") != length
                or payload.get("fps") != fps or payload.get("units") != "m"
                or payload.get("reference_frame") != "robot base link (base)"):
            raise ValueError(f"Cube sidecar timeline/frame mismatch: {path}")
        for key, width in (("position", 3), ("orientation_wxyz", 4)):
            values = payload.get(key, [])
            if len(values) != length or any(
                len(row) != width or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row)
                for row in values
            ):
                raise ValueError(f"Invalid {key} trajectory: {path}")
