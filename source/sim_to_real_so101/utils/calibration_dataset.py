"""Affine dataset relabeling without changing images, trajectories or source files."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def mapping_arrays(mapping):
    low = np.array([mapping[n]["joint_min"] for n in FOLLOWER_JOINTS], dtype=np.float64)
    high = np.array([mapping[n]["joint_max"] for n in FOLLOWER_JOINTS], dtype=np.float64)
    if not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Mapping bounds must be finite and increasing")
    return low, high


def units_to_degrees(units, mapping):
    low, high = mapping_arrays(mapping)
    scale = np.array([200.] * 5 + [100.])
    offset = np.array([100.] * 5 + [0.])
    return low + (np.asarray(units, dtype=np.float64) + offset) / scale * (high - low)


def degrees_to_units(degrees, mapping):
    low, high = mapping_arrays(mapping)
    return (np.asarray(degrees, dtype=np.float64) - low) / (high - low) * np.array([200.] * 5 + [100.]) - np.array([100.] * 5 + [0.])


def affine_relabel(old_mapping, new_mapping):
    bias = degrees_to_units(units_to_degrees(np.zeros(6), old_mapping), new_mapping)
    scale = degrees_to_units(units_to_degrees(np.ones(6), old_mapping), new_mapping) - bias
    return scale, bias


def relabel_dataset(source, destination, old_mapping, new_mapping):
    """Copy a LeRobot dataset and relabel joint values and their saved statistics."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError("Destination must be a new directory outside the source")
    before = snapshot(source)
    info = json.loads((source / "meta/info.json").read_text())
    for name in ("action", "observation.state"):
        if info["features"][name]["names"] != [n + ".pos" for n in FOLLOWER_JOINTS]:
            raise ValueError(f"Unexpected {name} joint order")
    scale, bias = affine_relabel(old_mapping, new_mapping)
    shutil.copytree(source, destination)
    maximum_error = 0.
    for path in sorted((destination / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for name in ("action", "observation.state"):
            original = np.array(table[name].to_pylist(), dtype=np.float64)
            values = (original * scale + bias).astype(np.float32)
            maximum_error = max(maximum_error, float(np.max(np.abs(
                units_to_degrees(original, old_mapping) - units_to_degrees(values, new_mapping)))))
            column = pa.array(values.tolist(), type=table.schema.field(name).type)
            table = table.set_column(table.schema.get_field_index(name), name, column)
        pq.write_table(table, path)

    def stats_value(value, stat):
        array = np.asarray(value, dtype=np.float64)
        if stat == "count":
            return value
        if stat == "std":
            return (array * scale).tolist()
        if stat in ("min", "max", "mean") or stat.startswith("q"):
            return (array * scale + bias).tolist()
        raise ValueError(f"Unsupported joint statistic {stat}")
    stats_path = destination / "meta/stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for name in ("action", "observation.state"):
            stats[name] = {stat: stats_value(value, stat) for stat, value in stats[name].items()}
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    for path in sorted((destination / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        for name in table.column_names:
            if any(name.startswith(f"stats/{feature}/") for feature in ("action", "observation.state")):
                stat = name.rsplit("/", 1)[-1]
                column = pa.array([stats_value(v, stat) for v in table[name].to_pylist()],
                                  type=table.schema.field(name).type)
                table = table.set_column(table.schema.get_field_index(name), name, column)
        pq.write_table(table, path)
    after = snapshot(source)
    if before != after:
        raise RuntimeError("Source changed during relabeling")
    copied = snapshot(destination)
    preserved = [name for name in before if not name.startswith(("data/", "meta/episodes/")) and name != "meta/stats.json"]
    if any(copied[name] != before[name] for name in preserved):
        raise RuntimeError("Relabeling changed images, cube trajectories or unrelated metadata")
    manifest = dict(method="A: inverse_new_mapping(old_mapping(units)); no clamping", source=str(source),
                    destination=str(destination), episodes=info["total_episodes"], frames=info["total_frames"],
                    source_hashes=before, source_unchanged=True, old_mapping=old_mapping, new_mapping=new_mapping,
                    scale=scale.tolist(), bias=bias.tolist(), max_physical_angle_error_degrees=maximum_error,
                    preserved_files=len(preserved), success="Original trajectories preserved; no new dynamics trial")
    (destination / "meta/calibration_relabel.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
