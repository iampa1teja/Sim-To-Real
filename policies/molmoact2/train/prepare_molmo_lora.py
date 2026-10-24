"""Prepare an independent as-real degree dataset and exact model-frame quantiles."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
DEFAULT_BOUNDS = [(-110, 110), (-100, 100), (-100, 90), (-95, 95), (-160, 160)]
TAG = "workshop_pick_place_v1_as_real"


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    result = {key: getattr(values, key)(axis=0).tolist() for key in ("min", "max", "mean", "std")}
    result["count"] = [len(values)]
    for q in (1, 10, 50, 90, 99):
        result[f"q{q:02d}"] = np.quantile(values, q / 100, axis=0, method="linear").tolist()
    return result


def prepare(source: Path, output: Path, setup: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    mapping = json.loads(setup.read_text()).get("sim_joint_mapping", {})
    bounds = np.array([[mapping.get(name, {}).get("joint_min", lo),
                        mapping.get(name, {}).get("joint_max", hi)]
                       for name, (lo, hi) in zip(JOINTS, DEFAULT_BOUNDS)])
    scale = np.r_[(bounds[:, 1] - bounds[:, 0]) / 200, 1].astype(np.float32)
    source_hashes = {}
    output.mkdir(parents=True)
    # Video payloads are immutable references; all numeric data and metadata are independent.
    shutil.copytree(source / "meta", output / "meta")
    (output / "videos").symlink_to((source / "videos").resolve(), target_is_directory=True)
    columns = {key: [] for key in ("observation.state", "action")}
    episodes = {}
    for path in sorted((source / "data").rglob("*.parquet")):
        relative = path.relative_to(source)
        source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        table = pq.read_table(path)
        episode_ids = np.array(table["episode_index"].to_pylist())
        for key in columns:
            values = np.array(table[key].to_pylist(), dtype=np.float32) * scale
            columns[key].append(values)
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field,
                                     pa.array(values.tolist(), type=field.type))
            for ep in np.unique(episode_ids):
                episodes.setdefault(int(ep), {}).setdefault(key, []).append(values[episode_ids == ep])
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)
    info = json.loads((output / "meta/info.json").read_text())
    arrays = {key: np.concatenate(parts) for key, parts in columns.items()}
    assert len(arrays["action"]) == info["total_frames"]
    assert len(episodes) == info["total_episodes"]
    raw_stats = json.loads((output / "meta/stats.json").read_text())
    for key, values in arrays.items():
        raw_stats[key] = stats(values)
    (output / "meta/stats.json").write_text(json.dumps(raw_stats, indent=2) + "\n")
    for path in sorted((output / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            for key, parts in episodes[int(row["episode_index"])].items():
                for statistic, value in stats(np.concatenate(parts)).items():
                    row[f"stats/{key}/{statistic}"] = value
        pq.write_table(pa.Table.from_pylist(rows), path)
    model_stats = {key: stats(values * SIGNS + OFFSETS) for key, values in arrays.items()}
    for entry in model_stats.values():
        entry["mask"] = [True] * 6
    metadata = {
        "action_key": "action", "state_key": "observation.state", "normalize_gripper": True,
        "action_horizon": 30, "n_action_steps": 30,
        "setup_type": "single so100/so101 robotic arm in molmoact2",
        "control_mode": "absolute joint pose",
        "camera_keys": ["observation.images.realsense_rgb", "observation.images.wrist_cam"],
        "action_stats": model_stats["action"], "state_stats": model_stats["observation.state"],
    }
    (output / "meta/molmo_norm_stats.json").write_text(
        json.dumps({"format": "molmoact2_norm_stats.v1", "norm_mode": "q01_q99",
                    "metadata_by_tag": {TAG: metadata}}, indent=2) + "\n")
    report = {
        "source": str(source.resolve()), "output": str(output.resolve()),
        "episodes": len(episodes), "frames": len(arrays["action"]),
        "input_joint_units": "lerobot_normalized", "arm_degrees_per_unit": scale.tolist(),
        "joint_signs": SIGNS.tolist(), "joint_offsets": OFFSETS.tolist(),
        "quantile_method": "exact numpy linear over every frame; no per-episode envelope",
        "action_representation": "absolute joint pose", "chunk_size": 30, "n_action_steps": 30,
        "source_data_sha256": source_hashes, "model_stats": model_stats,
        "videos": "symlink to unchanged source payloads",
        "pan_raw_degrees_min_max": [raw_stats["action"]["min"][0], raw_stats["action"]["max"][0]],
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episodes", "frames", "pan_raw_degrees_min_max")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/pick_place_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup", type=Path, default=Path("source/sim_to_real_so101/assets/real_setup.json"))
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.setup)
