"""Prepare an independent as-real degree dataset and exact model-frame quantiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SIGNS = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
OFFSETS = np.array([0, 90, 90, 0, 0, 0], dtype=np.float32)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TAG = "workshop_pick_place_v1_as_real"


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    result = {key: getattr(values, key)(axis=0).tolist() for key in ("min", "max", "mean", "std")}
    result["count"] = [len(values)]
    for q in (1, 10, 50, 90, 99):
        result[f"q{q:02d}"] = np.quantile(values, q / 100, axis=0, method="linear").tolist()
    return result


def source_mapping(source, arm_name, unit):
    repository = Path(__file__).resolve().parents[3]
    if not (repository / 'src/core').is_dir():
        repository = Path('/workspace/repo')
    sys.path.insert(0, str(repository / 'src'))
    from core.arms.spec import load_arm_spec
    from core.arms.calibration import resolve_calibration
    from core.datasets.sources import recorded_joint_mapping

    arm, _ = load_arm_spec(arm_name)
    if (arm.name != 'so101' or list(arm.joint_names) != JOINTS + ['gripper']
            or arm.lerobot.use_degrees):
        raise ValueError('This Molmo recipe requires the normalized six-joint SO-101 embodiment')
    saved, provenance = recorded_joint_mapping(source, arm)
    selected = resolve_calibration(arm, unit=unit).mapping if unit else None
    if saved is None:
        if selected is None:
            raise ValueError('Source needs a saved joint mapping or an explicit --unit for legacy data')
        saved, provenance = selected, f'unit:{unit}'
    elif selected is not None:
        for name in arm.joint_names:
            if any(not np.isclose(saved[name][key], selected[name][key], atol=1e-8, rtol=0)
                   for key in ('joint_min', 'joint_max')):
                raise ValueError('Explicit unit differs from the source mapping')
    for name, joint in zip(arm.joint_names, arm.action_joints):
        expected = (0, 100) if joint.role == 'gripper' else (-100, 100)
        row = saved[name]
        if (row.get('raw_min', expected[0]), row.get('raw_max', expected[1])) != expected:
            raise ValueError('This recipe requires conventional normalized source codec endpoints')
    return arm, saved, provenance


def prepare(source: Path, output: Path, *, arm_name='so101', unit=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError('Output must be outside the source dataset')
    arm, mapping, mapping_source = source_mapping(source, arm_name, unit)
    info = json.loads((source / 'meta/info.json').read_text())
    for key in ('observation.state', 'action'):
        if info['features'][key].get('names') != list(arm.feature_names):
            raise ValueError(f'{key} must declare SO-101 feature order')
    paths = sorted((source / 'data').rglob('*.parquet'))
    if not paths:
        raise ValueError('Source has no parquet frames')
    bounds = np.array([[mapping[name]['joint_min'], mapping[name]['joint_max']] for name in JOINTS])
    scale = np.r_[(bounds[:, 1] - bounds[:, 0]) / 200, 1].astype(np.float32)
    source_hashes = {}
    output.mkdir(parents=True)
    # Independent payloads remain valid after this container's source mount disappears.
    shutil.copytree(source / "meta", output / "meta")
    shutil.copytree(source / "videos", output / "videos")
    for name in ('task_meta', 'pick_place_meta', 'datagen_meta'):
        if (source / name).is_dir():
            shutil.copytree(source / name, output / name)
    prepared_mapping = {name: dict(row) for name, row in mapping.items()}
    for index, name in enumerate(JOINTS):
        prepared_mapping[name].update(raw_min=float(-100 * scale[index]), raw_max=float(100 * scale[index]))
    alignment_path = output / 'meta/recording_alignment.json'
    alignment = json.loads(alignment_path.read_text()) if alignment_path.exists() else {}
    if alignment_path.exists():
        alignment_path.rename(output / 'meta/source_recording_alignment.json')
    alignment.update(version=4, arm=arm.name, joint_names=list(arm.joint_names),
                     joint_roles=[joint.role for joint in arm.action_joints], joint_units=[joint.unit for joint in arm.action_joints],
                     use_degrees=False, joint_mapping=prepared_mapping,
                     recorded_action='absolute_joint_pose_in_molmo_centered_degrees',
                     preparation={'source': str(source), 'mapping_source': mapping_source})
    alignment_path.write_text(json.dumps(alignment, indent=2, allow_nan=False) + '\n')
    source_generation = output / 'meta/datagen.json'
    if source_generation.exists():
        source_generation.rename(output / 'meta/source_datagen.json')
    columns = {key: [] for key in ("observation.state", "action")}
    episodes = {}
    for path in paths:
        relative = path.relative_to(source)
        source_hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        table = pq.read_table(path)
        episode_ids = np.array(table["episode_index"].to_pylist())
        for key in columns:
            original = np.array(table[key].to_pylist(), dtype=np.float32)
            if original.ndim != 2 or original.shape[1] != 6 or not np.isfinite(original).all():
                raise ValueError(f'{path}: {key} needs finite six-joint values')
            values = original * scale
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
        "videos": "independent byte-identical copies of source payloads",
        "source_joint_mapping": mapping, "prepared_joint_mapping": prepared_mapping,
        "mapping_source": mapping_source,
        "pan_raw_degrees_min_max": [raw_stats["action"]["min"][0], raw_stats["action"]["max"][0]],
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("episodes", "frames", "pan_raw_degrees_min_max")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.getenv('S2R_DATASET'))
    parser.add_argument("--output", type=Path, default=os.getenv('S2R_OUTPUT'))
    parser.add_argument('--arm', default=os.getenv('S2R_ARM', 'so101'))
    parser.add_argument('--unit', default=os.getenv('S2R_UNIT'))
    args = parser.parse_args(argv)
    if args.source is None or args.output is None:
        parser.error('--source/--output or S2R_DATASET/S2R_OUTPUT are required')
    prepare(args.source, args.output, arm_name=args.arm, unit=args.unit)


if __name__ == "__main__":
    main()
