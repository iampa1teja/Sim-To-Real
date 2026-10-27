"""Follower fixture, physical zero anchors, and lossless dataset relabeling."""
import ast
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS, follower_mapping_from_calibration
from sim_to_real_so101.utils.calibration_dataset import (
    degrees_to_units, units_to_degrees, relabel_dataset, snapshot,
)

FIXTURE = ROOT / "tests/fixtures/rpd_robot.json"
SETUP = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())
MAPPING = SETUP["sim_joint_mapping"]


def test_persisted_mapping_equals_shared_derivation_from_follower_json_fixture():
    source = SETUP["sim_joint_mapping_source"]
    derived = follower_mapping_from_calibration(FIXTURE, gripper_closed_degrees=source["gripper_closed_degrees"])
    assert MAPPING == derived
    assert source["follower_id"] == "rpd_robot"
    fixture = json.loads(FIXTURE.read_text())
    for name in FOLLOWER_JOINTS:
        assert source["joints"][name]["range_min"] == fixture[name]["range_min"]
        assert source["joints"][name]["range_max"] == fixture[name]["range_max"]
        assert source["joints"][name]["method"]


def test_gripper_zero_is_usd_closing_stop_and_span_is_measured_encoder_travel():
    assert MAPPING["gripper"]["joint_min"] == pytest.approx(-10.000003814697266)
    assert MAPPING["gripper"]["joint_max"] - MAPPING["gripper"]["joint_min"] == (3514 - 2039) * 360 / 4096
    assert units_to_degrees([0] * 6, MAPPING)[-1] == MAPPING["gripper"]["joint_min"]
    # The calibrated span remains visible beyond USD limits; only the simulator clamps it.
    assert units_to_degrees([100] * 6, MAPPING)[-1] > 100


def test_wrist_encoder_2048_still_has_zero_degrees_without_ninety_degree_shift():
    units = 2048 / 4095 * 200 - 100
    assert units_to_degrees([0, 0, 0, 0, units, 0], MAPPING)[4] == pytest.approx(0, abs=1e-12)


def test_initial_pose_is_unchanged_and_representable_by_the_new_mapping():
    tree = ast.parse((ROOT / "source/sim_to_real_so101/assets/so101.py").read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "SO101_CFG" for t in n.targets))
    init = next(k.value for k in assignment.value.keywords if k.arg == "init_state")
    joints = next(k.value for k in init.keywords if k.arg == "joint_pos")
    pose = eval(compile(ast.Expression(joints), "init_state", "eval"), {"np": np})
    radians = np.array([pose[n] for n in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")])
    np.testing.assert_allclose(radians, [-.2736, -.6109, -.0745, 1.5148, -.0326036732051034, -.1465], atol=1e-12)
    raw = degrees_to_units(np.rad2deg(radians), MAPPING)
    assert np.all(raw[:5] >= -100) and np.all(raw[:5] <= 100) and 0 <= raw[-1] <= 100
    np.testing.assert_allclose(units_to_degrees(raw, MAPPING), np.rad2deg(radians), atol=1e-12)


def test_relabel_preserves_images_states_in_degrees_and_source_hashes(tmp_path):
    source, destination = tmp_path / "old", tmp_path / "new"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    (source / "videos").mkdir()
    (source / "videos/unchanged.mp4").write_bytes(b"frozen video bytes")
    values = np.array([[-100, -50, 0, 50, 100, 0], [0, 0, 0, 0, 0, 37], [100, 50, 90, -50, -100, 100]], np.float32)
    columns = {"action": values.tolist(), "observation.state": values.tolist()}
    path = "data/chunk-000/file-000.parquet"
    pq.write_table(pa.table(columns), source / path)
    names = [n + ".pos" for n in FOLLOWER_JOINTS]
    info = dict(total_episodes=1, total_frames=3, features={n: {"names": names} for n in columns})
    (source / "meta/info.json").write_text(json.dumps(info))
    stats = {n: dict(mean=values.mean(0).tolist(), min=values.min(0).tolist(), max=values.max(0).tolist(),
                     std=values.std(0).tolist(), count=[3]) for n in columns}
    (source / "meta/stats.json").write_text(json.dumps(stats))
    pq.write_table(pa.table({f"stats/{n}/{stat}": [value] for n, data in stats.items() for stat, value in data.items()}),
                   source / "meta/episodes/chunk-000/file-000.parquet")
    old = {n: {"joint_min": -100 if n != "gripper" else -10, "joint_max": 100} for n in FOLLOWER_JOINTS}
    before = snapshot(source)
    report = relabel_dataset(source, destination, old, MAPPING)
    assert before == snapshot(source)
    assert report["max_physical_angle_error_degrees"] < 1e-5
    assert (destination / "videos/unchanged.mp4").read_bytes() == b"frozen video bytes"
    relabeled = np.asarray(pq.read_table(destination / path)["observation.state"].to_pylist())
    np.testing.assert_allclose(units_to_degrees(relabeled, MAPPING), units_to_degrees(values, old), atol=1e-5)
    stats = json.loads((destination / "meta/stats.json").read_text())["observation.state"]
    np.testing.assert_allclose(stats["mean"], relabeled.mean(0), atol=1e-5)
    np.testing.assert_allclose(stats["std"], relabeled.std(0), atol=1e-5)
    with pytest.raises(ValueError, match="Destination"):
        relabel_dataset(source, source / "nested", old, MAPPING)
