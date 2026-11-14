"""Recording/replay preserve explicit arm order and mixed physical joint units."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.arms.spec import load_arm_spec
from utils.arms import mapping as evaluation
from core.eval import eval_set as benchmark, starts
from utils.datasets.sources import load_source, recorded_joint_mapping, replay_joint_mapping


@pytest.fixture
def arm(tmp_path, monkeypatch):
    directory = tmp_path / "arms/four_joint"
    directory.mkdir(parents=True)
    (directory / "robot.usda").write_text('#usda 1.0\n(defaultPrim = "Robot")\ndef Xform "Robot" {}\n')
    (directory / "arm.toml").write_text('''
[arm]
name = "four_joint"
usd_path = "robot.usda"
[arm.links]
base = "root_link"
ee = "tool"
[[arm.joints]]
name = "left_finger"
usd = "SlideL"
type = "prismatic"
role = "gripper"
limits = [0.0, 0.04]
initial = 0.02
[[arm.joints]]
name = "yaw"
usd = "Axis"
type = "revolute"
role = "arm"
limits = [-90.0, 90.0]
initial = 0.0
[[arm.joints]]
name = "extension"
usd = "Slide"
type = "prismatic"
role = "arm"
limits = [0.0, 0.5]
initial = 0.25
[[arm.joints]]
name = "right_finger"
usd = "SlideR"
type = "prismatic"
role = "gripper"
limits = [-0.04, 0.0]
initial = -0.02
[arm.gripper]
kind = "parallel"
joints = ["left_finger", "right_finger"]
contact_bodies = ["left", "right"]
friction_bodies = ["left", "right"]
[arm.lerobot]
follower_type = "test_follower"
leader_type = "test_leader"
calibration = "factory"
[arm.hardware]
motor = "factory"
encoder_counts_per_turn = 4096
zero_count = 2048
''')
    monkeypatch.setenv("S2R_ARMS_DIR", str(directory.parent))
    monkeypatch.delenv("S2R_FOLLOWER_CALIBRATION", raising=False)
    monkeypatch.delenv("S2R_LEADER_CALIBRATION", raising=False)
    return load_arm_spec("four_joint")[0]


def test_recorder_features_and_frames_follow_selected_arm(arm):
    from sim.recording.recorder import LeRobotRecorder

    recorder = LeRobotRecorder("move", "local/test", "/tmp/unused-contract", 30, "cpu", arm=arm)
    for feature in ("action", "observation.state"):
        assert recorder.dataset_features[feature]["shape"] == (4,)
        assert recorder.dataset_features[feature]["names"] == list(arm.feature_names)
    assert recorder.robot_type == "test_follower"
    action = np.array([25, 50, -20, 75], dtype=float)
    frame = recorder.make_sim_frame(action, action, {})
    action[:] = 0
    np.testing.assert_equal(frame["action"], [25, 50, -20, 75])
    with pytest.raises(ValueError, match="4 finite values"):
        recorder.make_sim_frame(np.zeros(6), np.zeros(4), {})


def test_task_start_conversion_handles_interleaved_parallel_grippers_and_slide(arm):
    state = [25, 50, -20, 75]
    actual = evaluation.dataset_state_to_radians(state, arm=arm)
    np.testing.assert_allclose(actual, [0.01, np.pi / 4, 0.2, -0.01], rtol=0, atol=1e-15)
    direct_units = replace(arm, lerobot=replace(arm.lerobot, use_degrees=True))
    codecs = arm.runtime_joint_limits
    for joint in arm.joints:
        if joint.role == "arm":
            codecs[joint.name].update(raw_min=joint.limits[0], raw_max=joint.limits[1])
    np.testing.assert_allclose(evaluation.dataset_state_to_radians(
        [25, 45, 0.2, 75], codecs, arm=direct_units), actual)
    with pytest.raises(ValueError, match="codec"):
        evaluation.dataset_state_to_radians([25, 45, 0.2, 75], arm=direct_units)


def test_replay_uses_recorded_mapping_and_refuses_conflicting_unit(tmp_path, arm, monkeypatch):
    mapping = arm.runtime_joint_limits
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    marker = root / "meta/recording_alignment.json"
    marker.write_text(json.dumps({"arm": arm.name, "joint_names": arm.joint_names, "joint_mapping": mapping}))
    saved, provenance = replay_joint_mapping(root, arm)
    assert saved == mapping
    assert provenance["mode"] == "recorded"
    assert provenance["path"] == str(marker)

    different = {name: dict(span) for name, span in mapping.items()}
    different["yaw"]["joint_max"] = 180
    monkeypatch.setattr("utils.arms.calibration.resolve_calibration",
                        lambda *args, **kwargs: SimpleNamespace(mapping=different))
    with pytest.raises(ValueError, match="differs from the dataset mapping"):
        replay_joint_mapping(root, arm, unit="another_unit")


def test_recorded_mapping_preserves_reversed_device_codec(tmp_path, arm, monkeypatch):
    mapping = arm.runtime_joint_limits
    mapping["left_finger"].update(raw_min=100, raw_max=0)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/recording_alignment.json").write_text(json.dumps({
        "arm": arm.name, "joint_names": arm.joint_names, "joint_mapping": mapping,
    }))
    saved, _ = replay_joint_mapping(tmp_path, arm)
    assert saved["left_finger"]["raw_min"] == 100
    assert saved["left_finger"]["raw_max"] == 0
    np.testing.assert_allclose(evaluation.dataset_state_to_radians([0, 0, 0, 0], saved, arm=arm),
                               [0.04, 0, 0.25, -0.04])
    # Matching physical spans alone cannot establish equivalent device units.
    monkeypatch.setattr("utils.arms.calibration.resolve_calibration",
                        lambda *args, **kwargs: SimpleNamespace(mapping=arm.runtime_joint_limits))
    with pytest.raises(ValueError, match="differs from the dataset mapping"):
        replay_joint_mapping(tmp_path, arm, unit="different_direction")


def test_explicit_default_codec_and_implicit_default_are_equivalent(tmp_path, arm, monkeypatch):
    mapping = arm.runtime_joint_limits
    mapping["left_finger"].update(raw_min=0, raw_max=100)
    mapping["yaw"].update(raw_min=-100, raw_max=100)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/recording_alignment.json").write_text(json.dumps({
        "arm": arm.name, "joint_names": arm.joint_names, "joint_mapping": mapping,
    }))
    monkeypatch.setattr("utils.arms.calibration.resolve_calibration",
                        lambda *args, **kwargs: SimpleNamespace(mapping=arm.runtime_joint_limits))
    saved, provenance = replay_joint_mapping(tmp_path, arm, unit="same_encoding")
    assert provenance["mode"] == "recorded"
    assert saved["left_finger"]["joint_max"] == .04


def test_recorded_mapping_requires_every_joint_span(tmp_path, arm):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/recording_alignment.json").write_text(json.dumps({
        "arm": arm.name, "joint_mapping": {"yaw": arm.runtime_joint_limits["yaw"]},
    }))
    with pytest.raises(ValueError, match="joint"):
        recorded_joint_mapping(tmp_path, arm)


def test_legacy_replay_requires_per_unit_selection_but_factory_needs_no_file(tmp_path, arm):
    per_unit = replace(arm, lerobot=replace(arm.lerobot, calibration="per_unit"))
    with pytest.raises(ValueError, match="requires a calibration unit"):
        replay_joint_mapping(tmp_path, per_unit)
    mapping, provenance = replay_joint_mapping(tmp_path, arm)
    assert provenance["mode"] == "factory"
    assert mapping["extension"] == {"joint_min": 0, "joint_max": 0.5}
    assert mapping["yaw"] == {"joint_min": -180, "joint_max": 179.912109375}


def test_recorded_mapping_rejects_same_dimension_different_joint_order(tmp_path, arm):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/recording_alignment.json").write_text(json.dumps({
        "arm": arm.name, "joint_names": list(reversed(arm.joint_names)), "joint_mapping": arm.runtime_joint_limits,
    }))
    with pytest.raises(ValueError, match="joint order"):
        recorded_joint_mapping(tmp_path, arm)


@pytest.mark.parametrize("key,value", [
    ("joint_units", ["deg"] * 4), ("joint_roles", ["arm"] * 4), ("use_degrees", True),
])
def test_recorded_mapping_rejects_different_joint_interpretation(tmp_path, arm, key, value):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/recording_alignment.json").write_text(json.dumps({
        "arm": arm.name, "joint_names": arm.joint_names, "joint_mapping": arm.runtime_joint_limits, key: value,
    }))
    with pytest.raises(ValueError, match=key):
        recorded_joint_mapping(tmp_path, arm)


def test_source_shapes_and_feature_order_follow_arm(tmp_path, arm):
    directory = tmp_path / "pick_place_meta"
    directory.mkdir()
    (directory / "episode_000000.json").write_text(json.dumps({
        "units": "m", "reference_frame": "robot base link (root_link)", "episode_index": 0, "num_frames": 2,
        "position": [[0.1, 0.2, 0.03]] * 2, "orientation_wxyz": [[1, 0, 0, 0]] * 2,
        "cube_size_m": [0.02, 0.02, 0.02],
    }))
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 2,
                                       "tasks": ["move"]}]),
        features={name: {"names": list(arm.feature_names)} for name in ("action", "observation.state")},
        hf_dataset={"action": np.zeros((2, 4)), "observation.state": np.ones((2, 4))},
    )
    episodes = load_source(tmp_path, "local/test", set(), dataset, arm=arm)
    assert episodes[0].actions.shape == (2, 4)
    dataset.features["action"]["names"] = list(reversed(arm.feature_names))
    with pytest.raises(ValueError, match="order"):
        load_source(tmp_path, "local/test", set(), dataset, arm=arm)


def test_parquet_starts_and_benchmark_validate_selected_joint_count(tmp_path, arm):
    import pyarrow as pa
    import pyarrow.parquet as pq

    chunk = tmp_path / "data/chunk-000"
    chunk.mkdir(parents=True)
    values = [25., 50., -20., 75.]
    pq.write_table(pa.table({"episode_index": [3, 3], "frame_index": [0, 1],
                            "observation.state": [values, [999.] * 4]}), chunk / "part.parquet")
    np.testing.assert_equal(starts.load_robot_starts(tmp_path, [3], arm=arm), [values])
    data = dict(schema_version=1, reference_frame="robot base link (base)", units="m", angle_units="rad",
                parameters={}, scene={}, training_region={},
                starts=[dict(id="train-0", split="train", x=.1, y=.2, yaw=0., arm_start_state=values)])
    data["eval_set_id"] = benchmark.eval_set_digest(data)
    assert benchmark.validate_eval_set(data, arm=arm) is data
    data["starts"][0]["arm_start_state"] = [0.] * 6
    data["eval_set_id"] = benchmark.eval_set_digest(data)
    with pytest.raises(ValueError, match="4 finite values"):
        benchmark.validate_eval_set(data, arm=arm)
