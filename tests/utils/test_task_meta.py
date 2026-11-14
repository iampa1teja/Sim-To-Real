"""Dataset sidecars retain every object's synchronized poses across both on-disk schemas."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.datasets.sources import load_source
from utils.datasets.task_meta import (
    load_object_poses, load_task_meta, migrate_dataset, read_task_meta,
    save_task_trajectory, sidecar_path, validate_task_trajectories,
)


POSES = [[0.1, 0.2, 0.03, 1., 0., 0., 0.], [0.2, 0.3, 0.04, 1., 0., 0., 0.]]
SIZES = {"lower": (0.02, 0.02, 0.02), "upper": (0.03, 0.03, 0.03)}


def legacy(root, index=0):
    path = root / "pick_place_meta" / f"episode_{index:06d}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "episode_index": index, "num_frames": 2, "fps": 30, "units": "m",
        "reference_frame": "robot base link (base)", "cube_size_m": [0.02] * 3,
        "position": [row[:3] for row in POSES], "orientation_wxyz": [row[3:] for row in POSES],
    }))
    return path


def test_multi_object_roundtrip_and_contract_frame_checks(tmp_path):
    poses = {"lower": np.asarray(POSES), "upper": np.asarray(POSES) + [0, 0, .1, 0, 0, 0, 0]}
    path = save_task_trajectory(tmp_path, 0, poses, SIZES, 30, task="stack", arm="synthetic")
    assert path == tmp_path / "task_meta/episode_000000.json"
    assert set(read_task_meta(path)["objects"]) == set(poses)
    for name in poses:
        np.testing.assert_array_equal(load_object_poses(tmp_path, 0, name), poses[name])
    with pytest.raises(ValueError, match="object name"):
        load_object_poses(tmp_path, 0)
    validate_task_trajectories(tmp_path, [2], 30, task="stack", arm="synthetic", tracked_objects=poses)
    for expected in ({"task": "other"}, {"arm": "other"}, {"tracked_objects": ["lower"]},
                     {"reference_frame": "world"}, {"num_frames": 3}, {"fps": 60}):
        with pytest.raises(ValueError, match="mismatch|tracked objects"):
            load_task_meta(tmp_path, 0, **expected)
    with pytest.raises(ValueError, match="committed episodes"):
        validate_task_trajectories(tmp_path, [2, 2], 30)


@pytest.mark.parametrize("bad_pose", [
    [float("nan"), 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0],
])
def test_invalid_pose_never_publishes_sidecar(tmp_path, bad_pose):
    with pytest.raises(ValueError, match="Invalid|Non-unit"):
        save_task_trajectory(tmp_path, 0, {"lower": [bad_pose]}, {"lower": SIZES["lower"]},
                             30, task="stack", arm="synthetic")
    assert not (tmp_path / "task_meta").exists()


def test_legacy_reader_and_idempotent_migration_preserve_values(tmp_path):
    old = legacy(tmp_path)
    before = old.read_bytes()
    task = SimpleNamespace(name="pick", spec=SimpleNamespace(tracked_objects=("block",)))
    data = load_task_meta(tmp_path, 0, task=task, arm="synthetic")
    assert data["schema_version"] == 2
    assert set(data["objects"]) == {"block"}
    assert data["task"] == "pick"
    np.testing.assert_array_equal(load_object_poses(tmp_path, 0, legacy_object="block"), POSES)
    assert migrate_dataset(tmp_path, task="pick", arm="synthetic", legacy_object="block") == 1
    assert old.read_bytes() == before
    assert sidecar_path(tmp_path / "pick_place_meta", 0) == tmp_path / "task_meta/episode_000000.json"
    assert migrate_dataset(tmp_path, task="pick", arm="synthetic", legacy_object="block") == 1
    np.testing.assert_array_equal(load_object_poses(tmp_path, 0, "block"), POSES)
    with pytest.raises(ValueError, match="legacy_object"):
        read_task_meta(old, task=SimpleNamespace(name="stack", spec=SimpleNamespace(tracked_objects=("a", "b"))))


def test_migration_validates_all_episodes_before_publishing(tmp_path):
    legacy(tmp_path)
    bad = legacy(tmp_path, 1)
    data = json.loads(bad.read_text())
    data["orientation_wxyz"][0] = [0, 0, 0, 0]
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Non-unit"):
        migrate_dataset(tmp_path, task="pick", arm="synthetic")
    assert not (tmp_path / "task_meta").exists()


def test_source_reads_multiple_objects_and_legacy_with_explicit_context(tmp_path):
    from utils.arms.arms import load_arm_spec

    arm = load_arm_spec("so101")[0]
    task = SimpleNamespace(name="stack", spec=SimpleNamespace(tracked_objects=tuple(SIZES)))
    poses = {name: POSES for name in SIZES}
    save_task_trajectory(tmp_path, 0, poses, SIZES, 30, task=task, arm=arm)
    dataset = SimpleNamespace(
        meta=SimpleNamespace(episodes=[{"episode_index": 0, "dataset_from_index": 0, "dataset_to_index": 2,
                                       "tasks": ["Stack the blocks"]}]),
        features={name: {"names": arm.feature_names} for name in ("action", "observation.state")},
        hf_dataset={"action": np.zeros((2, len(arm.joints))), "observation.state": np.ones((2, len(arm.joints)))},
    )
    source, = load_source(tmp_path, "local/test", set(), dataset, arm=arm, task=task)
    assert source.task == "Stack the blocks"
    assert source.task_name == "stack"
    assert source.object_sizes == SIZES
    for name in SIZES:
        np.testing.assert_array_equal(source.objects_base[name], POSES)
    (tmp_path / "task_meta/episode_000000.json").unlink()
    legacy(tmp_path)
    task = SimpleNamespace(name="pick", spec=SimpleNamespace(tracked_objects=("block",)))
    source, = load_source(tmp_path, "local/test", set(), dataset, arm=arm, task=task)
    assert set(source.objects_base) == {"block"}
    assert source.task_name == "pick"
    np.testing.assert_array_equal(source.objects_base["block"], POSES)


def test_teleop_save_keeps_all_objects_on_the_dataset_timeline(tmp_path):
    import torch
    from sim.teleop.loop import save_episode

    class Recorders:
        frame_count = 2

        def save_episode(self, sidecar):
            sidecar(tmp_path, 7)
            return 7

    arm = SimpleNamespace(name="synthetic", links=SimpleNamespace(base="root_link"))
    task = SimpleNamespace(name="stack")
    trajectory = [
        {"lower": torch.tensor(row), "upper": torch.tensor(row) + torch.tensor([0, 0, .1, 0, 0, 0, 0])}
        for row in POSES
    ]
    assert save_episode(Recorders(), trajectory, SIZES, 30, task=task, arm=arm) == 7
    assert trajectory == []
    data = load_task_meta(tmp_path, 7, task="stack", arm="synthetic",
                          reference_frame="robot base link (root_link)", num_frames=2)
    assert set(data["objects"]) == set(SIZES)
    np.testing.assert_allclose(load_object_poses(tmp_path, 7, "lower"), POSES)
    with pytest.raises(ValueError, match="datasets have 2"):
        save_episode(Recorders(), [], SIZES, 30, task=task, arm=arm)


def test_rerecord_replacement_supersedes_old_schema_in_same_slot(tmp_path):
    from utils.datasets.task_meta import replace_episode_meta

    old = legacy(tmp_path)
    legacy(tmp_path, 1)
    poses = {name: POSES for name in SIZES}
    save_task_trajectory(tmp_path, 1, poses, SIZES, 30, task="stack", arm="synthetic")
    replace_episode_meta(tmp_path, 1, 0)
    assert not old.exists()
    assert not (tmp_path / "pick_place_meta/episode_000001.json").exists()
    assert not (tmp_path / "task_meta/episode_000001.json").exists()
    data = load_task_meta(tmp_path, 0, task="stack", arm="synthetic", num_frames=2)
    assert set(data["objects"]) == set(SIZES)
    validate_task_trajectories(tmp_path, [2], 30)
