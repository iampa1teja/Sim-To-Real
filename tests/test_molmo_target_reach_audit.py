"""Checks for the continuous-bound allowance and benchmark reset geometry."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("target_reach", ROOT / "policies/molmoact2/eval/audit_molmo_target_reach.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def planar_chain(radius=0):
    chain = object.__new__(audit.Chain)
    chain.first = np.eye(4)
    chain.links = [np.eye(4) for _ in range(4)]
    chain.links[0][0, 3] = .1
    chain.links[1][0, 3] = .2
    chain.tail = np.array([radius, 0, 0, 1])
    chain.radius = radius
    return chain


def test_grid_allowance_preserves_an_exact_pose_between_grid_points():
    chain = planar_chain()
    bounds = np.array([[10., 20.], [30., 40.], [-90, 90], [-90, 90], [-90, 90]])
    target = chain.elbow(np.deg2rad(10.37), np.deg2rad(30.29))
    pan, lift = np.meshgrid(np.deg2rad(np.arange(10, 21)), np.deg2rad(np.arange(30, 41)))
    sampled_error = np.linalg.norm(chain.elbow(pan, lift) - target, axis=-1).min()
    assert sampled_error > 1e-4  # Sampling alone would incorrectly exclude this pose.
    lower, allowance = chain.lower_bounds(bounds, np.array([target]), 1)
    assert lower[0] == 0
    assert allowance >= sampled_error


def test_bound_covers_continuous_rotations_and_remains_conservative():
    chain = planar_chain(.08)
    bounds = np.tile([-90., 90.], (5, 1))
    target = np.array([0., 0., 1.])
    lower, _ = chain.lower_bounds(bounds, np.array([target]), 1)
    # This chain stays in z=0. Its first two links cannot get closer than .1 m
    # to the origin even if their relative rotation were unrestricted.
    assert lower[0] > .9
    rng = np.random.default_rng(7)
    poses = chain.fk(rng.uniform(bounds[:, 0], bounds[:, 1], size=(10000, 5)))
    assert np.linalg.norm(poses - target, axis=1).min() >= lower[0]


def test_recorded_training_orientation_is_preserved_when_solving_table_support():
    scene = dict(base_position_world=[0, 0, 0], base_quaternion_wxyz=[1, 0, 0, 0],
                 table_plane_world=[0, 0, 0], cube_size_m=[.02, .02, .02])
    start = dict(x=.1, y=.2, yaw=0, split="train",
                 orientation_wxyz=[np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0])
    np.testing.assert_allclose(audit.stored_target(start, scene), [.1, .2, np.sqrt(2) * .01 + .0002])
    generated = {key: value for key, value in start.items() if key != "orientation_wxyz"}
    generated["split"] = "id"
    np.testing.assert_allclose(audit.stored_target(generated, scene), [.1, .2, .0102])
