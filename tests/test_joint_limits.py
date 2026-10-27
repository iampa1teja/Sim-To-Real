"""Limits must cover recorded physical extrema and reach the actual PhysX API."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.joint_limits import (
    FOLLOWER_JOINTS, USD_JOINTS, apply_runtime_joint_limits, runtime_joint_limits,
)


def test_real_extrema_are_covered_without_full_turn_wrist_limits():
    observed = np.array([[-111.88477, 96.15235], [-106.69923, 99.93165], [-74.44337, 92.37306],
                         [-19.16016, 103.79884], [-117.07032, 39.55079], [-8.15431, 104.43360]])
    limits = runtime_joint_limits()
    actual = np.array([[limits[n]["joint_min"], limits[n]["joint_max"]] for n in FOLLOWER_JOINTS])
    assert np.all(actual[:, 0] < observed[:, 0])
    assert np.all(actual[:, 1] > observed[:, 1])
    assert actual[4, 0] > -180 and actual[4, 1] < 180
    source = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())
    assert actual[5, 0] < source["sim_joint_mapping_source"]["gripper_closed_degrees"]


def test_startup_uses_position_limit_api_with_joint_order_and_radians():
    calls = []
    robot = SimpleNamespace(joint_names=list(reversed(USD_JOINTS)),
                            write_joint_position_limit_to_sim=lambda values, **kw: calls.append((values, kw)))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=3, device="cpu")
    selected = torch.tensor([0, 2])
    apply_runtime_joint_limits(env, selected)
    values, kw = calls[0]
    assert values.shape == (2, 6, 2)
    assert kw["joint_ids"] == [5, 4, 3, 2, 1, 0]
    assert kw["env_ids"] is selected
    source = runtime_joint_limits()
    expected = np.array([[source[n]["joint_min"], source[n]["joint_max"]] for n in FOLLOWER_JOINTS])
    np.testing.assert_allclose(torch.rad2deg(values[0]).numpy(), expected, atol=2e-5)


@pytest.mark.parametrize("mutation", ["missing", "nonfinite", "reversed"])
def test_invalid_json_limits_cannot_reach_physx(mutation):
    limits = runtime_joint_limits()
    if mutation == "missing":
        del limits["gripper"]
    elif mutation == "nonfinite":
        limits["gripper"]["joint_min"] = float("nan")
    else:
        limits["gripper"] = {"joint_min": 10, "joint_max": -10}
    with pytest.raises(ValueError):
        runtime_joint_limits({"runtime_joint_limits": limits})
