"""CPU regressions for the opt-in simulated policy-pan adaptation.

The affine coefficients map physical USD pan into the checkpoint's training
pan coordinates. This is an experimental policy adaptation, separate from
the physical calibration tested in test_molmo_joint_frames.py. Its endpoint
anchors come from released SO101 action q01/q99 and the corresponding training
demonstration physical pan span, rather than the current cube/box positions.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]

SCALE = 0.4901697298
OFFSET = 3.5352321757
USD_SCALE = np.asarray([1.18037109375, 1, .95, .95, 1.6], dtype=np.float64)
USD_MIDPOINT = np.asarray([1.7578125, 0, -5, 0, 0], dtype=np.float64)
JOINT_ORDER = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def client(**overrides):
    from sim_to_real_so101.policy_client.contract import load_client_policy
    from sim_to_real_so101.policy_client.discovery import load_manifest
    options = dict(input_joint_units="usd_normalized",
                   arm_degrees_per_unit=",".join(map(str, USD_SCALE)),
                   arm_zero_degrees=",".join(map(str, USD_MIDPOINT)))
    options.update(overrides)
    return load_client_policy("molmoact2", load_manifest(ROOT / "policies/molmoact2"), options)


def adapted_client(**overrides):
    return client(**dict(pan_model_scale=SCALE, pan_model_offset=OFFSET, **overrides))


def images():
    return {"room": np.zeros((2, 3, 3), np.uint8),
            "wrist": np.ones((2, 3, 3), np.uint8)}


def normalized_pose(physical):
    physical = np.asarray(physical, dtype=np.float64)
    return np.r_[(physical[:5] - USD_MIDPOINT) / USD_SCALE, physical[5]]


@pytest.mark.parametrize("physical_pan,model_pan", [
    (0, 3.5352321757),
    (10, 8.4369294737),
    (-93.16213146, -42.1300246339),
    (91.84219407, 48.55349563198916),
])
def test_model_pan_matches_independent_training_endpoint_and_physical_anchors(physical_pan, model_pan):
    pose = normalized_pose([physical_pan, -20, -30, 15, -20, 37])
    built = adapted_client().build_request(pose, images(), "task")
    assert built["state"][0] == pytest.approx(model_pan, abs=1e-5)
    np.testing.assert_allclose(built["state"][1:], [110, 60, 15, -20, 37], rtol=0, atol=1e-5)


@pytest.mark.parametrize("model_pan,physical_pan", [
    (-42.1300246339, -93.16213146),
    (48.55349563198916, 91.84219407),
    (3.5352321757, 0),
    (8.4369294737, 10),
])
def test_inverse_pan_reconstructs_training_endpoints_and_physical_targets(model_pan, physical_pan):
    decoded = adapted_client().parse_response(
        {"actions": [[model_pan, 110, 60, 15, -20, 37]]}, JOINT_ORDER)[0]
    physical = decoded[:5] * USD_SCALE + USD_MIDPOINT
    np.testing.assert_allclose(physical, [physical_pan, -20, -30, 15, -20], rtol=0, atol=2e-5)
    assert decoded[5] == 37


def test_adaptation_preserves_every_other_joint_and_camera_exactly():
    base, adapted = client(), adapted_client()
    state = np.asarray([23, -49, 57, 18, -12, 33], dtype=np.float32)
    camera_frames = images()
    unadapted_request = base.build_request(state, camera_frames, "Pick the cube")
    adapted_request = adapted.build_request(state, camera_frames, "Pick the cube")
    np.testing.assert_array_equal(adapted_request["state"][1:], unadapted_request["state"][1:])
    assert adapted_request["instruction"] == unadapted_request["instruction"]
    for camera in camera_frames:
        np.testing.assert_array_equal(adapted_request["images"][camera], unadapted_request["images"][camera])
    response = {"actions": [[17, 123, 76, 51, -39, 27], [-12, 51, 118, 27, 12, 41]]}
    np.testing.assert_array_equal(adapted.parse_response(response, JOINT_ORDER)[:, 1:],
                                  base.parse_response(response, JOINT_ORDER)[:, 1:])


def test_default_joint_frame_and_metadata_are_unchanged():
    plain = client()
    built = plain.build_request(normalized_pose([10, -20, -30, 15, -20, 37]), images(), "task")
    np.testing.assert_allclose(built["state"], [10, 110, 60, 15, -20, 37], rtol=0, atol=1e-5)
    assert "pan_adaptation" not in plain.joint_transform


def test_adaptation_metadata_explicitly_distinguishes_policy_semantics_from_calibration():
    assert adapted_client().joint_transform["pan_adaptation"] == {
        "scale": SCALE, "offset": OFFSET,
        "status": "experimental policy adaptation; not physical calibration",
    }


@pytest.mark.parametrize("mode", ["lerobot_degrees", "lerobot_normalized", "model"])
def test_adaptation_rejects_real_or_already_model_coordinates(mode):
    with pytest.raises(ValueError, match="usd_normalized"):
        client(input_joint_units=mode, pan_model_scale=SCALE, pan_model_offset=OFFSET)


@pytest.mark.parametrize("options", [
    {"pan_model_scale": SCALE},
    {"pan_model_offset": OFFSET},
])
def test_partial_adaptation_is_rejected(options):
    with pytest.raises(ValueError, match="provided together"):
        client(**options)


@pytest.mark.parametrize("scale,offset", [
    (0, 0), (-1, 0), ("nan", 0), ("inf", 0), ("bad", 0),
    (1, "nan"), (1, "inf"), (1, "bad"),
    (True, 0), (np.bool_(True), 0), (1, False), (1, np.bool_(False)),
    (None, OFFSET), (SCALE, None), (None, None),
    ([], 0), (1, []),
])
def test_nonfinite_nonpositive_boolean_or_missing_parameters_are_rejected(scale, offset):
    with pytest.raises(ValueError, match="pan_model_"):
        client(pan_model_scale=scale, pan_model_offset=offset)


@pytest.mark.parametrize("scale,offset", [(1e-50, 0), (1e39, 0), (1, 1e39)])
def test_float32_noninvertible_or_overflowing_composed_transform_is_rejected(scale, offset):
    with pytest.raises(ValueError, match="finite, invertible float32"):
        client(pan_model_scale=scale, pan_model_offset=offset)


def test_string_parameters_use_the_same_manifest_configuration_path():
    configured = client(pan_model_scale=str(SCALE), pan_model_offset=str(OFFSET))
    built = configured.build_request(normalized_pose([10, 0, 0, 0, 0, 0]), images(), "task")
    assert built["state"][0] == pytest.approx(8.4369294737, abs=1e-5)
