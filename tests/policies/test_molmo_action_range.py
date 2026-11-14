"""Selected-axis extrapolation and calibrated client bounds; no motors or weights."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.arms.calibration import resolve_calibration
from core.arms.spec import load_arm_spec
from core.policy_client.contract import ArmInfo, load_client_policy
from core.policy_client.discovery import get_manifest
from core.rollout.real_rollout import clamp_action, joint_ranges

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
LOW = np.array([-42.13002463, 45.18258358, 35.40059183, 4.92978184, -65.57568618, -.30165569], np.float32)
HIGH = np.array([48.55349563, 186.1064668, 173.6076722, 93.41056418, 43.53107398, 44.74649337], np.float32)


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'src/core'))
    from policy_server.contract import load_server_policy
    policy = load_server_policy(ROOT / 'workspace/policies/molmoact2')
    module = __import__(type(policy).__module__, fromlist=['allow_action_extrapolation'])
    normalizer = SimpleNamespace(mode='q01_q99', q_low=LOW.copy(), q_high=HIGH.copy(),
                                 mask=np.ones(6, bool), zero_mask=np.zeros(6, bool))
    def decode(action, tag):
        import torch
        tensor = torch.is_tensor(action)
        raw = action.float().cpu().numpy() if tensor else np.asarray(action, np.float32)
        result = (np.clip(raw, -1, 1) + 1) * (HIGH - LOW) / 2 + LOW
        return torch.as_tensor(result, device=action.device) if tensor else result
    stats = SimpleNamespace(action_normalizers={'test': normalizer}, unnormalize_action=decode,
                            get_metadata=lambda tag: {'action_stats': {'names': NAMES}})
    model = SimpleNamespace(_get_robot_stats=lambda: stats)
    return module, model, stats, normalizer


@pytest.mark.parametrize('tensor', [False, True])
def test_only_selected_out_of_range_pan_is_extended_and_other_values_are_exact(fixture, tensor):
    import torch
    module, model, stats, normalizer = fixture
    values = np.array([[-2, -3, 4, -5, 6, 7], [-1, 0, 1, .5, -.5, .7],
                       [0, .2, .3, .4, .5, .6], [1, 1, 1, 1, 1, 1], [2, 3, -4, 5, -6, -7]], np.float32)
    raw = torch.tensor(values) if tensor else values
    before = stats.unnormalize_action(raw, 'test')
    module.allow_action_extrapolation(model, 'test', ('shoulder_pan',))
    after = stats.unnormalize_action(raw, 'test')
    if tensor:
        assert after.dtype == torch.float32 and after.device == raw.device
        before, after = before.numpy(), after.numpy()
    np.testing.assert_array_equal(after[:, 1:], before[:, 1:])
    np.testing.assert_array_equal(after[1:4], before[1:4])
    assert after[0, 0] < LOW[0] and after[-1, 0] > HIGH[0]
    np.testing.assert_array_equal(normalizer.q_low, LOW)
    np.testing.assert_array_equal(normalizer.q_high, HIGH)
    np.testing.assert_array_equal(stats.unnormalize_action(values, 'other'), before)


def test_extended_predictions_use_the_same_joint_translation_and_motor_bounds(fixture):
    module, model, stats, _ = fixture
    arm = load_arm_spec('so101')[0]
    cal = resolve_calibration(arm, unit='rpd_follower', environ={})
    manifest = get_manifest('molmoact2')
    client = load_client_policy('molmoact2', manifest, {'input_joint_units': 'lerobot_normalized'})
    client.configure(ArmInfo.from_arm(arm, cal.mapping))
    raw = np.zeros((4, 6), np.float32)
    raw[:, 0] = [-1.5, 1.5, -100, 100]
    old = client.parse_response({'actions': stats.unnormalize_action(raw, 'test')}, list(arm.feature_names))
    module.allow_action_extrapolation(model, 'test', ('shoulder_pan',))
    new = client.parse_response({'actions': stats.unnormalize_action(raw, 'test')}, list(arm.feature_names))
    assert new[0, 0] < old[0, 0] and new[1, 0] > old[1, 0]
    bounds = joint_ranges(arm, cal.mapping)
    sent = np.array([clamp_action(dict(zip(arm.feature_names, row)), arm.feature_names, bounds) for row in new])
    assert sent[-2, 0] == -100 and sent[-1, 0] == 100
    np.testing.assert_array_equal(new[:, 1:], old[:, 1:])


@pytest.mark.parametrize('joints', [('wrong',), ('shoulder_pan', 'shoulder_pan'), ()])
def test_unknown_duplicate_or_empty_axes_are_rejected(fixture, joints):
    module, model, _, _ = fixture
    with pytest.raises(ValueError, match='distinct checkpoint joints'):
        module.allow_action_extrapolation(model, 'test', joints)


def test_nonfinite_predictions_and_constant_axes_are_rejected(fixture):
    module, model, stats, normalizer = fixture
    module.allow_action_extrapolation(model, 'test', ('shoulder_pan',))
    for value in (np.nan, np.inf, -np.inf):
        raw = np.zeros((1, 6), np.float32); raw[0, 0] = value
        with pytest.raises(ValueError, match='Non-finite'):
            stats.unnormalize_action(raw, 'test')
    normalizer.zero_mask[0] = True
    with pytest.raises(ValueError, match='constant joint'):
        module.allow_action_extrapolation(model, 'test', ('shoulder_pan',))
