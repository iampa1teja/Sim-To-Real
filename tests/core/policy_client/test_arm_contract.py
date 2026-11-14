"""Selected-arm codecs configure clients before any request or transport opens."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from utils.arms.calibration import resolve_calibration
from utils.arms.spec import load_arm_spec
from core.policy_client.adapter import build_plugin_client
from core.policy_client.contract import ArmInfo, load_client_policy
from core.policy_client.discovery import get_manifest, load_manifest
from core.policy_client.remote_policy import RemotePolicy

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def info():
    arm = load_arm_spec('so101')[0]
    return ArmInfo.from_arm(arm, resolve_calibration(arm, unit='rpd_robot').mapping)


def client(info, options=None):
    manifest = get_manifest('molmoact2')
    result = load_client_policy(manifest.name, manifest, options or {})
    result.configure(info)
    return result


def test_molmo_configures_the_original_physical_frame_and_owns_wrist_offset(info):
    policy = client(info)
    low = np.array([info.joint_mapping[name]['joint_min'] for name in info.joint_names], dtype=np.float32)
    high = np.array([info.joint_mapping[name]['joint_max'] for name in info.joint_names], dtype=np.float32)
    expected_scale = np.array([1, -1, 1, 1, 1, 1], dtype=np.float32)
    expected_offset = np.array([0, 90, 90, 0, -90, 0], dtype=np.float32)
    expected_offset[:5] += expected_scale[:5] * ((high + low)[:5] / 2)
    expected_scale[:5] *= (high - low)[:5] / 200
    np.testing.assert_array_equal(policy.scale, expected_scale)
    np.testing.assert_array_equal(policy.offset, expected_offset)
    policy.configure(info)
    np.testing.assert_array_equal(policy.scale, expected_scale)
    np.testing.assert_array_equal(policy.offset, expected_offset)


def test_molmo_reorders_input_and_returned_actions_by_declared_names(info):
    order = [5, 2, 0, 4, 1, 3]
    permuted = replace(info, joint_names=tuple(info.joint_names[i] for i in order),
                       roles=tuple(info.roles[i] for i in order), types=tuple(info.types[i] for i in order))
    original, reordered = client(info), client(permuted)
    state = np.array([17, -25, 30, 45, 11, 54], dtype=np.float32)
    images = {name: np.zeros((2, 3, 3), dtype=np.uint8) for name in original.cameras}
    expected = original.build_request(state, images, 'pick')['state']
    actual = reordered.build_request(state[order], images, 'pick')['state']
    np.testing.assert_array_equal(actual, expected)
    returned = reordered.parse_response({'actions': actual[None]}, list(permuted.feature_names))
    np.testing.assert_allclose(returned[0], state[order], rtol=0, atol=1e-4)


def test_molmo_uses_explicit_degree_and_reversed_gripper_codecs(info):
    spans = {name: dict(span) for name, span in info.joint_mapping.items()}
    for name in info.joint_names[:-1]:
        spans[name].update(raw_min=-40., raw_max=60.)
    spans['gripper'].update(raw_min=100., raw_max=0.)
    encoded = replace(info, joint_mapping=spans)
    policy = client(encoded, {'wrist_roll_model_offset': 0})
    state = np.array([10., 10., 10., 10., 10., 25.], dtype=np.float32)
    images = {name: np.zeros((2, 2, 3), np.uint8) for name in policy.cameras}
    request = policy.build_request(state, images, '')
    midpoint = np.array([(spans[name]['joint_min'] + spans[name]['joint_max']) / 2
                         for name in info.joint_names[:-1]], dtype=np.float32)
    midpoint[1] = -midpoint[1] + 90
    midpoint[2] += 90
    np.testing.assert_allclose(request['state'], [*midpoint, 75], atol=1e-5, rtol=0)
    np.testing.assert_allclose(policy.parse_response({'actions': request['state'][None]}, list(info.feature_names)),
                               state[None], atol=1e-4, rtol=0)


def test_fixed_embodiment_refuses_before_opening_transport(info, monkeypatch):
    small = replace(info, joint_names=info.joint_names[:4], roles=info.roles[:4], types=info.types[:4],
                    joint_mapping={name: info.joint_mapping[name] for name in info.joint_names[:4]})
    monkeypatch.setattr('core.policy_client.adapter.PolicyClient', lambda *a, **k: pytest.fail('transport opened'))
    with pytest.raises(ValueError, match='requires 6 joints; selected arm has 4'):
        build_plugin_client('molmoact2', small.feature_names, arm_info=small)
    wrong = replace(info, joint_names=tuple('axis_' + str(i) for i in range(6)),
                    joint_mapping={'axis_' + str(i): info.joint_mapping[name]
                                   for i, name in enumerate(info.joint_names)})
    with pytest.raises(ValueError, match='SO-100/101 requires joints'):
        build_plugin_client('molmoact2', wrong.feature_names, arm_info=wrong)


def test_manifest_dimension_and_generic_default_horizon(tmp_path, info):
    source = ROOT / 'workspace/policies/example/policy.toml'
    folder = tmp_path / 'example'
    folder.mkdir()
    for value in ('0', 'true', '"6"', '"invalid"'):
        (folder / 'policy.toml').write_text(source.read_text().replace('[action]', '[action]\ndim = ' + value))
        with pytest.raises(ValueError, match='action.dim'):
            load_manifest(folder)
    assert get_manifest('example').action_dim == 'arm'
    policy = RemotePolicy(info.feature_names, policy_name='molmoact2', arm_info=info)
    assert policy._action_horizon == get_manifest('molmoact2').default_horizon == 30


def test_arm_info_rejects_partial_raw_codec(info):
    spans = {name: dict(span) for name, span in info.joint_mapping.items()}
    spans[info.joint_names[0]]['raw_min'] = -100
    with pytest.raises(ValueError, match='both raw codec endpoints'):
        replace(info, joint_mapping=spans)


def test_real_pan_workspace_expansion_maps_observations_and_actions_consistently():
    arm = load_arm_spec('so101')[0]
    selected = resolve_calibration(arm, unit='rpd_follower', environ={})
    info = ArmInfo.from_arm(arm, selected.mapping)
    normal = client(info, {'input_joint_units': 'lerobot_normalized'})
    wide = client(info, {'input_joint_units': 'lerobot_normalized',
                         'pan_model_scale': .5, 'pan_model_offset': 0.})
    images = {name: np.zeros((2, 3, 3), np.uint8) for name in wide.cameras}
    for pan in (-100., -50., 0., 50., 100.):
        state = np.array([pan, 5, -10, 20, 15, 40], np.float32)
        original = normal.build_request(state, images, '')['state']
        adapted = wide.build_request(state, images, '')['state']
        assert adapted[0] == original[0] * .5
        np.testing.assert_array_equal(adapted[1:], original[1:])
        np.testing.assert_allclose(wide.parse_response({'actions': adapted[None]}, list(info.feature_names)),
                                   state[None], atol=1e-4, rtol=0)
    model_targets = np.array([[-42.1300246, 90, 90, 0, 0, 50],
                              [48.5534956, 90, 90, 0, 0, 50]], np.float32)
    ordinary = normal.parse_response({'actions': model_targets}, list(info.feature_names))
    expanded = wide.parse_response({'actions': model_targets}, list(info.feature_names))
    np.testing.assert_allclose(expanded[:, 0], ordinary[:, 0] * 2, atol=1e-5)
    np.testing.assert_array_equal(expanded[:, 1:], ordinary[:, 1:])
    assert -100 < expanded[0, 0] < -70 and 80 < expanded[1, 0] < 100
    scales = wide.scale.copy(); offsets = wide.offset.copy()
    wide.configure(info)
    np.testing.assert_array_equal(wide.scale, scales)
    np.testing.assert_array_equal(wide.offset, offsets)
