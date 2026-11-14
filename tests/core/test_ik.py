"""IK options, manifest bounds and factories work without simulator libraries."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sim.actions.ik import IKOptions, _arm_bounds, make_ik, make_kinematics


@pytest.mark.parametrize('changes', [
    {'damping': 0.}, {'orientation_weight': -1.}, {'max_position_step_m': 0.},
    {'max_orientation_step_rad': float('nan')}, {'max_joint_step_rad': float('inf')},
    {'max_joint_step_m': -.01}, {'position_tolerance_m': 0.},
    {'orientation_tolerance_rad': -.01}, {'integral_gain': -.1},
    {'integral_gain': float('nan')}, {'max_integral_bias_rad': 0.},
    {'max_integral_bias_m': float('inf')},
])
def test_ik_options_reject_nonfinite_or_nonpositive_values(changes):
    with pytest.raises(ValueError, match='IK options must be finite and positive'):
        IKOptions(**changes)


def test_ik_options_allow_disabled_integral_feedback():
    assert IKOptions(integral_gain=0.).integral_gain == 0.


def test_arm_bounds_preserve_manifest_order_units_and_exclude_grippers(synthetic_arm):
    options = IKOptions(max_joint_step_rad=.12, max_joint_step_m=.007,
                        max_integral_bias_rad=.09, max_integral_bias_m=.003)
    bounds = _arm_bounds(synthetic_arm, options)
    assert bounds['ids'] == [0, 2]
    np.testing.assert_array_equal(bounds['step_caps'], [.12, .007])
    np.testing.assert_array_equal(bounds['bias_caps'], [.09, .003])
    np.testing.assert_allclose(bounds['lower'], [-math.pi, -.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(bounds['upper'], [math.pi, .2], rtol=0, atol=1e-15)
    changed = replace(synthetic_arm, joints=(
        replace(synthetic_arm.joints[0], limits=(-45., 90.)), *synthetic_arm.joints[1:]))
    bounds = _arm_bounds(changed, options)
    np.testing.assert_allclose(bounds['lower'], [-math.pi / 4, -.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(bounds['upper'], [math.pi / 2, .2], rtol=0, atol=1e-15)


def test_arm_bounds_require_an_arm_joint(synthetic_arm):
    arm = replace(synthetic_arm, joints=tuple(
        replace(joint, role='gripper') for joint in synthetic_arm.joints))
    with pytest.raises(ValueError, match='IK requires at least one joint with role arm'):
        _arm_bounds(arm, IKOptions())


@pytest.mark.parametrize('hook_name', ['make_ik', 'make_kinematics'])
def test_factories_choose_arm_hook_and_forward_arguments(synthetic_arm, monkeypatch, hook_name):
    import utils.arms.arms as arms_module
    import sim.actions.ik as ik_module

    sentinel, robot = object(), object()
    options = IKOptions()
    kwargs = {'position_offset_m': [.01, .02, .03], 'options': options}
    calls = []

    def hook(*args, **forwarded):
        calls.append((args, forwarded))
        return sentinel

    utils = SimpleNamespace(**{hook_name: hook})
    monkeypatch.setattr(arms_module, 'load_arm_utils', lambda arm: utils)
    def unexpected(*args, **kwargs):
        pytest.fail('An available arm hook must be used')
    monkeypatch.setattr(ik_module, 'ArticulationIK', unexpected)
    monkeypatch.setattr(ik_module, 'ArmKinematics', SimpleNamespace(from_arm=unexpected))
    if hook_name == 'make_ik':
        result = make_ik(robot, synthetic_arm, **kwargs)
        expected = (robot, synthetic_arm)
    else:
        result = make_kinematics(synthetic_arm, **kwargs)
        expected = (synthetic_arm,)
    assert result is sentinel
    assert calls == [(expected, kwargs)]


@pytest.mark.parametrize('utils', [None, SimpleNamespace()])
def test_factories_choose_defaults_without_an_arm_hook(synthetic_arm, monkeypatch, utils):
    import utils.arms.arms as arms_module
    import sim.actions.ik as ik_module

    live, offline, robot = object(), object(), object()
    calls = []
    def default_live(*args, **kwargs):
        calls.append(('live', args, kwargs))
        return live
    def default_offline(*args, **kwargs):
        calls.append(('offline', args, kwargs))
        return offline
    monkeypatch.setattr(arms_module, 'load_arm_utils', lambda arm: utils)
    monkeypatch.setattr(ik_module, 'ArticulationIK', default_live)
    monkeypatch.setattr(ik_module, 'ArmKinematics', SimpleNamespace(from_arm=default_offline))
    kwargs = {'position_offset_m': [.01, .02, .03]}
    assert make_ik(robot, synthetic_arm, **kwargs) is live
    assert make_kinematics(synthetic_arm, **kwargs) is offline
    assert calls == [('live', (robot, synthetic_arm), kwargs),
                     ('offline', (synthetic_arm,), kwargs)]
