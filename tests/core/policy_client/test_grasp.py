"""Optional grasp control selects an explicit role and training dataset."""
import json

import numpy as np
import pytest

from core.policy_client.contract import ArmInfo
from core.policy_client.grasp import GraspConfig, GraspController


def info():
    return ArmInfo('slide_hand', ('jaw_slide', 'axis', 'reach'), ('gripper', 'arm', 'arm'),
                   ('prismatic', 'revolute', 'prismatic'),
                   {'jaw_slide': dict(joint_min=0., joint_max=.04),
                    'axis': dict(joint_min=-90., joint_max=90.), 'reach': dict(joint_min=0., joint_max=.5)})


def test_latch_reads_role_named_features_in_dataset_order(tmp_path):
    metadata = tmp_path / 'meta'
    metadata.mkdir()
    names = ['axis.pos', 'jaw_slide.pos', 'reach.pos']
    (metadata / 'info.json').write_text(json.dumps({'features': {
        feature: {'names': names} for feature in ('action', 'observation.state')}}))
    (metadata / 'stats.json').write_text(json.dumps({
        'action': {'min': [-100, 10, -100], 'max': [100, 90, 100]},
        'observation.state': {'min': [-100, 5, -100]}}))
    arm = info()
    config = GraspConfig(gripper_latch=True, latch_dataset=str(tmp_path), latch_confirm_steps=1,
                         latch_steady_steps=1)
    control = GraspController(config, arm.feature_names, arm_info=arm)
    assert control.gripper_index == 0
    assert control.arm_indices == [1, 2]
    assert control.latch.empty_closed == 5
    target = np.array([20., 11., 12.])
    control.process(target, target, 0)
    sent = control.process(target, target, 1)
    np.testing.assert_array_equal(sent, [10, 11, 12])
    control.close()


def test_grasp_control_requires_configured_dataset():
    arm = info()
    with pytest.raises(ValueError, match='latch_dataset explicitly'):
        GraspController(GraspConfig(gripper_latch=True), arm.feature_names, arm_info=arm)
