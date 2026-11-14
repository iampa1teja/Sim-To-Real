"""Shared nonlinear IK retains each arm's bounds, joint order and gripper state."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
pytest.importorskip('newton')
pytest.importorskip('isaaclab')
from utils.arms.arms import load_arm_spec
from sim.actions.ik import ArmKinematics, IKOptions
from sim.actions.bounded_ik import BoundedPoseStep


@pytest.mark.parametrize('name,delta', [
    ('so101', [.12,.05,-.06,-.05,.04,0.]),
    ('sp7', [.12,.05,-.06,-.05,.04,.03,-.02,0.]),
])
def test_bounded_step_reaches_a_feasible_pose_without_moving_the_gripper(name,delta):
    arm = load_arm_spec(name)[0]
    kin = ArmKinematics.from_arm(arm)
    options = IKOptions(orientation_weight=.1)
    controller = BoundedPoseStep(arm,options,kin)
    initial = np.array([j.initial for j in arm.action_joints])
    requested = initial + np.array(delta)
    target = kin.fk(requested)
    target_q = Rotation.from_matrix(target[:3,:3]).as_quat()
    q = initial.copy()
    for _ in range(50):
        pose = kin.fk(q)
        result = controller.solve(torch.tensor(q[None],dtype=torch.float64),
            torch.tensor(pose[None,:3,3]),torch.tensor(Rotation.from_matrix(pose[:3,:3]).as_quat()[None]),
            None,torch.tensor(target[None,:3,3]),torch.tensor(target_q[None]))
        updated = result.joint_positions.numpy()[0]
        assert np.all(np.abs(updated[:-1]-q[:-1]) <= options.max_joint_step_rad+1e-6)
        assert updated[-1] == initial[-1]
        q = updated
    actual = kin.fk(q)
    assert np.linalg.norm(actual[:3,3]-target[:3,3]) < .002
    error = Rotation.from_matrix(target[:3,:3] @ actual[:3,:3].T).as_rotvec()
    axes = [i for i,name in enumerate(('x','y','z')) if name in arm.ik.orientation_axes]
    assert np.linalg.norm(error[axes]) < .02
