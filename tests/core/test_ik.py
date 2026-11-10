"""Constrained IK uses declared roles, units, frames and native Jacobian ordering."""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.arms.spec import IKSpec, load_arm_spec
from sim.datagen.ik import ArmKinematics, ArticulationIK, IKOptions, solve_damped_step


def planar(q):
    angle, length = q[0], 1 + q[2]
    position = torch.tensor([length * math.cos(angle), length * math.sin(angle), 0.], dtype=torch.float64)
    quaternion = torch.tensor([0., 0., math.sin(angle / 2), math.cos(angle / 2)], dtype=torch.float64)
    jacobian = torch.zeros((6, 4), dtype=torch.float64)
    jacobian[:3, 0] = torch.tensor([-position[1], position[0], 0.])
    jacobian[5, 0] = 1
    jacobian[:3, 2] = torch.tensor([math.cos(angle), math.sin(angle), 0.])
    return position, quaternion, jacobian


def test_four_joint_ik_converges_with_interleaved_parallel_gripper_and_prismatic(synthetic_arm):
    q = torch.tensor([0., .013, 0., .027], dtype=torch.float64)
    desired = torch.tensor([.08, .013, .10, .027], dtype=torch.float64)
    target, _, _ = planar(desired)
    initial_error = torch.linalg.norm(target - planar(q)[0])
    for _ in range(70):
        pose, rotation, jacobian = planar(q)
        result = solve_damped_step(synthetic_arm, q, pose, rotation, jacobian, target)
        next_q = result.joint_positions[0]
        assert abs(next_q[0] - q[0]) <= .08 + 1e-12
        assert abs(next_q[2] - q[2]) <= .005 + 1e-12
        torch.testing.assert_close(next_q[[1, 3]], q[[1, 3]], rtol=0, atol=0)
        q = next_q
    assert result.converged.item()
    assert result.position_error.item() < .001
    assert result.position_error.item() < initial_error / 100
    torch.testing.assert_close(q[[0, 2]], desired[[0, 2]], atol=1e-5, rtol=0)


def test_unreachable_prismatic_target_stops_at_runtime_limit(synthetic_arm):
    q = torch.tensor([0., .013, .199, .027], dtype=torch.float64)
    for _ in range(10):
        pose, rotation, jacobian = planar(q)
        result = solve_damped_step(synthetic_arm, q, pose, rotation, jacobian, [2., 0., 0.])
        q = result.joint_positions[0]
    assert q[2] == synthetic_arm.joints[2].limits[1]
    assert result.limited.item() and not result.converged.item()
    assert result.position_error.item() == pytest.approx(.8)


def test_orientation_constraints_ignore_uncontrolled_yaw_and_use_base_frame(synthetic_arm):
    q = torch.tensor([0., .01, 0., .03], dtype=torch.float64)
    jacobian = torch.zeros((6, 4), dtype=torch.float64)
    jacobian[4, 0] = 1  # world Y is base X after a +90-degree base rotation.
    angle = .2
    target = [0., math.sin(angle / 2), 0., math.cos(angle / 2)]
    base = [0., 0., math.sqrt(.5), math.sqrt(.5)]
    arm = replace(synthetic_arm, ik=IKSpec(('x',)))
    result = solve_damped_step(arm, q, [0., 0., 0.], [0., 0., 0., 1.], jacobian,
                               [0., 0., 0.], target, base_orientation=base)
    assert result.orientation_error.item() == pytest.approx(.2)
    assert result.joint_positions[0, 0] > 0
    ignored = replace(synthetic_arm, ik=IKSpec(('x', 'y')))
    result = solve_damped_step(ignored, q, [0., 0., 0.], [0., 0., 0., 1.], jacobian,
                               [0., 0., 0.], [0., 0., math.sin(angle / 2), math.cos(angle / 2)])
    assert result.converged.item() and result.orientation_error.item() == 0
    torch.testing.assert_close(result.joint_positions[0], q, atol=0, rtol=0)


def test_zero_jacobian_is_finite_and_does_not_report_unreachable_target_success(synthetic_arm):
    result = solve_damped_step(synthetic_arm, [0., .01, 0., .03], [0., 0., 0.], [0., 0., 0., 1.],
                               np.zeros((6, 4)), [1., 1., 1.])
    assert torch.isfinite(result.joint_positions).all() and not result.converged.item()
    for kwargs in ({'damping': 0}, {'max_joint_step_m': float('nan')}):
        with pytest.raises(ValueError, match='finite and positive'):
            IKOptions(**kwargs)


def test_native_adapter_accounts_for_tcp_offset_joint_permutation_and_base_dofs(synthetic_arm):
    q = torch.tensor([0., .013, .1, .027], dtype=torch.float64)
    position, orientation, jacobian = planar(q)
    # Native articulation order is Right, Slide, Pan, Left; manifest order remains Pan, Left, Slide, Right.
    ids = [2, 3, 1, 0]
    native_q = q[[3, 2, 0, 1]].unsqueeze(0)
    native_jacobian = torch.zeros((1, 2, 6, 10), dtype=torch.float64)
    for manifest, native in enumerate(ids):
        native_jacobian[0, 1, :, 6 + native] = jacobian[:, manifest]
    data = SimpleNamespace(
        body_link_pos_w=torch.stack((torch.zeros(3), position)).unsqueeze(0),
        body_link_quat_w=torch.stack((torch.tensor([0., 0., 0., 1.]), orientation)).unsqueeze(0),
        joint_pos=native_q, body_link_jacobian_w=native_jacobian,
    )
    robot = SimpleNamespace(data=data, is_fixed_base=False, num_base_dofs=6,
        find_joints=lambda names, preserve_order: (ids, list(synthetic_arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == synthetic_arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, synthetic_arm, position_offset_m=[0., .1, 0.], options=IKOptions(integral_gain=0))
    current, quaternion = solver.current_pose()
    torch.testing.assert_close(current[0], position + position.new_tensor([0., .1, 0.]), atol=0, rtol=0)
    target = current + current.new_tensor([0., .01, 0.])
    result = solver.solve(target)
    jacobian[0, 0] -= .1  # omega_z cross offset_y contributes -offset_y in linear X.
    expected = solve_damped_step(synthetic_arm, q, current, quaternion, jacobian, target)
    torch.testing.assert_close(result.joint_positions, expected.joint_positions, atol=1e-12, rtol=0)
    torch.testing.assert_close(robot.data.joint_pos, native_q, atol=0, rtol=0)


def stage_for_arm(arm, *, meters_per_unit=1., reverse_slide=False):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/Robot')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    for name, x in (('base', 0.), ('elbow', 1.), ('tool', 1.), ('left_link', 1.), ('right_link', 1.)):
        body = UsdGeom.Xform.Define(stage, '/Robot/' + name)
        body.AddTranslateOp().Set(Gf.Vec3d(x / meters_per_unit, 0., 0.))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/Robot/Pan')
    joint.CreateBody0Rel().SetTargets(['/Robot/base'])
    joint.CreateBody1Rel().SetTargets(['/Robot/elbow'])
    joint.CreateAxisAttr('Z')
    joint.CreateLocalPos1Attr(Gf.Vec3f(-1 / meters_per_unit, 0., 0.))
    joint = UsdPhysics.PrismaticJoint.Define(stage, '/Robot/Slide')
    joint.CreateBody0Rel().SetTargets(['/Robot/tool' if reverse_slide else '/Robot/elbow'])
    joint.CreateBody1Rel().SetTargets(['/Robot/elbow' if reverse_slide else '/Robot/tool'])
    joint.CreateAxisAttr('X')
    return stage


@pytest.mark.parametrize('meters_per_unit,reverse_slide', [(1., False), (.001, False), (1., True)])
def test_usd_fk_uses_authored_joint_axes_frames_units_and_direction(synthetic_arm, meters_per_unit, reverse_slide):
    kinematics = ArmKinematics(stage_for_arm(synthetic_arm, meters_per_unit=meters_per_unit,
                                             reverse_slide=reverse_slide), synthetic_arm)
    pose = kinematics.fk([math.pi / 2, .013, .1, .027])
    expected = 1 - .1 if reverse_slide else 1 + .1
    np.testing.assert_allclose(pose[:3, 3], [0., expected, 0.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pose[:3, :3], [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(pose, kinematics.fk([math.pi / 2, .04, .1, 0.]))


def test_so101_fk_zero_pose_matches_authored_frames_without_simulator(monkeypatch):
    monkeypatch.delenv('S2R_ARMS_DIR', raising=False)
    arm = load_arm_spec('so101')[0]
    assert arm.ik.orientation_axes == ('x', 'y')
    fk = ArmKinematics.from_arm(arm)
    pose = fk.fk(np.zeros(len(arm.joints)))
    assert pose.shape == (4, 4) and np.isfinite(pose).all()
    assert abs(np.linalg.det(pose[:3, :3]) - 1) < 1e-12


def biased_prismatic_servo(arm, *, gain=1.):
    """A CPU position servo with constant load: equilibrium is command minus 2 mm."""
    measured = torch.tensor([[0., .013, 0., .027]], dtype=torch.float64)
    positions = torch.zeros((1, 2, 3), dtype=torch.float64)
    rotations = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]], dtype=torch.float64)
    jacobian = torch.zeros((1, 1, 6, len(arm.joints)), dtype=torch.float64)
    jacobian[0, 0, 0, 2] = 1.
    data = SimpleNamespace(joint_pos=measured, body_link_pos_w=positions,
                            body_link_quat_w=rotations, body_link_jacobian_w=jacobian)
    robot = SimpleNamespace(data=data, is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(len(arm.joints))), list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, arm, options=IKOptions(integral_gain=gain), control_dt=1 / 30)
    return solver, measured, positions


def test_integral_bias_removes_pd_load_residual_without_changing_physics_or_grippers(synthetic_arm):
    errors = []
    for gain in (0., 1.):
        solver, measured, positions = biased_prismatic_servo(synthetic_arm, gain=gain)
        for _ in range(360):
            positions[0, 1, 0] = measured[0, 2]
            result = solver.solve([.05, 0., 0.])
            torch.testing.assert_close(result.joint_positions[:, [1, 3]], measured[:, [1, 3]], atol=0, rtol=0)
            assert abs(result.joint_positions[0, 2] - measured[0, 2]) <= .005 + 1e-12
            # Same fixed servo/load model in both runs. Only the encoded joint target changes.
            measured[0, 2] += .35 * (result.joint_positions[0, 2] - measured[0, 2] - .002)
        errors.append(abs(measured[0, 2].item() - .05))
    assert .002 < errors[0] < .0022
    assert errors[1] < 1e-5 and errors[1] < errors[0] / 100
    assert solver.integral_bias[0, 2] == pytest.approx(.002, abs=1e-5)


def test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits(synthetic_arm):
    solver, measured, positions = biased_prismatic_servo(synthetic_arm)
    for _ in range(30):
        solver.solve([1., 0., 0.])  # A blocked servo must not integrate beyond the allowed command lead.
    torch.testing.assert_close(solver.integral_bias, torch.zeros_like(measured), atol=0, rtol=0)
    for _ in range(20):
        solver.solve([.001, 0., 0.])
    before = solver.integral_bias
    assert before[0, 2] > 0
    for _ in range(10):
        solver.solve([.001, 0., 0.], advance=False)
    torch.testing.assert_close(solver.integral_bias, before, atol=0, rtol=0)
    measured[0, 2] = .2
    positions[0, 1, 0] = .2
    command = solver.solve([1., 0., 0.])
    assert command.joint_positions[0, 2] == .2
    assert solver.integral_bias[0, 2] == 0  # Hard-stop projection unwinds retained outward bias.
    solver.reset()
    assert solver.integral_bias is None
    solver.solve([.1, 0., 0.], advance=False)
    assert solver.integral_bias is None


def test_fk_jacobian_matches_analytic_base_frame_with_tcp_and_mixed_roles(synthetic_arm):
    arm = replace(synthetic_arm, ik=IKSpec(('z',), (.02, .03, .01)))
    kinematics = ArmKinematics(stage_for_arm(arm), arm)
    q = np.array([.4, .013, .06, .027])
    before = q.copy()
    current = kinematics.fk(q)
    jacobian = kinematics.jacobian(q)
    expected = np.zeros((6, 4))
    expected[:3, 0] = [-current[1, 3], current[0, 3], 0.]
    expected[5, 0] = 1.
    expected[:3, 2] = [math.cos(q[0]), math.sin(q[0]), 0.]
    np.testing.assert_allclose(jacobian, expected, atol=2e-10, rtol=0)
    np.testing.assert_array_equal(q, before)


def test_local_inverse_recovers_mixed_joint_command_and_preserves_parallel_gripper(synthetic_arm):
    arm = replace(synthetic_arm, ik=IKSpec(('z',)))
    kinematics = ArmKinematics(stage_for_arm(arm), arm)
    measured = np.array([0., .013, 0., .027])
    desired = np.array([.04, .001, .003, .039])
    target = kinematics.fk(desired)
    before = measured.copy()
    result = kinematics.inverse(target, measured)
    assert result.converged.item()
    assert result.position_error.item() <= 1e-5
    assert result.orientation_error.item() <= 1e-4
    np.testing.assert_allclose(result.joint_positions[0, [0, 2]], desired[[0, 2]], atol=2e-5, rtol=0)
    np.testing.assert_array_equal(result.joint_positions[0, [1, 3]], measured[[1, 3]])
    np.testing.assert_array_equal(measured, before)


@pytest.mark.parametrize('measured,target_slide', [(0., .15), (.199, .5)])
def test_local_inverse_final_dispatch_respects_measured_step_and_runtime_limits(
        synthetic_arm, measured, target_slide):
    arm = replace(synthetic_arm, ik=IKSpec(('z',)))
    kinematics = ArmKinematics(stage_for_arm(arm), arm)
    state = np.array([0., .013, measured, .027])
    target = kinematics.fk([.5, .013, target_slide, .027])
    result = kinematics.inverse(target, state)
    command = result.joint_positions[0].numpy()
    assert abs(command[0] - state[0]) <= .08 + 1e-12
    assert abs(command[2] - state[2]) <= .005 + 1e-12
    assert arm.joints[2].limits[0] <= command[2] <= arm.joints[2].limits[1]
    np.testing.assert_array_equal(command[[1, 3]], state[[1, 3]])
    assert result.limited.item() and not result.converged.item()
    expected_error = np.linalg.norm(target[:3, 3] - kinematics.fk(command)[:3, 3])
    assert result.position_error.item() == pytest.approx(expected_error, abs=1e-14)


def test_local_inverse_ignores_undeclared_orientation_and_rejects_invalid_inputs(synthetic_arm):
    arm = replace(synthetic_arm, ik=IKSpec(('x', 'y')))
    kinematics = ArmKinematics(stage_for_arm(arm), arm)
    measured = np.array([0., .013, 0., .027])
    target = kinematics.fk(measured)
    target[:3, :3] = np.diag([-1., -1., 1.])  # A pure yaw is outside the declared constraints.
    result = kinematics.inverse(target, measured)
    np.testing.assert_array_equal(result.joint_positions[0], measured)
    assert result.converged.item()
    for kwargs in ({'max_iterations': 0}, {'position_tolerance_m': float('nan')}):
        with pytest.raises(ValueError, match='positive'):
            kinematics.inverse(target, measured, **kwargs)
    malformed = target.copy()
    malformed[0, 0] = 2.
    with pytest.raises(ValueError, match='rigid 4x4'):
        kinematics.inverse(malformed, measured)


# Saved physical no-roll episode: datasets/v2_phase5_scripted_noroll_fast1,
# frames 190 (lift) and 334 (fast arc, worst one-step recovery). Values are decoded
# simulator radians. Keep these small real-command regressions independent of an
# untracked dataset; gripper commands are dispatched separately by Mimic.
@pytest.mark.parametrize('measured,target', [
    ([-1.163608319116648, -.7601840025618709, .9659029653945234, .6477574383318583,
      -.033707175924888984, .06433686438416306],
     [-1.163474640998018, -.7705220616360987, .9344297748614645, .6657402344146962,
      -.03245369241626948, -.1745329917784602]),
    ([1.5324260446621738, -.18931287358164015, .40377766621812344, .6384114379010836,
      -.037141974742046675, .06481570272093419],
     [1.5649960384805777, -.1854819999349474, .3676161233147698, .6300907638656686,
      -.03519884765900414, -.1745329917784602]),
])
def test_local_inverse_recovers_saved_physical_source_commands(monkeypatch, measured, target):
    monkeypatch.delenv('S2R_ARMS_DIR', raising=False)
    arm = load_arm_spec('so101')[0]
    kinematics = ArmKinematics.from_arm(arm)
    result = kinematics.inverse(kinematics.fk(target), measured)
    arm_ids = [index for index, joint in enumerate(arm.joints) if joint.role == 'arm']
    gripper_ids = [index for index, joint in enumerate(arm.joints) if joint.role == 'gripper']
    assert result.converged.item()
    assert result.position_error.item() <= 1e-5
    assert result.orientation_error.item() <= 1e-4
    np.testing.assert_allclose(result.joint_positions[0, arm_ids], np.asarray(target)[arm_ids], atol=2e-4, rtol=0)
    np.testing.assert_array_equal(result.joint_positions[0, gripper_ids], np.asarray(measured)[gripper_ids])
