"""Library IK/FK regressions; CPU only, with authored serial USD test arms.

Pose comparisons use metres/radians. Newton state storage is float32; FK checks
allow 2e-5 m, Jacobian differences 1e-4, and full-pose inversion 1e-4 m/1e-3 rad.
"""
from dataclasses import replace
import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.arms.spec import (ArmSpec, GripperSpec, HardwareSpec, IKSpec, InitialStateSpec,
                            JointSpec, LeRobotSpec, LinksSpec, load_arm_spec)
from sim.datagen.ik import ArmKinematics, ArticulationIK, IKOptions, _JointStep, _controller_cfg


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('pxr', 'isaaclab', 'newton'))
pytestmark = pytest.mark.skipif(not AVAILABLE, reason='Run inside s2r-sim with s2r-entrypoint python')


def arm_spec(directory, count=6, *, mixed=False, grippers=1, permute=True):
    """A complete manifest, including interleaved independent gripper joints."""
    arm_joints = [JointSpec(f'joint_{i}', f'Joint{i}', 'prismatic' if mixed and i == 0 else 'revolute',
                            'arm', (-.2, .2) if mixed and i == 0 else (-180., 180.), 0.)
                  for i in range(count)]
    grips = [JointSpec(f'grip_{i}', f'Grip{i}', 'prismatic', 'gripper', (0., .04), .02)
             for i in range(grippers)]
    joints = list(arm_joints)
    if permute:
        joints = joints[2::2] + joints[1::2] + joints[:1]
    for index, joint in enumerate(grips):
        joints.insert(min(1 + index * 3, len(joints)), joint)
    return ArmSpec(
        name='synthetic', directory=directory, usd_path=directory / 'arm.usda', utils=None,
        links=LinksSpec('base', 'tool'), joints=tuple(joints),
        gripper=(GripperSpec('parallel', tuple(j.name for j in grips),
                             tuple(f'finger_{i}' for i in range(grippers)),
                             tuple(f'finger_{i}' for i in range(grippers))) if grips else None),
        wrist_roll=None, cameras={}, appearance={}, actuators={}, spawn={},
        initial_state=InitialStateSpec(), lerobot=LeRobotSpec('test_follower', 'test_leader'),
        hardware=HardwareSpec('test', 4096, 2048),
        ik=IKSpec(('x', 'y') if count == 5 else ('x', 'y', 'z'), (.025, -.012, .018)),
    )


def stage_for_arm(arm, *, meters_per_unit=1., reverse_slide=False):
    """Author all bodies/joints at a known zero pose and export a self-contained USD.

    Both joint frames rotate 180 degrees for the reversed slide, so its +X
    coordinate moves along base -X. This tests axis direction independently of
    Newton's joint traversal order. Nonzero body COMs exercise COM-to-TCP shifts.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, '/Robot')
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(.3 / meters_per_unit, -.2 / meters_per_unit, .4 / meters_per_unit))
    root.AddRotateZOp().Set(35.)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    physical = sorted((j for j in arm.joints if j.role == 'arm'), key=lambda j: int(j.usd[5:]))
    increments = np.array([[0., 0., .04], [.10, .03, .10], [.16, 0., .02],
                           [.13, .02, 0.], [.07, .04, .02], [.05, 0., .03], [.03, .01, .04]])
    origins = np.cumsum(increments[:len(physical)], axis=0)
    positions = {'base': np.zeros(3)}
    for index, origin in enumerate(origins):
        positions['tool' if index == len(physical) - 1 else f'link_{index}'] = origin
    grippers = [j for j in arm.joints if j.role == 'gripper']
    for i, joint in enumerate(grippers):
        positions[f'finger_{i}'] = origins[-1] + [0., .025 * (1 if i % 2 == 0 else -1), .01]
    for name, position in positions.items():
        body = UsdGeom.Xform.Define(stage, '/Robot/' + name)
        body.AddTranslateOp().Set(Gf.Vec3d(*map(float, position / meters_per_unit)))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(1.)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(.011 / meters_per_unit, -.009 / meters_per_unit,
                                          .006 / meters_per_unit))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*([.01 / meters_per_unit ** 2] * 3)))
    axes = ('Z', 'Y', 'Y', 'X', 'Y', 'Z', 'X')
    for index, declared in enumerate(physical):
        left = 'base' if index == 0 else f'link_{index - 1}'
        right = 'tool' if index == len(physical) - 1 else f'link_{index}'
        kind = UsdPhysics.PrismaticJoint if declared.type == 'prismatic' else UsdPhysics.RevoluteJoint
        joint = kind.Define(stage, '/Robot/' + declared.usd)
        joint.CreateBody0Rel().SetTargets(['/Robot/' + left])
        joint.CreateBody1Rel().SetTargets(['/Robot/' + right])
        joint.CreateAxisAttr('X' if declared.type == 'prismatic' else axes[index])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, (positions[right] - positions[left]) / meters_per_unit)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        if declared.type == 'prismatic' and reverse_slide:
            for creator in (joint.CreateLocalRot0Attr, joint.CreateLocalRot1Attr):
                creator(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
        scale = 1. / meters_per_unit if declared.type == 'prismatic' else 1.
        joint.CreateLowerLimitAttr(float(declared.limits[0] * scale))
        joint.CreateUpperLimitAttr(float(declared.limits[1] * scale))
    for i, declared in enumerate(grippers):
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/Robot/' + declared.usd)
        joint.CreateBody0Rel().SetTargets(['/Robot/tool'])
        joint.CreateBody1Rel().SetTargets([f'/Robot/finger_{i}'])
        joint.CreateAxisAttr('Y')
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float,
            (positions[f'finger_{i}'] - positions['tool']) / meters_per_unit)))
        joint.CreateLowerLimitAttr(0.)
        joint.CreateUpperLimitAttr(.04 / meters_per_unit)
    arm.directory.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(arm.usd_path))
    return stage


def make_arm(tmp_path, count=6, *, mixed=False, grippers=1, permute=True,
             meters_per_unit=1., reverse_slide=False):
    arm = arm_spec(tmp_path, count, mixed=mixed, grippers=grippers, permute=permute)
    stage_for_arm(arm, meters_per_unit=meters_per_unit, reverse_slide=reverse_slide)
    return arm


def authored_pose(arm):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(arm.usd_path))
    cache = UsdGeom.XformCache()
    bodies = {prim.GetName(): prim
              for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies())
              if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in (arm.links.base, arm.links.ee)}
    base = np.asarray(cache.GetLocalToWorldTransform(bodies[arm.links.base])).T
    ee = np.asarray(cache.GetLocalToWorldTransform(bodies[arm.links.ee])).T
    expected = np.linalg.inv(base) @ ee
    expected[:3, 3] *= UsdGeom.GetStageMetersPerUnit(stage)
    expected[:3, 3] += expected[:3, :3] @ np.asarray(arm.ik.position_offset_m)
    return expected


def action_for(arm, physical_values, *, gripper_value=.013):
    return np.array([physical_values[int(j.usd[5:])] if j.role == 'arm'
                     else gripper_value + int(j.usd[4:]) * .01 for j in arm.joints], dtype=float)


def pose_tensors(matrix, dtype=torch.float64):
    from isaaclab.utils.math import quat_from_matrix
    tensor = torch.as_tensor(matrix, dtype=dtype)
    return tensor[:3, 3].unsqueeze(0), quat_from_matrix(tensor[:3, :3]).unsqueeze(0)


@pytest.mark.parametrize('count,mixed', [(5, True), (6, False), (7, False)])
def test_fk_matches_authored_frames_at_zero(tmp_path, count, mixed):
    """Authored zero transforms, including nonidentity base: atol 2e-5 m."""
    arm = make_arm(tmp_path, count, mixed=mixed)
    kinematics = ArmKinematics.from_arm(arm)
    pose = kinematics.fk(np.zeros(len(arm.joints)))
    np.testing.assert_allclose(pose, authored_pose(arm), rtol=0, atol=2e-5)
    assert pose.shape == (4, 4) and np.isfinite(pose).all()


def test_so101_fk_matches_authored_frames_at_zero(monkeypatch):
    """Production USD authored frame check: atol 2e-5 m."""
    monkeypatch.delenv('S2R_ARMS_DIR', raising=False)
    arm = load_arm_spec('so101')[0]
    kinematics = ArmKinematics.from_arm(arm)
    pose = kinematics.fk(np.zeros(len(arm.joints)))
    np.testing.assert_allclose(pose, authored_pose(arm), rtol=0, atol=2e-5)
    assert np.linalg.det(pose[:3, :3]) == pytest.approx(1., abs=1e-6)


@pytest.mark.parametrize('meters_per_unit,reverse_slide', [(1., False), (.001, False), (1., True), (.001, True)])
def test_fk_units_and_axis_direction(tmp_path, meters_per_unit, reverse_slide):
    """100 mm prismatic coordinate moves +/-100 mm regardless of stage units; 2e-5 m."""
    arm = make_arm(tmp_path, 5, mixed=True, meters_per_unit=meters_per_unit, reverse_slide=reverse_slide)
    kinematics = ArmKinematics.from_arm(arm)
    zero = kinematics.fk(np.zeros(len(arm.joints)))
    q = action_for(arm, [.1, 0., 0., 0., 0.])
    current = kinematics.fk(q)
    expected = zero.copy()
    expected[0, 3] += -.1 if reverse_slide else .1
    np.testing.assert_allclose(current, expected, atol=2e-5, rtol=0)
    q[[i for i, j in enumerate(arm.joints) if j.role == 'gripper']] = .035
    np.testing.assert_array_equal(current, kinematics.fk(q))


@pytest.mark.parametrize('meters_per_unit', [1., .001])
@pytest.mark.parametrize('count,mixed,grippers', [(5, True, 2), (6, False, 0), (7, False, 1)])
def test_jacobian_matches_fk_differences(tmp_path, count, mixed, grippers, meters_per_unit):
    """Central differences are only an oracle; all six rows agree to 1e-4."""
    arm = make_arm(tmp_path, count, mixed=mixed, grippers=grippers, meters_per_unit=meters_per_unit)
    kinematics = ArmKinematics.from_arm(arm)
    values = [.035 if mixed else .2, -.3, .25, .1, -.2, .15, -.1][:count]
    q = action_for(arm, values)
    original = q.copy()
    current = kinematics.fk(q)
    jacobian = kinematics.jacobian(q)
    oracle = np.zeros((6, len(arm.joints)))
    # 1 mm / 1 mrad is large enough to resolve float32 state storage and small
    # enough that the central-difference truncation error remains below 1e-4.
    epsilon = 1e-3
    for index, joint in enumerate(arm.joints):
        if joint.role != 'arm':
            continue
        step = np.zeros_like(q)
        step[index] = epsilon
        before, after = kinematics.fk(q - step), kinematics.fk(q + step)
        oracle[:3, index] = (after[:3, 3] - before[:3, 3]) / (2 * epsilon)
        angular = ((after[:3, :3] - before[:3, :3]) / (2 * epsilon)) @ current[:3, :3].T
        oracle[3:, index] = [angular[2, 1], angular[0, 2], angular[1, 0]]
    np.testing.assert_allclose(jacobian, oracle, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(q, original)
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            np.testing.assert_array_equal(jacobian[:, i], np.zeros(6))


@pytest.mark.parametrize('method', ['dls', 'adaptive_dls'])
def test_controller_config_maps_manifest_axes_damping_and_limit_avoidance(tmp_path, method):
    arm = arm_spec(tmp_path, 7)
    arm = replace(arm, ik=replace(arm.ik, method=method, orientation_axes=('z', 'x'),
                                 joint_limit_avoidance_gain=.002, joint_limit_avoidance_margin=.17))
    options = IKOptions(damping=.003, orientation_weight=.12)
    for use_orientation in (False, True):
        config = _controller_cfg(arm, options, use_orientation=use_orientation)
        assert config.command_type == ('pose' if use_orientation else 'position')
        assert config.use_relative_mode is False and config.ik_method == method
        assert config.ik_params['lambda_val' if method == 'dls' else 'lambda_min'] == .003
        assert tuple(config.orientation_weight) == (.12, 0., .12)
        assert config.joint_limit_avoidance_gain == .002
        assert config.joint_limit_avoidance_margin == .17


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_step_respects_caps_limits_and_grippers(tmp_path, dtype):
    """Per-joint caps/limits hold to 2e-7; gripper values remain bit-exact."""
    arm = arm_spec(tmp_path, 5, mixed=True, grippers=2)
    options = IKOptions(integral_gain=0.)
    q = torch.as_tensor(np.stack([action_for(arm, [0.] * 5)] * 2), dtype=dtype)
    rev = next(i for i, j in enumerate(arm.joints) if j.usd == 'Joint1')
    slide = next(i for i, j in enumerate(arm.joints) if j.usd == 'Joint0')
    q[1, rev] = math.pi - 1e-4
    jacobian = q.new_zeros((2, 6, len(arm.joints)))
    jacobian[:, 0, rev] = .001
    jacobian[:, 1, slide] = 1.
    position = q.new_zeros((2, 3))
    orientation = q.new_tensor([[0., 0., 0., 1.]]).expand(2, -1)
    result = _JointStep(arm, options, num_envs=2, device='cpu').solve(
        q, position, orientation, jacobian, position + q.new_tensor([1., 1., 0.]))
    assert result.joint_positions.dtype == dtype
    assert result.limited.all() and not result.converged.any()
    assert torch.isfinite(result.joint_positions).all()
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            torch.testing.assert_close(result.joint_positions[:, i], q[:, i], atol=0, rtol=0)
        else:
            cap = options.max_joint_step_rad if joint.type == 'revolute' else options.max_joint_step_m
            assert (abs(result.joint_positions[:, i] - q[:, i]) <= cap + 2e-7).all()
            factor = math.pi / 180 if joint.type == 'revolute' else 1.
            assert (result.joint_positions[:, i] >= joint.limits[0] * factor - 2e-7).all()
            assert (result.joint_positions[:, i] <= joint.limits[1] * factor + 2e-7).all()
    assert result.joint_positions[1, rev].item() == pytest.approx(math.pi, abs=2e-7)
    torch.testing.assert_close(result.position_error, q.new_full((2,), math.sqrt(2.)), atol=2e-7, rtol=0)


def test_five_joint_arm_leaves_undeclared_axis_free(tmp_path):
    """Pure base yaw yields zero motion (1e-10); declared tilt produces motion."""
    arm = arm_spec(tmp_path, 5)
    q = torch.zeros((1, len(arm.joints)), dtype=torch.float64)
    ids = [i for i, joint in enumerate(arm.joints) if joint.role == 'arm']
    jacobian = torch.zeros((1, 6, len(arm.joints)), dtype=q.dtype)
    jacobian[0, 3, ids[0]], jacobian[0, 4, ids[1]], jacobian[0, 5, ids[2]] = 1., 1., 1.
    p, quat = q.new_zeros((1, 3)), q.new_tensor([[0., 0., 0., 1.]])
    step = _JointStep(arm, IKOptions(integral_gain=0.), num_envs=1, device='cpu')
    yaw = q.new_tensor([[0., 0., math.sin(.3), math.cos(.3)]])
    ignored = step.solve(q, p, quat, jacobian, p, yaw)
    torch.testing.assert_close(ignored.joint_positions, q, atol=1e-10, rtol=0)
    assert ignored.converged.item() and ignored.orientation_error.item() == pytest.approx(0., abs=1e-10)
    tilt = q.new_tensor([[math.sin(.1), 0., 0., math.cos(.1)]])
    moved = step.solve(q, p, quat, jacobian, p, tilt)
    assert moved.joint_positions[0, ids[0]] > 1e-4
    assert moved.orientation_error.item() == pytest.approx(.2, abs=1e-7)
    assert not moved.converged.item()


@pytest.mark.parametrize('meters_per_unit', [1., .001])
@pytest.mark.parametrize('count', [6, 7])
def test_six_and_seven_joint_arms_reach_full_pose(tmp_path, count, meters_per_unit):
    """Full-pose command inversion reaches 1e-4 m / 1e-3 rad; grippers stay exact."""
    from isaaclab.utils.math import compute_pose_error
    arm = make_arm(tmp_path, count, grippers=2, meters_per_unit=meters_per_unit)
    kinematics = ArmKinematics.from_arm(arm)
    values = np.array([.2, -.3, .25, .1, -.2, .15, -.1][:count])
    measured = action_for(arm, values)
    desired = action_for(arm, values + np.array([.01, -.008, .006, -.004, .005, -.003, .004][:count]),
                         gripper_value=.002)
    target = kinematics.fk(desired)
    original = measured.copy()
    result = kinematics.inverse(target, measured)
    actual = kinematics.fk(result.joint_positions[0].numpy())
    p, quat = pose_tensors(actual)
    target_p, target_quat = pose_tensors(target)
    pe, re = compute_pose_error(p, quat, target_p, target_quat)
    assert pe.norm().item() < 1e-4
    assert re.norm().item() < 1e-3
    assert result.converged.item()
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            assert result.joint_positions[0, i].item() == measured[i]
    np.testing.assert_array_equal(measured, original)


def test_seven_joint_limit_avoidance_keeps_pose(tmp_path):
    """An active near-limit joint moves inward; TCP translation stays below 1e-4 m."""
    arm = make_arm(tmp_path, 7)
    arm = replace(arm, ik=replace(arm.ik, joint_limit_avoidance_gain=.002,
                                 joint_limit_avoidance_margin=.3))
    kinematics = ArmKinematics.from_arm(arm)
    values = [math.pi - .02, -.3, .25, .1, -.2, .15, -.1]
    measured = action_for(arm, values)
    q = torch.from_numpy(measured[None])
    pose = kinematics.fk(measured)
    p, quat = pose_tensors(pose)
    result = _JointStep(arm, IKOptions(integral_gain=0.), num_envs=1, device='cpu').solve(
        q, p, quat, torch.from_numpy(kinematics.jacobian(measured)[None]), p, quat)
    limit_joint = next(i for i, j in enumerate(arm.joints) if j.usd == 'Joint0')
    assert result.joint_positions[0, limit_joint].item() < measured[limit_joint] - 1e-5
    after = kinematics.fk(result.joint_positions[0].numpy())
    assert np.linalg.norm(after[:3, 3] - pose[:3, 3]) < 1e-4
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            assert result.joint_positions[0, i].item() == measured[i]


def legacy_arm(tmp_path):
    """The old 2-axis, two-gripper CPU adapter/servo contract."""
    base = arm_spec(tmp_path, 5)
    joints = (
        JointSpec('pan', 'Pan', 'revolute', 'arm', (-180., 180.), 0.),
        JointSpec('left', 'Left', 'prismatic', 'gripper', (0., .04), .02),
        JointSpec('slide', 'Slide', 'prismatic', 'arm', (-.2, .2), 0.),
        JointSpec('right', 'Right', 'prismatic', 'gripper', (0., .04), .02),
    )
    return replace(base, joints=joints, ik=IKSpec(),
                   gripper=GripperSpec('parallel', ('left', 'right'), ('left_link', 'right_link'),
                                       ('left_link', 'right_link')))


@pytest.mark.parametrize('fixed_base', [False, True])
def test_live_adapter_offset_permutation_and_base_dofs(tmp_path, fixed_base):
    """A translated/rotated base, TCP shift and joint permutation agree to 2e-7."""
    from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms
    arm = legacy_arm(tmp_path)
    q = torch.tensor([[0., .013, .1, .027]], dtype=torch.float64)
    ids = [2, 3, 1, 0]
    native_q = q[:, [3, 2, 0, 1]].clone()
    base_pos = q.new_tensor([[.3, -.2, .4]])
    base_quat = q.new_tensor([[0., 0., math.sqrt(.5), math.sqrt(.5)]])
    local_pos, local_quat = q.new_tensor([[1.1, 0., 0.]]), q.new_tensor([[0., 0., 0., 1.]])
    world_pos = base_pos + quat_apply(base_quat, local_pos)
    world_quat = quat_mul(base_quat, local_quat)
    local_jac = q.new_zeros((1, 6, 4))
    local_jac[:, 1, 0], local_jac[:, 5, 0], local_jac[:, 0, 2] = 1.1, 1., 1.
    world_jac = torch.cat([quat_apply(base_quat[:, None].expand(-1, 4, -1),
                                    local_jac[:, start:start + 3].transpose(1, 2)).transpose(1, 2)
                           for start in (0, 3)], dim=1)
    base_dofs = 0 if fixed_base else 6
    jacobian_body_id = 0 if fixed_base else 1
    native_jac = q.new_zeros((1, 2, 6, 4 + base_dofs))
    for manifest, native in enumerate(ids):
        native_jac[:, jacobian_body_id, :, base_dofs + native] = world_jac[:, :, manifest]
    data = SimpleNamespace(body_link_pos_w=torch.stack([base_pos, world_pos], dim=1),
        body_link_quat_w=torch.stack([base_quat, world_quat], dim=1),
        joint_pos=native_q, body_link_jacobian_w=native_jac)
    robot = SimpleNamespace(data=data, device='cpu', is_fixed_base=fixed_base, num_base_dofs=base_dofs,
        find_joints=lambda names, preserve_order: (ids, list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    options = IKOptions(integral_gain=0.)
    offset = q.new_tensor([[0., .1, .02]])
    solver = ArticulationIK(robot, arm, position_offset_m=offset[0].tolist(), options=options)
    current, quaternion = solver.current_pose()
    expected_current = world_pos + quat_apply(world_quat, offset)
    torch.testing.assert_close(current, expected_current, atol=1e-12, rtol=0)
    target_world = current + quat_apply(base_quat, q.new_tensor([[.005, .01, 0.]]))
    result = solver.solve(target_world)
    current_local, quat_local = subtract_frame_transforms(base_pos, base_quat, current, quaternion)
    target_local, _ = subtract_frame_transforms(base_pos, base_quat, target_world)
    shifted_local = local_jac.clone()
    shifted_local[:, 0, 0] -= .1
    expected = _JointStep(arm, options, num_envs=1, device='cpu').solve(
        q, current_local, quat_local, shifted_local, target_local)
    torch.testing.assert_close(result.joint_positions, expected.joint_positions, atol=2e-7, rtol=0)
    torch.testing.assert_close(robot.data.joint_pos, native_q, atol=0, rtol=0)


def biased_prismatic_servo(arm, *, gain=1.):
    """Unchanged CPU servo/load model: equilibrium is command minus 2 mm."""
    measured = torch.tensor([[0., .013, 0., .027]], dtype=torch.float64)
    positions = torch.zeros((1, 2, 3), dtype=torch.float64)
    rotations = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]], dtype=torch.float64)
    jacobian = torch.zeros((1, 1, 6, len(arm.joints)), dtype=torch.float64)
    jacobian[0, 0, 0, 2] = 1.
    data = SimpleNamespace(joint_pos=measured, body_link_pos_w=positions,
                            body_link_quat_w=rotations, body_link_jacobian_w=jacobian)
    robot = SimpleNamespace(data=data, device='cpu', is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(len(arm.joints))), list(arm.usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]))
    solver = ArticulationIK(robot, arm, options=IKOptions(integral_gain=gain), control_dt=1 / 30)
    return solver, measured, positions


def test_integral_bias_unchanged(tmp_path):
    """Same 2 mm servo load: feedback removes residual below 1e-5 m."""
    arm = legacy_arm(tmp_path)
    errors = []
    for gain in (0., 1.):
        solver, measured, positions = biased_prismatic_servo(arm, gain=gain)
        for _ in range(360):
            positions[0, 1, 0] = measured[0, 2]
            result = solver.solve([.05, 0., 0.])
            torch.testing.assert_close(result.joint_positions[:, [1, 3]], measured[:, [1, 3]], atol=0, rtol=0)
            assert abs(result.joint_positions[0, 2] - measured[0, 2]) <= .005 + 1e-12
            measured[0, 2] += .35 * (result.joint_positions[0, 2] - measured[0, 2] - .002)
        errors.append(abs(measured[0, 2].item() - .05))
    assert .0019 < errors[0] < .0022
    assert errors[1] < 1e-5 and errors[1] < errors[0] / 100
    assert solver.integral_bias[0, 2] == pytest.approx(.002, abs=1e-5)


def test_integral_preview_reset_and_antiwindup_at_lead_and_joint_limits(tmp_path):
    """Preview/reset/anti-windup retain old behavior; exact zero/stored-bias checks."""
    solver, measured, positions = biased_prismatic_servo(legacy_arm(tmp_path))
    for _ in range(30):
        solver.solve([1., 0., 0.])
    torch.testing.assert_close(solver.integral_bias, torch.zeros_like(measured), atol=0, rtol=0)
    for _ in range(20):
        solver.solve([.001, 0., 0.])
    before = solver.integral_bias
    assert before[0, 2] > 0
    before.zero_()
    assert solver.integral_bias[0, 2] > 0  # The accessor returns a copy.
    before = solver.integral_bias
    for _ in range(10):
        solver.solve([.001, 0., 0.], advance=False)
    torch.testing.assert_close(solver.integral_bias, before, atol=0, rtol=0)
    measured[0, 2] = .2
    positions[0, 1, 0] = .2
    command = solver.solve([1., 0., 0.])
    assert command.joint_positions[0, 2] == .2
    assert solver.integral_bias[0, 2] == 0
    solver.reset()
    assert solver.integral_bias is None
    solver.solve([.1, 0., 0.], advance=False)
    assert solver.integral_bias is None


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
def test_inverse_recovers_saved_source_commands(monkeypatch, measured, target):
    """Saved SO101 lift/fast-arc commands: 1e-5 m, 1e-4 rad, 2e-4 joint rad."""
    monkeypatch.delenv('S2R_ARMS_DIR', raising=False)
    arm = load_arm_spec('so101')[0]
    kinematics = ArmKinematics.from_arm(arm)
    result = kinematics.inverse(kinematics.fk(target), measured)
    arm_ids = [i for i, j in enumerate(arm.joints) if j.role == 'arm']
    grip_ids = [i for i, j in enumerate(arm.joints) if j.role == 'gripper']
    assert result.converged.item()
    assert result.position_error.item() <= 1e-5
    assert result.orientation_error.item() <= 1e-4
    np.testing.assert_allclose(result.joint_positions[0, arm_ids], np.asarray(target)[arm_ids], atol=2e-4, rtol=0)
    np.testing.assert_array_equal(result.joint_positions[0, grip_ids], np.asarray(measured)[grip_ids])


@pytest.mark.parametrize('bad_key,bad_value,error', [
    ('q', torch.zeros((1, 2)), 'IK joint_positions must have finite shape'),
    ('position', [[float('nan'), 0., 0.]], 'position must contain 3 finite values'),
    ('orientation', [[0., 0., 0., 0.]], 'orientation quaternion must be nonzero'),
    ('jacobian', torch.zeros((1, 5, 6)), 'IK jacobian must have finite shape'),
    ('target_position', [[0., float('inf'), 0.]], 'target_position must contain 3 finite values'),
    ('target_orientation', [[0., 0., 0., 0.]], 'target_orientation quaternion must be nonzero'),
])
def test_step_rejects_malformed_inputs(tmp_path, bad_key, bad_value, error):
    arm = arm_spec(tmp_path, 5)
    arguments = dict(q=torch.zeros((1, len(arm.joints)), dtype=torch.float64),
                     position=[[0., 0., 0.]], orientation=[[0., 0., 0., 1.]],
                     jacobian=torch.zeros((1, 6, len(arm.joints)), dtype=torch.float64),
                     target_position=[[.1, 0., 0.]], target_orientation=[[0., 0., 0., 1.]])
    arguments[bad_key] = bad_value
    step = _JointStep(arm, IKOptions(), num_envs=1, device='cpu')
    with pytest.raises(ValueError, match=error):
        step.solve(arguments['q'], arguments['position'], arguments['orientation'], arguments['jacobian'],
                   arguments['target_position'], arguments['target_orientation'])


def test_zero_jacobian_is_finite_without_claiming_success(tmp_path):
    """A rank-zero task cannot move joints or claim convergence; exact motion check."""
    arm = arm_spec(tmp_path, 7)
    q = torch.from_numpy(action_for(arm, [0.] * 7)[None])
    step = _JointStep(arm, IKOptions(), num_envs=1, device='cpu')
    result = step.solve(q, [[0., 0., 0.]], [[0., 0., 0., 1.]],
                        torch.zeros((1, 6, len(arm.joints))), [[1., 1., 1.]])
    torch.testing.assert_close(result.joint_positions, q, atol=0, rtol=0)
    assert torch.isfinite(result.joint_positions).all() and not result.converged.item()


@pytest.mark.parametrize('measured_slide', [0., .199])
def test_inverse_final_dispatch_respects_measured_step_and_limits(tmp_path, measured_slide):
    """After local iterations the physical dispatch window still holds to 1e-12."""
    arm = make_arm(tmp_path, 5, mixed=True, grippers=2)
    kinematics = ArmKinematics.from_arm(arm)
    measured = action_for(arm, [measured_slide, -.3, .25, .1, -.2])
    target = kinematics.fk(action_for(arm, [measured_slide + .15, .2, .35, .3, -.1]))
    result = kinematics.inverse(target, measured, max_iterations=32)
    command = result.joint_positions[0].numpy()
    assert result.limited.item() and not result.converged.item()
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            assert command[i] == measured[i]
            continue
        cap = .08 if joint.type == 'revolute' else .005
        assert abs(command[i] - measured[i]) <= cap + 1e-12
        factor = math.pi / 180 if joint.type == 'revolute' else 1.
        assert joint.limits[0] * factor <= command[i] <= joint.limits[1] * factor
    actual = kinematics.fk(command)
    assert result.position_error.item() == pytest.approx(np.linalg.norm(target[:3, 3] - actual[:3, 3]), abs=2e-7)


def test_inverse_ignores_undeclared_yaw_and_rejects_invalid_inputs(tmp_path):
    """Undeclared base yaw preserves all measured commands (1e-10)."""
    arm = make_arm(tmp_path, 5, mixed=True)
    kinematics = ArmKinematics.from_arm(arm)
    measured = action_for(arm, [0.] * 5)
    target = kinematics.fk(measured)
    target[:3, :3] = np.diag([-1., -1., 1.])
    result = kinematics.inverse(target, measured)
    np.testing.assert_allclose(result.joint_positions[0], measured, atol=1e-10, rtol=0)
    assert result.converged.item()
    for kwargs in ({'max_iterations': 0}, {'max_iterations': True},
                   {'position_tolerance_m': float('nan')}, {'orientation_tolerance_rad': 0.}):
        with pytest.raises(ValueError, match='positive'):
            kinematics.inverse(target, measured, **kwargs)
    malformed = target.copy()
    malformed[0, 0] = 2.
    with pytest.raises(ValueError, match='rigid 4x4'):
        kinematics.inverse(malformed, measured)
    for malformed in (measured[:-1], np.full_like(measured, float('nan'))):
        with pytest.raises(ValueError, match='finite action-order radians/metres'):
            kinematics.fk(malformed)
    outside = measured.copy()
    index = next(i for i, j in enumerate(arm.joints) if j.usd == 'Joint0')
    outside[index] = 1.
    with pytest.raises(ValueError, match='beyond the runtime limit'):
        kinematics.inverse(target, outside)


def test_fk_rejects_missing_declared_and_undeclared_path_joints(tmp_path):
    arm = make_arm(tmp_path, 6)
    missing = replace(arm, joints=(replace(arm.joints[0], usd='MissingJoint'), *arm.joints[1:]))
    with pytest.raises(ValueError, match='joint|Joint'):
        ArmKinematics.from_arm(missing)
    undeclared = replace(arm, joints=tuple(j for j in arm.joints if j.usd != 'Joint1'))
    with pytest.raises(ValueError, match='joint|Joint'):
        ArmKinematics.from_arm(undeclared)


def test_fk_rejects_authored_joint_frame_disagreement(tmp_path):
    """The authored-zero guard catches a 50 mm inconsistent joint frame."""
    from pxr import Gf, Usd, UsdPhysics
    arm = make_arm(tmp_path, 6)
    stage = Usd.Stage.Open(str(arm.usd_path))
    joint = UsdPhysics.Joint(stage.GetPrimAtPath('/Robot/Joint1'))
    original = joint.GetLocalPos0Attr().Get()
    joint.GetLocalPos0Attr().Set(original + Gf.Vec3f(.05, 0., 0.))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match='zero pose disagrees'):
        ArmKinematics.from_arm(arm)


def test_mimic_inverse_preserves_native_permutation_and_two_gripper_roles(tmp_path):
    """Actual Mimic command conversion recovers arm commands to 2e-4 rad."""
    from sim.datagen.mimic import S2RMimicEnv
    arm = make_arm(tmp_path, 6, grippers=2)
    kinematics = ArmKinematics.from_arm(arm)
    values = np.array([.2, -.3, .25, .1, -.2, .15])
    measured = action_for(arm, values)
    desired = action_for(arm, values + [.01, -.008, .006, -.004, .005, -.003])
    native_order = np.arange(len(arm.joints))[::-1]
    native = torch.as_tensor(measured[native_order][None], dtype=torch.float32)
    before = native.clone()
    ids = np.argsort(native_order).tolist()
    ik = SimpleNamespace(options=IKOptions(), joint_ids=ids,
                         robot=SimpleNamespace(data=SimpleNamespace(joint_pos=native)),
                         solve=lambda *args, **kwargs: pytest.fail('Command inversion must not advance integral state'))
    env = SimpleNamespace(num_envs=1, device='cpu', s2r_arm=arm, s2r_ik=ik, s2r_fk=kinematics)
    target = torch.as_tensor(kinematics.fk(desired), dtype=torch.float32)
    grip = torch.tensor([.005, .035])
    action = S2RMimicEnv.target_eef_pose_to_action(env, {'tool': target}, {'tool': grip})
    arm_ids = [i for i, j in enumerate(arm.joints) if j.role == 'arm']
    grip_ids = [i for i, j in enumerate(arm.joints) if j.role == 'gripper']
    torch.testing.assert_close(action[0, grip_ids], grip, atol=0, rtol=0)
    np.testing.assert_allclose(action[0, arm_ids], desired[arm_ids], atol=2e-4, rtol=0)
    torch.testing.assert_close(native, before, atol=0, rtol=0)
    assert action.dtype == native.dtype and action.shape == (1, len(arm.joints))


def test_step_caps_task_commands_before_library_solve(tmp_path):
    """Identity task rows expose position/orientation caps directly, within 2e-7."""
    arm = arm_spec(tmp_path, 6)
    options = IKOptions(damping=1e-4, orientation_weight=1., integral_gain=0.,
                        max_position_step_m=.02, max_orientation_step_rad=.06,
                        max_joint_step_rad=1.)
    q = torch.from_numpy(action_for(arm, [0.] * 6)[None])
    ids = [i for i, joint in enumerate(arm.joints) if joint.role == 'arm']
    jacobian = q.new_zeros((1, 6, len(arm.joints)))
    for row, index in enumerate(ids):
        jacobian[0, row, index] = 1.
    result = _JointStep(arm, options, num_envs=1, device='cpu').solve(
        q, [[0., 0., 0.]], [[0., 0., 0., 1.]], jacobian, [[.3, .4, 0.]],
        [[math.sin(.15), 0., 0., math.cos(.15)]])
    delta = (result.joint_positions - q)[0, ids]
    assert delta[:3].norm().item() == pytest.approx(.02, abs=2e-7)
    assert delta[3:].norm().item() == pytest.approx(.06, abs=2e-7)
    assert result.position_error.item() == pytest.approx(.5, abs=1e-12)
    assert result.orientation_error.item() == pytest.approx(.3, abs=1e-7)
    assert not result.limited.item()  # Only joint-step shrink/clamping sets limited.


@pytest.mark.parametrize('meters_per_unit,reverse_slide', [(1., False), (.001, False), (.001, True)])
def test_inverse_recovers_mixed_joint_command_in_metres(tmp_path, meters_per_unit, reverse_slide):
    """Mixed joints recover command poses in SI units: 1e-4 m / 1e-3 rad."""
    arm = make_arm(tmp_path, 5, mixed=True, grippers=2,
                   meters_per_unit=meters_per_unit, reverse_slide=reverse_slide)
    kinematics = ArmKinematics.from_arm(arm)
    values = np.array([.035, -.3, .25, .1, -.2])
    measured = action_for(arm, values)
    desired = action_for(arm, values + [.003, .006, -.005, .004, .003])
    result = kinematics.inverse(kinematics.fk(desired), measured)
    assert result.converged.item()
    assert result.position_error.item() < 1e-4
    assert result.orientation_error.item() < 1e-3
    for index, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            assert result.joint_positions[0, index].item() == measured[index]


def test_ignored_yaw_at_nonzero_float64_q_stays_within_library_precision(tmp_path):
    """Ignoring yaw adds at most float32 roundoff (1e-7 rad) to arm coordinates."""
    arm = arm_spec(tmp_path, 5)
    q = torch.from_numpy(action_for(arm, [.123456789, -.234567891, .345678912, -.456789123, .567891234])[None])
    jacobian = q.new_zeros((1, 6, len(arm.joints)))
    for row, index in enumerate(i for i, joint in enumerate(arm.joints) if joint.role == 'arm'):
        jacobian[0, row, index] = 1.
    result = _JointStep(arm, IKOptions(), num_envs=1, device='cpu').solve(
        q, [[0., 0., 0.]], [[0., 0., 0., 1.]], jacobian, [[0., 0., 0.]],
        [[0., 0., math.sin(.3), math.cos(.3)]])
    assert result.converged.item()
    torch.testing.assert_close(result.joint_positions, q, atol=1e-7, rtol=0)
    for i, joint in enumerate(arm.joints):
        if joint.role == 'gripper':
            assert result.joint_positions[0, i].item() == q[0, i].item()


@pytest.mark.parametrize('meters_per_unit', [1., .001])
def test_reversed_body_relationship_reports_joint_and_usd_fix(tmp_path, meters_per_unit):
    """Unsupported relationship reversal identifies the joint and required authoring fix."""
    from pxr import Gf, Usd, UsdPhysics

    arm = make_arm(tmp_path, 5, mixed=True, meters_per_unit=meters_per_unit)
    stage = Usd.Stage.Open(str(arm.usd_path))
    # Pin the declared base explicitly so the importer cannot choose the other
    # side of the reversed edge as the articulation's root.
    root_joint = UsdPhysics.FixedJoint.Define(stage, '/Robot/RootJoint')
    root_joint.CreateBody1Rel().SetTargets(['/Robot/base'])
    root_joint.CreateLocalPos0Attr(Gf.Vec3f(.3 / meters_per_unit, -.2 / meters_per_unit,
                                          .4 / meters_per_unit))
    root_joint.CreateLocalRot0Attr(Gf.Quatf(math.cos(math.radians(35.) / 2),
                                           Gf.Vec3f(0., 0., math.sin(math.radians(35.) / 2))))
    joint = UsdPhysics.Joint(stage.GetPrimAtPath('/Robot/Joint0'))
    left, right = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
    position0, position1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
    rotation0, rotation1 = joint.GetLocalRot0Attr().Get(), joint.GetLocalRot1Attr().Get()
    joint.GetBody0Rel().SetTargets(right)
    joint.GetBody1Rel().SetTargets(left)
    joint.GetLocalPos0Attr().Set(position1)
    joint.GetLocalPos1Attr().Set(position0)
    joint.CreateLocalRot0Attr(rotation1)
    joint.CreateLocalRot1Attr(rotation0)
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError) as caught:
        ArmKinematics.from_arm(arm)
    assert 'Joint0' in str(caught.value)
    assert 'swap body0/body1 and the local frames in the USD' in str(caught.value)
