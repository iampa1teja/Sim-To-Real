"""The unified solver matches frozen Phase-8 IK on measured multi-step histories.

Run in s2r-sim: s2r-entrypoint python -m pytest tests/sim/test_ik_oracle.py -q
The old SP7 hook and bowls' manual swap execute against the frozen modules, not the new factories.
"""
from dataclasses import fields
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

pytest.importorskip('newton')
pytest.importorskip('isaaclab')

from core.tasks.task import load_task
from utils.arms.arms import load_arm_spec
from sim.actions import ik
from sim.datagen.scripted import generation_ik_options

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'legacy_ik'
_KINEMATICS = {}


def kinematics(module, arm, offset):
    # Keep native models alive across cases; repeatedly importing the same CAD meshes
    # is unnecessary for a solver oracle and can crash the SDK mesh triangulator.
    key = module.__name__, arm.name, tuple(offset)
    if key not in _KINEMATICS:
        _KINEMATICS[key] = module.ArmKinematics.from_arm(arm, position_offset_m=offset)
    return _KINEMATICS[key]


def fixture_module(name):
    key = '_s2r_legacy_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, FIXTURES / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def robot(arm):
    count = len(arm.action_joints)
    return SimpleNamespace(device='cpu', is_fixed_base=True, num_base_dofs=0,
        find_joints=lambda names, preserve_order: (list(range(count)), list(arm.action_usd_joint_names)),
        find_bodies=lambda name: ([1 if name == arm.links.ee else 0], [name]),
        data=SimpleNamespace(joint_pos=torch.zeros(1, count), body_link_pos_w=torch.zeros(1, 2, 3),
            body_link_quat_w=torch.zeros(1, 2, 4), body_link_jacobian_w=torch.zeros(1, 1, 6, count)))


def measure(robot, kin, q, dtype):
    pose = kin.fk(q)
    robot.data.joint_pos = torch.tensor(q[None], dtype=dtype)
    robot.data.body_link_pos_w = torch.tensor(np.stack((np.zeros(3), pose[:3, 3]))[None], dtype=dtype)
    robot.data.body_link_quat_w = torch.tensor(np.stack(([0., 0., 0., 1.],
        Rotation.from_matrix(pose[:3, :3]).as_quat()))[None], dtype=dtype)
    robot.data.body_link_jacobian_w = torch.tensor(kin.jacobian(q)[None, None], dtype=dtype)


@pytest.mark.parametrize('arm_name,task_name', [
    ('so101', 'pick_place_cube'), ('so101', 'stack_bowls'), ('sp7', 'pick_place_cube'),
])
@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_live_commands_match_frozen_ik(monkeypatch, arm_name, task_name, seed, dtype):
    arm = load_arm_spec(arm_name)[0]
    task = load_task(task_name)
    old = fixture_module('ik')
    bounded = fixture_module('bounded_ik')
    old_sp7 = fixture_module('sp7_utils')
    options = generation_ik_options(task, SimpleNamespace(ik_integral_gain=float(seed % 2)))
    # These constants are the pre-unification datagen YAML override and SP7 hook semantics.
    offset = (-.0059, -.0002249, -.0994253) if task_name == 'stack_bowls' else arm.ik.position_offset_m
    legacy_options = old.IKOptions(**vars(options))
    old_robot, new_robot = robot(arm), robot(arm)
    if arm_name == 'sp7':
        old_sp7._KINEMATICS[(str(arm.usd_path), tuple(offset))] = kinematics(old, arm, offset)
    monkeypatch.setattr(ik, 'make_kinematics', lambda arm, position_offset_m=None:
        kinematics(ik, arm, arm.ik.position_offset_m if position_offset_m is None else position_offset_m))
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'sim.actions.ik', old)
        patch.setitem(sys.modules, 'sim.actions.bounded_ik', bounded)
        if arm_name == 'sp7':
            original = old_sp7.make_ik(old_robot, arm, position_offset_m=offset, options=legacy_options)
        else:
            original = old.ArticulationIK(old_robot, arm, position_offset_m=offset, options=legacy_options)
            if task_name == 'stack_bowls':
                original._step = bounded.BoundedPoseStep(arm, original.options,
                    kinematics(old, arm, offset))
    expert = task.spec.raw.get('expert', {})
    unified = ik.make_ik(new_robot, arm, position_offset_m=expert.get('control_offset'),
                         solver=expert.get('ik_solver'), options=options)
    assert unified.solver == ('dls' if task_name == 'pick_place_cube' and arm_name == 'so101' else 'bounded')
    np.testing.assert_array_equal(unified.position_offset_m, original.position_offset_m)
    old_fk = kinematics(old, arm, (0., 0., 0.))
    target_fk = kinematics(old, arm, offset)
    rng = np.random.default_rng(seed)
    q = np.array([joint.initial for joint in arm.action_joints])
    arm_ids = [i for i, joint in enumerate(arm.action_joints) if joint.role == 'arm']
    for step in range(20):
        measure(old_robot, old_fk, q, dtype)
        measure(new_robot, old_fk, q, dtype)
        requested = q.copy()
        requested[arm_ids] += rng.uniform(-.08, .08, len(arm_ids))
        for i in arm_ids:
            requested[i] = np.clip(requested[i], *np.deg2rad(arm.action_joints[i].limits))
        target = target_fk.fk(requested)
        position = torch.tensor(target[:3, 3], dtype=dtype)
        orientation = None if step % 7 == 0 else torch.tensor(Rotation.from_matrix(target[:3, :3]).as_quat(), dtype=dtype)
        for advance in (False, True):
            expected = original.solve(position, orientation, advance=advance)
            actual = unified.solve(position, orientation, advance=advance)
            for field in fields(expected):
                assert torch.equal(getattr(actual, field.name), getattr(expected, field.name)), (step, field.name)
            if original.integral_bias is not None:
                assert torch.equal(unified.integral_bias, original.integral_bias)
        q += .65 * (expected.joint_positions[0].numpy() - q)
        if step == 10:
            original.reset()
            unified.reset()
            assert unified.integral_bias is original.integral_bias is None
