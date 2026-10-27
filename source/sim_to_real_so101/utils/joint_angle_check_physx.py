"""Offline, zero-step replay of follower units into a separate PhysX arm.

No hardware devices are constructed or connected. The production interface is
used only for its tensor mapping, exactly as at recorder start synchronization.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np

from .joint_angle_check import compare_readback, finite_vector, sha256
from .joint_calibration import FOLLOWER_JOINTS, SETUP_PATH
from .joint_limits import USD_JOINTS, apply_runtime_joint_limits, runtime_joint_limits
from .recording_alignment import synced_start_pose


# Axes lie in actual flat faces, expressed in their rigid-link frames. The
# longitudinal direction is proximal-to-distal. These are diagnostic datums,
# not camera/gripper-contact adjustments.
SURFACES = {
    'upper_arm': dict(link='upper_arm', mesh='upper_arm_so101_v1/mesh',
                      normal_local=[0., 1., 0.], plane_offset_m=.012,
                      longitudinal_local=[-1., 0., 0.],
                      physical_face='Broad flat printed cheek between shoulder and elbow; '
                                    'place the phone long edge toward the elbow, avoiding raised screws and the round servo horn.'),
    'forearm': dict(link='lower_arm', mesh='under_arm_so101_v1/mesh',
                    normal_local=[0., -1., 0.], plane_offset_m=.012,
                    longitudinal_local=[-1., 0., 0.],
                    physical_face='Broad flat printed cheek between elbow and wrist; '
                                  'place the phone long edge from elbow toward wrist, avoiding the motor and raised fasteners.'),
    'housing': dict(link='gripper', mesh='wrist_roll_follower_so101_v1/mesh',
                    normal_local=[1., 0., 0.], plane_offset_m=-.015,
                    longitudinal_local=[0., 0., -1.], transverse_local=[0., 1., 0.],
                    physical_face='Flat stationary printed housing wall opposite the moving-jaw hinge, behind the fingers; '
                                  'place the long edge from wrist toward fingertips. For roll, turn the phone edge across '
                                  'the housing toward the moving-jaw hinge. Do not use the movable jaw, finger pad or a protruding part.'),
}


def rotation_xyzw(quaternion):
    x,y,z,w = np.asarray(quaternion, dtype=float)
    scale = x*x + y*y + z*z + w*w
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError('Invalid PhysX quaternion')
    return np.array([[1-2*(y*y+z*z)/scale, 2*(x*y-z*w)/scale, 2*(x*z+y*w)/scale],
                     [2*(x*y+z*w)/scale, 1-2*(x*x+z*z)/scale, 2*(y*z-x*w)/scale],
                     [2*(x*z-y*w)/scale, 2*(y*z+x*w)/scale, 1-2*(x*x+y*y)/scale]])


def expected_tilts(link_rotations):
    """Signed longitudinal elevation above gravity-horizontal, proximal-to-distal.

    The angle is atan2(v_z, |v_xy|), in [-90,90]. This folded gravity reading
    cannot establish pan and is singular for joint-offset inversion at vertical.
    A transverse housing reading by itself also has an arcsine branch ambiguity.
    """
    result = {}
    for name, surface in SURFACES.items():
        rotation = np.asarray(link_rotations[surface['link']])
        axis = rotation @ np.asarray(surface['longitudinal_local'])
        result[name] = dict(world_pitch_degrees=float(np.rad2deg(np.arctan2(axis[2], np.linalg.norm(axis[:2])))),
                            longitudinal_world=axis.tolist(),
                            phone_face_normal_world=(rotation @ surface['normal_local']).tolist())
        if name == 'housing':
            transverse = rotation @ np.asarray(surface['transverse_local'])
            result[name]['transverse_world_tilt_degrees'] = float(np.rad2deg(np.arctan2(transverse[2],np.linalg.norm(transverse[:2]))))
            result[name]['roll_axis_horizontal'] = bool(abs(result[name]['world_pitch_degrees']) <= 5.)
    return result


def check_in_physx(log_path, output_path):
    # Importing this module, asking for help, or summarizing never boots Isaac.
    from isaaclab.app import AppLauncher

    log_path, output_path = Path(log_path), Path(output_path)
    if log_path.resolve() == output_path.resolve():
        raise ValueError('PhysX output must not replace the raw capture')
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No follower states to check')
    states = [finite_vector(row['follower_units']) for row in rows]
    launcher = AppLauncher(headless=True, device='cpu', enable_cameras=False)
    try:
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext
        from sim_to_real_so101.assets.so101 import SO101_CFG
        from sim_to_real_so101.assets.real_setup import SETUP, robot_rotation
        from .lerobot_interface import LeRobotSO101Interface

        sim = SimulationContext(SimulationCfg(device='cpu'))
        cfg = SO101_CFG.copy()
        cfg.prim_path = '/World/JointAngleCheckRobot'
        cfg.init_state.pos = tuple(SETUP['robot']['position'])
        cfg.init_state.rot = robot_rotation()
        cfg.init_state.joint_pos = dict(SETUP['robot']['joint_positions'])
        robot = Articulation(cfg)
        sim.reset()
        print('[joint check] Separate PhysX arm initialized', flush=True)
        apply_runtime_joint_limits(SimpleNamespace(scene={'robot':robot}, device='cpu', num_envs=1))
        interface = LeRobotSO101Interface('cpu', '', '', {}, 30,
                                          joint_mapping=SETUP['sim_joint_mapping'])
        joint_ids = [robot.joint_names.index(name) for name in USD_JOINTS]
        bounds = runtime_joint_limits()
        limits = [[bounds[name]['joint_min'],bounds[name]['joint_max']] for name in FOLLOWER_JOINTS]
        backend_limits = np.rad2deg(robot.root_physx_view.get_dof_limits()[0,joint_ids].cpu().numpy())
        np.testing.assert_allclose(backend_limits, limits, atol=5e-5, rtol=0)
        before_time = float(sim.current_time)
        checks = []
        for index,state in enumerate(states):
            observation = dict(zip(interface.SO101_JOINT_ORDER, state))
            raw = interface.get_raw_actions_tensor(observation)
            mapped = synced_start_pose(interface.get_mapped_actions_vectorized(raw))
            robot.write_joint_state_to_sim(mapped[None,:],torch.zeros_like(mapped)[None,:],joint_ids=joint_ids)
            # Refresh kinematics only. There is deliberately no sim.step(),
            # env.step(), render(), target setter or real motor command here.
            sim.physics_sim_view.update_articulations_kinematic()
            actual = robot.root_physx_view.get_dof_positions()[0,joint_ids].clone()
            check = compare_readback(state, np.rad2deg(mapped.cpu().numpy()),
                                     np.rad2deg(actual.cpu().numpy()), limits)
            poses = robot.root_physx_view.get_link_transforms()[0].cpu().numpy()
            rotations = {name: rotation_xyzw(poses[robot.body_names.index(name),3:7])
                         for name in ('upper_arm','lower_arm','gripper')}
            check.update(source_frame=rows[index].get('frame',index), expected_surface_tilts=expected_tilts(rotations))
            checks.append(check)
        after_time = float(sim.current_time)
        if after_time != before_time:
            raise RuntimeError('Simulation time advanced during the joint-state read-back check')
        result = dict(kind='software mapping and PhysX read-back only; no physical offset estimate',
                      input_sha256=sha256(log_path), setup_sha256=sha256(SETUP_PATH),
                      source_states=len(states), passed=all(row['passed'] for row in checks),
                      simulation_time_before=before_time, simulation_time_after=after_time,
                      physics_steps_during_checks=0,
                      max_abs_difference_degrees=max(row['max_abs_difference_degrees'] for row in checks),
                      joint_order=list(FOLLOWER_JOINTS), runtime_limits_degrees=limits,
                      physx_runtime_limits_degrees=backend_limits.tolist(), surfaces=SURFACES, checks=checks)
        output_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key not in ('checks','surfaces')},indent=2), flush=True)
        return result
    except BaseException:
        # Kit shutdown can itself block after a backend exception. Preserve the
        # original error before cleanup so it remains visible in captured logs.
        traceback.print_exc()
        raise
    finally:
        # This worker owns only in-memory sim state and a completed JSON file.
        # Full stage teardown can block in Isaac 5.1's CPU/SDF cleanup. Use the
        # installed, documented immediate-close API; no render jobs exist here.
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
