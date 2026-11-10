"""Replay recorded commands with a free 8 g cube; save physics evidence, not a posed success.

Only frame zero is initialized from the recording. Subsequent robot and cube
states come from dynamics. No devices are opened and no datasets are rewritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/datasets/test'))
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--arm-state-root', type=Path,
                        help='Diagnostic: drive the five arm joints from recorded follower states; jaw still uses original commands')
    parser.add_argument('--prescribe-arm-states', action='store_true',
                        help='Diagnostic only: prescribe measured five-joint motion at physics ticks; jaw and cube remain dynamic')
    parser.add_argument('--robot-base-z-offset', type=float, default=0.,
                        help='Diagnostic mount correction in metres; recorded initial cube WORLD position stays unchanged')
    parser.add_argument('--robot-base-offset', type=float, nargs=3, default=(0., 0., 0.),
                        help='Diagnostic mount XYZ correction; initial cube WORLD position stays unchanged')
    parser.add_argument('--source-base-world-pose', type=float, nargs=7,
                        help='Recorded base XYZ/XYZW pose for replaying a dataset after mount calibration changes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--geometry', choices=('production', 'split_jaw', 'pads'), default='production')
    parser.add_argument('--friction', type=float)
    parser.add_argument('--condim', type=int, choices=(3, 4, 6))
    parser.add_argument('--substeps', type=int)
    parser.add_argument('--physics-dt', type=float)
    parser.add_argument('--effort', type=float)
    parser.add_argument('--stiffness', type=float)
    parser.add_argument('--damping', type=float)
    parser.add_argument('--armature', type=float)
    parser.add_argument('--solref', type=float, nargs=2)
    parser.add_argument('--solimp', type=float, nargs=5)
    parser.add_argument('--native-contacts', action='store_true')
    parser.add_argument('--impratio', type=float)
    parser.add_argument('--rolling-friction', type=float)
    parser.add_argument('--solver-tolerance', type=float)
    parser.add_argument('--hold-seconds', type=float, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument("--unit", default="rpd_follower", help="Calibration unit for the recorded fixture")
    args = parser.parse_args()
    if args.prescribe_arm_states and args.arm_state_root is None:
        parser.error('--prescribe-arm-states requires --arm-state-root')
    os.environ.update(S2R_ARM='so101', S2R_SCENE='rpd_meeting_room', S2R_TASK='pick_place_cube')
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'completed': False, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()}}
    app = env = None
    frames, contacts_log, contact_trace = [], [], []
    try:
        import numpy as np
        import pyarrow.parquet as pq

        data_paths = sorted(args.root.glob('data/*/*.parquet'))
        tables = [pq.read_table(path).to_pandas() for path in data_paths]
        import pandas as pd
        table = pd.concat(tables, ignore_index=True)
        table = table[table.episode_index == args.episode].sort_values('frame_index')
        states = np.stack(table['observation.state']).astype(np.float32)
        actions = np.stack(table['action']).astype(np.float32)
        from core.datasets.sources import load_sidecar
        from core.datasets.task_meta import sidecar_path
        poses = load_sidecar(args.root, args.episode)
        assert len(poses) == len(actions) == len(states)
        end = args.end_frame or len(actions)
        states, actions, poses = (x[args.start_frame:end] for x in (states, actions, poses))
        assert len(actions) > 1
        info = json.loads((args.root / 'meta/info.json').read_text())
        fps = info['fps']
        alignment = json.loads((args.root / 'meta/recording_alignment.json').read_text())
        result['source'] = {'root': str(args.root), 'episode': args.episode, 'frames': len(actions),
                            'fps': fps, 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in data_paths + [sidecar_path(args.root, args.episode)]}}

        from sim.app import launch, close_app
        app = launch(headless=True, enable_cameras=False)
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_newton.physics import NewtonManager
        import isaaclab.utils.math as math_utils
        import sim.scene.contact_materials as materials

        if args.friction is not None:
            materials.GRASP_FRICTION = args.friction
        if args.rolling_friction is not None:
            from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonMaterialCfg
            original_material = materials.contact_material

            def material(friction=materials.CONTACT_FRICTION, **kwargs):
                original = original_material(friction, **kwargs)
                fragments = list(original) if isinstance(original, (list, tuple)) else [original]
                return fragments + [NewtonMaterialCfg(rolling_friction=args.rolling_friction)]

            materials.contact_material = material
        original_properties = materials.contact_properties

        def properties():
            values = original_properties()
            if args.condim is not None:
                values[1].condim = args.condim
            if args.solref is not None:
                values[1].solref = tuple(args.solref)
            if args.solimp is not None:
                values[1].solimp = tuple(args.solimp)
            return values

        materials.contact_properties = properties
        if args.geometry != 'production':
            from grasp_collision_candidates import apply_moving_jaw_split
            if args.geometry == 'pads':
                from grasp_collision_candidates import apply_primitive_pads
            original_enable = materials.enable_collision_meshes

            def enable(root):
                authored = list(original_enable(root))
                replace = apply_moving_jaw_split if args.geometry == 'split_jaw' else apply_primitive_pads
                replaced, created = replace(root)
                for path in ([replaced] if isinstance(replaced, str) else (replaced or [])):
                    if path in authored:
                        authored.remove(path)
                return tuple(authored + list(created))

            materials.enable_collision_meshes = enable

        import sim.envs as envs
        from sim.envs.mdp import as_torch
        from sim.scene.geometry import table_plane
        from core.arms.mapping import JointMapping
        from core.arms.calibration import resolve_calibration
        from core.arms.spec import load_arm_spec
        arm, _ = load_arm_spec()
        selection = resolve_calibration(arm, unit=args.unit)
        assert selection.provenance['follower_calibration']['sha256'] == alignment['selected_calibrations']['follower']
        mapping = JointMapping.from_arm(arm, selection, device='cuda:0')
        q_source = mapping.to_sim(states)
        q_targets = mapping.to_sim(actions)
        if args.arm_state_root is not None:
            measured_paths = sorted(args.arm_state_root.glob('data/*/*.parquet'))
            measured = pd.concat([pq.read_table(p).to_pandas() for p in measured_paths], ignore_index=True)
            measured = measured[measured.episode_index == args.episode].sort_values('frame_index')
            actual = np.stack(measured['observation.state']).astype(np.float32)[args.start_frame:end]
            assert len(actual) == len(actions)
            measured_q = mapping.to_sim(actual)
            q_targets[:, :5] = measured_q[:, :5]
            q_source[0, :5] = measured_q[0, :5]
            result['source']['arm_state_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in measured_paths}
            result['source']['arm_motion'] = 'Measured real follower states used as five arm position targets; original jaw commands'
            if args.prescribe_arm_states:
                result['source']['arm_motion'] = 'DIAGNOSTIC prescribed measured real five-joint motion; jaw/cube dynamic; not original-command replay'
        else:
            result['source']['arm_motion'] = 'Original recorded simulation action targets, all six joints'
        env_id = envs.TASK.gym_id('teleop')
        cfg = parse_env_cfg(env_id, device='cuda:0', num_envs=1, use_fabric=True)
        cfg.seed = 101
        cfg.sim_joint_mapping = selection.mapping
        initial_base = tuple(cfg.scene.robot.init_state.pos)
        base_offset = np.asarray(args.robot_base_offset) + [0., 0., args.robot_base_z_offset]
        cfg.scene.robot.init_state.pos = tuple(np.asarray(initial_base) + base_offset)
        cfg.scene.camera_realsense_rgb = cfg.scene.camera_wrist_cam = cfg.scene.realsense_depth = None
        cfg.observations.visual = None
        cfg.sim.visualizer_cfgs = []
        assert abs(cfg.scene.blue_cube.spawn.mass_props.mass - 0.008) < 1e-12
        if args.physics_dt is not None:
            cfg.sim.dt = args.physics_dt
        cfg.decimation = round(1 / (fps * cfg.sim.dt))
        assert abs(cfg.decimation * cfg.sim.dt - 1/fps) < 1e-9
        cfg.sim.render_interval = cfg.decimation
        for attribute, value in [('effort_limit_sim', args.effort), ('stiffness', args.stiffness),
                                 ('damping', args.damping), ('armature', args.armature)]:
            if value is not None:
                setattr(cfg.scene.robot.actuators['gripper'], attribute, value)
        if args.substeps is not None:
            cfg.sim.physics.num_substeps = args.substeps
        if args.impratio is not None:
            cfg.sim.physics.solver_cfg.impratio = args.impratio
        if args.solver_tolerance is not None:
            cfg.sim.physics.solver_cfg.tolerance = args.solver_tolerance
        if args.native_contacts:
            cfg.sim.physics.solver_cfg.use_mujoco_contacts = True
        env = gym.make(env_id, cfg=cfg)
        e = env.unwrapped
        env.reset()
        robot, cube = e.scene['robot'], e.scene['blue_cube']
        base = robot.find_bodies('base')[0][0]
        ee = robot.find_bodies('gripper')[0][0]
        q0 = q_source[:1].clone()
        # The episode starts at a user-triggered spawn. Its initial velocity is
        # not recorded; zero is explicit, rather than inferring future impulses.
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        e.sim.forward()
        e.scene.update(0.0)
        base_pos = as_torch(robot.data.body_link_pos_w)[:, base].clone()
        base_quat = as_torch(robot.data.body_link_quat_w)[:, base].clone()
        # Sidecar coordinates refer to the recorded base, not a trial mount.
        source_base_pos = base_pos.clone()
        source_base_pos -= source_base_pos.new_tensor(base_offset)
        source_base_quat = base_quat
        if args.source_base_world_pose is not None:
            source_base_pos = base_pos.new_tensor(args.source_base_world_pose[:3]).reshape(1, 3)
            source_base_quat = base_quat.new_tensor(args.source_base_world_pose[3:]).reshape(1, 4)
        result['source']['base_world_pose_xyzw'] = torch.cat((source_base_pos, source_base_quat), dim=-1)[0].cpu().tolist()
        pose = torch.tensor(poses[:1], dtype=torch.float32, device=e.device)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            source_base_pos, source_base_quat, pose[:, :3], torch.cat((pose[:, 4:7], pose[:, 3:4]), dim=-1))
        cube.write_root_pose_to_sim_index(root_pose=torch.cat((pos_w, quat_w), dim=-1))
        cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=e.device))
        e.sim.forward()
        e.scene.update(0.0)
        model, solver = NewtonManager.get_model(), NewtonManager._solver
        mj = solver.mj_model
        shape_map = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
        labels = [model.shape_label[int(shape_map[g])] for g in range(mj.ngeom)]
        cube_geoms = {g for g, label in enumerate(labels) if '/CUBE/' in label}
        jaw_geoms = {g for g, label in enumerate(labels) if any(p in label for p in ('/Robot/jaw/', '/Robot/gripper/'))}
        table_geoms = {g for g, label in enumerate(labels) if 'tabletop' in label.lower()}
        jaw_joint = next(j for j in range(mj.njnt) if 'Jaw' in mj.joint(j).name)
        dof = int(mj.jnt_dofadr[jaw_joint])
        mass = [float(mj.body_mass[mj.geom_bodyid[g]]) for g in cube_geoms]
        assert len(mass) == 1 and abs(mass[0] - 0.008) < 1e-8, mass
        plane = np.asarray(table_plane(e.sim.stage, e.sim.stage.GetPrimAtPath(e.scene.env_prim_paths[0])))
        result['resolved'] = {
            'cube_mass_kg': mass, 'cube_inertia': [mj.body_inertia[mj.geom_bodyid[g]].tolist() for g in cube_geoms],
            'physics_dt': e.physics_dt, 'substeps': cfg.sim.physics.num_substeps,
            'solver_dt': e.physics_dt / cfg.sim.physics.num_substeps, 'table_plane': plane.tolist(),
            'jaw_armature': float(mj.dof_armature[dof]),
            'jaw_force_range': mj.jnt_actfrcrange[jaw_joint].tolist(),
            'joint_names': robot.joint_names,
            'contacts': [{'label': labels[g], 'type': int(mj.geom_type[g]), 'size': mj.geom_size[g].tolist(),
                          'friction': mj.geom_friction[g].tolist(), 'condim': int(mj.geom_condim[g]),
                          'solref': mj.geom_solref[g].tolist(), 'solimp': mj.geom_solimp[g].tolist()}
                         for g in sorted(cube_geoms | jaw_geoms)],
            'initial_velocity_assumption': 'zero; recording stores positions but no velocities',
            'mapping': selection.mapping,
            'cube_initial_world_m': pos_w[0].cpu().tolist(),
            'robot_base_position_m': base_pos[0].cpu().tolist(),
        }
        print('REPLAY_INIT ' + json.dumps(result['resolved']), flush=True)
        steps = len(actions) + round(args.hold_seconds * fps)
        with torch.inference_mode():
            for index in range(steps):
                pos = as_torch(cube.data.root_pos_w)[0].clone()
                quat = as_torch(cube.data.root_quat_w)[0].clone()
                ee_pos = as_torch(robot.data.body_link_pos_w)[0, ee].clone()
                ee_quat = as_torch(robot.data.body_link_quat_w)[0, ee].clone()
                local = math_utils.quat_apply_inverse(ee_quat[None], (pos-ee_pos)[None])[0]
                base_pose = torch.cat((as_torch(robot.data.body_link_pos_w)[0, base],
                                       as_torch(robot.data.body_link_quat_w)[0, base]))
                q = as_torch(robot.data.joint_pos)[0].clone()
                frames.append(np.r_[index / fps, q.cpu().numpy(), pos.cpu().numpy(), quat.cpu().numpy(),
                                    base_pose.cpu().numpy(), local.cpu().numpy(), ee_pos.cpu().numpy()])
                command = q_targets[min(index, len(actions)-1):min(index, len(actions)-1)+1]
                robot.set_joint_position_target(command)
                peak_pen = peak_force = peak_torque = 0.0
                contact_count = 0
                for tick in range(cfg.decimation):
                    if args.prescribe_arm_states:
                        current = min(index, len(measured_q)-1)
                        following = min(current+1, len(measured_q)-1)
                        fraction = tick / cfg.decimation
                        arm_q = (measured_q[current, :5] * (1-fraction)
                                 + measured_q[following, :5] * fraction).unsqueeze(0)
                        arm_v = ((measured_q[following, :5] - measured_q[current, :5]) * fps).unsqueeze(0)
                        robot.write_joint_state_to_sim_index(position=arm_q, velocity=arm_v,
                                                            joint_ids=[0, 1, 2, 3, 4])
                        # Reapply the recorded dynamic-jaw command after state writes.
                        robot.set_joint_position_target(command)
                    e.scene.write_data_to_sim()
                    e.sim.step(render=False)
                    e.scene.update(e.physics_dt)
                    contacts = solver.mjw_data.contact
                    count = int(solver.mjw_data.nacon.numpy()[0])
                    geoms = contacts.geom.numpy()[:count]
                    distances = contacts.dist.numpy()[:count]
                    addresses = contacts.efc_address.numpy()[:count, 0]
                    forces = solver.mjw_data.efc.force.numpy()[0]
                    selected = [i for i, pair in enumerate(geoms)
                                if ((int(pair[0]) in cube_geoms and int(pair[1]) in jaw_geoms)
                                    or (int(pair[1]) in cube_geoms and int(pair[0]) in jaw_geoms))]
                    normal = sum(max(0., float(forces[addresses[i]])) for i in selected if addresses[i] >= 0)
                    table_contacts = [i for i, pair in enumerate(geoms)
                                      if ((int(pair[0]) in table_geoms and int(pair[1]) in jaw_geoms)
                                          or (int(pair[1]) in table_geoms and int(pair[0]) in jaw_geoms))]
                    if 180 <= index <= 215 and (selected or table_contacts):
                        dims = contacts.dim.numpy()
                        bases = contacts.frame.numpy()
                        frictions = contacts.friction.numpy()
                        for i in selected + table_contacts:
                            address, dim = int(addresses[i]), int(dims[i])
                            contact_trace.append({'frame': index, 'physics_tick': tick,
                                'interaction': 'cube_gripper' if i in selected else 'table_gripper',
                                'geom_labels': [labels[int(g)] for g in geoms[i]],
                                'distance_m': float(distances[i]), 'dim': dim,
                                'basis': bases[i].tolist(), 'friction': frictions[i].tolist(),
                                'efc_address': address,
                                'constraint_force': forces[address:address+dim].tolist() if address >= 0 else []})
                    penetration = max((max(0., -float(distances[i])) for i in selected), default=0.)
                    peak_pen, peak_force = max(peak_pen, penetration), max(peak_force, normal)
                    peak_torque = max(peak_torque, abs(float(solver.mjw_data.qfrc_actuator.numpy()[0, dof])))
                    contact_count = max(contact_count, len(selected))
                contacts_log.append([peak_pen, peak_force, peak_torque, contact_count])
                if index % 60 == 0:
                    print(f'REPLAY_FRAME {index}/{steps} cube={pos.cpu().tolist()} pen_mm={peak_pen*1000:.4f} force={peak_force:.4f}', flush=True)
        array, diagnostics = np.asarray(frames), np.asarray(contacts_log)
        position = array[:, 7:10]
        height = position[:, 2] - position[:, :2] @ plane[:2] - plane[2]
        speed = np.linalg.norm(np.diff(position, axis=0), axis=1) * fps
        q_error = np.rad2deg(array[:len(q_source), 1:7] - q_source.cpu().numpy())
        result.update(completed=True, frames=len(array),
                      max_cube_speed_m_s=float(speed.max()), max_cube_table_center_height_m=float(height.max()),
                      final_cube_table_center_height_m=float(height[-1]),
                      final_cube_local_m=array[-1, 21:24].tolist(),
                      max_penetration_mm=float(diagnostics[:, 0].max()*1000),
                      peak_normal_force_n=float(diagnostics[:, 1].max()),
                      peak_jaw_torque_nm=float(diagnostics[:, 2].max()),
                      mean_abs_joint_error_deg=np.abs(q_error).mean(axis=0).tolist())
        raw = np.vstack((actions, np.repeat(actions[-1:], max(0, steps-len(actions)), axis=0)))
        np.savez_compressed(args.output / 'trajectory.npz', time_s=array[:, 0],
                            joint_pos_rad=array[:, 1:7], cube_pose_world_xyzw=array[:, 7:14],
                            robot_base_pose_world_xyzw=array[:, 14:21], cube_local_gripper=array[:, 21:24],
                            gripper_pos_world=array[:, 24:27], action_raw=raw,
                            penetration_m=diagnostics[:, 0], normal_force_n=diagnostics[:, 1],
                            jaw_torque_nm=diagnostics[:, 2], contact_count=diagnostics[:, 3])
    except BaseException:
        result['exception'] = traceback.format_exc()
    finally:
        try:
            if env is not None:
                try:
                    env.close()
                except BaseException:
                    result['cleanup_exception'] = traceback.format_exc()
            (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            (args.output / 'contact_trace.json').write_text(json.dumps(contact_trace) + '\n')
            print('REPLAY_RESULT ' + json.dumps(result), flush=True)
        finally:
            if app is not None:
                close_app(app, success=result.get("completed", False) and not result.get("exception") and not result.get("cleanup_exception"))
    return 0 if result.get('completed') and not result.get('exception') and not result.get('cleanup_exception') else 1


if __name__ == '__main__':
    raise SystemExit(main())
