"""Acquire a cube from the measured table, lift 50 mm, and hold closed for 30 s.

Uses production physics without physical devices. The --effort option compares
the Jaw cap alone; force and penetration are sampled every physics step.
"""

import argparse
import json
import os
from pathlib import Path
import traceback


def _forward_kinematics(stage, base_matrix, names):
    import numpy as np
    from pxr import UsdPhysics
    from scipy.spatial.transform import Rotation

    def frame(position, rotation):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat([*rotation.GetImaginary(), rotation.GetReal()]).as_matrix()
        matrix[:3, 3] = position
        return matrix

    pending = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)
               and str(prim.GetPath()).startswith("/World/envs/env_0/Robot/")]
    available, chain = {"base"}, []
    while pending:
        progressed = False
        for prim in pending[:]:
            joint = UsdPhysics.RevoluteJoint(prim)
            parent, child = joint.GetBody0Rel().GetTargets()[0].name, joint.GetBody1Rel().GetTargets()[0].name
            if parent not in available:
                continue
            axis = np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())]
            chain.append((parent, child, names.index(prim.GetName()), axis,
                          frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                          np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))))
            available.add(child)
            pending.remove(prim)
            progressed = True
        if not progressed:
            raise ValueError("Cannot resolve the arm joint chain")

    def forward(q):
        poses = {"base": base_matrix}
        for parent, child, index, axis, start, end in chain:
            rotate = np.eye(4)
            rotate[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
            poses[child] = poses[parent] @ start @ rotate @ end
        return poses["gripper"]

    return forward


def _table_and_lift_targets(forward, initial_q, local, plane, names, limits):
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    initial_ee = forward(initial_q)
    normal = np.r_[-plane[:2], 1.0]
    norm = np.linalg.norm(normal)
    normal /= norm
    support = 0.01 * np.abs(normal @ initial_ee[:3, :3]).sum()
    cube_center = initial_ee[:3, 3] + initial_ee[:3, :3] @ local
    distance = normal @ cube_center - plane[2] / norm
    table_ee = initial_ee.copy()
    table_ee[:3, 3] += (support + 0.0002 - distance) * normal
    lifting_ee = table_ee.copy()
    lifting_ee[:3, 3] += 0.05 * normal
    moving = [names.index(name) for name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")]

    def inverse(target, seed):
        def residual(values):
            q = seed.copy()
            q[moving] = values
            pose = forward(q)
            angle = Rotation.from_matrix(target[:3, :3].T @ pose[:3, :3]).as_rotvec()
            return np.r_[pose[:3, 3] - target[:3, 3], 0.001 * angle]
        solution = least_squares(residual, seed[moving], bounds=(limits[moving, 0], limits[moving, 1]),
                                 xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=300)
        q = seed.copy()
        q[moving] = solution.x
        assert np.linalg.norm(forward(q)[:3, 3] - target[:3, 3]) < 2e-4, solution.message
        return q

    table_q = inverse(table_ee, initial_q)
    lift_q = inverse(lifting_ee, table_q)
    return table_q, lift_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--effort", type=float)
    args = parser.parse_args()
    os.environ.update(S2R_ARM="so101", S2R_SCENE="rpd_meeting_room", S2R_TASK="pick_place_cube")
    result = {"passed": False, "seconds": args.seconds}
    app = env = None
    samples = []
    try:
        from sim.app import launch, close_app
        app = launch(headless=True, enable_cameras=False)
        import gymnasium as gym
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_newton.physics import NewtonManager
        import isaaclab.utils.math as math_utils
        import sim.envs as envs
        from sim.envs.mdp import as_torch
        from sim.scene.geometry import gripper_tip_local_offset, table_plane

        env_id = envs.TASK.gym_id("teleop")
        cfg = parse_env_cfg(env_id, device="cuda:0", num_envs=1, use_fabric=True)
        cfg.seed = 101
        cfg.scene.camera_realsense_rgb = cfg.scene.camera_wrist_cam = cfg.scene.realsense_depth = None
        cfg.observations.visual = None
        cfg.sim.visualizer_cfgs = []
        if args.effort is not None:
            cfg.scene.robot.actuators["gripper"].effort_limit_sim = args.effort
        result["effort_limit_nm"] = cfg.scene.robot.actuators["gripper"].effort_limit_sim
        result["impratio"] = cfg.sim.physics.solver_cfg.impratio
        result["cone"] = cfg.sim.physics.solver_cfg.cone
        env = gym.make(env_id, cfg=cfg)
        e = env.unwrapped
        env.reset()
        robot, cube = e.scene["robot"], e.scene["blue_cube"]
        jaw = robot.find_joints("Jaw")[0][0]
        ee = robot.find_bodies("gripper")[0][0]
        initial_jaw = 0.4
        target = as_torch(robot.data.joint_pos).clone()
        target[:, jaw] = initial_jaw
        local = np.asarray(gripper_tip_local_offset(
            e.sim.stage, 0.01, 0.003, "/World/envs/env_0/Robot/gripper"))
        base = robot.find_bodies("base")[0][0]
        base_matrix = np.eye(4)
        base_matrix[:3, :3] = Rotation.from_quat(as_torch(robot.data.body_link_quat_w)[0, base].cpu().numpy()).as_matrix()
        base_matrix[:3, 3] = as_torch(robot.data.body_link_pos_w)[0, base].cpu().numpy()
        names = list(robot.joint_names)
        forward = _forward_kinematics(e.sim.stage, base_matrix, names)
        plane = np.asarray(table_plane(e.sim.stage, e.sim.stage.GetPrimAtPath(e.scene.env_prim_paths[0])))
        table_q, lift_q = _table_and_lift_targets(
            forward, target[0].cpu().numpy(), local, plane, names,
            as_torch(robot.data.joint_pos_limits)[0].cpu().numpy())
        target = torch.as_tensor(table_q, device=e.device).float().unsqueeze(0)
        result.update(table_q_rad=table_q.tolist(), lift_q_rad=lift_q.tolist())
        robot.write_joint_state_to_sim(target, torch.zeros_like(target))
        robot.set_joint_position_target(target)
        e.sim.forward()
        ee_pos = as_torch(robot.data.body_link_pos_w)[:, ee].clone()
        ee_quat = as_torch(robot.data.body_link_quat_w)[:, ee].clone()
        local = ee_pos.new_tensor(local)
        pos = ee_pos + math_utils.quat_apply(ee_quat, local.unsqueeze(0))
        cube.write_root_pose_to_sim_index(root_pose=torch.cat((pos, ee_quat), dim=-1))
        cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=e.device))
        model, solver = NewtonManager.get_model(), NewtonManager._solver
        mj = solver.mj_model
        shape_map = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
        labels = [model.shape_label[int(shape_map[g])] for g in range(mj.ngeom)]
        cube_geoms = {g for g, label in enumerate(labels) if "/CUBE/" in label}
        jaw_geoms = {g for g, label in enumerate(labels) if any(
            part in label for part in ("/Robot/gripper/", "/Robot/jaw/"))}
        joint_id = next(j for j in range(mj.njnt) if "Jaw" in mj.joint(j).name)
        dof = int(mj.jnt_dofadr[joint_id])
        result.update(initial_cube_local_m=local.cpu().tolist(), cube_geoms=sorted(cube_geoms),
                      jaw_geoms=sorted(jaw_geoms), initial_jaw_rad=initial_jaw)
        result["shape_contact_stiffness"] = [
            {"label": labels[g], "kf": float(model.shape_material_kf.numpy()[shape_map[g]])}
            for g in sorted(cube_geoms | jaw_geoms)]
        result["initial_pose"] = {
            "gripper_world": ee_pos[0].cpu().tolist(),
            "jaw_world": as_torch(robot.data.body_link_pos_w)[0, robot.find_bodies("jaw")[0][0]].cpu().tolist(),
            "cube_world": as_torch(cube.data.root_pos_w)[0].cpu().tolist(),
            "cube_geom_size": [mj.geom_size[g].tolist() for g in cube_geoms],
            "cube_mass": [float(mj.body_mass[mj.geom_bodyid[g]]) for g in cube_geoms],
        }
        result["initial_contacts"] = []
        result["allocation"] = {"configured_njmax": cfg.sim.physics.solver_cfg.njmax,
                                "efc_force_shape": list(solver.mjw_data.efc.force.shape),
                                "contact_address_shape": list(solver.mjw_data.contact.efc_address.shape)}
        import carb
        result["render_settings"] = {key: carb.settings.get_settings().get(key) for key in
                                     ("/UJITSO/geometry", "/rtx-transient/hydra/geometrystreaming/active")}
        peak_force = peak_contact_force = peak_torque = max_penetration = steady_penetration = 0.0
        min_contact_steps = 0
        max_nefc = max_nacon = inactive_penetrating = 0
        acquisition_steps = round(4.0 / e.physics_dt)
        steps = acquisition_steps + round(args.seconds / e.physics_dt)
        with torch.inference_mode():
            for step in range(steps):
                time_s = step * e.physics_dt
                fraction = min(1.0, max(0.0, (time_s - 2.0) / 2.0))
                command = table_q + fraction * (lift_q - table_q)
                command[jaw] = initial_jaw + min(1.0, time_s) * (-np.pi / 12 - initial_jaw)
                robot.set_joint_position_target(torch.as_tensor(command, device=e.device).float().unsqueeze(0))
                e.scene.write_data_to_sim()
                e.sim.step(render=False)
                e.scene.update(e.physics_dt)
                contacts = solver.mjw_data.contact
                count = int(solver.mjw_data.nacon.numpy()[0])
                nefc = int(solver.mjw_data.nefc.numpy()[0])
                max_nefc, max_nacon = max(max_nefc, nefc), max(max_nacon, count)
                geoms = contacts.geom.numpy()[:count]
                distances = contacts.dist.numpy()[:count]
                addresses = contacts.efc_address.numpy()[:count, 0]
                forces = solver.mjw_data.efc.force.numpy()[0]
                selected = [i for i, pair in enumerate(geoms)
                            if ((int(pair[0]) in cube_geoms and int(pair[1]) in jaw_geoms) or
                                (int(pair[1]) in cube_geoms and int(pair[0]) in jaw_geoms))]
                normal_forces = [max(0.0, float(forces[addresses[i]])) for i in selected if addresses[i] >= 0]
                normal = sum(normal_forces)
                penetration = max([max(0.0, -float(distances[i])) for i in selected], default=0.0)
                inactive_penetrating += sum(distances[i] < -1e-5 and addresses[i] < 0 for i in selected)
                if step < 4:
                    result["initial_contacts"].append({"step": step, "nacon": count, "nefc": nefc, "contacts": [
                        {"labels": [labels[int(g)] for g in geoms[i]], "distance": float(distances[i]),
                         "address": int(addresses[i]),
                         "addresses": contacts.efc_address.numpy()[i].tolist(),
                         "dim": int(contacts.dim.numpy()[i]),
                         "type": int(contacts.type.numpy()[i]),
                         "friction": contacts.friction.numpy()[i].tolist(),
                         "includemargin": float(contacts.includemargin.numpy()[i]),
                         "solref": contacts.solref.numpy()[i].tolist(),
                         "solreffriction": contacts.solreffriction.numpy()[i].tolist(),
                         "solimp": contacts.solimp.numpy()[i].tolist(),
                         "normal": contacts.frame.numpy()[i, 0].tolist(),
                         "force": float(forces[addresses[i]]) if addresses[i] >= 0 else None}
                        for i in selected]})
                torque = abs(float(solver.mjw_data.qfrc_actuator.numpy()[0, dof]))
                peak_force = max(peak_force, normal)
                peak_contact_force = max(peak_contact_force, max(normal_forces, default=0.0))
                peak_torque = max(peak_torque, torque)
                max_penetration = max(max_penetration, penetration)
                if step >= acquisition_steps:
                    steady_penetration = max(steady_penetration, penetration)
                    min_contact_steps += int(normal > 0.01)
                relative_pos = math_utils.quat_apply_inverse(
                    as_torch(robot.data.body_link_quat_w)[:, ee],
                    as_torch(cube.data.root_pos_w) - as_torch(robot.data.body_link_pos_w)[:, ee])[0]
                samples.append([float(step + 1) * e.physics_dt, *relative_pos.cpu().tolist(),
                                normal, penetration, torque, float(as_torch(robot.data.joint_pos)[0, jaw])])
        array = np.asarray(samples)
        retained = bool(np.linalg.norm(array[-1, 1:4] - local.cpu().numpy()) < 0.02 and array[-1, 4] > 0.01)
        result.update(peak_cube_jaw_normal_force_n=peak_force, peak_single_contact_normal_force_n=peak_contact_force,
                      peak_jaw_torque_nm=peak_torque, max_penetration_mm=max_penetration * 1000,
                      steady_max_penetration_mm=steady_penetration * 1000,
                      final_cube_local_m=array[-1, 1:4].tolist(), final_jaw_rad=float(array[-1, 7]),
                      retained=retained, contact_steps_after_2s=min_contact_steps,
                      simulated_seconds=float(array[-1, 0]), hold_seconds=args.seconds)
        result.update(max_nefc=max_nefc, max_nacon=max_nacon,
                      inactive_penetrating_cube_contacts=int(inactive_penetrating))
        result["passed"] = retained and steady_penetration < 0.001 and min_contact_steps >= steps - acquisition_steps - 1
    except BaseException:
        result["exception"] = traceback.format_exc()
    finally:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if samples:
                import numpy as np
                np.save(str(args.output) + ".samples.npy", np.asarray(samples))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("GRASP_CHECK " + json.dumps(result), flush=True)
            if env is not None:
                env.close()
        finally:
            if app is not None:
                close_app(app, success=result["passed"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
