"""Ready-pose proposals through the existing library IK/FK adapter."""
import math

import numpy as np


def bounds(spec):
    return np.asarray([(math.radians(j.limits[0]), math.radians(j.limits[1]))
                       if j.type == 'revolute' else j.limits for j in spec.action_joints])


def reach_samples(spec, kinematics, count=100):
    limits = bounds(spec)
    rng = np.random.default_rng(101)
    poses = rng.uniform(limits[:, 0], limits[:, 1], size=(count, len(limits)))
    poses[0] = np.clip(0., limits[:, 0], limits[:, 1])
    for index,joint in enumerate(spec.action_joints):
        if joint.role=='gripper':
            poses[:,index]=poses[0,index]
    return poses, np.asarray([kinematics.fk(q)[:3, 3] for q in poses])


def downward_rotation(spec):
    from pxr import Gf
    approach = spec.gripper.approach_axis if spec.gripper is not None else None
    if approach is None:
        approach = np.asarray(spec.ik.position_offset_m)
        if np.linalg.norm(approach) < 1e-8:
            approach = np.array([0., 0., 1.])
    approach = np.asarray(approach, dtype=float)
    approach /= np.linalg.norm(approach)
    return np.asarray(Gf.Matrix3d().SetRotate(Gf.Rotation(Gf.Vec3d(*approach), Gf.Vec3d(0., 0., -1.)))).T


def ready_pose(spec, kinematics):
    from sim.datagen.ik import IKOptions

    from core.paths import DEFAULT_SCENE
    from core.scenes import load_scene_spec,load_mount_spec,mount_pose
    scene=load_scene_spec(DEFAULT_SCENE)
    mounting=mount_pose(scene,load_mount_spec(scene,spec),spec)
    cache_key=(tuple(spec.ik.position_offset_m),tuple(mounting.position),tuple(mounting.orientation_xyzw))
    cached=getattr(kinematics,'_import_ready_cache',{})
    if cache_key in cached:
        q,info=cached[cache_key]
        return q.copy(),info
    limits = bounds(spec)
    zero = np.clip(0., limits[:, 0], limits[:, 1])
    samples, positions = reach_samples(spec, kinematics)
    maximum = float(np.linalg.norm(positions[:, :2], axis=1).max())
    options = IKOptions(max_joint_step_rad=2 * math.pi, max_joint_step_m=1.,
                        max_position_step_m=.05, orientation_tolerance_rad=.01, integral_gain=0.)
    target = np.eye(4)
    target[:3, :3] = downward_rotation(spec)
    best, candidates = None, []
    from sim.arms.clearance import ready_route_validator
    try:
        validate_route=ready_route_validator(spec,kinematics)
        route_error=None
    except Exception as error:
        validate_route=None
        route_error=str(error)
    best_safe=None
    previous = zero.copy()
    # Sweep the declared range, including both endpoints. Nearby grid candidates
    # share a converged seed; random in-limit seeds resolve singular starts.
    for height in np.linspace(.10, .30, 5):
        for fraction in np.linspace(.3, .7, 5):
            target[:3, 3] = [maximum * fraction, 0., height]
            seeds = [previous, samples[int(np.argmin(np.linalg.norm(positions - target[:3, 3], axis=1)))]]
            result = None
            for seed in seeds:
                result = kinematics.inverse(target, seed, options=options, max_iterations=150,
                    position_tolerance_m=.001, orientation_tolerance_rad=.01)
                if result.converged.item():
                    break
            q = result.joint_positions[0].numpy().copy()
            margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q) /
                                  (limits[:, 1] - limits[:, 0])))
            # Gripper zeros do not count against arm joint margin.
            arm_ids = [i for i,j in enumerate(spec.action_joints) if j.role == 'arm']
            margin = float(np.min(np.minimum(q[arm_ids]-limits[arm_ids,0], limits[arm_ids,1]-q[arm_ids]) /
                                  (limits[arm_ids,1]-limits[arm_ids,0])))
            row = dict(target_m=target[:3, 3].tolist(), converged=bool(result.converged.item()),
                       position_error_m=float(result.position_error.item()),
                       orientation_error_rad=float(result.orientation_error.item()), limit_margin=margin,q=q.tolist())
            candidates.append(row)
            if row['converged']:
                previous = q
                if best is None or margin > best[0]:
                    best = margin, q, row
                if validate_route is not None:
                    row['workspace_route']=validate_route(q,target[:3,:3])
                    if row['workspace_route']['passed'] and (best_safe is None or margin>best_safe[0]):
                        best_safe=margin,q,row
                    if not row['workspace_route']['passed'] and row['workspace_route']['reason']!='ready geometry intersects tabletop':
                        # A different library IK seed can reach the same ready
                        # TCP on a branch that stays above the table on approach.
                        for seed in samples[::20]:
                            alternative=kinematics.inverse(target,seed,options=options,max_iterations=150,
                                position_tolerance_m=.001,orientation_tolerance_rad=.01)
                            if not alternative.converged.item():
                                continue
                            proposed=alternative.joint_positions[0].numpy().copy()
                            route=validate_route(proposed,target[:3,:3])
                            if route['passed']:
                                alt_margin=float(np.min((np.minimum(proposed-limits[:,0],limits[:,1]-proposed)/(limits[:,1]-limits[:,0]))[arm_ids]))
                                alt=dict(row,q=proposed.tolist(),limit_margin=alt_margin,workspace_route=route,alternative_seed=True)
                                candidates.append(alt)
                                if best_safe is None or alt_margin>best_safe[0]:
                                    best_safe=alt_margin,proposed,alt
                                break
    if best_safe is not None:
        best=best_safe
    info = dict(converged=best is not None, maximum_horizontal_reach_m=maximum, candidates=candidates,
                collision_safe_route=best_safe is not None)
    if route_error:
        info['workspace_route_error']=route_error
    elif best_safe is None:
        info['workspace_route_fallback']='No safe representative route; retain best joint-margin proposal for validation'
    if best is None:
        info.update(tag='TODO', fallback='zero clamped to limits')
        cached[cache_key]=(zero.copy(),info)
        kinematics._import_ready_cache=cached
        return zero, info
    info.update(tag='derived', selected=best[2])
    cached[cache_key]=(best[1].copy(),info)
    kinematics._import_ready_cache=cached
    return best[1], info


def forward_yaw(spec, q, kinematics=None):
    from sim.datagen.ik import ArmKinematics
    kinematics = kinematics or ArmKinematics.from_arm(spec)
    position = kinematics.fk(q)[:3, 3]
    return math.atan2(position[1], position[0])


def base_yaw_for_scene(spec, reference_spec, *, kinematics=None, reference_kinematics=None):
    from sim.datagen.ik import ArmKinematics
    kinematics = kinematics or ArmKinematics.from_arm(spec)
    reference_kinematics = reference_kinematics or ArmKinematics.from_arm(reference_spec)
    q, _ = ready_pose(spec, kinematics)
    reference_q, _ = ready_pose(reference_spec, reference_kinematics)
    return math.degrees(
        forward_yaw(reference_spec, reference_q, reference_kinematics) - forward_yaw(spec, q, kinematics))
