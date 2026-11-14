"""Live import checks in the default scene, with numerical failure evidence."""
import math
import os
from pathlib import Path

import numpy as np


def workspace_sweep(spec, kin, task, geometry):
    from core.benchmark.geometry import rotation_matrix
    from core.tasks.base import Pose
    from sim.datagen.ik import IKOptions

    rotation=rotation_matrix(geometry['base_quaternion_wxyz'])
    base=np.asarray(geometry['base_position_world'])
    plane=geometry['table_plane_world']
    tracked=task.spec.start.get('object',task.spec.tracked_objects[0])
    width=np.asarray(task.spec.objects[tracked].size)
    q=np.array([joint.initial for joint in spec.action_joints])
    orientation=kin.fk(q)[:3,:3]
    options=IKOptions(damping=.005,orientation_weight=.2,max_joint_step_rad=2*math.pi,
                      max_joint_step_m=1.,max_position_step_m=.02,integral_gain=0.)
    from sim.arms.ready_pose import reach_samples
    seeds,seed_positions=reach_samples(spec,kin,count=60)
    previous=q.copy()
    multistart_rng=np.random.default_rng(404)
    cells=[]
    start=task.spec.start
    for x in np.arange(start['xy_min'][0],start['xy_max'][0]+1e-8,.05):
        for y in np.arange(start['xy_min'][1],start['xy_max'][1]+1e-8,.05):
            local=task.start_pose_to_base(Pose((x,y,0.)),geometry)
            world=base+rotation@local.position
            world[2]=plane[0]*world[0]+plane[1]*world[1]+plane[2]+width[2]/2
            cells.append(('start',world))
    for name,obj in task.spec.objects.items():
        if name in task.spec.tracked_objects:
            continue
        row=geometry['objects'][name]
        size=obj.params.get('inner_half_size')
        if size is None:
            size=np.asarray(row.get('size',width))[:2]/2
        for x in np.arange(-size[0],size[0]+1e-8,.05):
            for y in np.arange(-size[1],size[1]+1e-8,.05):
                world=np.asarray(row['position_world'])+rotation_matrix(row['orientation_world_wxyz'])@np.array([x,y,obj.params.get('floor_z',0.)+width[2]/2])
                cells.append(('container',world))
    rows=[]
    for region,world in cells:
        target=np.eye(4);target[:3,:3]=orientation;target[:3,3]=rotation.T@(world-base)
        attempts=1
        result=kin.inverse(target,previous,options=options,max_iterations=150,
                           position_tolerance_m=.001,orientation_tolerance_rad=.01)
        if not result.converged.item():
            nearest=np.argsort(np.linalg.norm(seed_positions-target[:3,3],axis=1))[:5]
            for seed in seeds[nearest]:
                attempts+=1
                candidate=kin.inverse(target,seed,options=options,max_iterations=200,
                                     position_tolerance_m=.001,orientation_tolerance_rad=.01)
                if candidate.position_error.item()+.05*candidate.orientation_error.item() < result.position_error.item()+.05*result.orientation_error.item():
                    result=candidate
                if result.converged.item():
                    break
        if not result.converged.item():
            from sim.arms.ready_pose import bounds
            limits=bounds(spec)
            for _ in range(30):
                seed=multistart_rng.uniform(limits[:,0],limits[:,1])
                for index,joint in enumerate(spec.action_joints):
                    if joint.role=='gripper':
                        seed[index]=joint.initial
                attempts+=1
                candidate=kin.inverse(target,seed,options=options,max_iterations=300,
                                     position_tolerance_m=.001,orientation_tolerance_rad=.01)
                if candidate.position_error.item()+.05*candidate.orientation_error.item() < result.position_error.item()+.05*result.orientation_error.item():
                    result=candidate
                if result.converged.item():
                    break
        if result.converged.item():
            previous=result.joint_positions[0].numpy().copy()
        rows.append(dict(region=region,world_m=world.tolist(),base_m=target[:3,3].tolist(),
                         reached=bool(result.converged.item()),ik_seed_attempts=attempts,position_error_m=float(result.position_error.item()),
                         orientation_error_rad=float(result.orientation_error.item()),q=result.joint_positions[0].tolist()))
    reached=sum(row['reached'] for row in rows)
    return dict(name='reach_sweep',passed=bool(rows) and reached==len(rows),grid_spacing_m=.05,
                fraction_reached=reached/max(len(rows),1),cells=rows,
                ik_options=dict(damping=.005,orientation_weight=.2,multistart_seed=404),
                unreachable_cells=[row for row in rows if not row['reached']])


def validate_import(name,*,repo=None):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg
    from core.arms.spec import load_arm_spec
    from core.paths import DEFAULT_SCENE,DEFAULT_TASK
    from sim.datagen.ik import ArmKinematics,make_ik
    from core.tasks.base import Pose
    from sim.envs.mdp._common import as_torch
    from sim.envs.mdp.mimic import write_mimic_state
    from sim.eval.scene import capture_scene

    old={key:os.environ.get(key) for key in ('S2R_ARM','S2R_ARMS_DIR')}
    os.environ['S2R_ARM']=name
    if repo:
        os.environ['S2R_ARMS_DIR']=str(Path(repo)/'workspace'/'arms')
    spec,_=load_arm_spec(name)
    results=[dict(name='spec_action_dimension',passed=len(spec.joint_names)==sum(j.mimic is None for j in spec.joints),
                  action_dimension=len(spec.action_joints),physical_joints=len(spec.joints))]
    env=None
    def check(name,operation):
        try:
            row=operation();row.setdefault('name',name);results.append(row);return row
        except Exception as error:
            row=dict(name=name,passed=False,error=str(error));results.append(row);return row
    try:
        import sim.envs
        task=sim.envs.register_envs(scene=DEFAULT_SCENE,task=DEFAULT_TASK)
        cfg=parse_env_cfg(task.gym_id(task.spec.default_variant),device='cuda:0',num_envs=1)
        cfg.terminations=None
        cfg.episode_length_s=120.
        env=gym.make(task.gym_id(task.spec.default_variant),cfg=cfg)
        raw=env.unwrapped
        robot=raw.scene['robot']
        action_ids=robot.find_joints(list(spec.action_usd_joint_names),preserve_order=True)[0]
        obs,_=env.reset()
        target=torch.tensor([[joint.initial for joint in spec.action_joints]],device=raw.device,dtype=torch.float32)
        def hold():
            initial=as_torch(robot.data.joint_pos)[:,action_ids].clone()
            for _ in range(round(2/(cfg.decimation*cfg.sim.dt))):
                env.step(target)
            drift=(as_torch(robot.data.joint_pos)[:,action_ids]-initial).abs()[0].cpu().numpy()
            tolerance=np.array([math.radians(1) if j.type=='revolute' else .001 for j in spec.action_joints])
            return dict(passed=bool(np.all(drift<=tolerance)),duration_s=2.,
                drift_si={j.name:float(d) for j,d in zip(spec.action_joints,drift)},tolerance_si=tolerance.tolist())
        check('ready_pose_gravity_hold',hold)
        def mimic():
            errors=[]
            steps=0
            for repeat in range(2):
                env.reset()
                target[:]=target.new_tensor([[joint.initial for joint in spec.action_joints]])
                for name in spec.gripper_joint_names:
                    index=spec.joint_names.index(name)
                    joint=spec.action_joints[index]
                    limits=np.deg2rad(joint.limits) if joint.type=='revolute' else joint.limits
                    for coordinate in np.linspace(*limits,50):
                        target[:,index]=float(coordinate)
                        env.step(target);steps+=1
                        positions=as_torch(robot.data.joint_pos)[0]
                        for follower in spec.joints:
                            if follower.mimic and follower.mimic.joint==name:
                                follower_id=robot.find_joints([follower.usd])[0][0]
                                source_id=action_ids[index]
                                errors.append(float(abs(positions[follower_id]-positions[source_id]*follower.mimic.multiplier-follower.mimic.offset)))
            return dict(passed=max(errors,default=0.)<=.001,max_error_si=max(errors,default=0.),samples=len(errors),
                        reset_count=2,action_steps=steps,runtime=getattr(robot,'_s2r_mimic_runtime','manual'),tolerance_si=.001)
        check('mimic_tracking',mimic)
        kin=ArmKinematics.from_arm(spec)
        geometry=capture_scene(raw,task)
        sweep=check('reach_sweep',lambda:workspace_sweep(spec,kin,task,geometry))
        def grasp():
            source=spec.gripper.joints[0]
            source_index=spec.joint_names.index(source)
            opened=spec.gripper.open_position
            closed=spec.gripper.closed_position
            if opened is None or closed is None:
                raise ValueError('Measured gripper endpoints are required for grasp validation')
            reachable=[row for row in sweep.get('cells',[]) if row['reached'] and row['region']=='start']
            if not reachable:
                raise ValueError('No reachable default-task start cell for grasp validation')
            from sim.arms.ready_pose import bounds
            limits=bounds(spec)
            arm_indices=[i for i,joint in enumerate(spec.action_joints) if joint.role=='arm']
            def margin(row):
                q=np.asarray(row['q'])
                return float(np.min((np.minimum(q-limits[:,0],limits[:,1]-q)/(limits[:,1]-limits[:,0]))[arm_indices]))
            from sim.arms.clearance import ready_route_validator
            from sim.datagen.ik import IKOptions
            validator=ready_route_validator(spec,kin)
            ready=np.array([joint.initial for joint in spec.action_joints])
            desired=np.eye(4);desired[:3,:3]=kin.fk(ready)[:3,:3]
            selected=None
            object_name=task.spec.start.get('object',task.spec.tracked_objects[0])
            for candidate in sorted(reachable,key=margin,reverse=True):
                if not task.start_is_valid({object_name:Pose.from_lists(candidate['base_m'])},geometry):
                    continue
                desired[:3,3]=candidate['base_m']
                solved=kin.inverse(desired,ready,options=IKOptions(damping=.005,orientation_weight=.2,
                    max_joint_step_rad=2*math.pi,max_joint_step_m=1.,max_position_step_m=.02,integral_gain=0.),
                    max_iterations=150,position_tolerance_m=.001,orientation_tolerance_rad=.01)
                if solved.converged.item():
                    positions=solved.joint_positions[0].numpy().copy()
                    if validator.clearance(positions)>=-.0005:
                        selected=dict(candidate,q=positions.tolist())
                        break
            if selected is None:
                raise ValueError('No collision-clear grasp cell on the ready IK branch')
            command=target.new_tensor([selected['q']]);command[:,source_index]=opened
            robot.write_joint_state_to_sim_index(position=command,velocity=torch.zeros_like(command),joint_ids=action_ids)
            robot.set_joint_position_target_index(target=command,joint_ids=action_ids)
            write_mimic_state(raw)
            cube=raw.scene[object_name]
            object_pose=command.new_tensor([selected['world_m']+[0.,0.,0.,1.]])
            cube.write_root_pose_to_sim_index(root_pose=object_pose)
            cube.write_root_velocity_to_sim_index(root_velocity=command.new_zeros((1,6)))
            for _ in range(30):
                env.step(command)
            start_height=float(as_torch(cube.data.root_pos_w)[0,2])
            contact={name:0. for name in spec.gripper.contact_bodies}
            sensors=[value for value in raw.scene.sensors.values() if hasattr(value.data,'force_matrix_w')]
            def observe_contact():
                for sensor in sensors:
                    forces=as_torch(sensor.data.force_matrix_w)
                    if forces is None:
                        continue
                    magnitudes=forces.norm(dim=-1).sum(dim=-1)[0]
                    for body,force in zip(sensor.body_names,magnitudes):
                        if body in contact:
                            contact[body]=max(contact[body],float(force))
            for fraction in np.linspace(0.,1.,50):
                command[:,source_index]=opened+(closed-opened)*float(fraction)
                env.step(command);observe_contact()
            ik=make_ik(robot,spec)
            position,quaternion=ik.current_pose()
            goal=position.clone();goal[:,2]+=.05
            for _ in range(90):
                result=ik.solve(goal,quaternion)
                command=result.joint_positions;command[:,source_index]=closed
                env.step(command);observe_contact()
            lifted=float(as_torch(cube.data.root_pos_w)[0,2])-start_height
            return dict(passed=lifted>=.04 and all(value>0 for value in contact.values()),
                        commanded_lift_m=.05,measured_lift_m=lifted,contact_forces_n=contact,
                        start_cell_world_m=selected['world_m'],joint_limit_margin=margin(selected))
        check('gripper_cube_lift',grasp)
        def cameras():
            from PIL import Image
            directory=spec.directory/'validation'
            directory.mkdir(exist_ok=True)
            obs=raw.observation_manager.compute()
            files={}
            for camera in spec.cameras:
                pixels=obs['visual']['rgb_'+camera][0].detach().cpu().numpy()
                if np.issubdtype(pixels.dtype,np.floating):
                    pixels=np.clip(pixels*255,0,255).astype(np.uint8)
                destination=directory/(camera+'.png')
                Image.fromarray(pixels).save(destination)
                files[camera]=str(destination.relative_to(spec.directory))
            return dict(passed=len(files)==len(spec.cameras),rendered_frames=files)
        check('wrist_camera_frames',cameras)
    except Exception as error:
        results.append(dict(name='default_scene_spawn',passed=False,error=str(error)))
        for name in ('ready_pose_gravity_hold','mimic_tracking','reach_sweep','gripper_cube_lift','wrist_camera_frames'):
            if not any(row['name']==name for row in results):
                results.append(dict(name=name,passed=False,error='Unavailable after default scene spawn failure'))
    finally:
        if env is not None:
            env.close()
        for key,value in old.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value
    return results
