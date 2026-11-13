"""Collision-geometry checks for importer pose proposals, using Newton FK."""
import numpy as np


def ready_route_validator(spec,kinematics):
    """Check representative task approaches before accepting a ready IK branch.

    This is an importer proposal check, not a runtime planner. Reported full
    workspace reach is still measured separately by the live validation sweep.
    """
    import torch
    from pxr import Usd, UsdPhysics
    from isaaclab.utils.math import matrix_from_quat
    from core.paths import DEFAULT_SCENE,DEFAULT_TASK
    from core.scenes import load_scene_spec,load_mount_spec,mount_pose
    from core.scenes.anchors import quaternion_matrix,anchor_quaternion
    from core.tasks.base import Pose
    from core.tasks.loader import load_task
    from sim.arms.ready_pose import bounds
    from sim.datagen.ik import IKOptions
    from sim.scene.geometry import collision_body_points,table_plane

    scene=load_scene_spec(DEFAULT_SCENE)
    task=load_task(DEFAULT_TASK)
    mount=mount_pose(scene,load_mount_spec(scene,spec),spec)
    rotation=quaternion_matrix(mount.orientation_xyzw)
    room=Usd.Stage.Open(str(scene.room_usd))
    plane=table_plane(room,room.GetPseudoRoot(),scene.anchors['table'])
    stage=Usd.Stage.Open(str(spec.usd_path))
    body_prims={prim.GetName():prim for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    clouds={}
    for index,label in enumerate(kinematics._model.body_label):
        name=label.rsplit('/',1)[-1]
        if name==spec.links.base or name not in body_prims:
            continue
        try:
            clouds[index]=collision_body_points(stage,str(body_prims[name].GetPath()))
        except ValueError:
            continue
    if not clouds:
        raise ValueError('No collision geometry available for ready-route clearance')
    coordinates=kinematics._evaluate(np.array([j.initial for j in spec.action_joints]))
    base=coordinates[kinematics.base_id]
    base_world=rotation@(base[:3]*kinematics._meters_per_unit)+mount.position
    base_rotation=rotation@matrix_from_quat(torch.as_tensor(base[3:],dtype=torch.float64)).numpy()
    geometry=dict(base_position_world=base_world.tolist(),base_quaternion_wxyz=[
        *mount.orientation_xyzw[3:],*mount.orientation_xyzw[:3]],frames={})
    for name,anchor in scene.anchors.items():
        quat=anchor_quaternion(anchor)
        geometry['frames'][name]=dict(position_world=list(anchor.position),orientation_world_wxyz=[quat[3],*quat[:3]])
    start=task.spec.start
    low,high=np.asarray(start['xy_min']),np.asarray(start['xy_max'])
    centre=(low+high)/2
    corners=[np.array([x,y]) for x in (low[0],high[0]) for y in (low[1],high[1])]
    farthest=max(corners,key=lambda point:np.linalg.norm(point))
    object_name=start.get('object',task.spec.tracked_objects[0])
    width=task.spec.objects[object_name].size
    targets=[]
    for xy in (centre,farthest):
        local=task.start_pose_to_base(Pose.from_lists([*xy,0.]),geometry)
        world=base_world+base_rotation@local.position
        world[2]=plane[0]*world[0]+plane[1]*world[1]+plane[2]+width[2]/2+.0002
        for height in (.06,0.,.09):
            lifted=world.copy();lifted[2]+=height
            targets.append(base_rotation.T@(lifted-base_world))
    options=IKOptions(damping=.005,orientation_weight=.2,max_joint_step_rad=2*np.pi,
                      max_joint_step_m=1.,max_position_step_m=.02,integral_gain=0.)
    limits=bounds(spec)
    grip_ids=[i for i,joint in enumerate(spec.action_joints) if joint.role=='gripper']
    def clearance(q):
        poses=kinematics._evaluate(q)
        minimum=np.inf
        for index,cloud in clouds.items():
            body=poses[index]
            points=cloud@matrix_from_quat(torch.as_tensor(body[3:],dtype=torch.float64)).numpy().T
            points+=body[:3]*kinematics._meters_per_unit
            points=points@rotation.T+mount.position
            distance=points[:,2]-plane[0]*points[:,0]-plane[1]*points[:,1]-plane[2]
            minimum=min(minimum,float(distance.min()))
        return minimum
    def check(q,orientation):
        previous=q.copy()
        minimum=clearance(previous)
        if minimum<-.0005:
            return dict(passed=False,reason='ready geometry intersects tabletop',minimum_clearance_m=minimum)
        for position in targets:
            target=np.eye(4);target[:3,:3]=orientation;target[:3,3]=position
            result=kinematics.inverse(target,previous,options=options,max_iterations=100,
                position_tolerance_m=.004,orientation_tolerance_rad=.03)
            if not result.converged.item():
                return dict(passed=False,reason='task approach IK branch does not converge',
                            position_error_m=float(result.position_error.item()),
                            orientation_error_rad=float(result.orientation_error.item()))
            following=result.joint_positions[0].numpy().copy()
            for fraction in np.linspace(0.,1.,5):
                minimum=min(minimum,clearance(previous*(1-fraction)+following*fraction))
                if minimum<-.0005:
                    return dict(passed=False,reason='task approach branch intersects tabletop',minimum_clearance_m=minimum)
            previous=following
        return dict(passed=True,minimum_clearance_m=minimum,targets_base_m=[p.tolist() for p in targets],
                    scope='task start centre and farthest radial corner; approach, grasp and lift; pickup yaw')
    check.clearance=clearance
    return check
