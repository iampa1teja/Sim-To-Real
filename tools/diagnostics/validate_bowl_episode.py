import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from core.tasks.loader import load_task,load_task_module
from core.arms.spec import load_arm_spec
from core.arms.mapping import JointMapping
from core.datasets.sources import load_source,replay_joint_mapping
from core.scenes.anchors import quaternion_matrix


def terminal_pose_window(positions, quaternions, live_objects, names, fps, count=30):
    """Pair saved pre-action poses with the measured post-action endpoint.

    Thirty saved poses plus the actual success endpoint span thirty control
    intervals. Differencing the saved array before slicing instead includes an
    interval preceding this stability window.
    """
    if count < 2 or len(positions) < count or fps <= 0:
        raise ValueError('A complete positive-rate stability window is required')
    end_positions=np.asarray([live_objects[name]['position'] for name in names])
    end_quaternions=np.asarray([live_objects[name]['quat_xyzw'] for name in names])
    p=np.concatenate([positions[-count:],end_positions[None]],axis=0)
    q=np.concatenate([quaternions[-count:],end_quaternions[None]],axis=0)
    linear=np.diff(p,axis=0)*fps
    angular=(Rotation.from_quat(q[1:].reshape(-1,4))*Rotation.from_quat(q[:-1].reshape(-1,4)).inv()).as_rotvec().reshape(count,len(names),3)*fps
    return p,q,np.concatenate([linear,angular],axis=-1)


def main():
    root=Path('/workspace/datasets/stack_bowls_one_episode')
    out=Path('/workspace/s2r/outputs/stack_bowls')
    task=load_task('stack_bowls');arm=load_arm_spec('so101')[0]
    generation=json.loads((out/'generation-report.json').read_text())
    accepted=[a for a in generation['episodes'] if a['accepted']]
    assert generation['complete'] and len(accepted)==generation['accepted_episodes']==1
    row=accepted[0];assert row['final_success'] and row['last_state']['signals']['success']
    episodes=load_source(root,'local/stack_bowls_one_episode',set(),arm=arm,task=task)
    assert len(episodes)==1
    ep=episodes[0];assert ep.index==0 and ep.length==row['frames'] and ep.fps==30
    names=list(task.spec.tracked_objects)
    assert set(ep.objects_base)==set(names)
    world=row['waypoint_trace'][0]['objects']
    first=names[0];b=ep.objects_base[first][0]
    base_rotation=quaternion_matrix(world[first]['quat_xyzw']) @ quaternion_matrix(b[[4,5,6,3]]).T
    base_position=np.array(world[first]['position'])-base_rotation @ b[:3]
    p=np.stack([ep.objects_base[n][:,:3] @ base_rotation.T+base_position for n in names],1)
    base_q=np.stack([ep.objects_base[n][:,[4,5,6,3]] for n in names],1)
    q=(Rotation.from_matrix(base_rotation)*Rotation.from_quat(base_q.reshape(-1,4))).as_quat().reshape(ep.length,3,4)
    linear=np.diff(p,axis=0)*ep.fps
    angular=(Rotation.from_quat(q[1:].reshape(-1,4))*Rotation.from_quat(q[:-1].reshape(-1,4)).inv()).as_rotvec().reshape(ep.length-1,3,3)*ep.fps
    velocity=np.concatenate([linear,angular],axis=-1)
    geometry=load_task_module('geometry',task.name)
    terminal_p,terminal_q,terminal_velocity=terminal_pose_window(p,q,row['last_state']['objects'],names,ep.fps)
    paired,upright,stable=geometry.nesting_metrics(torch.tensor(terminal_p),torch.tensor(terminal_q),torch.tensor(terminal_velocity),task.spec.raw['success'])
    assert paired.all() and upright.all() and stable.all(), 'Last recorded second must be a stable nested stack'
    mapping_data,mapping_source=replay_joint_mapping(root,arm)
    mapping=JointMapping.from_arm(arm,mapping_data,device='cpu')
    commands=mapping.to_sim(ep.actions)
    roundtrip=mapping.to_raw(commands).cpu().numpy()
    assert np.allclose(roundtrip,ep.actions,rtol=0,atol=1e-3)
    info=json.loads((root/'meta/info.json').read_text())
    assert info['total_episodes']==1 and info['total_frames']==ep.length
    order=np.argsort(p[-1,:,2]);sorted_p=p[-1,order]
    up=np.array([quaternion_matrix(v)[:,2] for v in q[-1,order]])
    pitches=np.diff(sorted_p[:,2]);xy=np.linalg.norm(np.diff(sorted_p[:,:2],axis=0),axis=1)
    relative_tilts=np.degrees(np.arccos(np.clip(np.sum(up[1:]*up[:-1],axis=1),-1,1)))
    report=dict(passed=True,task=task.name,arm=arm.name,episodes=1,frames=ep.length,fps=ep.fps,
     duration_seconds=ep.length/ep.fps,accepted_attempt=row['attempt'],
     order_bottom_to_top=[names[i] for i in order],tracked_objects=names,
     criteria=task.spec.raw['success'],last_recorded_second_valid=True,
     stability_window='30 physical control intervals through the actual post-action success pose',
     stability_interval_count=len(terminal_velocity),
     max_final_second_linear_speed_m_s=float(np.linalg.norm(terminal_velocity[:,:,:3],axis=-1).max()),
     max_final_second_angular_speed_rad_s=float(np.linalg.norm(terminal_velocity[:,:,3:],axis=-1).max()),
     final_adjacent_xy_errors_m=xy.tolist(),final_nesting_pitches_m=pitches.tolist(),
     final_support_relative_tilts_deg=relative_tilts.tolist(),
     final_gravity_tilts_deg=np.degrees(np.arccos(np.clip(up[:,2],-1,1))).tolist(),
     live_final_signals=row['last_state']['signals'],
     max_recorded_object_step_m={n:float(np.linalg.norm(np.diff(p[:,i],axis=0),axis=1).max()) for i,n in enumerate(names)},
     action_codec_roundtrip_max_error=float(np.abs(roundtrip-ep.actions).max()),
     mapping_source=mapping_source,derived_robot_base_position_world_m=base_position.tolist(),
     source_asset=json.loads((task.spec.task_dir/'assets/provenance.json').read_text()),
     hashes={str(f.relative_to(Path('/workspace/s2r'))):hashlib.sha256(f.read_bytes()).hexdigest()
     for f in [task.spec.task_dir/'task.toml',task.spec.task_dir/'expert.py',task.spec.task_dir/'mdp.py',
     task.spec.objects['yellow_bowl'].usd,Path('/workspace/s2r/configs/datagen/stack_bowls.yaml')]})
    tilts=np.degrees(np.arccos(np.clip(1-2*(terminal_q[:,:,0]**2+terminal_q[:,:,1]**2),-1,1)))
    report['final_second_max_gravity_tilt_degrees']={n:float(tilts[:,i].max()) for i,n in enumerate(names)}
    report['also_meets_12_degree_world_upright_check']=bool(np.all(tilts<=12.))
    assert report['also_meets_12_degree_world_upright_check']
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
