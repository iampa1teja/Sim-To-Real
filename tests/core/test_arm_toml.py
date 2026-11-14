"""Generated arm manifests load with injected simulator measurements on the CPU."""
from pathlib import Path
import tomllib

import pytest

from core.arms.arm_toml import ImportChoices,build_arm_toml,build_report
from core.arms.urdf import UrdfModel,expand_xacro,find_packages
from core.arms.spec import load_arm_spec

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'/'urdf'


@pytest.mark.parametrize('package,joint_count,gripper,kind',[
    ('five_jaw_pkg',5,'GripJaw','jaw'),('seven_parallel_pkg',7,'Grip_L','parallel')])
def test_generated_manifest_loads_and_reports_every_field(package,joint_count,gripper,kind,tmp_path,monkeypatch):
    path=next((FIXTURES/package).glob('robot.urdf*'));packages=find_packages(path)
    model=UrdfModel.from_xml(expand_xacro(path,packages),path.parent,packages)
    choices=ImportChoices('test_import','Tool_0',(gripper,),
        rename={gripper:'opening'},measurements=dict(closed_gap_m=0.,open_gap_m=.09,
        closed_position=0.,open_position=.045 if kind=='parallel' else 1.2,
        closing_axis=[1.,0.,0.],approach_axis=[0.,0.,1.],position_offset_m=[0.,0.,.07]),
        ready_info=dict(converged=True),camera={'Camera_1':'wrist_cam'} if joint_count==7 else {})
    directory=tmp_path/'arms'/'test_import';(directory/'usd').mkdir(parents=True)
    (directory/'usd'/'test_import.usd').write_text('dummy path for pure loader')
    (directory/'utils.py').write_text('')
    text=build_arm_toml(model,choices)
    raw=tomllib.loads(text)['arm']
    assert [j['usd'] for j in raw['joints'][:joint_count]]==[f'Axis_{i+1}' for i in range(joint_count)]
    assert raw['gripper']['kind']==kind
    assert raw['gripper']['joints']==['opening']
    assert raw['ik']['orientation_axes']==['x','y']+(['z'] if joint_count>5 else [])
    assert raw['ik']['joint_limit_avoidance_gain']==(.5 if joint_count==7 else 0.)
    (directory/'arm.toml').write_text(text)
    monkeypatch.setenv('S2R_ARMS_DIR',str(directory.parent))
    spec,_=load_arm_spec('test_import')
    assert len(spec.action_joints)==joint_count+1
    assert spec.gripper.open_gap_m==.09
    assert spec.ik.position_offset_m==(0.,0.,.07)
    if joint_count==7:
        assert spec.mimic_joints==((8,7,1.,0.),)
        assert spec.joints[-1].mimic.joint=='opening'
        assert spec.joints[2].limits==pytest.approx((-180.,60.1605684887))
        assert spec.joints[7].limits==(0.,.045)
        assert spec.hardware.joint_ids['opening']=='8'
    report=build_report(model,choices,[dict(name='test',passed=True,numeric_error_m=0.)])
    assert report['fields']['arm.links.ee']['tag']=='from-flag'
    assert report['fields']['arm.gripper.open_gap_m']['tag']=='measured'
    assert report['fields']['arm.gripper.open_gap_m']['confirm']
    assert report['fields']['arm.ik.orientation_axes']['tag']=='confirm'
    if joint_count==7:
        assert report['fields']['arm.hardware.Grip_L.counts_per_metre']['tag']=='TODO'
    assert report['checks'][0]['passed']


def test_six_arm_joint_proposal_uses_full_orientation():
    from dataclasses import replace
    import math
    path=FIXTURES/'five_jaw_pkg'/'robot.urdf'
    model=UrdfModel.from_xml(path.read_text(),path.parent,{})
    model.joints['ToolMount']=replace(model.joints['ToolMount'],type='revolute',lower=-math.pi,upper=math.pi)
    choices=ImportChoices('six_axes','Tool_0',('GripJaw',))
    raw=tomllib.loads(build_arm_toml(model,choices))['arm']
    assert raw['ik']['orientation_axes']==['x','y','z']
    assert raw['ik']['joint_limit_avoidance_gain']==0.
