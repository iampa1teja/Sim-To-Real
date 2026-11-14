"""URDF parsing/expansion remains CPU-only and does not infer semantic roles."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from utils.arms.urdf import UrdfModel,expand_xacro,find_packages
from utils.arms.arm_toml import ImportChoices

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'/'urdf'


def load_fixture(package):
    directory=FIXTURES/package
    path=next(directory.glob('robot.urdf*'))
    packages=find_packages(path)
    return UrdfModel.from_xml(expand_xacro(path,packages),directory,packages)


def test_split_xacro_packages_control_fixed_chain_axes_and_meshes():
    model=load_fixture('seven_parallel_pkg')
    assert model.root=='world' and model.default_base()=='Base_0'
    chain=model.chain('Base_0','Tool_0')
    assert len(chain)==15
    assert len(model.movable_chain('Base_0','Tool_0'))==7
    assert [j.name for j in model.subtree_joints('Tool_0')]==['Grip_L','Grip_R']
    assert model.joints['Grip_R'].mimic==('Grip_L',1.,0.)
    assert model.joints['Grip_L'].ros2_control['id']=='8'
    assert model.joints['Axis_3'].lower==-math.pi
    assert model.joints['Axis_3'].upper==1.05
    np.testing.assert_array_equal(model.joints['Axis_3'].axis,[-1,0,0])
    link=model.links['Finger_L']
    assert link.collisions[0].mesh_path==link.visuals[0].mesh_path
    assert link.collisions[0].mesh_path.is_file()
    assert not np.allclose(link.inertial.inertia,np.diag(np.diag(link.inertial.inertia)))
    assert model.chain('Tool_0','Camera_1')[0].type=='fixed'


def test_mesh_scale_and_five_joint_jaw():
    model=load_fixture('five_jaw_pkg')
    assert len(model.movable_chain('Base_0','Tool_0'))==5
    np.testing.assert_allclose(model.links['Jaw_1'].visuals[0].scale,[1,1,1.1])
    assert model.joints['GripJaw'].type=='revolute'


@pytest.mark.parametrize('fixture,error',[
    ('bad_missing_mesh','Missing mesh.*missing.stl'),
    ('bad_unknown_package','Missing package.*missing'),
    ('bad_closed_loop','Closed loop.*'),
    ('bad_mimic_of_mimic','mimic of a mimic'),
])
def test_bad_descriptions_name_the_element(fixture,error):
    path=FIXTURES/(fixture+'.urdf')
    with pytest.raises(ValueError,match=error):
        UrdfModel.from_xml(path.read_text(),path.parent,{})


def test_gripper_on_chain_is_rejected_without_name_inference():
    path=FIXTURES/'bad_gripper_on_chain.urdf'
    model=UrdfModel.from_xml(path.read_text(),path.parent,{})
    with pytest.raises(ValueError,match='--gripper.*base-to-ee chain'):
        ImportChoices('bad','C',('Grip',)).selected_joints(model)


def test_xacro_missing_package_names_it(tmp_path):
    path=tmp_path/'robot.xacro'
    path.write_text('<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:include filename="$(find absent_pkg)/arm.xacro"/></robot>')
    with pytest.raises(ValueError,match='Missing package.*absent_pkg'):
        expand_xacro(path,{})


def test_fk_oracle_with_nontrivial_origins_axes_and_mimic(tmp_path):
    xml='''<robot name="Oracle"><link name="A"/><link name="B"/><link name="C"/><link name="D"/>
    <joint name="Turn" type="revolute"><parent link="A"/><child link="B"/><origin xyz="1 2 3"/>
    <axis xyz="0 0 -2"/><limit lower="-3.2" upper="3.2" effort="2" velocity="1"/></joint>
    <joint name="Slide" type="prismatic"><parent link="B"/><child link="C"/><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/>
    <axis xyz="1 0 0"/><limit lower="0" upper="0.2" effort="2" velocity="1"/></joint>
    <joint name="Follower" type="prismatic"><parent link="B"/><child link="D"/><axis xyz="0 -1 0"/>
    <limit lower="-0.2" upper="0.4" effort="2" velocity="1"/><mimic joint="Slide" multiplier="-2" offset="0.03"/></joint></robot>'''
    model=UrdfModel.from_xml(xml,tmp_path,{})
    actual=model.fk({'Turn':math.pi/2,'Slide':.1},'A','C')
    np.testing.assert_allclose(actual[:3,3],[1.1,1.,3.],atol=1e-12)
    np.testing.assert_allclose(actual[:3,:3],np.eye(3),atol=1e-12)
    follower=model.fk({'Turn':0.,'Slide':.1},'A','D')
    np.testing.assert_allclose(follower[:3,3],[1.,2.17,3.],atol=1e-12)


@pytest.mark.parametrize('kind',['planar','floating'])
def test_unsupported_joint_types_name_the_joint(kind,tmp_path):
    with pytest.raises(ValueError,match=f'BadJoint.*{kind}'):
        UrdfModel.from_xml(f'<robot><link name="A"/><link name="B"/><joint name="BadJoint" type="{kind}"><parent link="A"/><child link="B"/></joint></robot>',tmp_path,{})


def test_continuous_joint_uses_explicit_degree_bounds(tmp_path):
    xml='<robot><link name="A"/><link name="B"/><joint name="Spin" type="continuous"><parent link="A"/><child link="B"/><axis xyz="0 1 1"/><limit effort="4" velocity="2"/></joint></robot>'
    model=UrdfModel.from_xml(xml,tmp_path,{},continuous_limit_deg=270.)
    joint=model.joints['Spin']
    assert joint.type=='revolute' and joint.continuous
    assert (joint.lower,joint.upper)==pytest.approx((-1.5*math.pi,1.5*math.pi))
    assert np.linalg.norm(joint.axis)==pytest.approx(1.)


@pytest.mark.parametrize('extension',['stl','dae','obj'])
def test_mesh_formats_origins_and_scales_are_preserved(extension,tmp_path):
    mesh=tmp_path/f'Mesh_1.{extension}';mesh.write_text('format contents are converted by the SDK')
    xml=f'<robot><link name="Link_1"><visual><origin xyz=".1 .2 .3" rpy="0 0 1.5707963267948966"/><geometry><mesh filename="Mesh_1.{extension}" scale=".5 2 3"/></geometry></visual></link></robot>'
    model=UrdfModel.from_xml(xml,tmp_path,{})
    geometry=model.links['Link_1'].visuals[0]
    assert geometry.mesh_path==mesh
    np.testing.assert_allclose(geometry.origin[:3,3],[.1,.2,.3])
    np.testing.assert_allclose(geometry.origin[:3,:3]@[1,0,0],[0,1,0],atol=1e-12)
    np.testing.assert_array_equal(geometry.scale,[.5,2,3])


def test_multiple_end_effector_subtrees_need_explicit_roles():
    model=load_fixture('seven_parallel_pkg')
    with pytest.raises(ValueError,match='More than one end effector|unassigned movable'):
        ImportChoices('bad','Link_6',('Axis_7',)).selected_joints(model)


def test_fixed_frame_zero_axis_is_ignored(tmp_path):
    xml='<robot><link name="A"/><link name="B"/><joint name="Frame" type="fixed"><parent link="A"/><child link="B"/><axis xyz="0 0 0"/></joint></robot>'
    model=UrdfModel.from_xml(xml,tmp_path,{})
    np.testing.assert_array_equal(model.fk({},'A','B'),np.eye(4))


def test_deep_package_ancestor_and_sibling_resolution(tmp_path):
    package=tmp_path/'robot_pkg';sibling=tmp_path/'shared_pkg'
    directory=package/'robots/variant/description/urdf';directory.mkdir(parents=True)
    sibling.mkdir()
    (package/'package.xml').write_text('<package><name>DeepRobot</name></package>')
    (sibling/'package.xml').write_text('<package><name>SharedParts</name></package>')
    source=directory/'robot.xacro';source.write_text('<robot/>')
    assert find_packages(source)['DeepRobot']==package
    assert find_packages(source)['SharedParts']==sibling


def test_empty_control_block_is_still_hardware_provenance(tmp_path):
    xml='<robot><link name="A"/><ros2_control name="system" type="system"><hardware><plugin>test</plugin></hardware></ros2_control></robot>'
    assert UrdfModel.from_xml(xml,tmp_path,{}).has_ros2_control
