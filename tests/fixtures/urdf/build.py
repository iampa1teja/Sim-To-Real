"""Rebuild tiny synthetic ROS packages; ASCII geometry keeps the fixtures reviewable."""
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent


def box_stl(path, size):
    vertices = [(x*size[0]/2, y*size[1]/2, z*size[2]/2)
                for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    faces = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
             (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    lines = ['solid box']
    for face in faces:
        lines += [' facet normal 0 0 0', '  outer loop']
        lines += ['   vertex ' + ' '.join(map(str, vertices[index])) for index in face]
        lines += ['  endloop', ' endfacet']
    path.write_text('\n'.join(lines + ['endsolid box']) + '\n')


def cylinder_stl(path, radius=.012, length=.025, segments=12):
    lines = ['solid cylinder']
    for index in range(segments):
        a, b = [2*math.pi*k/segments for k in (index,index+1)]
        p, q = (radius*math.cos(a),radius*math.sin(a)), (radius*math.cos(b),radius*math.sin(b))
        for points in (((*p,-length/2),(*q,-length/2),(*q,length/2)),
                       ((*p,-length/2),(*q,length/2),(*p,length/2)),
                       ((0,0,-length/2),(*q,-length/2),(*p,-length/2)),
                       ((0,0,length/2),(*p,length/2),(*q,length/2))):
            lines += [' facet normal 0 0 0', '  outer loop']
            lines += ['   vertex ' + ' '.join(map(str, point)) for point in points]
            lines += ['  endloop', ' endfacet']
    path.write_text('\n'.join(lines + ['endsolid cylinder']) + '\n')


def alternate_meshes(directory):
    vertices=[(x*.01,y*.01,z*.01) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    faces=[(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
           (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]
    obj=['# Tiny box visual fixture']+['v '+' '.join(map(str,v)) for v in vertices]
    obj+=['f '+' '.join(str(i+1) for i in face) for face in faces]
    (directory/'camera.obj').write_text('\n'.join(obj)+'\n')
    values=' '.join(str(value) for vertex in vertices for value in vertex)
    indices=' '.join(str(index) for face in faces for index in face)
    (directory/'camera.dae').write_text(f'''<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><created>2026-01-01T00:00:00Z</created><modified>2026-01-01T00:00:00Z</modified><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_geometries><geometry id="Box"><mesh><source id="Positions"><float_array id="PositionsArray" count="24">{values}</float_array><technique_common><accessor source="#PositionsArray" count="8" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="Vertices"><input semantic="POSITION" source="#Positions"/></vertices><triangles count="12"><input semantic="VERTEX" source="#Vertices" offset="0"/><p>{indices}</p></triangles></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="Scene"><node id="BoxNode"><instance_geometry url="#Box"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#Scene"/></scene></COLLADA>\n''')


def link(name, mesh='body.stl', xyz='0 0 0', *, scale=None, mass=.08, package=None):
    filename = f'package://{package}/meshes/{mesh}' if package else f'meshes/{mesh}'
    scale_attr = f' scale="{scale}"' if scale else ''
    return f'''<link name="{name}">
 <inertial><origin xyz="0.005 0 0" rpy="0.1 0.2 0.3"/><mass value="{mass}"/>
 <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.00012" iyz="0" izz="0.00015"/></inertial>
 <visual><origin xyz="{xyz}" rpy="0 0 0"/><geometry><mesh filename="{filename}"{scale_attr}/></geometry></visual>
 <collision><origin xyz="{xyz}" rpy="0 0 0"/><geometry><mesh filename="{filename}"{scale_attr}/></geometry></collision>
</link>'''


def joint(name, parent, child, kind='fixed', xyz='0 0 0', axis='0 1 0', limits='-2.9 2.9', mimic=''):
    low, high = limits.split()
    return f'''<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>
 <origin xyz="{xyz}" rpy="0 0 0"/><axis xyz="{axis}"/>
 <limit lower="{low}" upper="{high}" effort="{20 if kind != 'prismatic' else 15}" velocity="{2 if kind != 'prismatic' else .2}"/>{mimic}</joint>'''


def build():
    for pkg, count in [('five_jaw_pkg',5), ('seven_parallel_pkg',7)]:
        directory = ROOT/pkg
        (directory/'meshes').mkdir(parents=True,exist_ok=True)
        box_stl(directory/'meshes/body.stl',(.035,.035,.035))
        box_stl(directory/'meshes/finger.stl',(.008,.025,.07))
        box_stl(directory/'meshes/link.stl',(.10,.025,.025))
        cylinder_stl(directory/'meshes/motor.stl')
        alternate_meshes(directory/'meshes')
        (directory/'package.xml').write_text(f'<package format="3"><name>{pkg}</name><version>0.0.1</version><description>Synthetic import acceptance arm</description><maintainer email="test@example.org">S2R</maintainer><license>MIT</license></package>\n')
        package = pkg if count == 7 else None
        rows = ['<link name="world"/>',link('Base_0',mass=.4,package=package),joint('WorldMount','world','Base_0')]
        lengths = [0., .06, .14, .14, .10, .06, .04]
        axes = ['0 0 1','0 1 0','-1 0 0','0 1 0','1 0 0','0 1 0','0 0 1'] if count==7 else ['0 0 1','0 1 0','0 1 0','0 1 0','1 0 0']
        previous = 'Base_0'
        for i in range(count):
            bracket, child = f'Bracket_{i+1}', f'Link_{i+1}'
            rows += [link(bracket,mesh='motor.stl',mass=.025,package=package),
                     joint(f'FixedBracket_{i+1}',previous,bracket,xyz='0 0 .025' if i==0 else f'{lengths[i]} 0 0'),
                     link(child,mesh='link.stl',xyz='.05 0 0',package=package),
                     joint(f'Axis_{i+1}',bracket,child,'revolute',axis=axes[i],
                           limits=f'{-math.pi} 1.05' if i==2 else f'{-math.pi} {math.pi}')]
            previous=child
        rows += [link('Tool_0',package=package),joint('ToolMount',previous,'Tool_0',xyz='.05 0 0')]
        if count==7:
            rows += [link('Finger_L',mesh='finger.stl',xyz='0 0 .035',mass=.025,package=package),
                     link('Finger_R',mesh='finger.stl',xyz='0 0 .035',mass=.025,package=package),
                     joint('Grip_L','Tool_0','Finger_L','prismatic',xyz='.004 0 .01',axis='1 0 0',limits='0 .045'),
                     joint('Grip_R','Tool_0','Finger_R','prismatic',xyz='-.004 0 .01',axis='-1 0 0',limits='0 .045',
                           mimic='<mimic joint="Grip_L" multiplier="1" offset="0"/>'),
                     link('Camera_1',mass=.015,package=package),joint('CameraMount','Tool_0','Camera_1',xyz='0 -.05 .02')]
            # Valid OBJ/DAE visuals exercise the converter's format handling.
            camera_index=next(i for i,row in enumerate(rows) if row.startswith('<link name="Camera_1"'))
            extra=''.join(f'<visual><origin xyz="0 0 .005" rpy=".1 0 .2"/><geometry><mesh filename="package://{pkg}/meshes/camera.{ext}" scale=".5 .5 .5"/></geometry></visual>' for ext in ('obj','dae'))
            rows[camera_index]=rows[camera_index].replace('</link>',extra+'</link>')
            # Split the full robot across includes and use both package URI forms.
            header='<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="seven_parallel">'
            arm_end = next(i for i,row in enumerate(rows) if row.startswith('<link name="Finger_L"'))
            (directory/'arm.xacro').write_text(header+'\n<xacro:property name="housing_mass" value="0.025"/>\n'+'\n'.join(rows[:arm_end]).replace('mass value="0.025"','mass value="${housing_mass}"')+'\n</robot>\n')
            (directory/'gripper.xacro').write_text(header+'\n'+'\n'.join(rows[arm_end:])+'\n</robot>\n')
            controls=''.join(f'<joint name="{name}"><param name="id">{i+1}</param><command_interface name="position"/><state_interface name="position"/></joint>'
                             for i,name in enumerate([f'Axis_{j+1}' for j in range(count)]+['Grip_L','Grip_R']))
            (directory/'ros2_control.xacro').write_text(header+f'\n<ros2_control name="TestSystem" type="system"><hardware><plugin>test/Placeholder</plugin></hardware>{controls}</ros2_control></robot>\n')
            includes=''.join(f'<xacro:include filename="$(find {pkg})/{part}.xacro"/>\n' for part in ('arm','gripper','ros2_control'))
            (directory/'robot.urdf.xacro').write_text(header+'\n'+includes+'</robot>\n')
        else:
            rows += [link('Jaw_1',mesh='finger.stl',xyz='0 0 .035',scale='1 1 1.1',mass=.025),
                     joint('GripJaw','Tool_0','Jaw_1','revolute',xyz='.015 0 .01',axis='0 -1 0',limits='0 1.2')]
            (directory/'robot.urdf').write_text('<robot name="five_jaw">\n'+'\n'.join(rows)+'\n</robot>\n')
    bad = {
      'bad_missing_mesh': '<link name="A"><visual><geometry><mesh filename="missing.stl"/></geometry></visual></link>',
      'bad_unknown_package': '<link name="A"><visual><geometry><mesh filename="package://missing/shape.stl"/></geometry></visual></link>',
      'bad_closed_loop': '<link name="A"/><link name="B"/>'+joint('AB','A','B')+joint('BA','B','A'),
      'bad_mimic_of_mimic': '<link name="A"/><link name="B"/><link name="C"/><link name="D"/>'+joint('J1','A','B','prismatic',limits='0 .1')+joint('J2','B','C','prismatic',limits='0 .1',mimic='<mimic joint="J1"/>')+joint('J3','C','D','prismatic',limits='0 .1',mimic='<mimic joint="J2"/>'),
      'bad_gripper_on_chain': '<link name="A"/><link name="B"/><link name="C"/>'+joint('Arm','A','B','revolute')+joint('Grip','B','C','prismatic',limits='0 .1'),
    }
    for name,content in bad.items():
        (ROOT/f'{name}.urdf').write_text(f'<robot name="{name}">{content}</robot>\n')


if __name__=='__main__':
    build()
