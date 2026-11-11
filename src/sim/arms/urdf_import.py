"""Import a ROS description package into a simulation arm folder.

SDK imports are local and occur only after sim.app.launch. A failed simulator
step is recorded in import-report.json; source/flag errors fail before publish.
"""
import argparse
import json
import hashlib
from pathlib import Path
import shutil

import numpy as np


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, target = value.partition('=')
        if not separator or not name or not target or name in result:
            raise ValueError(f'{label} expects unique NAME=VALUE pairs: {value!r}')
        result[name] = target
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('name')
    result.add_argument('--urdf', type=Path, required=True)
    result.add_argument('--ee_link', required=True)
    result.add_argument('--gripper', action='append', default=[])
    result.add_argument('--base_link')
    result.add_argument('--camera', action='append', default=[])
    result.add_argument('--package_path', action='append', default=[])
    result.add_argument('--ros2_control', type=Path)
    result.add_argument('--rename', action='append', default=[])
    result.add_argument('--grasp_depth_m', type=float, default=.01)
    result.add_argument('--continuous_limit_deg', type=float, default=180.)
    result.add_argument('--force', action='store_true')
    result.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    return result


def stage_sources(args):
    import tempfile
    import tomllib
    import xml.etree.ElementTree as ET
    from core.arms.urdf import UrdfModel, expand_xacro, find_packages
    from core.arms.arm_toml import ImportChoices

    if not args.name.isidentifier() or args.name.startswith('_'):
        raise ValueError(f'Arm name {args.name!r} must be a public Python identifier')
    if not np.isfinite(args.grasp_depth_m) or args.grasp_depth_m < 0:
        raise ValueError('--grasp_depth_m must be finite and nonnegative')
    source = args.urdf.expanduser().resolve(strict=True)
    packages = find_packages(source)
    packages.update({name: Path(path).expanduser().resolve(strict=True)
                     for name, path in _pairs(args.package_path, '--package_path').items()})
    target = args.repo.resolve() / 'arms' / args.name
    if target.exists():
        if not args.force:
            raise FileExistsError(f'{target} exists; --force only overwrites an imported arm')
        try:
            imported = tomllib.loads((target/'arm.toml').read_text())['arm']['import']
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(f'--force refuses to overwrite {target}: missing [arm.import]') from error
        if not isinstance(imported, dict):
            raise ValueError(f'--force refuses to overwrite {target}: invalid [arm.import]')
    # Validate expansion, mesh resolution and semantic flags before publishing.
    text = expand_xacro(source, packages)
    if args.ros2_control is not None:
        root, control = ET.fromstring(text), ET.parse(args.ros2_control).getroot()
        blocks = [control] if control.tag == 'ros2_control' else control.findall('ros2_control')
        if not blocks:
            raise ValueError(f'{args.ros2_control}: missing <ros2_control> element')
        root.extend(blocks)
        text = ET.tostring(root, encoding='unicode')
    model = UrdfModel.from_xml(text, source.parent, packages, continuous_limit_deg=args.continuous_limit_deg)
    choices = ImportChoices(args.name, args.ee_link, tuple(args.gripper), urdf=str(source),
        base_link=args.base_link, camera=_pairs(args.camera, '--camera'),
        package_path={name:str(path) for name,path in packages.items()},
        ros2_control=str(args.ros2_control) if args.ros2_control else None,
        rename=_pairs(args.rename,'--rename'), grasp_depth_m=args.grasp_depth_m,
        continuous_limit_deg=args.continuous_limit_deg, force=args.force)
    choices.selected_joints(model)
    # Keep all explicitly supplied packages and every package referenced by an
    # include/mesh; copying nearby unrelated packages is unnecessary provenance.
    input_root = next((directory for directory in (source.parent, *source.parents)
                       if (directory/'package.xml').is_file()), source.parent)
    roots = {input_root, *(Path(value).resolve() for value in _pairs(args.package_path, '--package_path').values())}
    for package, directory in packages.items():
        if f'package://{package}/' in text or f'$(find {package})' in source.read_text():
            roots.add(directory)
    for link in model.links.values():
        for geometry in (*link.visuals, *link.collisions):
            if geometry.mesh_path is not None:
                mesh_root = next((p for p in geometry.mesh_path.parents if (p/'package.xml').is_file()), None)
                if mesh_root is not None:
                    roots.add(mesh_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.'+args.name+'-import-', dir=target.parent))
    temporary.chmod(0o755)
    copied = []
    try:
        for index, directory in enumerate(sorted(roots)):
            destination = temporary/'source'/f'{index}_{directory.name}'
            shutil.copytree(directory, destination, ignore=shutil.ignore_patterns('.git','__pycache__','.venv'))
            copied.append((directory, destination.relative_to(temporary)))
        if args.ros2_control:
            destination = temporary/'source'/'external_ros2_control.xml'
            shutil.copy2(args.ros2_control, destination)
        for path in (temporary/'source').rglob('*'):
            if path.is_file():
                choices.source_hashes[str(path.relative_to(temporary))] = hashlib.sha256(path.read_bytes()).hexdigest()
        # The expanded original preserves provenance. A second converter input
        # resolves mesh URIs against the copied sources and bounds continuous axes.
        (temporary/'source'/f'{args.name}.urdf').write_text(text)
        document = ET.fromstring(text)
        for element in document.findall('.//mesh'):
            from core.arms.urdf import resolve_mesh
            original = resolve_mesh(element.attrib['filename'], source.parent, packages)
            candidates = [(root, relative) for root,relative in copied if original.is_relative_to(root)]
            if not candidates:
                destination = temporary/'source'/'external_meshes'/hashlib.sha256(str(original).encode()).hexdigest()[:12]/original.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(original,destination)
                choices.source_hashes[str(destination.relative_to(temporary))] = hashlib.sha256(original.read_bytes()).hexdigest()
                resolved = target/destination.relative_to(temporary)
            else:
                root, relative = max(candidates,key=lambda item:len(item[0].parts))
                resolved = target/relative/original.relative_to(root)
            element.set('filename',str(resolved))
        for element in document.findall('joint'):
            joint = model.joints[element.attrib['name']]
            if joint.continuous:
                element.set('type','revolute')
                limit = element.find('limit')
                if limit is None:
                    limit = ET.SubElement(element,'limit')
                limit.set('lower',str(joint.lower)); limit.set('upper',str(joint.upper))
                limit.set('effort',str(joint.effort)); limit.set('velocity',str(joint.velocity))
        (temporary/'source'/'conversion.urdf').write_text(ET.tostring(document,encoding='unicode'))
        (temporary/'usd').mkdir()
        (temporary/f'{args.name}_utils.py').write_text('"""Optional owner-supplied arm hooks. No hardware driver is generated."""\n')
        if target.exists():
            shutil.rmtree(target)
        temporary.rename(target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return model, dict(directory=target, source=target/'source'/'conversion.urdf',
                       usd=target/'usd'/f'{args.name}.usd', choices=choices)


def convert_to_usd(model, paths):
    from pxr import Usd, UsdPhysics
    choices = paths['choices']
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        cfg = UrdfConverterCfg(asset_path=str(paths['source']), usd_dir=str(paths['usd'].parent/'converted'),
            usd_file_name=paths['usd'].name, fix_base=True, merge_fixed_joints=False,
            collision_type='Convex Hull', make_instanceable=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='position'),
            run_asset_transformer=False)
        converter = UrdfConverter(cfg)
        generated = Path(converter.usd_path)
        choices.conversion_path = 'isaaclab.sim.converters.UrdfConverter (3.0 URDFImporterConfig)'
    except Exception as error:
        choices.fallbacks.append(dict(step='convert', error=str(error), fallback='Isaac Sim URDFImporter extension'))
        from isaaclab.sim import enable_extension
        enable_extension('isaacsim.asset.importer.urdf')
        from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
        cfg = URDFImporterConfig(urdf_path=str(paths['source']), usd_path=str(paths['usd'].parent/'fallback'),
                                 fix_base=True, merge_fixed_joints=False, joint_target_type='position',
                                 collision_type='Convex Hull', run_asset_transformer=False)
        generated = Path(URDFImporter(cfg).import_urdf())
        choices.conversion_path = 'isaacsim.asset.importer.urdf.URDFImporter'
    stage = Usd.Stage.Open(str(generated))
    if stage is None:
        raise ValueError(f'Converter did not produce a readable USD: {generated}')
    # Flatten layer references into the deliverable; keep converter materials and
    # mesh assets in usd/ as needed. Never merge named fixed links.
    stage.Flatten().Export(str(paths['usd']))
    stage = Usd.Stage.Open(str(paths['usd']))
    # Isaac Sim 3.0 nests links in Geometry/. The scene contract addresses named
    # bodies directly below the articulation root. Reparent through USD's
    # namespace editor so joint relationships follow, retaining world transforms.
    from pxr import UsdGeom
    root=stage.GetDefaultPrim()
    transforms=UsdGeom.XformCache()
    root_inverse=transforms.GetLocalToWorldTransform(root).GetInverse()
    links=[]
    for name in model.links:
        matches=[prim for prim in stage.Traverse() if prim.GetName()==name and prim.IsA(UsdGeom.Xform)]
        if len(matches)!=1:
            raise ValueError(f'Converted link {name!r} must resolve uniquely; found {len(matches)}')
        prim=matches[0]
        links.append((str(prim.GetPath()),name,transforms.GetLocalToWorldTransform(prim)*root_inverse))
    for source,name,transform in sorted(links,key=lambda row:len(row[0]),reverse=True):
        destination=str(root.GetPath().AppendChild(name))
        if source!=destination:
            editor=Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath(source,destination)
            if not editor.ApplyEdits():
                raise ValueError(f'Cannot preserve named converted link {name!r} at articulation root')
        xform=UsdGeom.Xformable(stage.GetPrimAtPath(destination))
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(transform)
    _, fingers = choices.selected_joints(model)
    bodies = {joint.child for joint in fingers}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            ancestor = prim
            while ancestor and ancestor.GetName() not in bodies:
                ancestor = ancestor.GetParent()
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set('convexDecomposition' if ancestor else 'convexHull')
    stage.GetRootLayer().Save()
    return paths['usd']


def conversion_checks(model, paths, spec):
    from pxr import Usd, UsdGeom
    from sim.datagen.ik import ArmKinematics
    stage = Usd.Stage.Open(str(paths['usd']))
    names = {prim.GetName() for prim in stage.Traverse()}
    missing_links = sorted(set(model.links)-names)
    missing_joints = sorted({joint.name for joint in model.joints.values() if joint.type != 'fixed'}-names)
    checks = [dict(name='conversion_names', passed=not missing_links and not missing_joints,
                   missing_links=missing_links, missing_joints=missing_joints,
                   metres_per_unit=UsdGeom.GetStageMetersPerUnit(stage))]
    kinematics = ArmKinematics.from_arm(spec, position_offset_m=[0.,0.,0.])
    from sim.arms.ready_pose import bounds
    limits = bounds(spec)
    rng = np.random.default_rng(101)
    samples = np.vstack([np.zeros(len(limits)),rng.uniform(limits[:,0],limits[:,1],size=(20,len(limits)))])
    results = []
    inverse_names = {v:k for k,v in paths['choices'].rename.items()}
    for q in samples:
        oracle = model.fk({inverse_names.get(j.name,j.name):float(v) for j,v in zip(spec.action_joints,q)},
                          spec.links.base,spec.links.ee)
        actual = kinematics.fk(q)
        distance = float(np.linalg.norm(actual[:3,3]-oracle[:3,3]))
        angle = float(np.arccos(np.clip((np.trace(actual[:3,:3].T@oracle[:3,:3])-1)/2,-1.,1.)))
        results.append(dict(position_error_m=distance,orientation_error_rad=angle,q=q.tolist()))
    checks.append(dict(name='fk_cross_check', passed=all(r['position_error_m']<=.001 and r['orientation_error_rad']<=.01 for r in results),
                       poses=results, max_position_error_m=max(r['position_error_m'] for r in results),
                       max_orientation_error_rad=max(r['orientation_error_rad'] for r in results)))
    return checks


def measure_gripper(stage, spec, *, grasp_depth_m=.01):
    import torch
    from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
    from sim.datagen.ik import ArmKinematics
    from sim.scene.geometry import collision_body_points
    from sim.arms.ready_pose import bounds

    kin = ArmKinematics.from_arm(spec, position_offset_m=[0.,0.,0.])
    grip = spec.gripper
    limits = bounds(spec)
    source_id = spec.joint_names.index(grip.joints[0])
    q = np.clip(0.,limits[:,0],limits[:,1])
    body_names = list(grip.contact_bodies)
    if len(body_names) == 1:
        body_names.append(spec.links.ee)
    labels = {label.rsplit('/',1)[-1]:index for index,label in enumerate(kin._model.body_label)}
    default = stage.GetDefaultPrim()
    body_prims = {prim.GetName():prim for prim in stage.Traverse() if prim.GetName() in body_names}
    clouds = {name:collision_body_points(stage,str(body_prims[name].GetPath())) for name in body_names}
    def points(coordinate):
        q[source_id]=coordinate
        poses = torch.as_tensor(kin._evaluate(q),dtype=torch.float64)
        ee=poses[kin.ee_id]
        result={}
        for name in body_names:
            body=poses[labels[name]]
            position,rotation = subtract_frame_transforms(ee[:3]*kin._meters_per_unit,ee[3:],
                                                          body[:3]*kin._meters_per_unit,body[3:])
            result[name]=clouds[name]@matrix_from_quat(rotation).numpy().T+position.numpy()
        return result
    low, high = limits[source_id]
    first,last=points(low),points(high)
    motion=last[body_names[0]].mean(axis=0)-first[body_names[0]].mean(axis=0)
    if np.linalg.norm(motion)<1e-8:
        raise ValueError('Selected gripper source produces no measurable finger motion')
    closing=motion/np.linalg.norm(motion)
    centroid=np.concatenate(list(first.values())).mean(axis=0)
    approach=centroid-closing*np.dot(centroid,closing)
    if np.linalg.norm(approach)<1e-8:
        raise ValueError('Cannot derive an approach axis perpendicular to gripper closing motion')
    approach/=np.linalg.norm(approach)
    sampled=[]
    for coordinate in np.linspace(low,high,21):
        cloud=points(coordinate)
        tips={name:p[(p@approach)>=(p@approach).max()-.02] for name,p in cloud.items()}
        ordered=sorted(tips,key=lambda name:float(tips[name].mean(axis=0)@closing))
        left,right=tips[ordered[0]],tips[ordered[-1]]
        a,b=float((left@closing).max()),float((right@closing).min())
        gap=max(0.,b-a)
        centre=np.concatenate([left[np.isclose(left@closing,a,atol=1e-6)],
                               right[np.isclose(right@closing,b,atol=1e-6)]]).mean(axis=0)
        tip=float(max((p@approach).max() for p in tips.values()))
        centre += approach*(tip-centre@approach-grasp_depth_m)
        sampled.append(dict(q=float(coordinate),gap_m=gap,centre_m=centre.tolist(),
                            fingertips={name:p.tolist() for name,p in tips.items()}))
    closed=min(sampled,key=lambda row:row['gap_m'])
    opened=max(sampled,key=lambda row:row['gap_m'])
    if opened['gap_m']<=closed['gap_m']+1e-8:
        raise ValueError('Finger collision sweep has no measurable gap range')
    return dict(closed_gap_m=closed['gap_m'],open_gap_m=opened['gap_m'],
                closed_position=closed['q'],open_position=opened['q'],closing_axis=closing.tolist(),
                approach_axis=approach.tolist(),position_offset_m=closed['centre_m'],
                fingertip_points=closed['fingertips'],sweep=sampled)


def measure_cameras(stage,model,choices):
    import torch
    from pxr import UsdGeom
    from isaaclab.utils.math import euler_xyz_from_quat,quat_from_matrix
    result={}
    names={prim.GetName():prim for prim in stage.Traverse() if prim.GetName() in model.links}
    for link,name in choices.camera.items():
        incoming=next((joint for joint in model.joints.values() if joint.child==link),None)
        parent=incoming.parent if incoming is not None and incoming.type=='fixed' else link
        matrix,_=UsdGeom.XformCache().ComputeRelativeTransform(names[link],names[parent])
        values=np.asarray(matrix).T.copy()
        rotation=torch.as_tensor(values[:3,:3],dtype=torch.float64)
        euler=euler_xyz_from_quat(quat_from_matrix(rotation).unsqueeze(0))
        result[name]=dict(parent_link=parent,position=(values[:3,3]*UsdGeom.GetStageMetersPerUnit(stage)).tolist(),
                          euler_degrees=[float(np.degrees(value[0])) for value in euler])
    return result


def derive_actuators(spec, kinematics, model, *, sample_positions=None):
    import math
    import newton
    import warp as wp
    from sim.arms.ready_pose import reach_samples

    kin=kinematics
    positions=sample_positions if sample_positions is not None else reach_samples(spec,kin,count=100)[0]
    maximum=np.zeros(kin._model.joint_dof_count)
    gravity_buffer=wp.zeros(kin._model.joint_dof_count,dtype=wp.float32,device=kin._model.device)
    for q in positions:
        kin._evaluate(q)
        newton.eval_inverse_dynamics_passive(kin._model,kin._state,gravity_force=gravity_buffer)
        gravity=gravity_buffer.numpy()
        maximum=np.maximum(maximum,np.abs(gravity))
    ready=np.array([joint.initial for joint in spec.action_joints])
    kin._evaluate(ready)
    matrix=newton.eval_mass_matrix(kin._model,kin._state).numpy()[0]
    indices={joint.name:(int(dof),i) for i,(joint,dof) in enumerate(zip(spec.action_joints,kin._dof_ids))}
    result={}
    for joint in spec.joints:
        source=joint.mimic.joint if joint.mimic else joint.name
        dof,_=indices[source]
        effort=model.joints[joint.usd].effort
        if effort<=0:
            raise ValueError(f'joint {joint.usd!r}: URDF effort is absent or zero; actuator gain needs confirmation')
        inertia=float(matrix[dof,dof])
        if not math.isfinite(inertia) or inertia<=0:
            raise ValueError(f'joint {joint.usd!r}: Newton effective inertia is not positive')
        sag=math.radians(.5) if joint.type=='revolute' else .0005
        stiffness=min(max(float(maximum[dof])/sag,16*inertia/.5**2),effort/.02)
        if joint.role=='gripper':
            stroke=abs((math.radians(joint.limits[1]-joint.limits[0]) if joint.type=='revolute' else joint.limits[1]-joint.limits[0]))
            stiffness=max(stiffness,16*inertia/.5**2,effort/max(stroke,1e-6))
        damping=2*math.sqrt(stiffness*inertia)
        result[joint.usd]=dict(effort_limit_sim=effort,stiffness=stiffness,damping=damping,
            evidence=dict(worst_gravity_torque_or_force=float(maximum[dof]),effective_inertia=inertia,
                          sag_limit_si=sag,samples=len(positions),rule='critical damping; gravity sag; effort cap',
                          stiffness_clamped=bool(float(maximum[dof])/sag>effort/.02)))
    return result


def _write(paths, model, checks):
    from core.arms.arm_toml import build_arm_toml,build_report
    choices=paths['choices']
    directory=paths['directory']
    (directory/'arm.toml').write_text(build_arm_toml(model,choices))
    (directory/'arm.toml').chmod(0o644)
    report=build_report(model,choices,checks)
    (directory/choices.report_path).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (directory/choices.report_path).chmod(0o644)
    return report


def _spec(paths):
    import tomllib
    from core.arms.spec import _parse_spec
    return _parse_spec(paths['choices'].name,paths['directory'],tomllib.loads((paths['directory']/'arm.toml').read_text())['arm'])


def main(argv=None):
    import sys
    from sim.app import launch,close_app
    args=parser().parse_args(argv)
    try:
        model,paths=stage_sources(args)
    except Exception as error:
        parser().exit(2,f'URDF sources/flags: {error}\n')
    choices=paths['choices']
    choices.command_line=list(sys.argv if argv is None else argv)
    checks=[]
    app=None
    try:
        app=launch(headless=True,enable_cameras=True)
        def step(name,operation,fallback):
            try:
                result=operation()
                checks.append(dict(name=name,passed=True))
                _write(paths,model,checks)
                return result
            except Exception as error:
                checks.append(dict(name=name,passed=False,error=str(error)))
                choices.fallbacks.append(dict(step=name,error=str(error),fallback=fallback))
                _write(paths,model,checks)
                return None
        usd=step('convert_to_usd',lambda:convert_to_usd(model,paths),'Write manifest and report; no replacement kinematics or physics')
        if usd is not None:
            _write(paths,model,checks)
            spec=step('load_spec',lambda:_spec(paths),'Keep generated manifest for owner review')
            if spec is not None:
                rows=step('conversion_cross_check',lambda:conversion_checks(model,paths,spec),'Record FK mismatch and continue')
                checks.extend(rows or [])
                from pxr import Usd
                measured=step('measure_gripper',lambda:measure_gripper(Usd.Stage.Open(str(usd)),spec,grasp_depth_m=args.grasp_depth_m),
                              'Keep zero TCP with TODO; no gap map until measured')
                cameras=step('measure_cameras',lambda:measure_cameras(Usd.Stage.Open(str(usd)),model,choices),'Keep declared link and confirm camera transform')
                choices.cameras=cameras or {}
                if measured:
                    choices.measurements=measured
                    _write(paths,model,checks); spec=_spec(paths)
                from sim.datagen.ik import ArmKinematics
                kin=step('newton_kinematics',lambda:ArmKinematics.from_arm(spec),'Keep clamped-zero ready pose and TODO actuator gains')
                if kin is not None:
                    from sim.arms.ready_pose import ready_pose,base_yaw_for_scene
                    ready=step('ready_pose',lambda:ready_pose(spec,kin),'Use zero clamped into limits, tagged TODO')
                    if ready:
                        q,choices.ready_info=ready
                        choices.ready_positions={joint.usd:float(value) for joint,value in zip(spec.action_joints,q)}
                        if not choices.ready_info['converged']:
                            choices.fallbacks.append(dict(step='ready_pose',fallback='zero clamped into limits',error='No grid target converged'))
                        _write(paths,model,checks); spec=_spec(paths)
                    from core.arms.spec import load_arm_spec
                    from core.paths import DEFAULT_ARM
                    yaw=step('base_yaw',lambda:base_yaw_for_scene(spec,load_arm_spec(DEFAULT_ARM)[0],kinematics=kin),
                             'Keep zero base yaw, tagged TODO')
                    if yaw is not None:
                        choices.base_yaw_degrees=float(yaw); choices.yaw_info=dict(derived=True,confirm=True)
                    gains=step('derive_actuators',lambda:derive_actuators(spec,kin,model,
                        sample_positions=np.array([row['q'] for row in choices.ready_info.get('candidates',[]) if row['converged']])
                            if choices.ready_info.get('converged') else None),'Keep zero gains tagged TODO; do not modify production physics')
                    if gains:
                        choices.actuators=gains
                _write(paths,model,checks)
                step('hardware_placeholders',lambda:write_hardware_placeholders(model,paths),'Record hardware TODOs; simulation uses declared joint units only')
                rows=step('validate_import',lambda:validate_import(args.name,repo=args.repo),'Keep failed validation evidence in report')
                if rows and any(not row['passed'] for row in rows):
                    correction=step('workspace_mount_fallback',lambda:derive_workspace_mount(_spec(paths),paths),
                                    'Keep default anchor and report unreachable/unstable workspace')
                    if correction:
                        choices.fallbacks.append(dict(step='validate_import',error='Default mount intersects tabletop',
                            fallback='Derived world translation from base collision support; confirm mount',
                            previous_checks=rows,mount=correction))
                        rows=step('validate_with_derived_mount',lambda:validate_import(args.name,repo=args.repo),
                                  'Keep failed check evidence; never change physics')
                checks.extend(rows or [])
                for row in rows or []:
                    if row['name']=='mimic_tracking':
                        choices.mimic_runtime=row.get('runtime','manual')
    except Exception as error:
        checks.append(dict(name='simulator',passed=False,error=str(error)))
        choices.fallbacks.append(dict(step='simulator',error=str(error),fallback='Write arm folder with unresolved checks'))
    finally:
        report=_write(paths,model,checks)
        for key,row in report['fields'].items():
            if row['tag'] in ('confirm','TODO') or row.get('confirm'):
                print(f"{row['tag'].upper()} {key}: {row['value']} ({row['reason']})")
        for row in checks:
            if not row['passed']:
                print('FAILED '+row['name']+': '+json.dumps(row))
        import os
        owner=args.repo.stat()
        for path in (paths['directory'],*paths['directory'].rglob('*')):
            try:
                os.chown(path,owner.st_uid,owner.st_gid,follow_symlinks=False)
            except PermissionError:
                pass
        print(f"Import report: {paths['directory']/choices.report_path}",flush=True)
        if app is not None:
            close_app(app,success=True)
    return 0


def write_hardware_placeholders(model,paths):
    choices=paths['choices']
    arm,fingers=choices.selected_joints(model)
    if not any(j.ros2_control for j in (*arm,*fingers)) and not choices.ros2_control:
        return None
    mapping={}
    excluded=[]
    for joint in (*arm,*fingers):
        if joint.mimic:
            continue
        name=choices.rename.get(joint.name,joint.name)
        if joint.type=='prismatic':
            if 'counts_per_metre' not in joint.ros2_control:
                excluded.append(name);continue
            low,high=joint.lower,joint.upper
        else:
            low,high=-180.,(4095-2048)*360/4096
        mapping[name]=dict(joint_min=low,joint_max=high)
    payload=dict(schema_version=1,arm=choices.name,unit_id='factory',mapping=mapping,
                 source=dict(kind='urdf_import_placeholder',confirm=True,excluded_prismatic_joints=excluded))
    directory=paths['directory']/'calibration'/'units'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'factory.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload


def derive_workspace_mount(spec,paths):
    from pxr import Usd
    import torch
    from isaaclab.utils.math import matrix_from_quat
    from core.paths import DEFAULT_SCENE
    from core.scenes import load_scene_spec,load_mount_spec,mount_pose
    from core.scenes.anchors import quaternion_matrix
    from sim.datagen.ik import ArmKinematics
    from sim.scene.geometry import collision_body_points,table_plane
    import tomllib

    scene=load_scene_spec(DEFAULT_SCENE)
    destination=scene.directory/'mounts'/(spec.name+'.toml')
    if destination.exists():
        data=tomllib.loads(destination.read_text())
        if data.get('mount',{}).get('source',{}).get('importer')!='urdf':
            return None
    mount=load_mount_spec(scene,spec)
    pose=mount_pose(scene,mount,spec)
    room=Usd.Stage.Open(str(scene.room_usd))
    plane=table_plane(room,room.GetPseudoRoot(),scene.anchors['table'])
    stage=Usd.Stage.Open(str(spec.usd_path))
    points=collision_body_points(stage,str(stage.GetDefaultPrim().GetPath())+'/'+spec.links.base)
    kin=ArmKinematics.from_arm(spec)
    coordinates=kin._evaluate(np.array([joint.initial for joint in spec.action_joints]))
    body=coordinates[kin.base_id]
    points=points@matrix_from_quat(torch.tensor(body[3:])).numpy().T+body[:3]*kin._meters_per_unit
    points=points@quaternion_matrix(pose.orientation_xyzw).T+pose.position
    clearance=points[:,2]-plane[0]*points[:,0]-plane[1]*points[:,1]-plane[2]
    correction=max(0.,-float(clearance.min())+.0002)
    if correction<=.001:
        return None
    # A mount is the specified fallback for workspace placement. This only fits
    # the imported base support; neither room nor robot physics is changed.
    current=np.asarray(mount.translation_correction_m,dtype=float)
    current[2]+=correction
    destination.parent.mkdir(exist_ok=True)
    destination.write_text('[mount]\nanchor = '+json.dumps(mount.anchor)+'\ntranslation_correction_m = '+json.dumps(current.tolist())+
        '\n\n[mount.source]\nimporter = "urdf"\ntag = "derived"\nconfirm = true\nreason = "Base collision support fitted above measured default tabletop"\n')
    destination.chmod(0o644)
    return dict(path=str(destination),translation_correction_m=current.tolist(),base_penetration_m=-float(clearance.min()),tag='derived',confirm=True)


def validate_import(name,*,repo=None):
    from sim.arms.validation import validate_import as validate
    return validate(name,repo=repo)


if __name__=='__main__':
    raise SystemExit(main())
