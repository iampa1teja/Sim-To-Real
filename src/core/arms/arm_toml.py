"""Render an imported arm manifest and field-by-field evidence without SDKs."""
from dataclasses import dataclass, field
import math
import json

import numpy as np

from core.arms.urdf import UrdfModel

IMPORTER_VERSION = '1'


@dataclass
class ImportChoices:
    name: str
    ee_link: str
    gripper: tuple[str, ...]
    urdf: str = ''
    base_link: str | None = None
    camera: dict[str, str] = field(default_factory=dict)  # link -> camera name
    package_path: dict[str, str] = field(default_factory=dict)
    ros2_control: str | None = None
    rename: dict[str, str] = field(default_factory=dict)
    grasp_depth_m: float = .01
    continuous_limit_deg: float = 180.
    force: bool = False
    measurements: dict = field(default_factory=dict)
    ready_positions: dict[str, float] = field(default_factory=dict)
    ready_info: dict = field(default_factory=dict)
    base_yaw_degrees: float = 0.
    yaw_info: dict = field(default_factory=dict)
    actuators: dict = field(default_factory=dict)
    cameras: dict = field(default_factory=dict)
    source_hashes: dict = field(default_factory=dict)
    command_line: list[str] = field(default_factory=list)
    conversion_path: str = 'pending'
    report_path: str = 'import-report.json'
    fallbacks: list = field(default_factory=list)
    mimic_runtime: str = 'manual'

    def selected_joints(self, model):
        base = self.base_link or model.default_base()
        arm = model.movable_chain(base, self.ee_link)
        if not arm:
            raise ValueError(f'ee link {self.ee_link!r} must be below a movable arm chain')
        subtree = {joint.name: joint for joint in model.subtree_joints(self.ee_link)}
        if not self.gripper:
            raise ValueError('--gripper is required; subtree proposals must be confirmed explicitly')
        if len(self.gripper) != len(set(self.gripper)):
            raise ValueError('Duplicate --gripper flags')
        for name in self.gripper:
            if name not in subtree:
                raise ValueError(f'--gripper {name!r} must be in the ee subtree, not on the base-to-ee chain')
            if subtree[name].mimic is not None:
                raise ValueError(f'--gripper {name!r} is a mimic; select its source explicitly')
        selected = set(self.gripper)
        fingers = [model.joints[name] for name in self.gripper]
        fingers += [joint for joint in subtree.values() if joint.mimic is not None and joint.mimic[0] in selected]
        undeclared = {joint.name for joint in model.subtree_joints(base)} - {joint.name for joint in (*arm,*fingers)}
        if undeclared:
            raise ValueError(f'More than one end effector or unassigned movable ee subtree joints: {sorted(undeclared)}')
        if any(joint.mimic is not None for joint in arm):
            raise ValueError('Arm-joint mimics are unsupported; only gripper mimics are accepted')
        unknown = set(self.rename) - {joint.name for joint in (*arm, *fingers)}
        if unknown:
            raise ValueError(f'--rename names unknown selected joints: {sorted(unknown)}')
        names = [self.rename.get(j.name, j.name) for j in (*arm, *fingers)]
        if len(names) != len(set(names)):
            raise ValueError('--rename creates duplicate joint names')
        for link, name in self.camera.items():
            if link not in model.links:
                raise ValueError(f'--camera link {link!r} does not exist')
            chain = model.chain(self.ee_link, link)
            if any(joint.type != 'fixed' for joint in chain):
                raise ValueError(f'--camera {link!r} must be fixed below the ee')
            if not name or list(self.camera.values()).count(name) != 1:
                raise ValueError(f'Duplicate or empty camera name {name!r}')
        return arm, fingers


def _literal(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(str(key)) + ' = ' + _literal(item) for key, item in value.items()) + '}'
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(_literal(item) for item in value) + ']'
    return json.dumps(value, allow_nan=False)


def _render(model: UrdfModel, choices: ImportChoices):
    arm, fingers = choices.selected_joints(model)
    fields, lines = {}, ['# Generated from URDF. Read import-report.json before confirming this arm.']
    def table(path):
        lines.extend(('', '[' + path + ']'))
    def emit(path, key, value, tag, reason, *, confirm=False):
        fields[path + '.' + key] = dict(value=json.loads(_literal_json(value)), tag=tag, reason=reason)
        if confirm:
            fields[path + '.' + key]['confirm'] = True
        lines.append(f'# {tag}' + ('; confirm' if confirm else '') + ': ' + reason.replace('\n', ' '))
        lines.append(json.dumps(key) + ' = ' + _literal(value))
    table('arm')
    emit('arm', 'name', choices.name, 'from-flag', '--name')
    emit('arm', 'usd_path', f'usd/{choices.name}.usd', 'derived', 'Converted USD in this arm folder')
    emit('arm', 'utils', f'{choices.name}_utils.py', 'derived', 'Optional hooks placeholder')
    table('arm.links')
    emit('arm.links', 'base', choices.base_link or model.default_base(),
         'from-flag' if choices.base_link else 'confirm', '--base_link or structural fixed-root proposal')
    emit('arm.links', 'ee', choices.ee_link, 'from-flag', '--ee_link')
    positions = {joint.name: float(np.clip(0., joint.lower, joint.upper)) for joint in (*arm, *fingers)}
    positions.update(choices.ready_positions)
    for joint in fingers:
        if joint.mimic is not None:
            source, multiplier, offset = joint.mimic
            positions[joint.name] = positions[source] * multiplier + offset
    for index, joint in enumerate((*arm, *fingers)):
        path = f'arm.joints.{index}'
        lines.extend(('', '[[arm.joints]]'))
        emit(path, 'name', choices.rename.get(joint.name, joint.name),
             'from-flag' if joint.name in choices.rename else 'derived', 'URDF name or --rename')
        emit(path, 'usd', joint.name, 'derived', 'Unmerged converted URDF joint name')
        emit(path, 'type', joint.type, 'derived', 'URDF type; continuous joints receive declared finite limits')
        emit(path, 'role', 'arm' if index < len(arm) else 'gripper', 'from-flag', 'base-to-ee chain from --ee_link / --gripper subtree')
        limits = [math.degrees(v) for v in (joint.lower, joint.upper)] if joint.type == 'revolute' else [joint.lower, joint.upper]
        emit(path, 'limits', limits, 'confirm' if joint.continuous else 'derived',
             '--continuous_limit_deg' if joint.continuous else 'URDF radians converted to degrees, prismatic metres preserved')
        emit(path, 'initial', positions[joint.name], 'derived' if choices.ready_info.get('converged') else 'TODO',
             'Converged ready pose' if choices.ready_info.get('converged') else 'Ready-pose fallback: zero clamped to URDF limits')
        if joint.mimic is not None:
            source, multiplier, offset = joint.mimic
            emit(path, 'mimic', dict(joint=choices.rename.get(source, source), multiplier=multiplier, offset=offset),
                 'derived', 'URDF mimic source, multiplier and SI offset')
    table('arm.gripper')
    parallel = all(joint.type == 'prismatic' for joint in fingers)
    emit('arm.gripper', 'kind', 'parallel' if parallel else 'jaw', 'derived', 'Types of explicitly selected gripper joints')
    emit('arm.gripper', 'joints', [choices.rename.get(j.name, j.name) for j in fingers if j.mimic is None], 'from-flag', '--gripper sources only')
    bodies = [j.child for j in fingers]
    emit('arm.gripper', 'contact_bodies', bodies, 'confirm', 'Child links of selected fingers including mimic joints')
    emit('arm.gripper', 'friction_bodies', list(dict.fromkeys((*bodies, choices.ee_link))), 'derived', 'Contact bodies and ee link')
    for key in ('closed_gap_m', 'open_gap_m', 'closed_position', 'open_position', 'closing_axis', 'approach_axis'):
        if key in choices.measurements:
            emit('arm.gripper', key, choices.measurements[key], 'measured', 'Converted collision geometry sweep', confirm=True)
        else:
            fields['arm.gripper.' + key] = dict(value=None, tag='TODO', reason='Collision geometry measurement not available')
    table('arm.ik')
    emit('arm.ik', 'orientation_axes', ['x', 'y'] if len(arm) == 5 else ['x', 'y', 'z'] if len(arm) >= 6 else [],
         'confirm', 'Joint-count structural proposal; confirm controllable orientation axes')
    emit('arm.ik', 'method', 'dls', 'derived', 'Isaac Lab library IK')
    emit('arm.ik', 'joint_limit_avoidance_gain', .5 if len(arm) > 6 else 0., 'derived', 'Spare joint for seven-axis redundancy')
    emit('arm.ik', 'position_offset_m', choices.measurements.get('position_offset_m', [0., 0., 0.]),
         'measured' if 'position_offset_m' in choices.measurements else 'TODO',
         'Measured inner fingertip centre minus grasp depth along approach axis' if choices.measurements else 'Grasp-centre measurement unavailable', confirm=True)
    for link, name in choices.camera.items():
        path = 'arm.cameras.' + json.dumps(name)
        table(path)
        values = dict(parent_link=link, position=[0., 0., 0.], euler_degrees=[0., 0., 0.])
        values.update(choices.cameras.get(name, {}))
        values.update(width=640, height=480, horizontal_aperture=20.955,
                      vertical_aperture=20.955 * 480/640, focal_length=20.955 / (2 * math.tan(math.pi/6)))
        for key, value in values.items():
            emit(path, key, value, 'confirm', 'Fixed camera transform and default 640x480 60-degree horizontal FOV')
    for index, joint in enumerate((*arm, *fingers)):
        path = f'arm.actuators.joint_{index}'
        table(path)
        values = choices.actuators.get(joint.name)
        if values is None:
            values = dict(effort_limit_sim=joint.effort, stiffness=0., damping=0.)
        emit(path, 'joint_names_expr', [joint.name], 'derived', 'One actuator per converted joint')
        for key, value in values.items():
            if key != 'evidence':
                emit(path, key, value, 'derived' if joint.name in choices.actuators else 'TODO',
                     'URDF effort; Newton gravity torque and mass matrix' if joint.name in choices.actuators else 'Actuator derivation unavailable')
    table('arm.spawn')
    emit('arm.spawn', 'activate_contact_sensors', False, 'derived', 'Template default')
    table('arm.spawn.articulation_props')
    emit('arm.spawn.articulation_props', 'self_collision_enabled', False, 'derived', 'Template default')
    table('arm.initial_state')
    emit('arm.initial_state', 'position', [0., 0., 0.], 'derived', 'Template default; scene anchor owns placement')
    emit('arm.initial_state', 'rotation_relative_to_anchor', True, 'derived', 'Apply the ready-forward yaw relative to the shared mount anchor')
    emit('arm.initial_state', 'rotation_euler_degrees', [0., 0., choices.base_yaw_degrees],
         'derived' if choices.yaw_info.get('derived') else 'TODO', 'Forward direction aligned to default reference arm', confirm=True)
    if model.has_ros2_control or choices.ros2_control:
        table('arm.hardware')
        for key, value in dict(motor='unconfigured', encoder_counts_per_turn=4096, zero_count=2048,
                joint_ids={choices.rename.get(j.name, j.name): j.ros2_control['id']
                           for j in (*arm, *fingers) if 'id' in j.ros2_control}).items():
            emit('arm.hardware', key, value, 'confirm', 'ros2_control ids; assumed encoder defaults require hardware confirmation')
        for joint in (*arm, *fingers):
            if joint.type == 'prismatic' and 'counts_per_metre' not in joint.ros2_control:
                fields[f'arm.hardware.{joint.name}.counts_per_metre'] = dict(value='TODO', tag='TODO',
                    reason='URDF linear stroke does not imply rotary motor transmission; excluded from factory unit')
    fields['arm.lerobot.driver'] = dict(value='TODO', tag='TODO', reason='Real rollout needs an owner-supplied driver')
    lines.extend(('', '# TODO: Real rollout needs an owner-supplied driver.', '# [arm.lerobot]',
                  '# follower_type = "TODO"', '# leader_type = "TODO"'))
    table('arm.import')
    for key, value in dict(source_hashes=choices.source_hashes, command_line=choices.command_line,
                          importer_version=IMPORTER_VERSION, conversion_path=choices.conversion_path,
                          report_path=choices.report_path, mimic_runtime=choices.mimic_runtime).items():
        emit('arm.import', key, value, 'derived', 'Import provenance; no hardware or control configuration')
    return '\n'.join(lines) + '\n', fields


def _literal_json(value):
    return json.dumps(value, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item(), allow_nan=False)


def build_arm_toml(model, choices):
    return _render(model, choices)[0]


def build_report(model, choices, checks):
    _, fields = _render(model, choices)
    return dict(importer_version=IMPORTER_VERSION, arm=choices.name, fields=fields,
                checks=checks, fallbacks=choices.fallbacks, measurements=choices.measurements,
                ready_pose=choices.ready_info, base_yaw=choices.yaw_info,
                actuator_evidence={name: values.get('evidence', {}) for name, values in choices.actuators.items()})
