"""Pure expanded-URDF description, units and test/report FK oracle.

Runtime FK/IK belongs to sim.datagen.ik. This oracle is intentionally used only
for conversion cross-checks and reports, never for control.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _numbers(text, count, label):
    try:
        values = np.asarray([float(v) for v in text.split()], dtype=float)
    except (ValueError, AttributeError) as error:
        raise ValueError(f'{label}: expected {count} finite numbers') from error
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f'{label}: expected {count} finite numbers')
    return values


def _number(value, label, default=None):
    if value is None and default is not None:
        return default
    return float(_numbers(str(value), 1, label)[0])


def _origin(element, label):
    result = np.eye(4)
    if element is None:
        return result
    result[:3, 3] = _numbers(element.get('xyz', '0 0 0'), 3, f'{label} xyz')
    r, p, y = _numbers(element.get('rpy', '0 0 0'), 3, f'{label} rpy')
    cr, cp, cy, sr, sp, sy = math.cos(r), math.cos(p), math.cos(y), math.sin(r), math.sin(p), math.sin(y)
    result[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    return result


@dataclass(frozen=True)
class UrdfInertial:
    mass: float
    origin: np.ndarray  # COM transform in link frame
    inertia: np.ndarray  # tensor rotated into link frame, about COM


@dataclass(frozen=True)
class UrdfGeometry:
    kind: str
    mesh_path: Path | None
    scale: np.ndarray
    origin: np.ndarray
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float
    mimic: tuple[str, float, float] | None = None
    ros2_control: dict = field(default_factory=dict)
    continuous: bool = False


@dataclass(frozen=True)
class UrdfLink:
    name: str
    inertial: UrdfInertial | None
    visuals: tuple[UrdfGeometry, ...]
    collisions: tuple[UrdfGeometry, ...]


def resolve_mesh(filename, base_dir, package_paths):
    if filename.startswith('package://'):
        package, separator, relative = filename[len('package://'):].partition('/')
        if package not in package_paths:
            raise ValueError(f'Missing package {package!r} for mesh {filename!r}; use --package_path {package}=dir')
        if not separator:
            raise ValueError(f'Mesh URI needs a path: {filename!r}')
        path = Path(package_paths[package]) / relative
    elif filename.startswith('file://'):
        path = Path(filename[len('file://'):])
    elif '://' in filename:
        raise ValueError(f'Unsupported mesh URI {filename!r}')
    else:
        path = Path(base_dir) / filename
    if not path.is_file():
        raise ValueError(f'Missing mesh file {filename!r}: {path}')
    return path.resolve()


def _geometry(element, base_dir, packages, label):
    geometry = element.find('geometry')
    if geometry is None or len(geometry) != 1:
        raise ValueError(f'{label}: expected exactly one geometry element')
    shape = geometry[0]
    scale = np.ones(3)
    path, parameters = None, {}
    if shape.tag == 'mesh':
        path = resolve_mesh(shape.get('filename', ''), base_dir, packages)
        scale = _numbers(shape.get('scale', '1 1 1'), 3, f'{label} mesh scale')
        if np.any(scale == 0):
            raise ValueError(f'{label}: mesh scale must be nonzero')
    elif shape.tag == 'box':
        parameters['size'] = _numbers(shape.get('size'), 3, f'{label} box size').tolist()
        if min(parameters['size']) <= 0:
            raise ValueError(f'{label}: box size must be positive')
    elif shape.tag in ('cylinder', 'sphere'):
        for key in ('radius', 'length') if shape.tag == 'cylinder' else ('radius',):
            parameters[key] = _number(shape.get(key), f'{label} {key}')
            if parameters[key] <= 0:
                raise ValueError(f'{label}: {key} must be positive')
    else:
        raise ValueError(f'{label}: unsupported geometry element <{shape.tag}>')
    return UrdfGeometry(shape.tag, path, scale, _origin(element.find('origin'), label), parameters)


@dataclass
class UrdfModel:
    name: str
    links: dict[str, UrdfLink]
    joints: dict[str, UrdfJoint]
    root: str
    has_ros2_control: bool = False

    @property
    def root_link(self):
        return self.root

    @classmethod
    def from_xml(cls, text, base_dir, package_paths=None, *, continuous_limit_deg=180.):
        packages = package_paths or {}
        try:
            robot = ET.fromstring(text)
        except ET.ParseError as error:
            raise ValueError(f'Invalid URDF XML: {error}') from error
        if robot.tag != 'robot':
            raise ValueError(f'Expected <robot>, found <{robot.tag}>')
        if not math.isfinite(continuous_limit_deg) or continuous_limit_deg <= 0:
            raise ValueError('continuous_limit_deg must be finite and positive')
        controls = {}
        for control in robot.findall('ros2_control'):
            for element in control.findall('joint'):
                name = element.get('name')
                controls.setdefault(name, {}).update({p.get('name'): (p.text or '').strip()
                                                      for p in element.findall('param')})
                for interface in (*element.findall('command_interface'), *element.findall('state_interface')):
                    controls[name].update({f'{interface.get("name")}.{p.get("name")}': (p.text or '').strip()
                                           for p in interface.findall('param')})
        links = {}
        for element in robot.findall('link'):
            name = element.get('name')
            if not name or name in links:
                raise ValueError(f'Duplicate or missing link name {name!r}')
            inertial_element = element.find('inertial')
            inertial = None
            if inertial_element is not None:
                origin = _origin(inertial_element.find('origin'), f'link {name} inertial')
                mass_element, tensor = inertial_element.find('mass'), inertial_element.find('inertia')
                if mass_element is None or tensor is None:
                    raise ValueError(f'link {name!r}: inertial requires mass and inertia')
                mass = _number(mass_element.get('value'), f'link {name} mass')
                if mass <= 0:
                    raise ValueError(f'link {name!r}: mass must be positive')
                xx, xy, xz, yy, yz, zz = [_number(tensor.get(key), f'link {name} {key}')
                                         for key in ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')]
                matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                if np.linalg.eigvalsh(matrix).min() < -1e-12:
                    raise ValueError(f'link {name!r}: inertia tensor must be positive semidefinite')
                inertial = UrdfInertial(mass, origin, origin[:3, :3] @ matrix @ origin[:3, :3].T)
            links[name] = UrdfLink(name, inertial,
                tuple(_geometry(e, base_dir, packages, f'link {name} visual') for e in element.findall('visual')),
                tuple(_geometry(e, base_dir, packages, f'link {name} collision') for e in element.findall('collision')))
        if not links:
            raise ValueError('URDF contains no <link> elements')
        joints, incoming = {}, {}
        for element in robot.findall('joint'):
            name, kind = element.get('name'), element.get('type')
            if not name or name in joints:
                raise ValueError(f'Duplicate or missing joint name {name!r}')
            if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
                raise ValueError(f'joint {name!r}: unsupported type {kind!r}')
            parent_element, child_element = element.find('parent'), element.find('child')
            if parent_element is None or child_element is None:
                raise ValueError(f'joint {name!r}: requires parent and child')
            parent, child = parent_element.get('link'), child_element.get('link')
            if parent not in links or child not in links:
                raise ValueError(f'joint {name!r}: unknown parent {parent!r} or child {child!r}')
            if child in incoming:
                raise ValueError(f'Closed loop at joint {name!r}, link {child!r} has multiple parents')
            incoming[child] = name
            axis_element = element.find('axis')
            axis = _numbers(axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0',
                            3, f'joint {name} axis')
            if np.linalg.norm(axis) < 1e-12:
                if kind=='fixed':
                    axis=np.array([1.,0.,0.])  # fixed axes have no physical meaning
                else:
                    raise ValueError(f'joint {name!r}: axis is zero')
            axis /= np.linalg.norm(axis)
            limit = element.find('limit')
            lower = upper = effort = velocity = 0.
            if kind != 'fixed':
                if kind == 'continuous':
                    lower, upper = -math.radians(continuous_limit_deg), math.radians(continuous_limit_deg)
                elif limit is None:
                    raise ValueError(f'joint {name!r}: missing <limit>')
                else:
                    lower = _number(limit.get('lower'), f'joint {name} lower')
                    upper = _number(limit.get('upper'), f'joint {name} upper')
                if lower >= upper:
                    raise ValueError(f'joint {name!r}: limits must increase')
                effort = _number(limit.get('effort'), f'joint {name} effort', 0.) if limit is not None else 0.
                velocity = _number(limit.get('velocity'), f'joint {name} velocity', 0.) if limit is not None else 0.
                if min(effort, velocity) < 0:
                    raise ValueError(f'joint {name!r}: effort and velocity must be nonnegative')
            mimic_element = element.find('mimic')
            mimic = None if mimic_element is None else (mimic_element.get('joint'),
                _number(mimic_element.get('multiplier'), f'joint {name} mimic multiplier', 1.),
                _number(mimic_element.get('offset'), f'joint {name} mimic offset', 0.))
            joints[name] = UrdfJoint(name, 'revolute' if kind == 'continuous' else kind, parent, child,
                _origin(element.find('origin'), f'joint {name}'), axis, lower, upper,
                effort, velocity, mimic, controls.get(name, {}), kind == 'continuous')
        for joint in joints.values():
            if joint.mimic is not None:
                source = joints.get(joint.mimic[0])
                if source is None:
                    raise ValueError(f'joint {joint.name!r}: missing mimic source {joint.mimic[0]!r}')
                if source.mimic is not None:
                    raise ValueError(f'joint {joint.name!r}: mimic of a mimic is unsupported')
                if source.type != joint.type or joint.type == 'fixed':
                    raise ValueError(f'joint {joint.name!r}: mimic source must have the same movable type')
        # Validate all components, including cycles disconnected from the root.
        for link in links:
            seen, current = set(), link
            while current in incoming:
                if current in seen:
                    raise ValueError(f'Closed loop at link {current!r}, joint {incoming[current]!r}')
                seen.add(current)
                current = joints[incoming[current]].parent
        roots = set(links) - set(incoming)
        if len(roots) != 1:
            raise ValueError(f'URDF must be one connected tree; roots: {sorted(roots)}')
        return cls(robot.get('name', 'robot'), links, joints, roots.pop(), bool(robot.findall('ros2_control')))

    def chain(self, base_link, ee_link):
        if base_link not in self.links or ee_link not in self.links:
            raise ValueError(f'Unknown base {base_link!r} or ee link {ee_link!r}')
        incoming = {joint.child: joint for joint in self.joints.values()}
        result, current = [], ee_link
        while current != base_link:
            if current not in incoming:
                raise ValueError(f'ee link {ee_link!r} is not below base {base_link!r}')
            joint = incoming[current]
            result.append(joint)
            current = joint.parent
        return list(reversed(result))

    def movable_chain(self, base_link, ee_link):
        return [joint for joint in self.chain(base_link, ee_link) if joint.type != 'fixed']

    def default_base(self):
        children = [joint for joint in self.joints.values() if joint.parent == self.root]
        return children[0].child if len(children) == 1 and children[0].type == 'fixed' else self.root

    def subtree_joints(self, link):
        if link not in self.links:
            raise ValueError(f'Unknown subtree link {link!r}')
        result, queue = [], [link]
        while queue:
            parent = queue.pop(0)
            for joint in self.joints.values():
                if joint.parent == parent:
                    queue.append(joint.child)
                    if joint.type != 'fixed':
                        result.append(joint)
        return result

    def fk(self, q, base, tip):
        """URDF report oracle only; q is a name-to-SI-coordinate dictionary."""
        transform = np.eye(4)
        for joint in self.chain(base, tip):
            coordinate = q.get(joint.name, 0.)
            if joint.mimic is not None:
                source, multiplier, offset = joint.mimic
                coordinate = q.get(source, 0.) * multiplier + offset
            motion = np.eye(4)
            if joint.type == 'revolute':
                x, y, z = joint.axis
                cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                motion[:3, :3] += math.sin(coordinate) * cross + (1 - math.cos(coordinate)) * (cross @ cross)
            elif joint.type == 'prismatic':
                motion[:3, 3] = coordinate * joint.axis
            transform = transform @ joint.origin @ motion
        return transform


def find_packages(start):
    """Discover package.xml names in ancestors and nearby sibling package folders."""
    start = Path(start).resolve()
    start = start.parent if start.is_file() else start
    candidates = set()
    # ROS packages are normally adjacent under src/, with URDF one or two levels
    # below their package root. Avoid recursively scanning unrelated workspaces.
    ancestors=(start,*start.parents)
    for directory in ancestors:
        candidates.add(directory/'package.xml')
    nearest=next((directory for directory in ancestors if (directory/'package.xml').is_file()),None)
    neighbours=(start,*list(start.parents)[:3]) if nearest is None else (start,nearest,nearest.parent)
    for directory in neighbours:
        candidates.add(directory / 'package.xml')
        candidates.update(directory.glob('*/package.xml'))
        candidates.update(directory.glob('*/*/package.xml'))
    packages = {}
    for manifest in sorted(candidates):
        if manifest.is_file():
            try:
                name = ET.parse(manifest).getroot().findtext('name')
            except ET.ParseError as error:
                raise ValueError(f'Invalid package manifest {manifest}: {error}') from error
            if name:
                packages.setdefault(name.strip(), manifest.parent.resolve())
    return packages


def expand_xacro(path, package_paths):
    """Use xacro itself, including includes/properties and ros2_control XML.

    Resolve $(find) through the supplied package map without depending on ROS or
    ament. Restore xacro's resolver even when expansion fails.
    """
    import xacro
    import xacro.substitution_args

    path = Path(path)
    original = xacro.substitution_args._eval_find

    def find(name):
        if name not in package_paths:
            raise ValueError(f'Missing package {name!r} in xacro; use --package_path {name}=dir')
        return str(Path(package_paths[name]).resolve())

    xacro.substitution_args._eval_find = find
    try:
        return xacro.process_file(str(path)).toxml()
    except Exception as error:
        raise ValueError(f'xacro expansion failed for {path}: {error}') from error
    finally:
        xacro.substitution_args._eval_find = original
