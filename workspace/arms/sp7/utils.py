"""SP7 appearance using the same fitted beige and motor colours as SO-101."""

_KINEMATICS = {}


def _prepare_gripper_guides(root):
    """Separate sliding guides from jaws so overlapping CAD guide faces cannot jam.

    The URDF fingers contain the opposing slide carriages as well as the pads.
    Their guide faces overlap by about 0.13 mm in the source CAD. Represent each
    region with its conservative convex envelope and filter only guide/guide
    and guide/opposing-jaw pairs. Jaw/jaw and every distant arm pair stay active.
    The split is 25 mm along the finger body Y axis, behind the grasp surface.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    cache = UsdGeom.XformCache()
    parts = {}
    for body_name in ('gripper_left_link', 'gripper_right_link'):
        body = root.GetChild(body_name)
        if not body:
            continue
        existing = [body.GetChild('s2r_guide_collision'), body.GetChild('s2r_jaw_collision')]
        if all(existing):
            parts[body_name] = existing
            continue
        meshes = [p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Mesh)
                  and p.HasAPI(UsdPhysics.CollisionAPI)
                  and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        if len(meshes) != 1:
            raise ValueError(f'Expected one original collision mesh for {body_name}, got {len(meshes)}')
        source = UsdGeom.Mesh(meshes[0])
        transform = cache.ComputeRelativeTransform(source.GetPrim(), body)[0]
        vertices = np.array([transform.Transform(Gf.Vec3d(*map(float, p))) for p in source.GetPointsAttr().Get()])
        counts = list(source.GetFaceVertexCountsAttr().Get())
        indices = list(source.GetFaceVertexIndicesAttr().Get())
        faces = []; start = 0
        for count in counts:
            faces.append(vertices[indices[start:start + count]])
            start += count
        created = []
        for name, direction in [('guide', -1), ('jaw', 1)]:
            clipped = []
            for polygon in faces:
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    da, db = direction * (a[1] - .025), direction * (b[1] - .025)
                    if da >= 0:
                        clipped.append(a)
                    if (da >= 0) != (db >= 0):
                        clipped.append(a + (b - a) * da / (da - db))
            points = np.unique(np.round(clipped, 9), axis=0)
            hull = ConvexHull(points)
            mesh = UsdGeom.Mesh.Define(root.GetStage(), body.GetPath().AppendChild(f's2r_{name}_collision'))
            mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
            mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
            mesh.CreateFaceVertexIndicesAttr(hull.simplices.flatten().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            created.append(mesh.GetPrim())
        UsdPhysics.CollisionAPI(source.GetPrim()).CreateCollisionEnabledAttr(False)
        parts[body_name] = created
    if len(parts) == 2:
        left, right = parts['gripper_left_link'], parts['gripper_right_link']
        for first, second in [(left[0], right[0]), (left[0], right[1]), (right[0], left[1])]:
            UsdPhysics.FilteredPairsAPI.Apply(first).CreateFilteredPairsRel().AddTarget(second.GetPath())


def _prepare_wrist_cavities(root):
    """Keep the concave wrist brackets hollow using baked URDF mesh pieces."""
    import json
    from pathlib import Path
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if not all(root.GetChild(name) for name in ('link5', 'link7')):
        return
    path = Path(__file__).parent / 'urdf' / 'derived' / 'wrist-collisions.json'
    data = json.loads(path.read_text())
    for name, pieces in data['parts'].items():
        body = root.GetChild(name)
        if body.GetChild('s2r_wrist_collision_0000'):
            continue
        for prim in Usd.PrimRange(body):
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        for index, part in enumerate(pieces):
            mesh = UsdGeom.Mesh.Define(root.GetStage(), body.GetPath().AppendChild(f's2r_wrist_collision_{index:04d}'))
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in part['points_m']])
            mesh.CreateFaceVertexCountsAttr([3] * len(part['faces']))
            mesh.CreateFaceVertexIndicesAttr([v for face in part['faces'] for v in face])
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateVisibilityAttr('invisible')
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')


def prepare_collisions(root):
    """Keep self-contact active except within welded or joint-adjacent assemblies.

    The URDF retains separate bodies for motor housings and mounting parts.
    Filter adjacency after grouping fixed joints, so these connected parts do
    not fight their own joints. Distant pairs such as gripper/base stay active.
    """
    from itertools import combinations
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    _prepare_gripper_guides(root)
    _prepare_wrist_cavities(root)

    # The builder's large default gap produces tens of thousands of speculative
    # contacts between separated finger hulls. A 2 mm detection gap retains real
    # self-contact while keeping the contact buffers available for near pairs.
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.CreateAttribute('newton:contactGap', Sdf.ValueTypeNames.Float).Set(.002)

    bodies = {str(p.GetPath()): p for p in Usd.PrimRange(root)
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    parents = {path: path for path in bodies}

    def find(path):
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    joints = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        first, second = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(first) != 1 or len(second) != 1:
            continue
        a, b = str(first[0]), str(second[0])
        if a not in bodies or b not in bodies:
            continue
        joints.append((a, b))
        if prim.IsA(UsdPhysics.FixedJoint):
            parents[find(b)] = find(a)
    adjacent = {frozenset((find(a), find(b))) for a, b in joints if find(a) != find(b)}
    for a, b in combinations(bodies, 2):
        if find(a) == find(b) or frozenset((find(a), find(b))) in adjacent:
            UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())


def make_kinematics(arm, *, position_offset_m=None):
    from sim.datagen.ik import ArmKinematics
    offset = tuple(arm.ik.position_offset_m if position_offset_m is None else position_offset_m)
    key = (str(arm.usd_path), offset)
    if key not in _KINEMATICS:
        _KINEMATICS[key] = ArmKinematics.from_arm(arm, position_offset_m=offset)
    return _KINEMATICS[key]


def make_ik(robot, arm, **kwargs):
    from sim.datagen.bounded_ik import BoundedPoseStep
    from sim.datagen.ik import ArticulationIK
    controller = ArticulationIK(robot, arm, **kwargs)
    controller._step = BoundedPoseStep(arm, controller.options,
        make_kinematics(arm, position_offset_m=controller.position_offset_m))
    controller.library_name = 'SciPy bounded least_squares; Newton FK and analytic Jacobians'
    controller.method_name = 'bounded_nonlinear_pose'
    controller.mode_name = 'measured_pose_feedback'
    return controller


def apply_appearance(stage, robot_root, look):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    materials = {}
    for name, color, roughness in (
        ('BeigePlastic', look['robot_beige'], look['robot_plastic_roughness']),
        ('MotorCharcoal', look['motor_charcoal'], look['motor_roughness']),
    ):
        material = UsdShade.Material.Define(stage, robot_root + '/Looks/' + name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[name] = material
    root = stage.GetPrimAtPath(robot_root)
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        body = str(prim.GetPath())[len(robot_root):].strip('/').split('/')[0]
        material = materials['MotorCharcoal' if body.lower().startswith('servo') else 'BeigePlastic']
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants)
