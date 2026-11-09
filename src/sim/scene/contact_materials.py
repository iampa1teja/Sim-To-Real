"""Author collision schemas and contact materials before Newton imports the scene."""

from pxr import Usd, UsdGeom, UsdPhysics


CONTACT_FRICTION = 0.5
GRASP_FRICTION = 1.5
GRASP_ROLLING_FRICTION = 0.002
CONTACT_SOLREF = (0.004, 1.0)
CONTACT_SOLIMP = (0.999, 0.999, 0.001, 0.5, 2.0)


def contact_material(friction=CONTACT_FRICTION, *, rolling_friction=None):
    """A solver-common material; Newton uses dynamic friction for its shape coefficient."""
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )
    if rolling_friction is None:
        return material
    from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonMaterialCfg

    return [material, NewtonMaterialCfg(rolling_friction=rolling_friction)]


def contact_properties():
    """Stiff frictional contacts shared by the cube, jaws and measured tabletop."""
    from isaaclab.sim.schemas.schemas_cfg import UsdPhysicsCollisionCfg
    from isaaclab_newton.sim.schemas.schemas_cfg import MujocoCollisionCfg

    return [UsdPhysicsCollisionCfg(), MujocoCollisionCfg(
        condim=6, solref=CONTACT_SOLREF, solimp=CONTACT_SOLIMP)]


def enable_collision_meshes(root):
    """Move inherited collision intent from legacy Xform containers onto actual meshes.

    The arm USD puts CollisionAPI on instanceable ``collisions`` Xforms. Newton only
    imports meshes carrying CollisionAPI themselves. Deinstance those containers in
    the composed runtime stage so their children can receive the defining schema;
    the source asset, visual instances, transforms and joint topology stay intact.
    """
    containers = [prim for prim in Usd.PrimRange(root)
                  if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim)]
    authored = []
    for container in containers:
        enabled = UsdPhysics.CollisionAPI(container).GetCollisionEnabledAttr().Get()
        # MJWarp needs full-dimensional convex meshes. PhysX SDFs are unsupported,
        # and decomposing this asset generates planar fragments MuJoCo rejects.
        # Author runtime hulls on the existing mesh children; retain the source USD.
        approximation = "convexHull"
        if container.IsInstance():
            container.SetInstanceable(False)
        for mesh in Usd.PrimRange(container):
            if not mesh.IsA(UsdGeom.Mesh) or mesh.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(bool(enabled))
            UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
            authored.append(str(mesh.GetPath()))
        # The actual meshes now own collision. Retaining the legacy aggregate would
        # also register its children as PhysX mesh-merge members during rendering.
        for schema in ("PhysxMeshMergeCollisionAPI", "PhysxSDFMeshCollisionAPI", "PhysxConvexHullCollisionAPI"):
            container.RemoveAppliedSchema(schema)
        container.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        container.RemoveAPI(UsdPhysics.CollisionAPI)
    for split in (_split_fixed_finger, _split_moving_finger):
        replaced, pieces = split(root)
        if replaced in authored:
            authored.remove(replaced)
        authored.extend(pieces)
    from sim.scene.gripper_collisions import apply_primitive_pads

    replaced, pieces = apply_primitive_pads(root)
    authored = [path for path in authored if path not in replaced]
    authored.extend(pieces)
    return tuple(authored)


def _split_fixed_finger(root):
    """Keep the SO101 finger opening out of the proximal body's convex hull.

    The CAD finger face is x=-0.0079 m below local z=-0.04 m. A hull of
    the entire L-shaped part fills that opening and overlaps a valid cube
    placement by several millimetres. Two hulls retain that contact face.
    """
    return _split_link_mesh(root, link_name="gripper", mesh_name="wrist_roll_follower_so101_v1",
                            axis=2, coordinate=-0.04)


def _split_moving_finger(root):
    """Keep the moving finger's proximal body out of its distal contact hull."""
    return _split_link_mesh(root, link_name="jaw", mesh_name="moving_jaw_so101_v1",
                            axis=1, coordinate=-0.04)


def _split_link_mesh(root, *, link_name, mesh_name, axis, coordinate):
    """Clip the source CAD at a link-local plane and cook two convex pieces."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from pxr import Gf, Vt

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    if not link:
        return None, ()
    source = next((prim for prim in Usd.PrimRange(link)
                   if prim.IsA(UsdGeom.Mesh) and
                   f"/collisions/{mesh_name}/" in str(prim.GetPath())), None)
    if source is None or not source.HasAPI(UsdPhysics.CollisionAPI) or not (
            UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get()):
        return None, ()
    mesh = UsdGeom.Mesh(source)
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(source, link)
    points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, point)))
                         for point in mesh.GetPointsAttr().Get()])
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    intersections, offset = [], 0
    for count in mesh.GetFaceVertexCountsAttr().Get():
        face = indices[offset:offset + count]
        offset += count
        for index in range(count):
            p, q = points[face[index]], points[face[(index + 1) % count]]
            if (p[axis] - coordinate) * (q[axis] - coordinate) < 0:
                intersections.append(p + (coordinate - p[axis]) / (q[axis] - p[axis]) * (q - p))
    if not intersections:
        raise ValueError(f"Split plane does not cross collision mesh {source.GetPath()}")
    intersections = np.asarray(intersections)
    inverse, created = transform.GetInverse(), []
    for name, mask in (("finger_contact", points[:, axis] <= coordinate),
                       ("body_contact", points[:, axis] >= coordinate)):
        piece = np.unique(np.round(np.vstack((points[mask], intersections)), 10), axis=0)
        if np.linalg.matrix_rank(piece - piece.mean(axis=0), tol=1e-10) < 3:
            raise ValueError(f"Split creates a planar collision part at {source.GetPath()}/{name}")
        hull = ConvexHull(piece)
        vertices = piece[hull.vertices]
        remap = {int(index): position for position, index in enumerate(hull.vertices)}
        faces = hull.simplices.copy()
        # Qhull reports outward plane normals; orient every triangle accordingly.
        normals = np.cross(piece[faces[:, 1]] - piece[faces[:, 0]],
                           piece[faces[:, 2]] - piece[faces[:, 0]])
        for face, normal, equation in zip(faces, normals, hull.equations):
            if np.dot(normal, equation[:3]) < 0:
                face[1], face[2] = face[2], face[1]
        local = [inverse.Transform(Gf.Vec3d(*map(float, point))) for point in vertices]
        path = source.GetPath().GetParentPath().AppendChild(name)
        part = UsdGeom.Mesh.Define(root.GetStage(), path)
        part.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in local]))
        part.CreateFaceVertexCountsAttr([3] * len(faces))
        part.CreateFaceVertexIndicesAttr([remap[int(index)] for index in faces.reshape(-1)])
        part.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(part.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(part.GetPrim()).CreateApproximationAttr("convexHull")
        created.append(str(path))
    UsdPhysics.CollisionAPI(source).CreateCollisionEnabledAttr(False)
    return str(source.GetPath()), tuple(created)


def bind_contact_material(root, predicate, *, friction=CONTACT_FRICTION, rolling_friction=None):
    """Bind the contact material only to selected enabled collision geometry."""
    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material_from_fragments
    from isaaclab.sim.utils import bind_physics_material
    from isaaclab_newton.sim.schemas.schemas import apply_mujoco_collision

    colliders = [prim for prim in Usd.PrimRange(root)
                 if prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI)
                 and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                 and predicate(prim)]
    if not colliders:
        return ()
    stage = root.GetStage()
    material_path = str(root.GetPath().AppendChild("ContactMaterial"))
    spawn_physics_material_from_fragments(
        material_path, contact_material(friction, rolling_friction=rolling_friction), stage=stage)
    _, mujoco_properties = contact_properties()
    for prim in colliders:
        apply_mujoco_collision(mujoco_properties, str(prim.GetPath()), stage=stage)
        bind_physics_material(
            str(prim.GetPath()), material_path, stage=stage, stronger_than_descendants=True,
        )
    return tuple(str(prim.GetPath()) for prim in colliders)
