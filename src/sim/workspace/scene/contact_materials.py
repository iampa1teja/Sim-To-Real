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


def enable_collision_meshes(root, arm=None):
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
    from utils.arms.spec import load_arm_spec, load_arm_utils

    arm = arm or load_arm_spec()[0]
    utilities = load_arm_utils(arm)
    prepare = getattr(utilities, "prepare_collisions", None)
    if prepare is not None:
        changes = prepare(root)
        if changes is not None:
            replaced, pieces = changes
            authored = [path for path in authored if path not in replaced]
            authored.extend(pieces)
    return tuple(authored)


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
