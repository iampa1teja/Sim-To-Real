"""Measurements of static scene geometry, cached for the lifetime of a USD stage."""

import weakref

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from utils.arms.spec import load_arm_spec
from utils.scene import load_scene_spec


def fixed_jaw_mesh_name() -> str:
    gripper = load_arm_spec()[0].gripper
    if gripper is None or not gripper.tip_mesh:
        raise ValueError("Jaw-relative placement needs [arm.gripper] tip_mesh")
    return gripper.tip_mesh


_GEOMETRY_CACHE = weakref.WeakKeyDictionary()


def clear_geometry_cache(stage):
    """Invalidate measurements after editing static meshes or table transforms."""
    _GEOMETRY_CACHE.pop(stage, None)


def _jaw_points(stage, gripper_path, mesh_name):
    cache = _GEOMETRY_CACHE.setdefault(stage, {})
    key = ("jaw", gripper_path, mesh_name)
    if key not in cache:
        points = _measure_jaw_points(stage, gripper_path, mesh_name)
        points.setflags(write=False)
        cache[key] = points
    return cache[key]


def _measure_jaw_points(stage, gripper_path, mesh_name):
    gripper = stage.GetPrimAtPath(gripper_path)
    if not gripper.IsValid():
        raise RuntimeError(f"Fixed-jaw link not found: {gripper_path}")
    marker = f"/visuals/{mesh_name or fixed_jaw_mesh_name()}/"
    mesh_prim = next(
        (p for p in Usd.PrimRange(gripper, Usd.TraverseInstanceProxies())
         if p.IsA(UsdGeom.Mesh) and marker in str(p.GetPath())),
        None,
    )
    if mesh_prim is None:
        raise RuntimeError(f"Fixed-jaw mesh matching {marker!r} not found under {gripper_path}; cannot measure jaw tip.")

    local_transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh_prim, gripper)
    jaw_points = np.array([
        local_transform.Transform(Gf.Vec3d(*map(float, p)))
        for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
    ])
    jaw_points *= UsdGeom.GetStageMetersPerUnit(stage)
    return jaw_points


def collision_body_points(stage, body_path):
    """Static collision vertices in a rigid body's local metre frame."""
    cache = _GEOMETRY_CACHE.setdefault(stage, {})
    key = ('collision', str(body_path))
    if key not in cache:
        body = stage.GetPrimAtPath(body_path)
        if not body.IsValid():
            raise ValueError(f'Contact body not found: {body_path}')
        clouds = []
        transforms = UsdGeom.XformCache()
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            collider = prim
            while collider and collider != body and not collider.HasAPI(UsdPhysics.CollisionAPI):
                collider = collider.GetParent()
            if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
                continue
            if prim.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            else:
                extent = UsdGeom.Boundable(prim).ComputeExtent(Usd.TimeCode.Default())
                if extent is None:
                    continue
                points = [[x, y, z] for x in (extent[0][0], extent[1][0])
                          for y in (extent[0][1], extent[1][1]) for z in (extent[0][2], extent[1][2])]
            relative, _ = transforms.ComputeRelativeTransform(prim, body)
            clouds.extend(relative.Transform(Gf.Vec3d(*map(float, point))) for point in points)
        if not clouds:
            raise ValueError(f'No collision geometry on contact body {body_path}')
        points = np.asarray(clouds) * UsdGeom.GetStageMetersPerUnit(stage)
        points.setflags(write=False)
        cache[key] = points
    return cache[key]


def parallel_fingertip_points(stage, robot_path, ee_name, contact_bodies, approach_axis,
                              *, tip_depth_m=.02, body_to_ee=None):
    """Collision tip bands per contact body, transformed into the ee metre frame.

    Cache body-local vertices, never moving finger-to-ee transforms. Live callers
    may supply measured body transforms rather than authored stage transforms.
    """
    axis = np.asarray(approach_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    ee = stage.GetPrimAtPath(str(robot_path) + '/' + ee_name)
    transforms = UsdGeom.XformCache()
    result = {}
    for name in contact_bodies:
        path = str(robot_path) + '/' + name
        points = collision_body_points(stage, path)
        if body_to_ee is not None:
            matrix = body_to_ee[name]
            local = points @ matrix[:3, :3].T + matrix[:3, 3]
        else:
            relative, _ = transforms.ComputeRelativeTransform(stage.GetPrimAtPath(path), ee)
            matrix = np.asarray(relative).T.copy()
            matrix[:3, 3] *= UsdGeom.GetStageMetersPerUnit(stage)
            local = points @ matrix[:3, :3].T + matrix[:3, 3]
        extent = local @ axis
        result[name] = local[extent >= extent.max() - tip_depth_m]
    return result


def gripper_tip_local_offset(stage, half_size: float, clearance_m: float, gripper_path: str,
                             mesh_name: str | None = None, *, gripper=None,
                             body_to_ee=None) -> tuple[float, float, float]:
    if gripper is not None and gripper.kind == 'parallel':
        if gripper.approach_axis is None or gripper.closing_axis is None:
            raise ValueError('Parallel placement needs measured approach_axis and closing_axis')
        robot_path, ee_name = gripper_path.rsplit('/', 1)
        clouds = parallel_fingertip_points(stage, robot_path, ee_name, gripper.contact_bodies,
                                          gripper.approach_axis, body_to_ee=body_to_ee)
        points = np.concatenate(list(clouds.values()))
        closing = np.asarray(gripper.closing_axis, dtype=float)
        closing /= np.linalg.norm(closing)
        # Spawn beside the outer finger surface with explicit clearance, so a
        # closed ready gripper never begins interpenetrating the object.
        centre = points.mean(axis=0)
        centre += closing * ((points @ closing).max() + half_size + clearance_m - centre @ closing)
        return tuple(centre)

    jaw_points = _jaw_points(stage, gripper_path, mesh_name or fixed_jaw_mesh_name())
    tip_z = jaw_points[:, 2].min()
    tip_band = jaw_points[jaw_points[:, 2] <= tip_z + 2 * half_size]
    return (
        float(tip_band[:, 0].max() + half_size + clearance_m),
        float((tip_band[:, 1].min() + tip_band[:, 1].max()) / 2),
        float(tip_z + half_size),
    )


def table_plane(stage, root_prim, anchor=None):
    """Return z = ax + by + c for a static tabletop in world metres.

    Cache keys include the stage and environment path; clones never share a world
    plane accidentally. Call clear_geometry_cache after changing scene geometry.
    """
    anchor = load_scene_spec().anchors.get("table") if anchor is None else anchor
    if anchor is None:
        raise ValueError("Scene has no table anchor; cannot measure spawn clearance")
    cache = _GEOMETRY_CACHE.setdefault(stage, {})
    key = ("table", str(root_prim.GetPath()), anchor.prim_name, tuple(anchor.position), tuple(anchor.up))
    if key not in cache:
        cache[key] = _measure_table_plane(stage, root_prim, anchor)
    return cache[key].copy()


def anchor_prim(stage, root_prim, anchor):
    """Resolve a declared prim path or prim name within one environment."""
    if not anchor.prim_name:
        return None
    path = anchor.prim_name
    if "/" in path:
        prim = stage.GetPrimAtPath(path if path.startswith("/") else
                                   str(root_prim.GetPath().AppendPath(path)))
        return prim if prim.IsValid() else None
    return next((prim for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
                 if prim.GetName() == path), None)


def _measure_table_plane(stage, root_prim, anchor):
    table = anchor_prim(stage, root_prim, anchor)
    if table is None:
        if anchor.prim_name:
            raise RuntimeError(f"Table anchor prim {anchor.prim_name!r} not found; cannot measure spawn clearance")
        normal = np.asarray(anchor.up, dtype=float)
        if abs(normal[2]) < 1e-12:
            raise ValueError("Table anchor must have a nonvertical surface")
        return np.array([-normal[0] / normal[2], -normal[1] / normal[2],
                         np.dot(normal, anchor.position) / normal[2]])
    if not table.IsA(UsdGeom.Mesh):
        raise ValueError(f"Table anchor {anchor.prim_name!r} must reference a Mesh")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, point)))
                       for point in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    return np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)[0]


def cache_scene_geometry(env, env_ids):
    """Measure scene anchors and arm tip geometry once before resets."""
    from sim.workspace.envs.mdp._common import _env_prim_path, env_indices

    arm = getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]
    scene = getattr(env.cfg, "scene_spec", None) or load_scene_spec()
    table = scene.anchors.get("table")
    stage = env.sim.stage
    robot = env.scene["robot"]
    for index in env_indices(env, env_ids).tolist():
        if table is not None:
            table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]), table)
        if arm.gripper is not None and arm.gripper.kind == 'parallel':
            for name in arm.gripper.contact_bodies:
                collision_body_points(stage, _env_prim_path(env, robot, index) + '/' + name)
        elif arm.gripper is not None and arm.gripper.tip_mesh:
            _jaw_points(stage, _env_prim_path(env, robot, index) + "/" + arm.links.ee, arm.gripper.tip_mesh)
