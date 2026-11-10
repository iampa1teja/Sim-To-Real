"""Measurements of static scene geometry, cached for the lifetime of a USD stage."""

import weakref

import numpy as np
from pxr import Gf, Usd, UsdGeom

from core.arms.spec import load_arm_spec


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


def gripper_tip_local_offset(stage, half_size: float, clearance_m: float, gripper_path: str,
                             mesh_name: str | None = None) -> tuple[float, float, float]:
    jaw_points = _jaw_points(stage, gripper_path, mesh_name or fixed_jaw_mesh_name())
    tip_z = jaw_points[:, 2].min()
    tip_band = jaw_points[jaw_points[:, 2] <= tip_z + 2 * half_size]
    return (
        float(tip_band[:, 0].max() + half_size + clearance_m),
        float((tip_band[:, 1].min() + tip_band[:, 1].max()) / 2),
        float(tip_z + half_size),
    )


def table_plane(stage, root_prim):
    """Return z = ax + by + c for a static tabletop in world metres.

    Cache keys include the stage and environment path; clones never share a world
    plane accidentally. Call clear_geometry_cache after changing scene geometry.
    """
    cache = _GEOMETRY_CACHE.setdefault(stage, {})
    key = ("table", str(root_prim.GetPath()))
    if key not in cache:
        cache[key] = _measure_table_plane(stage, root_prim)
    return cache[key].copy()


def _measure_table_plane(stage, root_prim):
    table = next((p for p in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
                  if p.GetName() == "Tabletop" and p.IsA(UsdGeom.Mesh)), None)
    if table is None:
        raise RuntimeError("No Tabletop mesh found; cannot measure spawn clearance.")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, point)))
                       for point in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    return np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)[0]


def cache_scene_geometry(env, env_ids):
    """Measure static geometry once at startup, before episode reset/spawn events."""
    from sim.envs.mdp.resets import _env_prim_path, env_indices
    arm, _ = load_arm_spec()
    if arm.gripper is None or not arm.gripper.tip_mesh:
        return

    stage = env.sim.stage
    robot = env.scene["robot"]
    mesh_name = fixed_jaw_mesh_name()
    for index in env_indices(env, env_ids).tolist():
        table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]))
        _jaw_points(stage, _env_prim_path(env, robot, index) + "/" + arm.links.ee, mesh_name)
