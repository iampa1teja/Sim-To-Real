import numpy as np
from pxr import Gf, Usd, UsdGeom

from core.arms.spec import load_arm_spec


def fixed_jaw_mesh_name() -> str:
    links = load_arm_spec()[0].get("links", {})
    if "fixed_jaw_mesh" not in links:
        raise KeyError("arm.toml needs [arm.links] fixed_jaw_mesh (the fixed-jaw mesh name under the ee link's visuals)")
    return links["fixed_jaw_mesh"]


def gripper_tip_local_offset(stage, half_size: float, clearance_m: float, gripper_path: str, mesh_name: str | None = None) -> tuple[float, float, float]:
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
    tip_z = jaw_points[:, 2].min()
    tip_band = jaw_points[jaw_points[:, 2] <= tip_z + 2 * half_size]
    return (
        float(tip_band[:, 0].max() + half_size + clearance_m),
        float((tip_band[:, 1].min() + tip_band[:, 1].max()) / 2),
        float(tip_z + half_size),
    )


def table_plane(stage, root_prim):
    table = next((p for p in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
                  if p.GetName() == "Tabletop" and p.IsA(UsdGeom.Mesh)), None)
    if table is None:
        raise RuntimeError("No Tabletop mesh found; cannot measure spawn clearance.")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, point)))
                       for point in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    return np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)[0]
