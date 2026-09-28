"""USD measurements shared by registered-object resets and the cube spawner."""

import numpy as np
from pxr import Gf, Usd, UsdGeom


def gripper_tip_local_offset(
    stage, half_size: float, clearance_m: float, gripper_path: str
) -> tuple[float, float, float]:
    """Measure the fixed jaw's mesh geometry and return a gripper-local offset.

    Args:
        stage: USD stage containing the specified gripper link.
        half_size: Cube half width in metres.
        clearance_m: Requested jaw-to-cube surface clearance in metres.
        gripper_path: Exact fixed-jaw link path, matching the live frame sensor.

    Returns:
        Gripper-local XYZ offset in metres, measured from the fixed-jaw mesh.
    """
    gripper = stage.GetPrimAtPath(gripper_path)
    if not gripper.IsValid():
        raise RuntimeError(f"Fixed-jaw link not found: {gripper_path}")
    mesh_prim = next(
        (p for p in Usd.PrimRange(gripper, Usd.TraverseInstanceProxies())
         if p.IsA(UsdGeom.Mesh) and "/visuals/wrist_roll_follower_so101_v1/" in str(p.GetPath())),
        None,
    )
    if mesh_prim is None:
        raise RuntimeError("Fixed-jaw mesh not found; cannot measure jaw tip.")

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
    """Fit z = ax + by + c in world metres from the root's measured tabletop."""
    table = next((p for p in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
                  if p.GetName() == "Tabletop" and p.IsA(UsdGeom.Mesh)), None)
    if table is None:
        raise RuntimeError("No Tabletop mesh found; cannot measure spawn clearance.")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, point)))
                       for point in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(stage)
    return np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)[0]
