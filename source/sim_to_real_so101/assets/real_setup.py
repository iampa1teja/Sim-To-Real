"""Shared placement for the packaged scene and every MyRoom launcher."""
import json
from pathlib import Path

from pxr import Gf, Sdf, UsdShade

ASSET_DIR = Path(__file__).resolve().parent
SETUP = json.loads((ASSET_DIR / "real_setup.json").read_text())
DEFAULT_SETUP_USDZ = str(ASSET_DIR / "usd" / "real_setup.usdz")


def quaternion_tuple(rotation):
    q = rotation.GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def robot_rotation():
    tilt = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*SETUP["robot"]["table_normal"]))
    yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), SETUP["robot"]["yaw_degrees"])
    return quaternion_tuple(yaw * tilt)


def camera_rotation():
    camera = SETUP["realsense"]
    forward = (Gf.Vec3d(*camera['target']) - Gf.Vec3d(*camera['position'])).GetNormalized()
    # Match the nearly horizontal table edge in the Cheese reference.
    table_long = Gf.Vec3d(0.996801706, 0.079914694, 0.00263459).GetNormalized()
    right = (-table_long + Gf.Dot(table_long, forward) * forward).GetNormalized()
    up = Gf.Cross(right, forward).GetNormalized()
    transform = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*camera["position"]), Gf.Vec3d(*camera["target"]), up
    ).GetInverse()
    return quaternion_tuple(transform.ExtractRotation())


def apply_robot_appearance(stage, robot_root):
    """Apply the real arm's ivory plastic, charcoal motors and yellow wrist mount."""
    look = SETUP['appearance']
    for material_name, color in (
        ('material_a_3d_printed', look['robot_ivory']),
        ('material_sts3215', look['motor_charcoal']),
    ):
        prim = stage.GetPrimAtPath(f'{robot_root}/Looks/{material_name}/Shader')
        if not prim.IsValid(): continue
        shader = UsdShade.Shader(prim)
        is_preview = shader.GetIdAttr().Get() == 'UsdPreviewSurface'
        shader.CreateInput('diffuseColor' if is_preview else 'diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness' if is_preview else 'reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(0.55)
    # Make just the camera mount editable; keep lens/PCB/motor materials intact.
    for suffix in ('/gripper/visuals', '/gripper/visuals/camera_mount'):
        prim = stage.GetPrimAtPath(robot_root + suffix)
        if prim.IsValid() and prim.IsInstance(): prim.SetInstanceable(False)
    mount = stage.GetPrimAtPath(robot_root + '/gripper/visuals/camera_mount/mount')
    if mount.IsValid() and not mount.IsInstanceProxy():
        material = UsdShade.Material.Define(stage, robot_root + '/Looks/WristMountYellow')
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*look['wrist_mount_yellow']))
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mount).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
