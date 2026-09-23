"""Shared placement for the packaged scene and every MyRoom launcher."""
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

ASSET_DIR = Path(__file__).resolve().parent
SETUP = json.loads((ASSET_DIR / "real_setup.json").read_text())
DEFAULT_SETUP_USDZ = str(ASSET_DIR / "usd" / "digital_twin.usdz")


def quaternion_tuple(rotation):
    q = rotation.GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def robot_rotation():
    tilt = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*SETUP["robot"]["table_normal"]))
    yaw = Gf.Rotation(Gf.Vec3d(0, 0, 1), SETUP["robot"]["yaw_degrees"])
    return quaternion_tuple(yaw * tilt)


def camera_rotation():
    camera = SETUP["realsense"]
    if "euler_degrees" in camera:
        # USD/OpenGL camera convention: -Z forward, +Y up; XYZ degrees.
        x, y, z = camera["euler_degrees"]
        rotation = (Gf.Rotation(Gf.Vec3d(1, 0, 0), x)
                    * Gf.Rotation(Gf.Vec3d(0, 1, 0), y)
                    * Gf.Rotation(Gf.Vec3d(0, 0, 1), z))
        return quaternion_tuple(rotation)
    forward = (Gf.Vec3d(*camera['target']) - Gf.Vec3d(*camera['position'])).GetNormalized()
    # Match the nearly horizontal table edge in the Cheese reference.
    table_long = Gf.Vec3d(0.996801706, 0.079914694, 0.00263459).GetNormalized()
    right = (-table_long + Gf.Dot(table_long, forward) * forward).GetNormalized()
    up = Gf.Cross(right, forward).GetNormalized()
    transform = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*camera["position"]), Gf.Vec3d(*camera["target"]), up
    ).GetInverse()
    return quaternion_tuple(transform.ExtractRotation())


def apply_room_lighting(stage, room_root="/RealSetup/Room"):
    """One emitter at the scanned tube above the north-wall whiteboard."""
    room = stage.GetPrimAtPath(room_root)
    for prim in reversed(list(Usd.PrimRange(room))):
        if prim.HasAPI(UsdLux.LightAPI):
            stage.RemovePrim(prim.GetPath())
    values = SETUP["lighting"]
    stage.DefinePrim(room_root + "/Lighting", "Xform")
    light = UsdLux.RectLight.Define(stage, room_root + "/Lighting/Tubelight")
    light.CreateWidthAttr(values["length"])
    light.CreateHeightAttr(values["diffuser_width"])
    light.CreateIntensityAttr(values["intensity"])
    light.CreateNormalizeAttr(False)
    light.CreateColorAttr(Gf.Vec3f(*values["color"]))
    light.CreateEnableColorTemperatureAttr(False)
    transform = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*values["position"]), Gf.Vec3d(*values["target"]), Gf.Vec3d(0, 0, 1)
    ).GetInverse()
    xform = UsdGeom.Xformable(light)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)
    return light


def apply_room_appearance(stage, room_root="/RealSetup/Room"):
    """Author the fitted albedo before packaging its texture into the USDZ."""
    look = SETUP["appearance"]
    material_root = room_root + "/Materials"
    texture = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/Texture"))
    texture.GetInput("file").Set(Sdf.AssetPath(str(ASSET_DIR / look["table_texture"])))
    texture.GetInput("scale").Set(Gf.Vec4f(*look["table_albedo_scale"]))
    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0))
    surface = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/Surface"))
    surface.GetInput("roughness").Set(look["table_roughness"])
    surface.GetInput("clearcoat").Set(look["table_clearcoat"])
    wall = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/M_f94acc284ce2/ScanColor"))
    wall.GetInput("scale").Set(Gf.Vec4f(*look["backdrop_texture_scale"]))
    wall.GetInput("bias").Set(Gf.Vec4f(*look["backdrop_texture_bias"]))


def apply_robot_appearance(stage, robot_root, arm_color=None):
    """Apply the real arm's beige plastic, charcoal motors and yellow wrist mount."""
    look = SETUP['appearance']
    arm_color = look['robot_beige'] if arm_color is None else arm_color
    for material_name, color, roughness in (
        ('material_a_3d_printed', arm_color, look['robot_plastic_roughness']),
        ('material_sts3215', look['motor_charcoal'], look['motor_roughness']),
    ):
        prim = stage.GetPrimAtPath(f'{robot_root}/Looks/{material_name}/Shader')
        if not prim.IsValid(): continue
        shader = UsdShade.Shader(prim)
        is_preview = shader.GetIdAttr().Get() == 'UsdPreviewSurface'
        shader.CreateInput('diffuseColor' if is_preview else 'diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness' if is_preview else 'reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(roughness)
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
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(look['wrist_mount_roughness'])
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mount).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
