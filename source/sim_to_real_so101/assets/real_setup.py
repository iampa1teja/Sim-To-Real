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
    """Apply the fitted tube light and optional broad window reflection."""
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
    fill = values.get("soft_fill")
    if fill:
        window = UsdLux.RectLight.Define(stage, room_root + "/Lighting/SoftFill")
        window.CreateWidthAttr(fill["width"])
        window.CreateHeightAttr(fill["height"])
        window.CreateIntensityAttr(fill["intensity"])
        window.CreateNormalizeAttr(False)
        window.CreateColorAttr(Gf.Vec3f(*fill["color"]))
        matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*fill["position"]), Gf.Vec3d(*fill["target"]), Gf.Vec3d(0, 0, 1)
        ).GetInverse()
        xf = UsdGeom.Xformable(window)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(matrix)
    return light


def apply_room_appearance(stage, room_root="/RealSetup/Room"):
    """Author the fitted albedo before packaging its texture into the USDZ."""
    look = SETUP["appearance"]
    material_root = room_root + "/Materials"
    texture = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/Texture"))
    texture.GetInput("file").Set(Sdf.AssetPath(str(ASSET_DIR / look["table_texture"])))
    texture.GetInput("scale").Set(Gf.Vec4f(*look["table_albedo_scale"]))
    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(*look.get("table_albedo_bias", [0, 0, 0, 0]))
    )
    surface = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/Surface"))
    surface.GetInput("roughness").Set(look["table_roughness"])
    surface.GetInput("clearcoat").Set(look["table_clearcoat"])
    # Select photographic coordinates or conventional wood UVs.
    uv = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/UV"))
    uv.GetInput("varname").Set("photo_st" if "table_photo_camera" in look else "st")
    if "table_photo_camera" in look:
        apply_table_photo_projection(stage, room_root)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("mirror")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("mirror")
    transform = UsdShade.Shader.Define(stage, material_root + "/RealWalnut/GrainScale")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*look.get("table_uv_scale", [1, 1])))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.GetInput("st").ConnectToSource(transform.ConnectableAPI(), "result")
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


def apply_opencv_intrinsics(prim, intrinsics):
    """Use RTX's native OpenCV model; USD aperture offsets alone are ignored."""
    # Isaac Lab's USD registry may omit this renderer schema. Author its
    # schema token and typed properties explicitly;
    # this is the same USD representation used by the native camera API.
    prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole:"
    # RTX samples at half-integer raster coordinates; RealSense/OpenCV uses
    # integer pixel centers. Offset the lens center, not the measured K.
    prim.CreateAttribute("calibration:pixelCenterOffset", Sdf.ValueTypeNames.Float).Set(0.5)
    for key in ("fx", "fy", "cx", "cy"):
        value = float(intrinsics[key]) + (0.5 if key in ("cx", "cy") else 0.0)
        prim.CreateAttribute(prefix + key, Sdf.ValueTypeNames.Float).Set(value)
    prim.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(intrinsics["width"], intrinsics["height"]))
    # The connected D435I color stream reports zero distortion. Do not silently
    # reinterpret inverse Brown-Conrady coefficients as the forward OpenCV model.
    coefficients = intrinsics.get("coeffs", [0.0] * 5)
    if any(coefficients) and intrinsics.get("model") == "distortion.inverse_brown_conrady":
        raise ValueError("Nonzero inverse Brown-Conrady distortion requires conversion before use.")
    for index, key in enumerate(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")):
        prim.CreateAttribute(prefix + key, Sdf.ValueTypeNames.Float).Set(float(coefficients[index]) if index < len(coefficients) else 0.0)


def apply_table_photo_projection(stage, room_root):
    """Project a cleaned real photograph onto the existing table surface.

    Subdivide only the visual mesh so ordinary perspective-correct UV
    interpolation approximates a projective texture to subpixel accuracy.
    Collision meshes and the tabletop outline/holes remain unchanged.
    """
    import hashlib
    import numpy as np
    from pxr import Vt

    photo = SETUP['appearance'].get('table_photo_camera')
    if not photo:
        return
    prim = stage.GetPrimAtPath(room_root + '/Visual/Furniture/Tabletop')
    mesh = UsdGeom.Mesh(prim)
    signature = hashlib.sha256(json.dumps(photo, sort_keys=True).encode()).hexdigest()
    if prim.GetCustomDataByKey('photoProjection') != signature:
        for child in list(prim.GetChildren()):
            if child.IsA(UsdGeom.Subset):
                stage.RemovePrim(child.GetPath())
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        original_uv = np.asarray(UsdGeom.PrimvarsAPI(prim).GetPrimvar('st').ComputeFlattened(), dtype=float)
        triangles, uv_triangles = [], []
        cursor = 0
        for count in counts:
            face = indices[cursor:cursor+count]
            uv_face = original_uv[cursor:cursor+count]
            for j in range(1, count-1):
                tri = points[face[[0,j,j+1]]]
                uv = uv_face[[0,j,j+1]]
                length = max(np.linalg.norm(tri[a]-tri[b]) for a,b in ((0,1),(1,2),(2,0)))
                divisions = max(1, int(np.ceil(length/.015)))
                def vertex(i,k):
                    w=np.array([1-(i+k)/divisions,i/divisions,k/divisions])
                    return w@tri,w@uv
                for i in range(divisions):
                    for k in range(divisions-i):
                        cells=[((i,k),(i+1,k),(i,k+1))]
                        if k<divisions-i-1:
                            cells.append(((i+1,k),(i+1,k+1),(i,k+1)))
                        for cell in cells:
                            vals=[vertex(*index) for index in cell]
                            triangles.append([v[0] for v in vals]);uv_triangles.append([v[1] for v in vals])
            cursor += count
        triangles=np.asarray(triangles);new_points=triangles.reshape(-1,3)
        # Convert local points to the room frame; environment clone offsets do
        # not change the photographic material's world-space registration.
        transform,_=UsdGeom.XformCache().ComputeRelativeTransform(prim,stage.GetPrimAtPath(room_root))
        matrix=np.asarray(transform)
        world=np.c_[new_points,np.ones(len(new_points))]@matrix
        x,y,z=photo['euler_degrees']
        rotation=(Gf.Rotation(Gf.Vec3d(1,0,0),x)*Gf.Rotation(Gf.Vec3d(0,1,0),y)*Gf.Rotation(Gf.Vec3d(0,0,1),z))
        camera=(world[:,:3]-photo['position'])@np.asarray(Gf.Matrix3d(rotation)).T
        depth=-camera[:,2]
        safe_depth=np.where(abs(depth)<1e-5,1e-5,depth)
        photo_uv=np.c_[(photo['fx']*camera[:,0]/safe_depth+photo['cx']+.5)/photo['width'],
                       1-(-photo['fy']*camera[:,1]/safe_depth+photo['cy']+.5)/photo['height']]
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(new_points.astype(np.float32)))
        mesh.GetFaceVertexCountsAttr().Set([3]*len(triangles))
        mesh.GetFaceVertexIndicesAttr().Set(np.arange(len(new_points),dtype=np.int32))
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.repeat(normals,3,axis=0).astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        pva=UsdGeom.PrimvarsAPI(prim)
        st=pva.GetPrimvar('st');st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv_triangles).reshape(-1,2).astype(np.float32)))
        st.SetInterpolation(UsdGeom.Tokens.faceVarying)
        uv=pva.CreatePrimvar('photo_st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
        uv.Set(Vt.Vec2fArray.FromNumpy(photo_uv.astype(np.float32)))
        prim.SetCustomDataByKey('photoProjection',signature)
    # A single wood material covers the entire tabletop. Mirror wrapping
    # continues the same grain beyond the reference camera's footprint.
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Subset):
            child.SetActive(False)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(room_root+'/Materials/RealWalnut')),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants)
