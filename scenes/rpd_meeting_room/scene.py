"""Measured meeting-room material edits that do not belong to generic scene code.

The imported mesh/material names and the fitted photographic coordinates are
owned by this scene. Camera, lighting, mount and solver data remain declarative.
"""
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade


def apply_room_appearance(stage, room_root, look, asset_dir):
    """Author the fitted albedo before packaging its texture into the USDZ."""
    material_root = room_root + "/Materials"
    texture = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/RealWalnut/Texture"))
    texture.GetInput("file").Set(Sdf.AssetPath(str(asset_dir / look["table_texture"])))
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
        apply_table_photo_projection(stage, room_root, look)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("mirror")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("mirror")
    transform = UsdShade.Shader.Define(stage, material_root + "/RealWalnut/GrainScale")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*look.get("table_uv_scale", [1, 1])))
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.GetInput("st").ConnectToSource(transform.ConnectableAPI(), "result")
    if "table_uv_frame" in look:
        apply_table_planar_uv(stage, room_root, look["table_uv_frame"])
        for wrap in ("wrapS", "wrapT"):
            texture.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set(look.get("table_texture_wrap", "mirror"))
    wall = UsdShade.Shader(stage.GetPrimAtPath(material_root + "/M_f94acc284ce2/ScanColor"))
    wall.GetInput("scale").Set(Gf.Vec4f(*look["backdrop_texture_scale"]))
    wall.GetInput("bias").Set(Gf.Vec4f(*look["backdrop_texture_bias"]))
    if "wall_texture" in look:
        from pxr import Vt
        wall.GetInput("file").Set(Sdf.AssetPath(str(asset_dir / look["wall_texture"])))
        # Clean per-corner UVs: s along the wall base, t = 0 at the wall base.
        wall_mesh = stage.GetPrimAtPath(room_root + "/Visual/Architecture/SouthWall_0")
        st = UsdGeom.PrimvarsAPI(wall_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in look["wall_st_faceVarying"]]))
    for path in look.get("hide_room_prims", []):
        prim = stage.GetPrimAtPath(room_root + path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()



def apply_table_planar_uv(stage, room_root, frame):
    """Camera-independent planar UVs for the physical-scale table albedo.

    st = ((P - origin).u / size_u + 0.5, (P - origin).v / size_v + 0.5) per face vertex,
    in the room frame; u runs along the table's long edge (grain direction).
    """
    import numpy as np
    from pxr import Vt

    prim = stage.GetPrimAtPath(room_root + '/Visual/Furniture/Tabletop')
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, stage.GetPrimAtPath(room_root))
    points = (np.c_[points, np.ones(len(points))] @ np.asarray(transform))[:, :3]
    corners = points[np.asarray(mesh.GetFaceVertexIndicesAttr().Get())] - np.asarray(frame["origin"])
    st = np.stack([corners @ np.asarray(frame["u_axis"]) / frame["size_u"] + 0.5,
                   corners @ np.asarray(frame["v_axis"]) / frame["size_v"] + 0.5], axis=-1)
    UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    ).Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
    # A single wood material covers the entire tabletop.
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Subset):
            child.SetActive(False)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(room_root + '/Materials/RealWalnut')),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants)



def apply_table_photo_projection(stage, room_root, look):
    """Project a cleaned real photograph onto the existing table surface.

    Subdivide only the visual mesh so ordinary perspective-correct UV
    interpolation approximates a projective texture to subpixel accuracy.
    Collision meshes and the tabletop outline/holes remain unchanged.
    """
    import hashlib
    import numpy as np
    from pxr import Vt

    photo = look.get('table_photo_camera')
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
                            triangles.append([v[0] for v in vals])
                            uv_triangles.append([v[1] for v in vals])
            cursor += count
        triangles = np.asarray(triangles)
        new_points = triangles.reshape(-1, 3)
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
        st = pva.GetPrimvar('st')
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv_triangles).reshape(-1, 2).astype(np.float32)))
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



def configure_scene_updates(env, env_ids, scene_spec):
    """Preserve the measured scene's startup order: views, appearance, lighting."""
    from sim.envs.scene_builder import hide_default_viewports
    from sim.envs.mdp._common import _env_prim_path
    from sim.scene.appearance import apply_scene_lighting, scene_content_root

    hide_default_viewports(env.sim.stage)
    if Path(env.cfg.scene.room.spawn.usd_path).resolve() != scene_spec.room_usd:
        return
    room = env.scene["room"]
    look = scene_spec.appearance.get("appearance", {})
    for index in range(env.num_envs):
        root = scene_content_root(_env_prim_path(env, room, index), scene_spec.room_root)
        apply_room_appearance(env.sim.stage, root, look, scene_spec.directory)
        apply_scene_lighting(env.sim.stage, root, scene_spec.lighting)


def customize(env_cfg, scene_spec):
    """Keep the scene-specific UV/material hook at its established startup event."""
    from isaaclab.managers import EventTermCfg

    env_cfg.events.configure_scene_updates = EventTermCfg(
        func=configure_scene_updates, mode="startup", params={"scene_spec": scene_spec},
    )
