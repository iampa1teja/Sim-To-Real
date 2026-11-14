"""Preserve the supplied USD visual in a centred metre-scale hollow asset.

Run in s2r-sim: s2r-entrypoint python workspace/tasks/stack_bowls/build_asset.py.
The full authored visual and normals are retained. Convex wall sectors retain the
cavity; a single convex hull of the whole bowl would incorrectly fill it.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt


def mesh_prim(stage, path, vertices, faces):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    return mesh


def main():
    root = Path(__file__).resolve().parent
    source = root/'assets/bowl.source.usdc'
    data = source.read_bytes()
    supplied = Usd.Stage.Open(str(source))
    if not supplied or not supplied.GetDefaultPrim():
        raise ValueError('The supplied bowl USD needs a default prim')
    if UsdGeom.GetStageUpAxis(supplied) != UsdGeom.Tokens.z:
        raise ValueError('The supplied bowl must open along its authored Z axis')
    transforms = UsdGeom.XformCache()
    meshes = []
    visual_faces = visual_normals = 0
    for prim in Usd.PrimRange(supplied.GetDefaultPrim()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('The supplied bowl must have triangular faces')
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim))
        points = (np.column_stack((points, np.ones(len(points)))) @ matrix)[:, :3]
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, ::-1]
        meshes.append(trimesh.Trimesh(points, faces, process=True))
        visual_faces += len(faces)
        visual_normals += len(mesh.GetNormalsAttr().Get() or [])
    if not meshes:
        raise ValueError('No bowl mesh found in supplied USD')
    original = trimesh.util.concatenate(meshes)
    source_bounds = original.bounds.copy()
    centre = (original.bounds[0] + original.bounds[1]) / 2
    scale = .05/original.extents[2]
    original.apply_translation(-centre)
    original.apply_scale(scale)
    if not np.allclose(original.extents,[.12,.12,.05],atol=.001):
        raise ValueError(f'Supplied bowl proportions differ from task dimensions: {original.extents}')
    bottom = original.bounds[0,2]
    upward = (original.face_normals[:,2] > .99999) & (
        original.triangles_center[:,2] > bottom+1e-6) & (
        original.triangles_center[:,2] < bottom+.0125)
    if not upward.any():
        raise ValueError('Could not identify the actual inner bowl floor')
    heights = np.round(original.triangles_center[upward,2],7)
    values, inverse = np.unique(heights,return_inverse=True)
    floor_height = float(values[np.bincount(inverse,weights=original.area_faces[upward]).argmax()]-bottom)
    path = root/'assets/bowl.usd'
    temporary = root/'assets/bowl.rebuild.usdc'
    temporary.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    body = UsdGeom.Xform.Define(stage, '/Bowl')
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(.055)
    visual_prim = UsdGeom.Xform.Define(stage, '/Bowl/Visual')
    transform = Gf.Matrix4d().SetScale(scale)
    transform.SetTranslateOnly(Gf.Vec3d(*(-centre*scale)))
    visual_prim.AddTransformOp().Set(transform)
    Sdf.CopySpec(supplied.Flatten(), supplied.GetDefaultPrim().GetPath(),
                 stage.GetRootLayer(), '/Bowl/Visual/Source')
    extras = [prim.GetPath() for prim in Usd.PrimRange(stage.GetPrimAtPath('/Bowl/Visual/Source'))
              if prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdGeom.Camera)]
    for prim_path in reversed(extras):
        stage.RemovePrim(prim_path)
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/Bowl/Visual/Source')):
        for relationship in list(prim.GetRelationships()):
            if relationship.GetName().startswith('material:binding'):
                prim.RemoveProperty(relationship.GetName())
        for schema in [UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI,
                       UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI]:
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)
    radii = {}

    def radius(z):
        if z not in radii:
            # Exact intersections with the original triangles, not STL vertex bins.
            lines = trimesh.intersections.mesh_plane(original, [0, 0, 1], [0, 0, z-.025])
            r = np.linalg.norm(lines.reshape(-1, 3)[:, :2], axis=1)
            radii[z] = (float(r.min()), float(r.max()))
        return radii[z]

    pieces = []
    # A closed convex bottom, using the inner floor measured from this mesh.
    points = []
    for z, r in [(0., radius(1e-7)[1]), (floor_height, radius(floor_height+1e-7)[1])]:
        for a in np.linspace(0, 2*np.pi, 97)[:-1]:
            points.append((r*np.cos(a), r*np.sin(a), z-.025))
    pieces.append(trimesh.convex.convex_hull(np.asarray(points)))
    levels = [floor_height+1e-7]+[z for z in [.008,.016,.024,.034,.042,.048]
        if z > floor_height+1e-7]+[.0499999]
    for low, high in zip(levels, levels[1:]):
        for sector in range(24):
            points = []
            for z in np.linspace(low, high, 7):
                inner, outer = radius(float(z))
                for angle in np.linspace(sector*2*np.pi/24, (sector+1)*2*np.pi/24, 5):
                    for r in (inner, outer):
                        points.append((r*np.cos(angle), r*np.sin(angle), z-.025))
            pieces.append(trimesh.convex.convex_hull(np.asarray(points)))
    for i, hull in enumerate(pieces):
        mesh = mesh_prim(stage, f'/Bowl/Collisions/part_{i:03d}', hull.vertices, hull.faces)
        mesh.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
    stage.GetRootLayer().Save()
    temporary.replace(path)
    metadata = dict(source_usd='~/pavan/bowl.usdc', source_copy=str(source.relative_to(root)),
                    source_sha256=hashlib.sha256(data).hexdigest(),
                    source_units='millimetre-sized coordinates normalized to task dimensions',
                    source_declared_meters_per_unit=UsdGeom.GetStageMetersPerUnit(supplied),
                    source_bounds=source_bounds.tolist(), source_centre=centre.tolist(), scale_to_metres=scale,
                    size_m=original.extents.tolist(), mass_kg=.055,
                    source_triangles=visual_faces, visual_triangles=visual_faces,
                    authored_visual_normals=visual_normals, visual_method='Full supplied USD mesh; no decimation',
                    removed_embedded_lights_and_cameras=len(extras),
                    floor_thickness_m=floor_height,
                    convex_pieces=len(pieces), collision_method='24 angular sectors x wall bands + measured solid floor',
                    origin='bounding-box centre; bottom z=-0.025 m',
                    color_order='any')
    (root/'assets'/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
