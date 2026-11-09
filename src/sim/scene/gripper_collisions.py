"""CAD-fitted SO101 contact pads and support geometry.

Importing this module changes no stage or production configuration.
Call apply_primitive_pads after the legacy finger hulls are split and before
contact materials are bound. All coordinates are link-local metres.
"""


def _mesh_triangles(root, link_name, mesh_name):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    link = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild(link_name))
    mesh = next(prim for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies())
                if prim.IsA(UsdGeom.Mesh) and f"/visuals/{mesh_name}/" in str(prim.GetPath()))
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh, link)
    points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, point)))
                         for point in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
    points *= UsdGeom.GetStageMetersPerUnit(root.GetStage())
    indices = UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get()
    triangles, offset = [], 0
    for count in UsdGeom.Mesh(mesh).GetFaceVertexCountsAttr().Get():
        face = indices[offset:offset + count]
        offset += count
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
    return points[np.asarray(triangles)], str(mesh.GetPath())


def _covered_by_triangles(points, triangles):
    """Membership in the actual planar CAD triangles, including their boundaries."""
    import numpy as np

    covered = np.zeros(len(points), dtype=bool)
    for triangle in triangles:
        a, b, c = triangle
        u, v = b - a, c - a
        determinant = u[0] * v[1] - u[1] * v[0]
        if abs(determinant) < 1e-16:
            continue
        w = points - a
        alpha = (w[:, 0] * v[1] - w[:, 1] * v[0]) / determinant
        beta = (u[0] * w[:, 1] - u[1] * w[:, 0]) / determinant
        covered |= (alpha >= -1e-8) & (beta >= -1e-8) & (alpha + beta <= 1 + 1e-8)
    return covered


def _rectangle_covered_area(rectangle, triangles):
    """Exact polygon clipping verifies that a proposed pad face adds no area."""
    import numpy as np

    total = 0.0
    for triangle in triangles:
        polygon = list(triangle)
        for axis, coordinate, sign in ((0, rectangle[0, 0], 1), (0, rectangle[1, 0], -1),
                                       (1, rectangle[0, 1], 1), (1, rectangle[1, 1], -1)):
            clipped = []
            for p, q in zip(polygon, polygon[1:] + polygon[:1]):
                inside_p = sign * (p[axis] - coordinate) >= -1e-13
                inside_q = sign * (q[axis] - coordinate) >= -1e-13
                if inside_p:
                    clipped.append(p)
                if inside_p != inside_q:
                    fraction = (coordinate - p[axis]) / (q[axis] - p[axis])
                    clipped.append(p + fraction * (q - p))
            polygon = clipped
            if not polygon:
                break
        if len(polygon) >= 3:
            vertices = np.asarray(polygon)
            total += abs(float(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                                      - vertices[:, 1] * np.roll(vertices[:, 0], -1)))) / 2
    return total


def _inscribed_pad_face(triangles, plane_x, long_axis):
    """Find a rectangle strictly inside one measured CAD step, at its actual X."""
    import numpy as np

    plane_triangles = triangles[np.max(np.abs(triangles[:, :, 0] - plane_x), axis=1) < 2e-6]
    if len(plane_triangles) < 2:
        raise ValueError(f"Missing CAD contact plane X={plane_x}")
    width_axis = 3 - long_axis
    planar = plane_triangles[:, :, [width_axis, long_axis]]
    bounds = np.stack((planar.min(axis=(0, 1)), planar.max(axis=(0, 1))))
    # The CAD has tapered edges and tooth relief. A bounding box would widen it.
    # Select covered grid cells, then verify the continuous rectangle by clipping.
    axes = [np.linspace(bounds[0, axis] + 0.00005, bounds[1, axis] - 0.00005,
                        max(3, int(np.ceil(np.diff(bounds[:, axis])[0] / 0.0001)) + 1))
            for axis in (0, 1)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    covered = _covered_by_triangles(grid.reshape(-1, 2), planar).reshape(grid.shape[:2])
    centers = (grid[:-1, :-1] + grid[1:, 1:]) / 2
    cells = (covered[:-1, :-1] & covered[1:, :-1] & covered[:-1, 1:] & covered[1:, 1:]
             & _covered_by_triangles(centers.reshape(-1, 2), planar).reshape(centers.shape[:2]))
    heights = np.zeros(cells.shape[1], dtype=int)
    best, best_area = None, 0
    for row, valid in enumerate(cells):
        heights = np.where(valid, heights + 1, 0)
        stack = []
        for column, height in enumerate(np.r_[heights, 0]):
            start = column
            while stack and stack[-1][1] > height:
                begin, old_height = stack.pop()
                area = old_height * (column - begin)
                if area > best_area:
                    best_area, best = area, (row - old_height + 1, row + 1, begin, column)
                start = begin
            if not stack or stack[-1][1] < height:
                stack.append((start, int(height)))
    if best is None:
        raise ValueError(f"No inscribed CAD pad on X={plane_x}")
    i0, i1, j0, j1 = best
    rectangle = np.array([[axes[0][i0], axes[1][j0]], [axes[0][i1], axes[1][j1]]])
    rectangle[0] += 0.00002
    rectangle[1] -= 0.00002
    area = float(np.prod(rectangle[1] - rectangle[0]))
    covered_area = _rectangle_covered_area(rectangle, planar)
    if abs(covered_area - area) > max(1e-12, area * 1e-6):
        raise ValueError(f"Pad face crosses CAD relief: covered {covered_area}/{area}")
    actual_x = float(plane_triangles[:, :, 0].mean())
    return actual_x, width_axis, rectangle, {
        "cad_plane_x_m": actual_x,
        "maximum_cad_plane_residual_mm": float(np.abs(plane_triangles[:, :, 0] - actual_x).max() * 1000),
        "contact_face_area_mm2": area * 1e6,
        "continuous_triangle_coverage_ratio": covered_area / area,
        "width_axis": width_axis, "long_axis": long_axis,
        "width_long_bounds_m": rectangle.tolist(),
    }


def _clipped_cad_points(triangles, constraints):
    """Clip CAD triangles to axis halfspaces, retaining exact cut intersections."""
    import numpy as np

    polygons = triangles
    for axis, coordinate, sign in constraints:
        next_polygons = []
        for polygon in polygons:
            values = sign * (polygon[:, axis] - coordinate)
            if np.all(values >= -1e-12):
                next_polygons.append(polygon)
                continue
            if np.all(values < -1e-12):
                continue
            clipped = []
            for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
                inside_p = sign * (p[axis] - coordinate) >= -1e-12
                inside_q = sign * (q[axis] - coordinate) >= -1e-12
                if inside_p:
                    clipped.append(p)
                if inside_p != inside_q:
                    fraction = (coordinate - p[axis]) / (q[axis] - p[axis])
                    clipped.append(p + fraction * (q - p))
            if len(clipped) >= 3:
                next_polygons.append(np.asarray(clipped))
        polygons = next_polygons
    return np.unique(np.round(np.vstack(polygons), 10), axis=0)


def _author_cad_support(root, link, name, points):
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                       points[faces[:, 2]] - points[faces[:, 0]])
    for face, normal, plane in zip(faces, normals, hull.equations):
        if np.dot(normal, plane[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    vertices = points[hull.vertices]
    remap = {int(index): position for position, index in enumerate(hull.vertices)}
    path = link.GetPath().AppendChild(name)
    mesh = UsdGeom.Mesh.Define(root.GetStage(), path)
    metre = UsdGeom.GetStageMetersPerUnit(root.GetStage())
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, point / metre)) for point in vertices]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([remap[int(index)] for index in faces.reshape(-1)])
    mesh.CreateVisibilityAttr("invisible")
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return str(path)


def apply_primitive_pads(root):
    """Candidate: two inscribed analytic boxes on each finger's real distal steps.

    The two planes on each finger are fitted to CAD, not extended across the
    opening. Proximal body hulls, CAD visuals, masses and joints remain intact.
    Returns ``(disabled_hull_paths, created_collider_paths)`` for material binding.
    """
    import json
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics

    stage = root.GetStage()
    metre = UsdGeom.GetStageMetersPerUnit(stage)
    replaced, created = [], []
    for link_name, mesh_name, long_axis, planes, boundaries, outward_sign in (
            ("gripper", "wrist_roll_follower_so101_v1", 2, (-0.0079, -0.0099),
             (-0.1034243, -0.09442528, -0.08442528), 1),
            ("jaw", "moving_jaw_so101_v1", 1, (-0.0123, -0.0103),
             (-0.080999, -0.072, -0.062), -1)):
        link = stage.GetPrimAtPath(root.GetPath().AppendChild(link_name))
        if not link:
            continue
        hull_path = link.GetPath().AppendPath(f"collisions/{mesh_name}/finger_contact")
        hull = stage.GetPrimAtPath(hull_path)
        if not hull or not hull.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(hull).GetCollisionEnabledAttr().Get():
            continue
        triangles, source = _mesh_triangles(root, link_name, mesh_name)
        for index, plane in enumerate(planes):
            x, width_axis, rectangle, report = _inscribed_pad_face(triangles, plane, long_axis)
            center = np.zeros(3)
            dimensions = np.zeros(3)
            dimensions[0] = 0.001
            center[0] = x - outward_sign * dimensions[0] / 2
            center[[width_axis, long_axis]] = rectangle.mean(axis=0)
            dimensions[[width_axis, long_axis]] = rectangle[1] - rectangle[0]
            path = link.GetPath().AppendChild(f"contact_pad_{index}")
            pad = UsdGeom.Cube.Define(stage, path)
            pad.CreateSizeAttr(1.0)
            pad.AddTranslateOp().Set(Gf.Vec3d(*map(float, center / metre)))
            pad.AddScaleOp().Set(Gf.Vec3f(*map(float, dimensions / metre)))
            pad.CreateVisibilityAttr("invisible")
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim()).CreateCollisionEnabledAttr(True)
            report.update(source_mesh=source, link_frame=link_name,
                          box_center_m=center.tolist(), box_dimensions_m=dimensions.tolist(),
                          outward_sign=outward_sign, normal_axis=0,
                          contact_plane_offset_error_mm=abs(center[0] + outward_sign * dimensions[0] / 2 - x) * 1000)
            pad.GetPrim().SetCustomDataByKey("s2rPadFit", json.dumps(report, sort_keys=True))
            created.append(str(path))
        # Preserve the exact CAD nose and the rest of the distal finger. In the
        # two contact bands, backing hulls end 0.2 mm behind each 1 mm pad.
        # This keeps fingertip/table collisions while boxes own the inner faces.
        nose, first_step, second_step = boundaries
        supports = (
            ("contact_nose", ((long_axis, nose, -1),)),
            ("contact_back_0", ((long_axis, nose, 1), (long_axis, first_step, -1),
                                (0, planes[0] - outward_sign * 0.0012, -outward_sign))),
            ("contact_back_1", ((long_axis, first_step, 1), (long_axis, second_step, -1),
                                (0, planes[1] - outward_sign * 0.0012, -outward_sign))),
            ("contact_upper", ((long_axis, second_step, 1), (long_axis, -0.04, -1))),
        )
        for name, constraints in supports:
            points = _clipped_cad_points(triangles, constraints)
            created.append(_author_cad_support(root, link, name, points))
        UsdPhysics.CollisionAPI(hull).CreateCollisionEnabledAttr(False)
        replaced.append(str(hull_path))
    return tuple(replaced), tuple(created)
