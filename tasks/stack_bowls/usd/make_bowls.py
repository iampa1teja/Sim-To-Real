"""Regenerate the bottom-origin USDA bowls with separate convex colliders."""

import math
from pathlib import Path
import tomllib


_SEGMENTS = 16


def _ring(radius, z):
    return [
        (round(radius * math.cos(i * math.tau / _SEGMENTS), 8),
         round(radius * math.sin(i * math.tau / _SEGMENTS), 8), z)
        for i in range(_SEGMENTS)
    ]


def _mesh(name, vertices, faces, *, collision=False, color=None):
    schemas = ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )'
    counts = [len(face) for face in faces]
    indices = [i for face in faces for i in face]
    lines = [f'    def Mesh "{name}"' + (schemas if collision else ""), "    {",
             '        uniform token subdivisionScheme = "none"',
             f"        point3f[] points = {vertices}",
             f"        int[] faceVertexCounts = {counts}",
             f"        int[] faceVertexIndices = {indices}"]
    if collision:
        lines.extend(['        token visibility = "invisible"',
                      '        bool physics:collisionEnabled = true',
                      '        uniform token physics:approximation = "convexHull"'])
    if color is not None:
        lines.append(f"        color3f[] primvars:displayColor = [{color}]")
    return "\n".join([*lines, "    }"])


def _surface(rings):
    vertices = [point for ring in rings for point in ring]
    faces = [tuple(reversed(range(_SEGMENTS)))]
    for level in range(len(rings) - 1):
        first = level * _SEGMENTS
        for i in range(_SEGMENTS):
            j = (i + 1) % _SEGMENTS
            faces.append((first + i, first + j, first + _SEGMENTS + j, first + _SEGMENTS + i))
    faces.append(tuple(range(len(vertices) - _SEGMENTS, len(vertices))))
    return vertices, faces


def _floor_collider(radius, thickness):
    return f'''    def Cylinder "FloorCollision" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {thickness}
        double radius = {radius}
        double3 xformOp:translate = (0, 0, {thickness / 2})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        token visibility = "invisible"
        bool physics:collisionEnabled = true
    }}'''


def _bowl(name, radius, height, floor, inner_radius, mass, color):
    # The wall flares out from the floor. The interior stays wider than
    # the small bowl's rim, including near the large bowl's floor.
    base_radius = radius * 0.8
    inner_base = inner_radius * 0.9
    rings = [_ring(base_radius, 0), _ring(radius, height),
             _ring(inner_radius, height), _ring(inner_base, floor)]
    vertices, faces = _surface(rings)
    # A cylinder gives Newton an analytic contact against the triangulated table.
    meshes = [_mesh("Surface", vertices, faces, color=color), _floor_collider(base_radius, floor)]
    for i in range(_SEGMENTS):
        j = (i + 1) % _SEGMENTS
        vertices = [ring[k] for ring in rings for k in (i, j)]
        faces = [(0, 1, 3, 2), (2, 3, 5, 4), (4, 5, 7, 6), (6, 7, 1, 0),
                 (0, 2, 4, 6), (1, 7, 5, 3)]
        meshes.append(_mesh(f"WallCollision_{i:02d}", vertices, faces, collision=True))
    return f'''#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float3[] extent = [({-radius}, {-radius}, 0), ({radius}, {radius}, {height})]
    bool physics:rigidBodyEnabled = true
    float physics:mass = {mass}

''' + "\n\n".join(meshes) + "\n}\n"


def main():
    folder = Path(__file__).resolve().parent
    objects = tomllib.loads((folder.parent / "task.toml").read_text(encoding="utf-8"))["task"]["objects"]
    specs = [("small_bowl", "SmallBowl", 0.0025, 0.017, (0.12, 0.30, 0.66)),
             ("large_bowl", "LargeBowl", 0.003, 0.03, (0.72, 0.32, 0.12))]
    for key, prim, floor, inner, color in specs:
        obj = objects[key]
        width, depth, height = obj["size_outer"]
        if width != depth:
            raise ValueError(f"{key} needs a circular footprint")
        output = folder.parent / obj["usd"]
        output.write_text(_bowl(prim, width / 2, height, floor, inner, obj["mass"], color), encoding="utf-8")


if __name__ == "__main__":
    main()
