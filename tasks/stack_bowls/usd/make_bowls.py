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


def _mesh(name, vertices, faces, *, color):
    counts = [len(face) for face in faces]
    indices = [i for face in faces for i in face]
    lines = [f'    def Mesh "{name}"', "    {",
             '        uniform token subdivisionScheme = "none"',
             f"        point3f[] points = {vertices}",
             f"        int[] faceVertexCounts = {counts}",
             f"        int[] faceVertexIndices = {indices}"]
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


def _wall_collider(index, *, radius, thickness, length, height, rise, tilt):
    angle = (index + 0.5) * math.tau / _SEGMENTS
    x, y = radius * math.cos(angle), radius * math.sin(angle)
    return f'''    def Cube "WallCollision_{index:02d}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        double3 xformOp:translate = ({x:.9f}, {y:.9f}, {height / 2:.9f})
        float xformOp:rotateZ = {math.degrees(angle):.9f}
        float xformOp:rotateY = {tilt:.9f}
        float3 xformOp:scale = ({thickness:.9f}, {length:.9f}, {rise:.9f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:rotateY", "xformOp:scale"]
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
    # Tilt the boxes along the bowl wall and overlap the ends slightly to close its seams.
    top, bottom = (radius + inner_radius) / 2, (base_radius + inner_base) / 2
    middle = (top + bottom) / 2
    thickness = ((radius - inner_radius) + (base_radius - inner_base)) / 2
    length = 2 * middle * math.tan(math.pi / _SEGMENTS) * 1.02
    rise = math.hypot(height, top - bottom)
    tilt = math.degrees(math.atan2(top - bottom, height))
    meshes.extend(_wall_collider(i, radius=middle, thickness=thickness, length=length,
                                 height=height, rise=rise, tilt=tilt) for i in range(_SEGMENTS))
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
