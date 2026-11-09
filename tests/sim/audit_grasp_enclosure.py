"""CPU-only CAD/trajectory enclosure audit; no app, physics, or GPU initialization.

Opens the USD in memory, applies the existing runtime collision promotion there,
and checks its finger hulls against captured oriented cube corners. No USD or
trajectory is saved back. Translation estimates are diagnostic, not calibration.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def build_fk(stage, base_pose, names):
    """Return q_rad -> {body_name: world 4x4}; all quaternions use XYZW."""
    def frame(position, quaternion):
        return pose_matrix([*position, *quaternion.GetImaginary(), quaternion.GetReal()])

    pending = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    chain, available = [], {"base"}
    while pending:
        old_count = len(pending)
        for prim in pending[:]:
            joint = UsdPhysics.RevoluteJoint(prim)
            parent = joint.GetBody0Rel().GetTargets()[0].name
            child = joint.GetBody1Rel().GetTargets()[0].name
            if parent in available:
                chain.append((parent, child, names.index(prim.GetName()),
                              np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())],
                              frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                              np.linalg.inv(frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()))))
                available.add(child)
                pending.remove(prim)
        if len(pending) == old_count:
            raise ValueError("Unresolved arm joint chain")

    def forward(q):
        poses = {"base": pose_matrix(base_pose)}
        for parent, child, index, axis, before, after in chain:
            rotate = np.eye(4)
            rotate[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
            poses[child] = poses[parent] @ before @ rotate @ after
        return poses

    return forward


def mesh_points(stage, root, link, marker):
    body = stage.GetPrimAtPath(root.GetPath().AppendChild(link))
    prim = next(p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                if p.IsA(UsdGeom.Mesh) and marker in str(p.GetPath()))
    transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, body)
    points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, point)))
                         for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    return points * UsdGeom.GetStageMetersPerUnit(stage), str(prim.GetPath())


def transformed_planes(planes, transform):
    normals = planes[:, :3] @ transform[:3, :3].T
    return np.c_[normals, planes[:, 3] - normals @ transform[:3, 3]]


def support_and_intersection(planes, cube_planes, cube_min, cube_max):
    bounds = [(None, None), (None, None), (float(cube_min[2]), float(cube_max[2]))]
    low = linprog([0, 1, 0], A_ub=planes[:, :3], b_ub=-planes[:, 3], bounds=bounds, method="highs")
    high = linprog([0, -1, 0], A_ub=planes[:, :3], b_ub=-planes[:, 3], bounds=bounds, method="highs")
    both = np.vstack((planes, cube_planes))
    intersection = linprog([0, 0, 0], A_ub=both[:, :3], b_ub=-both[:, 3],
                          bounds=[(None, None)] * 3, method="highs")
    if not low.success or not high.success:
        return {"depth_slabs_overlap": False, "intersects_cube": bool(intersection.success)}
    minimum_y, maximum_y = float(low.fun), float(-high.fun)
    return {
        "depth_slabs_overlap": True,
        "minimum_y_mm": minimum_y * 1000,
        "maximum_y_mm": maximum_y * 1000,
        "y_separation_lower_bound_mm": max(0., minimum_y - cube_max[1], cube_min[1] - maximum_y) * 1000,
        "intersects_cube": bool(intersection.success),
    }


def audit(args):
    from sim.scene.contact_materials import enable_collision_meshes

    source_hash = hashlib.sha256(args.asset.read_bytes()).hexdigest()
    stage = Usd.Stage.Open(str(args.asset))
    root = stage.GetDefaultPrim()
    names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    with np.load(args.trajectory, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    identical = {}
    for link, mesh in (("gripper", "wrist_roll_follower_so101_v1"), ("jaw", "moving_jaw_so101_v1")):
        visual, _ = mesh_points(stage, root, link, f"/visuals/{mesh}/")
        collision, _ = mesh_points(stage, root, link, f"/collisions/{mesh}/")
        identical[link] = {"points": len(visual), "equal_visual_and_collision_arrays": bool(np.array_equal(visual, collision))}
    enable_collision_meshes(root)
    fixed, fixed_path = mesh_points(stage, root, "gripper", "/collisions/wrist_roll_follower_so101_v1/finger_contact")
    whole_moving, whole_moving_path = mesh_points(stage, root, "jaw", "/collisions/moving_jaw_so101_v1/mesh")
    moving_marker = ("finger_contact" if args.moving_geometry == "split" else "mesh")
    moving, moving_path = mesh_points(stage, root, "jaw", f"/collisions/moving_jaw_so101_v1/{moving_marker}")
    fixed_planes, moving_planes = ConvexHull(fixed).equations, ConvexHull(moving).equations
    whole_moving_planes = ConvexHull(whole_moving).equations
    corners = np.array(np.meshgrid(*[[-args.half_size, args.half_size]] * 3)).reshape(3, -1).T
    rows = []
    for index in range(args.start_frame, min(args.end_frame, len(data["time_s"]))):
        fk = build_fk(stage, data["robot_base_pose_world_xyzw"][index], names)
        poses = fk(data["joint_pos_rad"][index])
        gripper = poses["gripper"]
        inverse = np.linalg.inv(gripper)
        cube_transform = inverse @ pose_matrix(data["cube_pose_world_xyzw"][index])
        cube_corners = corners @ cube_transform[:3, :3].T + cube_transform[:3, 3]
        cube_min, cube_max = cube_corners.min(0), cube_corners.max(0)
        axes = cube_transform[:3, :3].T
        normals = np.vstack((axes, -axes))
        offsets = -normals @ cube_transform[:3, 3] - args.half_size
        cube_planes = np.c_[normals, offsets]
        fixed_result = support_and_intersection(fixed_planes, cube_planes, cube_min, cube_max)
        moving_result = support_and_intersection(transformed_planes(moving_planes, inverse @ poses["jaw"]),
                                                  cube_planes, cube_min, cube_max)
        whole_moving_result = support_and_intersection(transformed_planes(whole_moving_planes, inverse @ poses["jaw"]),
                                                        cube_planes, cube_min, cube_max)
        for result, planes in ((fixed_result, fixed_planes),
                               (moving_result, transformed_planes(moving_planes, inverse @ poses["jaw"])),
                               (whole_moving_result, transformed_planes(whole_moving_planes, inverse @ poses["jaw"]))):
            signed = (cube_corners @ planes[:, :3].T + planes[:, 3]) / np.linalg.norm(planes[:, :3], axis=1)
            result["separating_face_gap_lower_bound_mm"] = float(max(0., signed.min(axis=0).max()) * 1000)
        rows.append({
            "frame": index, "time_s": float(data["time_s"][index]),
            "jaw_degrees": float(np.rad2deg(data["joint_pos_rad"][index, 5])),
            "cube_center_gripper_mm": (cube_transform[:3, 3] * 1000).tolist(),
            "cube_orientation_gripper_xyzw": Rotation.from_matrix(cube_transform[:3, :3]).as_quat().tolist(),
            "cube_corner_min_gripper_mm": (cube_min * 1000).tolist(),
            "cube_corner_max_gripper_mm": (cube_max * 1000).tolist(),
            "gripper_local_y_axis_world": gripper[:3, 1].tolist(),
            "fk_error_um": float(np.linalg.norm(gripper[:3, 3] - data["gripper_pos_world"][index]) * 1e6),
            "fixed_finger": fixed_result, "moving_finger": moving_result,
            "moving_whole_hull": whole_moving_result,
            "normal_force_during_following_control_interval_n": float(data["normal_force_n"][index]),
            "penetration_during_following_control_interval_mm": float(data["penetration_m"][index] * 1000),
        })
    reference = next(row for row in rows if row["frame"] == args.reference_frame)
    fixed_support = reference["fixed_finger"]
    y_axis = np.asarray(reference["gripper_local_y_axis_world"])
    minimum_shift = -fixed_support["y_separation_lower_bound_mm"] / 1000 * y_axis
    center_y = reference["cube_center_gripper_mm"][1]
    pad_mid_y = (fixed_support["minimum_y_mm"] + fixed_support["maximum_y_mm"]) / 2
    centering_shift = (center_y - pad_mid_y) / 1000 * y_axis
    assert hashlib.sha256(args.asset.read_bytes()).hexdigest() == source_hash
    return {
        "kind": "CPU-only CAD hull and oriented-cube geometry audit; no simulator or GPU",
        "asset": str(args.asset), "asset_sha256": source_hash,
        "trajectory": str(args.trajectory), "trajectory_sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "collision_visual_identity": identical,
        "runtime_collision_geometry": {"fixed": fixed_path, "moving": moving_path},
        "moving_geometry_variant": args.moving_geometry,
        "whole_moving_mesh_for_comparison": whole_moving_path,
        "maximum_fk_error_um": max(row["fk_error_um"] for row in rows),
        "fixed_intersection_frames": [row["frame"] for row in rows if row["fixed_finger"]["intersects_cube"]],
        "moving_intersection_frames": [row["frame"] for row in rows if row["moving_finger"]["intersects_cube"]],
        "bilateral_intersection_frames": [row["frame"] for row in rows
                                           if row["moving_finger"]["intersects_cube"] and row["fixed_finger"]["intersects_cube"]],
        "translation_diagnostic": {
            "reference_frame": args.reference_frame,
            "minimum_world_robot_translation_for_y_projection_edge_overlap_mm": (minimum_shift * 1000).tolist(),
            "world_robot_translation_to_center_finger_y_projection_on_cube_mm": (centering_shift * 1000).tolist(),
            "centering_shift_norm_mm": float(np.linalg.norm(centering_shift) * 1000),
            "interpretation": "Edge overlap is only a necessary one-dimensional condition, not a grasp. "
                              "Centering is a diagnostic offset, not evidence of a calibration error. "
                              "Neither translation is applied; the real camera must independently justify a mount/placement correction.",
        },
        "limitations": "CAD convex hull intersection ignores contact margins and softness. "
                       "Rows use pre-action states; force and penetration summarize the following 1/30 s interval. "
                       "The cube trajectory is actual free physics, not forced source motion.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--asset", type=Path, default=Path("arms/so101/usd/SO-ARM101-USD.usd"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=145)
    parser.add_argument("--end-frame", type=int, default=211)
    parser.add_argument("--reference-frame", type=int, default=188)
    parser.add_argument("--moving-geometry", choices=("whole", "split"), default="whole")
    parser.add_argument("--half-size", type=float, default=.01)
    args = parser.parse_args()
    report = audit(args)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
