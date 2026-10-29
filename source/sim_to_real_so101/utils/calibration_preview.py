"""Isaac-only collision evidence and room-camera contact sheets; no LeRobot."""
import json
from pathlib import Path

import numpy as np

from .calibration_poses import box_vertices, check_clearance, quaternion_matrix
from .joint_calibration import FOLLOWER_JOINTS


class PoseGeometry:
    """Conservative collision boxes carried by actual PhysX rigid-link poses."""

    def __init__(self, renderer):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.renderer = renderer
        self.robot = renderer.robot
        self.stage = renderer.env.unwrapped.sim.stage
        cache = UsdGeom.XformCache()
        self.local_boxes = {}
        self.local_hulls = {}
        body_paths = {}
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in self.robot.body_names:
                if "/env_0/Robot/" in str(prim.GetPath()):
                    body_paths[prim.GetName()] = prim
        if set(body_paths) != set(self.robot.body_names):
            raise ValueError("Could not identify all PhysX body prims for collision checking")
        for name, body in body_paths.items():
            inverse = cache.GetLocalToWorldTransform(body).GetInverse()
            vertices = []
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
                    continue
                transform = cache.GetLocalToWorldTransform(prim) * inverse
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points:
                    # Gf uses row-vector matrices, including translation in row 4.
                    matrix = np.asarray(transform)
                    local = np.asarray(points, dtype=float)
                    vertices.append(local @ matrix[:3, :3] + matrix[3, :3])
            if not vertices:
                raise ValueError(f"No collision mesh vertices for {name}; refusing an incomplete preview")
            from scipy.spatial import ConvexHull
            points = np.concatenate(vertices)
            self.local_hulls[name] = points[ConvexHull(points).vertices]
            self.local_boxes[name] = box_vertices(points.min(0), points.max(0))
        self.adjacent = set()
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) and "/env_0/Robot/" in str(prim.GetPath()):
                joint = UsdPhysics.Joint(prim)
                a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if a and b:
                    self.adjacent.add(frozenset((a[0].name, b[0].name)))
        from .joint_calibration import SETUP_PATH
        setup = json.loads(SETUP_PATH.read_text())
        self.mount_point = np.asarray(setup["robot"]["position"])
        self.table_normal = np.asarray(setup["robot"]["table_normal"])
        self.table_normal /= np.linalg.norm(self.table_normal)
        tabletop = self.stage.GetPrimAtPath("/World/envs/env_0/Room/Room/Visual/Furniture/Tabletop")
        table_vertices = []
        for prim in Usd.PrimRange(tabletop, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
                table_vertices.append(points @ matrix[:3, :3] + matrix[3, :3])
        if not table_vertices:
            raise ValueError("Cannot identify the tabletop mesh; refusing an incomplete preview")
        self.table_point = self.table_normal * np.max(np.concatenate(table_vertices) @ self.table_normal)

    def check(self, safety):
        transforms = self.robot.root_physx_view.get_link_transforms()[0].detach().cpu().numpy()
        if transforms.shape != (len(self.robot.body_names), 7) or not np.isfinite(transforms).all():
            raise RuntimeError("Invalid PhysX link transforms")
        boxes, evidence, hulls = {}, {}, {}
        for i, name in enumerate(self.robot.body_names):
            xyz, xyzw = transforms[i, :3], transforms[i, 3:]
            rotation = quaternion_matrix(xyzw)
            boxes[name] = (self.local_boxes[name] @ rotation.T + xyz, rotation)
            hulls[name] = self.local_hulls[name] @ rotation.T + xyz
            evidence[name] = transforms[i].tolist()
        exempt = {frozenset(pair) for pair in safety.get("mechanical_interface_exemptions", [])}
        result = check_clearance(boxes, self.adjacent | exempt, self.table_point, self.table_normal,
                                 margin_m=safety["margin_m"], mounted=safety.get("mounted_table_exemptions", ["base"]), hulls=hulls)
        result["mechanical_interface_exemptions"] = safety.get("mechanical_interface_exemptions", [])
        result["method"] = "collision convex hulls transformed by PhysX; OBB/SAT broad phase and GJK separating-plane clearance lower bound"
        result["physx_link_transforms_xyzw"] = evidence
        result["table_plane"] = {"point": self.table_point.tolist(), "normal": self.table_normal.tolist(),
                                 "source": "upper support plane of the room Tabletop mesh, with measured real_setup.json table normal"}
        support = boxes["gripper"][0].mean(0)
        normal = self.table_normal / np.linalg.norm(self.table_normal)
        relative = support - self.mount_point
        height = float((support - self.table_point) @ normal)
        reach = float(np.linalg.norm(relative - (relative @ normal) * normal))
        masses = self.robot.root_physx_view.get_masses()[0].detach().cpu().numpy()
        force = float(sum(mass * 9.81 for name, mass in zip(self.robot.body_names, masses) if name != "base"))
        plausible = (force <= safety["max_support_force_n"] and reach <= safety["max_support_reach_m"]
                     and .02 <= height <= safety["max_support_height_m"])
        result["hand_hold"] = dict(plausible=plausible, support_force_n=force,
                                   support_reach_m=reach, support_height_m=height,
                                   rule="supported arm: total moving weight and gripper support point within configured reach/height")
        if not plausible:
            result["status"] = "REJECT"
            result["reasons"].append("hand hold: support force, reach or height outside configured envelope")
        return result


def preview(renderer, poses, identity, output):
    """Render every reference and write both quantitative and visual evidence."""
    from PIL import Image, ImageDraw

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    geometry = PoseGeometry(renderer)
    rows, tiles = [], []
    for name, target in poses:
        shown = renderer.show(name, target)
        result = geometry.check(identity["preview"])
        camera = renderer.env.unwrapped.scene["camera_realsense_rgb"]
        renderer.render_for(.15)
        camera.update(renderer.env.unwrapped.step_dt, force_recompute=True)
        pixels = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        image = Image.fromarray(pixels).convert("RGB")
        image.save(output / f"{name}.png")
        tile = Image.new("RGB", (640, 550), "white")
        tile.paste(image.resize((640, 480)), (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 485), f"{name}: {result['status']}", fill="green" if result["status"] == "OK" else "red")
        draw.text((10, 505), ", ".join(f"{n}={v:g}" for n, v in zip(FOLLOWER_JOINTS, target)), fill="black")
        draw.text((10, 525), (result["reasons"] or ["table/self clearance >= 2 cm; supported hand envelope OK"])[0][:100], fill="black")
        tiles.append(tile)
        rows.append(dict(name=name, target_deg=target.tolist(), shown_physx_deg=shown, **result))
        print(f"{name}: {result['status']} " + "; ".join(result["reasons"]), flush=True)
    sheet = Image.new("RGB", (1280, 550 * ((len(tiles) + 1) // 2)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (640 * (i % 2), 550 * (i // 2)))
    sheet.save(output / "contact-sheet.png")
    result = dict(pose_set=identity, poses=rows, all_ok=all(row["status"] == "OK" for row in rows),
                  no_robot=True, camera="camera_realsense_rgb (same pose, focal scale 0.35 for full-arm reference view)", contact_sheet=str(output / "contact-sheet.png"))
    (output / "preview.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
