"""CPU-only static tool-point bounds; never commands the arm or changes an eval.

Positive lower bounds exclude the specified fixed diagnostic point at the
stored cube centre, not all finger contact, pushing, overshoot or task success.
Numerically found poses ignore orientation, collisions and the approach path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.grasp_diagnostics import _tip_offsets
from sim_to_real_so101.utils.pick_place_benchmark import EvalSetStarts
from sim_to_real_so101.utils.pick_place_eval import (
    resting_pose, rotation_matrix, upright_orientation,
)


def rotation_z(angle):
    angle = np.asarray(angle)
    matrix = np.broadcast_to(np.eye(4), (*angle.shape, 4, 4)).copy()
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix[..., 0, 0] = matrix[..., 1, 1] = cosine
    matrix[..., 1, 0], matrix[..., 0, 1] = sine, -sine
    return matrix


def transform(quaternion, position):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        [*quaternion.GetImaginary(), quaternion.GetReal()]
    ).as_matrix()
    matrix[:3, 3] = position
    return matrix


def stored_target(start, scene):
    """Mirror the reset's train/non-train orientation rule in the stored scene."""
    quaternion = start.get("orientation_wxyz", [math.cos(start["yaw"] / 2), 0, 0,
                                                math.sin(start["yaw"] / 2)])
    if start["split"] != "train":
        quaternion = upright_orientation(scene["base_quaternion_wxyz"],
                                         scene["table_plane_world"], quaternion)
    world, _ = resting_pose([start["x"], start["y"], 0], quaternion,
                           scene["base_position_world"], scene["base_quaternion_wxyz"],
                           scene["table_plane_world"], scene["cube_size_m"])
    return rotation_matrix(scene["base_quaternion_wxyz"]).T @ (
        world - scene["base_position_world"])


class Chain:
    def __init__(self, stage, cube_size):
        if UsdGeom.GetStageMetersPerUnit(stage) != 1:
            raise ValueError("This audit requires metre-authored USD geometry")
        root = str(stage.GetDefaultPrim().GetPath())
        names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
        anchors, inverses, limits = [], [], []
        for name in names:
            joint = stage.GetPrimAtPath(root + "/joints/" + name)
            if joint.GetAttribute("physics:axis").Get() != "Z":
                raise ValueError("Unexpected joint axis")
            anchors.append(transform(joint.GetAttribute("physics:localRot0").Get(),
                                     joint.GetAttribute("physics:localPos0").Get()))
            inverses.append(np.linalg.inv(transform(
                joint.GetAttribute("physics:localRot1").Get(),
                joint.GetAttribute("physics:localPos1").Get())))
            limits.append([joint.GetAttribute("physics:lowerLimit").Get(),
                           joint.GetAttribute("physics:upperLimit").Get()])
        self.physical = np.asarray(limits)
        self.first = anchors[0]
        self.links = [inverses[i] @ anchors[i + 1] for i in range(4)]
        self.final_inverse = inverses[-1]
        self.tip = _tip_offsets(stage, root, cube_size / 2)[0] + [cube_size / 2, 0, 0]
        self.tail = inverses[-1] @ np.r_[self.tip, 1]
        self.radius = sum(np.linalg.norm(self.links[i][:3, 3]) for i in (2, 3))
        self.radius += np.linalg.norm(self.tail[:3])
        cache = UsdGeom.XformCache()
        base = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root + "/base"))).T
        gripper = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root + "/gripper"))).T
        reference = np.linalg.inv(base) @ gripper
        zero = np.eye(4)
        for anchor, inverse in zip(anchors, inverses):
            zero = zero @ anchor @ inverse
        self.zero_error_m = float(np.linalg.norm(zero[:3, 3] - reference[:3, 3]))
        self.zero_error_rad = float(Rotation.from_matrix(
            zero[:3, :3].T @ reference[:3, :3]).magnitude())
        if self.zero_error_m > 2e-6 or self.zero_error_rad > 2e-5:
            raise ValueError("Joint-chain zero pose disagrees with authored link transforms")
        np.testing.assert_allclose(self.fk(np.zeros(5)),
                                   (reference @ np.r_[self.tip, 1])[:3], atol=2e-6)
        np.testing.assert_allclose(self.gripper_rotation(np.zeros(5)), reference[:3, :3], atol=2e-6)

    def elbow(self, pan, lift):
        return (self.first @ rotation_z(pan) @ self.links[0] @ rotation_z(lift)
                @ self.links[1])[..., :3, 3]

    def fk(self, degrees):
        angles = np.deg2rad(np.asarray(degrees))
        matrix = self.first @ rotation_z(angles[..., 0])
        for index, link in enumerate(self.links):
            matrix = matrix @ link @ rotation_z(angles[..., index + 1])
        return (matrix @ self.tail)[..., :3]

    def gripper_rotation(self, degrees):
        angles = np.deg2rad(np.asarray(degrees))
        matrix = self.first @ rotation_z(angles[..., 0])
        for index, link in enumerate(self.links):
            matrix = matrix @ link @ rotation_z(angles[..., index + 1])
        return (matrix @ self.final_inverse)[..., :3, :3]

    def search_cube_aligned(self, bounds, target, cube_rotation, point_angles):
        """Probe an extra orientation constraint; this is not required by the task.

        A failed numerical search does not prove infeasibility. A found pose
        still lacks collision and path checks and is never sent to the robot.
        """
        rng = np.random.default_rng(20261023)
        faces = []
        for quarter_turn in range(4):
            target_rotation = cube_rotation @ Rotation.from_euler(
                "z", quarter_turn * 90, degrees=True).as_matrix()

            def residual(angles):
                rotation_error = Rotation.from_matrix(
                    target_rotation.T @ self.gripper_rotation(angles)).as_rotvec()
                return np.r_[self.fk(angles) - target, .05 * rotation_error]

            initial_points = [point_angles, bounds.mean(axis=1),
                              *rng.uniform(bounds[:, 0], bounds[:, 1], size=(6, 5))]
            fits = [least_squares(residual, initial, bounds=(bounds[:, 0], bounds[:, 1]),
                                  ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=350)
                    for initial in initial_points]
            best = min(fits, key=lambda fit: np.linalg.norm(fit.fun))
            faces.append(dict(cube_face_quarter_turn=quarter_turn,
                              position_error_mm=float(np.linalg.norm(best.fun[:3]) * 1000),
                              orientation_error_degrees=float(np.linalg.norm(best.fun[3:]) / .05 * 180 / np.pi),
                              joints_degrees=best.x.tolist(), weighted_cost=float(np.linalg.norm(best.fun))))
        return dict(found_within_1mm_5degrees=any(
            face["position_error_mm"] < 1 and face["orientation_error_degrees"] < 5 for face in faces),
            best=min(faces, key=lambda face: face["weighted_cost"]), all_faces=faces)

    def lower_bounds(self, bounds, targets, grid_step_degrees):
        # All downstream rotations remain inside a ball about the elbow axis.
        # Every allowed pan/lift pair is within half a grid cell of a sample.
        grids = [np.linspace(*np.deg2rad(bounds[i]),
                             int(np.ceil(np.ptp(bounds[i]) / grid_step_degrees)) + 1)
                 for i in (0, 1)]
        half_cells = [float(np.diff(grid).max() / 2) for grid in grids]
        pan_radius = sum(np.linalg.norm(link[:3, 3]) for link in self.links[:2])
        lift_radius = np.linalg.norm(self.links[1][:3, 3])
        allowance = (2 * pan_radius * math.sin(half_cells[0] / 2)
                     + 2 * lift_radius * math.sin(half_cells[1] / 2))
        distances = np.full(len(targets), np.inf)
        for index in range(0, len(grids[0]), 16):
            pivots = self.elbow(grids[0][index:index + 16, None], grids[1][None, :]).reshape(-1, 3)
            for target_index, target in enumerate(targets):
                distances[target_index] = min(distances[target_index],
                                             np.linalg.norm(pivots - target, axis=1).min())
        # Extra numerical margin; the analytic allowance, not the grid alone,
        # makes this a bound over the continuous pan/lift intervals.
        return np.maximum(distances - self.radius - allowance - 1e-7, 0), allowance


def local_path(value):
    prefix = "/workspace/Sim-to-Real-SO-101-Workshop/"
    return ROOT / value[len(prefix):] if value.startswith(prefix) else Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/pick_place_molmo_so101.yaml")
    parser.add_argument("--grid-step-deg", type=float, default=.125)
    parser.add_argument("--check-orientation", action="store_true",
                        help="also search cube-aligned poses; this adds an orientation constraint the task does not require")
    args = parser.parse_args()
    if not math.isfinite(args.grid_step_deg) or not 0 < args.grid_step_deg <= 1:
        parser.error("--grid-step-deg must be finite and in (0, 1]")
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["policy"]["checkpoint"] != "molmoact2_so101":
        raise ValueError("This diagnostic is for the released SO-101 checkpoint")
    options = cfg["policy"]["options"]
    stats_path = Path(cfg["policy"]["models_dir"]) / cfg["policy"]["checkpoint"] / "norm_stats.json"
    stats_file = json.loads(stats_path.read_text())
    if stats_file["norm_mode"] != "q01_q99":
        raise ValueError("Unexpected checkpoint normalization")
    stats = stats_file["metadata_by_tag"][options["norm_tag"]]["action_stats"]
    schedule = EvalSetStarts(local_path(cfg["eval"]["eval_set"]), seed=1984, selection="one_per_zone")
    data = schedule.data
    if len(set(data["scene"]["cube_size_m"])) != 1:
        raise ValueError("Diagnostic point assumes a cube")
    stage = Usd.Stage.Open(str(local_path(cfg["eval"]["robot_usd"])))
    chain = Chain(stage, data["scene"]["cube_size_m"][0])
    bounds = np.asarray([stats["q01"][:5], stats["q99"][:5]]).T.copy()
    pan_scale = float(options.get("pan_model_scale", 1))
    pan_offset = float(options.get("pan_model_offset", 0))
    if not math.isfinite(pan_scale) or pan_scale <= 0 or not math.isfinite(pan_offset):
        raise ValueError("Invalid pan adaptation")
    bounds[0] = (bounds[0] - pan_offset) / pan_scale
    bounds[1] = 90 - bounds[1, ::-1]
    bounds[2] -= 90
    bounds[:, 0] = np.maximum(bounds[:, 0], chain.physical[:, 0])
    bounds[:, 1] = np.minimum(bounds[:, 1], chain.physical[:, 1])
    profiles = {"baseline": bounds, "shoulder_gain_075": bounds.copy(), "physical": chain.physical}
    # A hypothetical range change, not an enabled policy option or a controller.
    profiles["shoulder_gain_075"][1, 1] = min(bounds[1, 1] / .75, chain.physical[1, 1])
    targets = np.asarray([stored_target(start, data["scene"]) for start in data["starts"]])
    indices = [next(i for i, start in enumerate(data["starts"]) if start["id"] == item["id"])
               for item in schedule.schedule]
    rng = np.random.default_rng(20261023)
    samples = rng.uniform(chain.physical[:, 0], chain.physical[:, 1], size=(10000, 5))
    radii = np.linalg.norm(chain.fk(samples) - chain.elbow(
        np.deg2rad(samples[:, 0]), np.deg2rad(samples[:, 1])), axis=1)
    if radii.max() > chain.radius + 1e-12:
        raise ValueError("Downstream-ball inclusion check failed")
    source_paths = [args.config, stats_path, local_path(cfg["eval"]["eval_set"]), Path(__file__)]
    source_paths += [Path(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath]
    source_paths += [ROOT / "source/sim_to_real_so101/utils" / name for name in
                     ("grasp_diagnostics.py", "pick_place_eval.py", "pick_place_benchmark.py")]
    report = dict(scope=__doc__, eval_set_id=data["eval_set_id"],
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source_paths},
                  checks=dict(zero_position_error_m=chain.zero_error_m,
                              zero_rotation_error_rad=chain.zero_error_rad,
                              random_ball_points=10000, maximum_sampled_downstream_m=float(radii.max())),
                  downstream_radius_m=chain.radius, tool_offset_m=chain.tip.tolist(), profiles={})
    output = Path(tempfile.mkdtemp(prefix="molmo-target-reach-", dir="/dev/shm")) / "report.json"
    for label, limits in profiles.items():
        lower, allowance = chain.lower_bounds(limits, targets, args.grid_step_deg)
        rows = []
        for index in indices:
            best_error, best_angles = float("inf"), None
            initial_points = [limits.mean(axis=1), *rng.uniform(limits[:, 0], limits[:, 1], size=(15, 5))]
            for initial in initial_points:
                fit = least_squares(lambda angles: chain.fk(angles) - targets[index], initial,
                                    bounds=(limits[:, 0], limits[:, 1]), ftol=1e-11, xtol=1e-11,
                                    gtol=1e-11, max_nfev=350)
                error = float(np.linalg.norm(fit.fun))
                if error < best_error:
                    best_error, best_angles = error, fit.x
                if error < 1e-7:
                    break
            if best_error + 2e-6 < lower[index]:
                raise ValueError("Found pose contradicts conservative lower bound")
            start = data["starts"][index]
            rows.append(dict(start_id=start["id"], zone=start["zone"],
                             target_base_m=targets[index].tolist(), lower_mm=float(lower[index] * 1000),
                             found_error_mm=best_error * 1000, found_joint_degrees=best_angles.tolist()))
            if args.check_orientation:
                quaternion = start.get("orientation_wxyz", [math.cos(start["yaw"] / 2), 0, 0,
                                                            math.sin(start["yaw"] / 2)])
                if start["split"] != "train":
                    quaternion = upright_orientation(data["scene"]["base_quaternion_wxyz"],
                                                     data["scene"]["table_plane_world"], quaternion)
                rows[-1]["cube_aligned_pose_search"] = chain.search_cube_aligned(
                    limits, targets[index], rotation_matrix(quaternion), best_angles)
        report["profiles"][label] = dict(bounds_degrees=limits.tolist(),
            grid_cell_allowance_mm=allowance * 1000, smoke_starts=rows,
            full_set_bounds=[dict(start_id=start["id"], zone=start["zone"], lower_mm=float(value * 1000))
                             for start, value in zip(data["starts"], lower)])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(label, "stored starts with bound >1 mm:", int((lower > .001).sum()),
              "; >30 mm:", int((lower > .03).sum()), flush=True)
        for row in rows:
            print(f"  {row['zone']}: lower={row['lower_mm']:.2f} mm; found={row['found_error_mm']:.2f} mm", flush=True)
            if args.check_orientation:
                search = row["cube_aligned_pose_search"]
                best = search["best"]
                print(f"    cube-aligned pose found: {search['found_within_1mm_5degrees']}; "
                      f"best search residual: {best['position_error_mm']:.2f} mm / "
                      f"{best['orientation_error_degrees']:.2f} degrees", flush=True)
    print(f"RAM audit report: {output}")


if __name__ == "__main__":
    main()
