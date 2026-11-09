"""Summarize captured grasp replays on CPU, preserving diagnostic/original distinctions."""

import argparse
import json
from pathlib import Path

import numpy as np


def rotation_matrices_xyzw(quaternions):
    q = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)


def motion_class(arguments):
    if arguments.get("prescribe_arm_states"):
        return "prescribed_measured_arm_motion"
    if arguments.get("arm_state_root"):
        return "measured_state_position_targets"
    return "original_recorded_commands"


def trajectory_metrics(data, result, clearance_m, stable_frames, relative_drift_m):
    cube = data["cube_pose_world_xyzw"]
    plane = np.asarray(result["resolved"]["table_plane"])
    normal = np.r_[-plane[:2], 1.]
    center_height = cube[:, :3] @ normal - plane[2]
    rotation = rotation_matrices_xyzw(cube[:, 3:])
    support = .01 * np.abs(np.einsum("j,njk->nk", normal, rotation)).sum(axis=1)
    bottom_clearance = (center_height - support) / np.linalg.norm(normal)
    local = data["cube_local_gripper"]
    force = data["normal_force_n"]
    intervals = np.diff(data["time_s"])
    fps = float(result.get("source", {}).get("fps", round(1 / np.median(intervals))))
    windows = []
    for start in range(len(cube) - stable_frames + 1):
        end = start + stable_frames
        if np.min(bottom_clearance[start:end]) < clearance_m:
            continue
        drift = np.linalg.norm(local[start:end] - local[start], axis=1).max()
        if drift <= relative_drift_m:
            windows.append({"start_frame": start, "end_frame_exclusive": end,
                            "start_time_s": float(data["time_s"][start]),
                            "duration_s": (stable_frames - 1) / fps,
                            "minimum_cube_bottom_clearance_m": float(bottom_clearance[start:end].min()),
                            "maximum_relative_position_drift_m": float(drift),
                            "minimum_combined_contact_force_n": float(force[start:end].min())})
            break
    return {"maximum_cube_center_height_above_table_mm": float(center_height.max() * 1000),
            "maximum_cube_center_lift_above_start_mm": float((center_height.max() - center_height[0]) * 1000),
            "maximum_cube_bottom_clearance_mm": float(bottom_clearance.max() * 1000),
            "lift_and_carry_metric_pass": bool(windows), "qualifying_windows": windows}


def summarize(directory, args):
    reference_result = json.loads((directory / "baseline/result.json").read_text())
    with np.load(directory / "baseline/trajectory.npz", allow_pickle=False) as archive:
        reference_initial = archive["cube_pose_world_xyzw"][0, :3].copy()
        reference_actions = archive["action_raw"].copy()
        reference_base = archive["robot_base_pose_world_xyzw"][0, :3].copy()
    invalid = dict(value.split("=", 1) for value in args.invalid_run)
    rows = []
    for path in sorted(directory.glob("*/result.json")):
        result = json.loads(path.read_text())
        arguments = result.get("arguments", {})
        offset = np.asarray(arguments.get("robot_base_offset", (0., 0., 0.))) + [
            0., 0., float(arguments.get("robot_base_z_offset", 0.))]
        has_offset = bool(np.any(offset != 0))
        row = {"run": path.parent.name, "result_path": str(path),
               "completed": bool(result.get("completed")),
               "motion_class": motion_class(arguments),
               "command_timeline_class": ("recorded_prefix_then_appended_hold_diagnostic"
                                          if arguments.get("hold_seconds", 0) else "recorded_episode"),
               "robot_base_offset_mm": (offset * 1000).tolist(),
               "additional_robot_base_offset_mm": (offset * 1000).tolist(),
               "robot_base_z_offset_mm": float(offset[2] * 1000),
               "scene_class": "diagnostic_mount_offset" if has_offset else "configured_original_mount",
               "invalid_reason": invalid.get(path.parent.name),
               "friction_override": arguments.get("friction"), "condim": arguments.get("condim"),
               "substeps": result.get("resolved", {}).get("substeps"),
               "geometry": arguments.get("geometry"),
               "peak_penetration_mm": result.get("max_penetration_mm"),
               "peak_combined_normal_force_n": result.get("peak_normal_force_n"),
               "peak_jaw_torque_nm": result.get("peak_jaw_torque_nm")}
        trajectory = path.parent / "trajectory.npz"
        if row["completed"] and trajectory.exists():
            with np.load(trajectory, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            observed_offset = data["robot_base_pose_world_xyzw"][0, :3] - reference_base
            has_offset = bool(np.linalg.norm(observed_offset) > 1e-6)
            row["robot_base_offset_mm"] = (observed_offset * 1000).tolist()
            row["robot_base_z_offset_mm"] = float(observed_offset[2] * 1000)
            row["scene_class"] = "diagnostic_mount_offset" if has_offset else "configured_original_mount"
            row["mount_offset_source"] = "Captured base world position minus recorded baseline, including any configured correction"
            row.update(trajectory_metrics(data, result, args.clearance_m, args.stable_frames, args.relative_drift_m))
            row["initial_cube_world_difference_from_baseline_m"] = float(
                np.linalg.norm(data["cube_pose_world_xyzw"][0, :3] - reference_initial))
            row["same_source_hashes_as_baseline"] = (result["source"]["sha256"] == reference_result["source"]["sha256"])
            row["stored_actions_equal_baseline"] = (data["action_raw"].shape == reference_actions.shape
                                                     and np.array_equal(data["action_raw"], reference_actions))
            row["cube_mass_kg"] = result["resolved"]["cube_mass_kg"]
            valid_original = (not row["invalid_reason"] and row["motion_class"] == "original_recorded_commands"
                              and row["same_source_hashes_as_baseline"] and row["stored_actions_equal_baseline"]
                              and row["initial_cube_world_difference_from_baseline_m"] <= 1e-6
                              and all(abs(mass - .008) < 1e-8 for mass in row["cube_mass_kg"]))
            row["original_command_success_under_this_scene"] = bool(valid_original and row["lift_and_carry_metric_pass"])
            row["original_command_success_under_original_mount"] = bool(
                row["original_command_success_under_this_scene"] and not has_offset)
        else:
            row["lift_and_carry_metric_pass"] = False
            row["original_command_success_under_this_scene"] = False
            row["original_command_success_under_original_mount"] = False
        rows.append(row)
    return {
        "criterion": {"cube_mass_kg": .008, "minimum_cube_bottom_table_clearance_m": args.clearance_m,
                      "minimum_consecutive_frames": args.stable_frames, "fps": 30,
                      "maximum_position_drift_relative_to_gripper_m": args.relative_drift_m,
                      "cube_support_extent": "Oriented 20 mm cube, evaluated against measured table plane",
                      "original_replay_requirements": "Original commands, identical source hashes/actions, identical initial cube world position within1um, 8g cube, completed and not invalid",
                      "diagnostic_policy": "Measured-target/prescribed motion cannot establish original-command success; changed mount cannot establish success under original mount",
                      "limitations": "This tests lift and carry, not successful release/placement. Combined contact force cannot establish simultaneous contact at both fingers."},
        "runs": rows,
        "original_command_successes_under_original_mount": [row["run"] for row in rows if row["original_command_success_under_original_mount"]],
        "original_command_successes_under_diagnostic_mount": [row["run"] for row in rows if row["original_command_success_under_this_scene"] and row["scene_class"] == "diagnostic_mount_offset"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/grasp_replay"))
    parser.add_argument("--report", type=Path, default=Path("outputs/grasp_replay/sweep_summary.json"))
    parser.add_argument("--clearance-m", type=float, default=.05)
    parser.add_argument("--stable-frames", type=int, default=16,
                        help="16 captured states span 0.5 seconds at 30 Hz")
    parser.add_argument("--relative-drift-m", type=float, default=.002)
    parser.add_argument("--invalid-run", action="append", default=[], metavar="RUN=REASON")
    args = parser.parse_args()
    report = summarize(args.directory, args)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("run | motion | mount_xyz_mm | lift_mm | bottom_clearance_mm | penetration_mm | lift_carry_metric | original_episode_lift_carry")
    for row in report["runs"]:
        lift = row.get("maximum_cube_center_lift_above_start_mm", float("nan"))
        bottom = row.get("maximum_cube_bottom_clearance_mm", float("nan"))
        penetration = row.get("peak_penetration_mm")
        note = " INVALID: " + row["invalid_reason"] if row["invalid_reason"] else ""
        mount = ",".join(f"{value:.3f}" for value in row["robot_base_offset_mm"])
        print(f"{row['run']} | {row['motion_class']} | [{mount}] | "
              f"{lift:.3f} | {bottom:.3f} | {penetration} | {row['lift_and_carry_metric_pass']} | "
              f"{row['original_command_success_under_this_scene']}{note}")
    print("ORIGINAL_MOUNT_SUCCESSES " + str(report["original_command_successes_under_original_mount"]))
    print("DIAGNOSTIC_MOUNT_SUCCESSES " + str(report["original_command_successes_under_diagnostic_mount"]))


if __name__ == "__main__":
    main()
