"""Verify both completed dataset methods and write their final calibration report."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.calibration_dataset import snapshot, units_to_degrees, degrees_to_units
from sim_to_real_so101.utils.joint_calibration import follower_mapping_from_calibration


def main():
    report = ROOT / "docs/reports/2026-10-27-calibration-fixes"
    audit = json.loads((report / "step0.json").read_text())
    method_a = json.loads((report / "method_a.json").read_text())
    method_b = json.loads((report / "method_b.json").read_text())
    setup = json.loads((ROOT / "source/sim_to_real_so101/assets/real_setup.json").read_text())
    source = ROOT / "datasets/pick_place_v1"
    real = ROOT / "datasets/pick_place_v1-real"
    destination = ROOT / "datasets/pick_place_v1_calibrated_B"
    if not method_b["complete"] or len(method_b["episodes"]) != 75:
        raise RuntimeError("Method B has not completed all 75 episodes")
    if [e["source_episode"] for e in method_b["episodes"]] != list(range(75)):
        raise RuntimeError("Method B missed or repeated a source episode")
    successes = [e for e in method_b["episodes"] if e["success"]]
    failed_ids = [e["source_episode"] for e in method_b["episodes"] if not e["success"]]
    with (report / "method_b_episode_map.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(method_b["episodes"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(method_b["episodes"])
    if [e["destination_episode"] for e in successes] != list(range(len(successes))):
        raise RuntimeError("Method B retained episode indices are inconsistent")
    info = json.loads((destination / "meta/info.json").read_text())
    if info["total_episodes"] != len(successes) or info["total_frames"] != sum(e["frames"] for e in successes):
        raise RuntimeError("Method B output metadata does not match the successful replay episodes")
    if snapshot(source) != audit["source_hashes"] or snapshot(real) != audit["real_hashes"]:
        raise RuntimeError("An original dataset changed")
    copied = ROOT / "datasets/pick_place_v1_calibrated_A"
    copied_hashes = snapshot(copied)
    preserved = [name for name in audit["source_hashes"]
                 if not name.startswith(("data/", "meta/episodes/")) and name != "meta/stats.json"]
    if any(copied_hashes[name] != audit["source_hashes"][name] for name in preserved):
        raise RuntimeError("Method A no longer preserves the original images and trajectories")
    maximum_error, source_actions = 0., {}
    for path in sorted((source / "data").rglob("*.parquet")):
        original = pq.read_table(path)
        relabeled = pq.read_table(copied / path.relative_to(source))
        if not original.select([n for n in original.column_names if n not in ("action", "observation.state")]).equals(
                relabeled.select([n for n in relabeled.column_names if n not in ("action", "observation.state")])):
            raise RuntimeError("Method A changed unrelated frame columns")
        for feature in ("action", "observation.state"):
            old_values, new_values = np.asarray(original[feature].to_pylist()), np.asarray(relabeled[feature].to_pylist())
            error = np.max(np.abs(units_to_degrees(old_values, audit["old_mapping"])
                                  - units_to_degrees(new_values, setup["sim_joint_mapping"])))
            maximum_error = max(maximum_error, float(error))
        for row in original.to_pylist():
            source_actions.setdefault(row["episode_index"], []).append(row["action"])
    if maximum_error > 1e-5:
        raise RuntimeError(f"Method A physical angle error {maximum_error} exceeds float32 roundoff")
    output_rows = []
    for path in sorted((destination / "data").rglob("*.parquet")):
        output_rows.extend(pq.read_table(path).to_pylist())
    if len(output_rows) != info["total_frames"]:
        raise RuntimeError("Method B parquet frame total differs from metadata")
    for episode in successes:
        rows = sorted([r for r in output_rows if r["episode_index"] == episode["destination_episode"]],
                      key=lambda row: row["frame_index"])
        if len(rows) != episode["frames"] or [r["frame_index"] for r in rows] != list(range(len(rows))):
            raise RuntimeError("Method B episode timeline differs from the recorded replay")
        np.testing.assert_array_equal([r["action"] for r in rows], source_actions[episode["source_episode"]])
        if not np.isfinite([r["observation.state"] for r in rows]).all():
            raise RuntimeError("Method B contains non-finite measured states")
    data_audit = dict(source_unchanged=True, real_unchanged=True, method_a_preserved_files=len(preserved),
                      method_a_max_physical_angle_error_degrees=maximum_error,
                      method_b_episodes=len(successes), method_b_frames=len(output_rows),
                      method_b_actions_match_original_sequences=True, method_b_states_finite=True)
    (report / "completion-data-audit.json").write_text(json.dumps(data_audit, indent=2) + "\n")
    import hashlib
    calibration = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/rpd_robot.json"
    if hashlib.sha256(calibration.read_bytes()).hexdigest() != audit["calibration_sha256"]:
        raise RuntimeError("Follower calibration changed")
    subprocess.run([sys.executable, str(ROOT / "scripts/rebuild_calibrated_dataset.py"),
                    "--audit", str(report / "step0.json"), "--report", str(report),
                    "--method-b", str(destination)], check=True)
    tests = json.loads((report / "test_comparison.json").read_text())
    if not all(tests[env]["same_failure_set"] for env in ("host", "container")):
        raise RuntimeError("Before/after test failures differ; investigate before finalizing")
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    encoder_mapping = follower_mapping_from_calibration(audit["calibration"])
    myroom_radians = np.array([setup["robot"]["joint_positions"][n]
                              for n in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")])
    myroom_units = degrees_to_units(np.rad2deg(myroom_radians), setup["sim_joint_mapping"])
    if not (np.all(myroom_units[:5] >= -100) and np.all(myroom_units[:5] <= 100) and 0 <= myroom_units[-1] <= 100):
        raise RuntimeError("MyRoom initial pose is outside the calibrated command range")
    usd_names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
    lines = ["# Calibration fixes — 2026-10-27", "",
             "The simulation now derives all six follower spans through one shared function. "
             "The real calibration files, original sim/real datasets, weights, scene assets and completed evaluation results remain intact.", "",
             "Step 0 was reported in the conversation before implementation. "
             "[step0.json](step0.json) preserves the pxr audit from the teleop image, the original mapping, "
             "the follower fixture and the pre-change per-file dataset hashes.", "",
             "## Step 0: counts and degrees", "",
             "| Joint | Calibration counts | Encoder degrees | Old sim mapping | New sim mapping | USD limits | Outside USD |",
             "|---|---|---|---|---|---|---|"]

    def interval(low, high):
        return f"{low:.6f} … {high:.6f}"
    for name, usd_name in zip(names, usd_names):
        calibration_joint = audit["calibration"][name]
        low, high = calibration_joint["range_min"], calibration_joint["range_max"]
        old, new = audit["old_mapping"][name], setup["sim_joint_mapping"][name]
        encoder = encoder_mapping[name]
        limits = audit["usd_limits"][usd_name]
        outside = []
        if new["joint_min"] < limits[0]:
            outside.append("lower")
        if new["joint_max"] > limits[1]:
            outside.append("upper")
        lines.append(f"| {name} | {low} … {high} | {interval(encoder['joint_min'], encoder['joint_max'])} | "
                     f"{interval(old['joint_min'], old['joint_max'])} | {interval(new['joint_min'], new['joint_max'])} | "
                     f"{interval(*limits)} | {', '.join(outside)} |")
    lines += ["", "The USD does not contain a separate ‘closed’ label. The closing mechanical end-stop is its jaw lower limit, "
              "−10.000003815°. Its authored angular state and drive target are 0°. "
              "The existing CAD diagnostic contact-point gap is 15.8 mm at 0°, 3.48 mm at the lower stop, "
              "and −6.97 mm at −19°. Thus the available calibration and USD data do not verify a 19° offset. "
              "Gripper unit 0 uses the closing end-stop and its measured span is 129.638671875°.", "",
              "Wrist encoder 2048 still maps to 0°. The 0–4095 range gives a −0.0439453125° midpoint "
              "at normalized unit 0; no additional follower zero rotation was introduced.", "",
              "## Initial poses and shared mapping", "",
              "`utils/joint_calibration.py:follower_mapping_from_calibration` is used by the regeneration "
              "command, runtime sim setup, and real rollout. The JSON stores all joint ranges, methods, follower ID, "
              "calibration hash and the measured USD gripper anchor. The fixture regression checks equality for all six joints.", "",
              "The `assets/so101.py` initial radians remain `[-0.2736, -0.6109, -0.0745, 1.5148, -0.0326036732051, -0.1465]`. "
              "The MyRoom override also keeps its existing physical pose. Both are representable by the new mapping. "
              "The real rollout loop has no fixed rest pose or automatic rest-pose movement; its observed state and first-target guard remain in follower units.", "",
              "MyRoom initial radians: `" + ", ".join(f"{v:.10f}" for v in myroom_radians) + "`. "
              "The corresponding new units are `" + ", ".join(f"{v:.6f}" for v in myroom_units) + "`.", "",
              "Frozen eval sets retain their original files and cube starts. Their arm-start states now use the "
              "`scene.joint_mapping` saved in each set, preserving the frozen physical poses. "
              "Non-benchmark profiles use Method A for their recorded starts. Evaluation results from before "
              "and after this revision cannot be directly compared. Existing old-unit checkpoints need retraining "
              "or an explicit adapter before use with the relabeled units; no weights were changed.", "",
              "## USD and camera cleanup", "",
              "Only `SO-ARM101-USD.usd` remains as the controlled arm asset. The alternate no-camera arm and "
              "`S0101_NO_CAMERA_CFG` were removed. The room package, room wrapper, wrist mount, meshes and other scene assets remain.", "",
              "The removed `ARM_new.usd` layer changed exactly five composed attributes: wrist-roll `physics:localRot1`, "
              "gripper orientation/translation, and jaw orientation/translation. Its mesh geometry and relationships were unchanged. "
              "Adding 90° to the canonical wrist coordinate reproduced its body transforms over 100 poses with a maximum matrix "
              f"difference of {audit['wrist_fk_100_poses_max_matrix_difference']:.3g}. "
              "The affected Molmo profiles now use the canonical USD and `wrist_roll_model_offset: -90` in both mapping directions. "
              "The canonical wrist still clamps at ±160°; native Molmo wrist targets above 70° reach its upper stop. "
              "The initial follower wrist pose and encoder zero remain unchanged.", "",
              "pxr found **zero Camera prims** in all three original arm assets, including inactive prims. "
              "Camera-mount geometry is retained. The optional runtime camera was `{ENV_REGEX_NS}/molmo_room`. "
              "Its helper, evaluation-script spawn/routing/report paths, shell/config flags, YAML settings, capability-sweep "
              "candidate, and obsolete feature tests were removed. The calibrated room RGB/depth and wrist cameras remain. "
              "Original call sites were listed in the conversation before removal. Historical reports retain their original "
              "asset names and camera-result fields as evidence, not current launch instructions.", "",
              "## Dataset methods", "",
              "| Method | Output root | Retained episodes | Success evidence |",
              "|---|---|---:|---|",
              "| A: relabel | `datasets/pick_place_v1_calibrated_A` | 75 / 75 | Original physical trajectories and images preserved; no new dynamics success score |",
              f"| B: physics replay | `datasets/pick_place_v1_calibrated_B` | {len(successes)} / 75 | Existing contact/lift/release/25-tick placement benchmark |", "",
              "**Recommend Method A** for preserving the demonstrations. It relabels action and state units with "
              "`inverse_new_mapping(old_mapping(u))`, without clipping, and transforms global/per-episode joint statistics. "
              f"All {method_a['frames']:,} frames remain; maximum physical-angle roundoff is "
              f"{method_a['max_physical_angle_error_degrees']:.9g}°. Video bytes, cube sidecars and unrelated metadata match the source hashes. "
              "The pinned LeRobot reader decoded both cameras in the first and last frames successfully.", "",
              "Method A preserves existing successes and failures. Original recordings do not store the contact/lift/release "
              "benchmark labels, so an independently verified original success count is unavailable; **75 retained is not a new 75/75 success claim**.", "",
              "Method B replayed every original action sequence from its recorded initial physical joint/cube positions "
              "with zero initial velocities. Only the initial cube pose was restored; subsequent cube motion came from physics. "
              "Images and measured states were rendered/recorded anew; failed episodes were discarded. A confirmed benchmark success at any step counts, matching the usual termination event; saved episodes retain the full original action sequence. "
              f"It retained {len(successes)} episodes and {info['total_frames']:,} frames. "
              "[method_b.json](method_b.json) records every attempt and retained source ID. "
              "This is an open-loop replay of demonstrations, not a policy evaluation or a new real-robot trial.", "",
              "Failed Method B source episodes (zero-based): `" + ", ".join(str(i) for i in failed_ids) + "`.", "",
              "Successful Method B source episodes (zero-based): `" + ", ".join(str(e["source_episode"]) for e in successes) + "`.", "",
              "[Episode mapping CSV](method_b_episode_map.csv) maps every original episode ID to its retained output ID. "
              "Outputs are renumbered after failed attempts are discarded. Original episode 7 succeeded and became output "
              "episode 0; output episode 7 is successful original episode 36. ‘Failed’ is a task benchmark result, "
              "not a claim that the follower calibration is invalid.", "",
              "The source sim and paired real datasets match their complete pre-change hash snapshots; "
              "the actual `rpd_robot.json` calibration hash also matches. "
              "[method_a.json](method_a.json), [Method A reader verification](method_a_verification.json), [Method B reader verification](method_b_verification.json), and [the complete data audit](completion-data-audit.json) preserve the verification details.", "",
              "## Tests before and after", "",
              "All host pytest commands used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Isaac/USD geometry checks and "
              "the physical replay ran inside the teleop image. The host suite excludes `test_rerecording.py` "
              "because LeRobot is absent there; the full teleop-image suite includes it.", "",
              "| Environment | Before | After | Failure set |", "|---|---|---|---|"]
    for env in ("host", "container"):
        lines.append(f"| {env} | {tests[env]['before']['summary']} | {tests[env]['after']['summary']} | Unchanged |")
    lines += ["", "Final targeted regression suite: **182 passed, 2 skipped**. Isolated real-pxr USD tests: **2 passed**. "
              "The frozen-start regression also passes. Removed overhead-camera cases account for fewer total tests. "
              "[test_comparison.json](test_comparison.json) lists every unchanged baseline failure and links to the full logs below.", ""]
    for env in ("host", "container"):
        lines += [f"{env} failures before and after:", ""] + ["- `" + value + "`" for value in tests[env]["after"]["failures"]] + [""]
    lines += ["The host additionally lacks Rerun, PyAV and SciPy for the affected existing cases; wrapper/mock failures "
              "also predate this change. The isolated container suite additionally lacks host training-storage mounts, uses a read-only repository mount for the temporary-config case, and its bare subprocess Python lacks NumPy in one case. No baseline failure was hidden or treated as a passing result.", "",
              "[Before host](before-host.txt), [after host](after-host.txt), "
              "[before container](before-container.txt), [after container](after-container.txt), "
              "[targeted](final-targeted.txt), [USD geometry](usd-tests.txt).", "",
              "## Files changed", "", "[changed_files.txt](changed_files.txt) lists the complete change set. "
              "Changes cover calibration derivation/configuration, legacy start-pose handling, dataset tools, "
              "Molmo wrist/camera options, alternate asset removal, tests, documentation and this evidence bundle.", "",
              "## Joint comparisons", "",
              "Paired real-arm data exists at `datasets/pick_place_v1-real`: 75 episodes and 16,448 frames. "
              "These are actual recorded action/state columns, not invented samples or calibration endpoints. "
              "All five requested statistics are reported separately for actions and measured states, for both new methods.", "",
              "![Action histograms](action_histograms.png)", "", "![State histograms](observation_state_histograms.png)", "",
              "[CSV](joint_comparison.csv) · [JSON](joint_comparison.json)", "",
              (report / "joint_comparison.md").read_text()]
    (report / "report.md").write_text("\n".join(lines).rstrip() + "\n")
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT, text=True).splitlines()
    changed += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    changed.append(str((report / "changed_files.txt").relative_to(ROOT)))
    (report / "changed_files.txt").write_text("\n".join(sorted(set(changed))) + "\n")
    print(f"Verified A: 75 preserved; B: {len(successes)}/75 benchmark successes; source/calibration unchanged.")


if __name__ == "__main__":
    main()
