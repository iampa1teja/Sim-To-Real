"""Read-only cube-face yaw and SO101 pan/roll review at each demo's first lift.

Requires NumPy and PyArrow, without Isaac Sim, Torch or a GPU. This review does
not reconstruct gripper forward kinematics, contact, release or task success.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "alignment_eval_utils", REPO / "source/sim_to_real_so101/utils/pick_place_eval.py"
)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)
SETUP_PATH = REPO / "source/sim_to_real_so101/assets/real_setup.json"


def square_angle(degrees):
    """Signed face-angle difference, accounting for a square's 90-degree symmetry."""
    return (degrees + 45.0) % 90.0 - 45.0


def cube_face_yaw(quaternion):
    """Yaw of the most horizontal cube-local axis in the recorded base frame.

    A cube may rest on any face. Euler yaw of local X is ill-conditioned when
    local X is vertical; selecting the largest XY projection avoids that case.
    Face comparisons use square_angle(), so choosing the other horizontal face
    has the same result for a level cube. Tilted cubes still need video review.
    """
    rotation = eval_utils.rotation_matrix(quaternion)
    axis = int(np.argmax(np.linalg.norm(rotation[:2], axis=0)))
    return float(np.degrees(np.arctan2(rotation[1, axis], rotation[0, axis])))


def review_episode(row, states, joint_mapping, neutral_roll_deg, lift_m):
    count = row["num_frames"]
    if (row.get("reference_frame") != "robot base link (base)"
            or row.get("units") != "m" or not isinstance(count, int) or count < 1):
        raise ValueError("Expected a nonempty metre-valued base-link sidecar")
    positions = np.asarray(row["position"], dtype=float)
    quaternions = np.asarray(row["orientation_wxyz"], dtype=float)
    states = np.asarray(states, dtype=float)
    if (positions.shape != (count, 3) or quaternions.shape != (count, 4)
            or states.shape != (count, 6) or not np.isfinite(positions).all()
            or not np.isfinite(quaternions).all() or not np.isfinite(states).all()):
        raise ValueError("Sidecar and state timelines must have finite, matching frames")
    if np.any(np.linalg.norm(quaternions, axis=1) == 0):
        raise ValueError("Zero cube quaternion")
    fps = float(row["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Sidecar FPS must be finite and positive")
    # Sidecars contain base-frame positions, not world height or contact forces.
    # This is a first base-Z rise candidate, not the simulator's grasp predicate.
    candidates = np.flatnonzero(positions[:, 2] - positions[0, 2] > lift_m)
    if not candidates.size:
        return None
    frame = int(candidates[0])
    initial = np.degrees(eval_utils.dataset_state_to_radians(states[0], joint_mapping))
    joints = np.degrees(eval_utils.dataset_state_to_radians(states[frame], joint_mapping))
    yaw = cube_face_yaw(quaternions[frame])
    initial_yaw = cube_face_yaw(quaternions[0])
    return {
        "frame": frame, "time_s": frame / fps, "cube_yaw_deg": yaw,
        "pan_deg": float(joints[0]), "roll_deg": float(joints[4]),
        "face_minus_pan_deg": square_angle(yaw - joints[0]),
        "start_face_minus_pan_deg": square_angle(initial_yaw - initial[0]),
        "roll_change_deg": float(joints[4] - initial[4]),
        "roll_from_neutral_deg": float(joints[4] - neutral_roll_deg),
    }


def review_dataset(root, lift_m=0.01):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    count = info["total_episodes"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("Dataset must contain at least one saved episode")
    expected_names = [name + ".pos" for name in eval_utils.SO101_JOINTS]
    if info["features"]["observation.state"].get("names") != expected_names:
        raise ValueError("Dataset observation.state joint names/order differ from the recorder")
    setup = json.loads(SETUP_PATH.read_text())
    joint_mapping = setup.get("sim_joint_mapping", {})
    neutral_roll = float(np.degrees(setup["robot"]["joint_positions"]["Wrist_Roll"]))
    frames = {}
    for path in sorted(root.glob("data/*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state"])
        for row in table.to_pylist():
            episode, frame = int(row["episode_index"]), int(row["frame_index"])
            key = (episode, frame)
            if key in frames:
                raise ValueError(f"Duplicate episode/frame: {key}")
            frames[key] = row["observation.state"]
    results = []
    expected_episodes = set(range(count))
    sidecars = sorted((root / "pick_place_meta").glob("episode_*.json"))
    seen = set()
    for path in sidecars:
        row = json.loads(path.read_text())
        episode = row["episode_index"]
        if episode in seen or path.name != f"episode_{episode:06d}.json":
            raise ValueError(f"Duplicate/misnamed sidecar: {path}")
        seen.add(episode)
        if float(row["fps"]) != float(info["fps"]):
            raise ValueError(f"Dataset/sidecar FPS mismatch: {path}")
        keys = [(episode, frame) for frame in range(row["num_frames"])]
        if any(key not in frames for key in keys):
            raise ValueError(f"Missing state frames for {path}")
        states = [frames.pop(key) for key in keys]
        results.append((episode, review_episode(row, states, joint_mapping, neutral_roll, lift_m)))
    if seen != expected_episodes or frames:
        raise ValueError("Dataset episodes, state frames and cube sidecars do not agree")
    return results, neutral_roll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--lift_m", type=float, default=0.01,
                        help="Base-Z rise above frame zero for the review frame (default 0.01 m)")
    args = parser.parse_args()
    if not np.isfinite(args.lift_m) or args.lift_m <= 0:
        parser.error("--lift_m must be finite and positive")
    try:
        results, neutral = review_dataset(args.dataset, args.lift_m)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(1, f"Alignment review failed: {exc}\n")
    print(f"Read-only review; first base-Z rise > {args.lift_m:g} m; "
          f"calibrated neutral roll {neutral:.2f} deg")
    print("episode frame time_s cube_yaw_deg pan_deg roll_deg face-minus-pan_deg "
          "start-face-minus-pan_deg roll-change_deg roll-from-neutral_deg")
    deltas = []
    for episode, row in results:
        if row is None:
            print(f"{episode:06d} REVIEW: no base-Z lift candidate")
            continue
        deltas.append(row["face_minus_pan_deg"])
        print(f"{episode:06d} {row['frame']} " + " ".join(f"{row[key]:.2f}" for key in (
            "time_s", "cube_yaw_deg", "pan_deg", "roll_deg", "face_minus_pan_deg",
            "start_face_minus_pan_deg", "roll_change_deg", "roll_from_neutral_deg")))
    if deltas:
        phase = np.deg2rad(np.asarray(deltas) * 4)
        vector = np.mean(np.exp(1j * phase))
        if abs(vector) < 1e-6:
            print("Face-minus-pan distribution has no stable 90-degree circular centre")
        else:
            centre = float(np.rad2deg(np.angle(vector)) / 4)
            spread = np.abs(square_angle(np.asarray(deltas) - centre))
            print(f"Face-minus-pan circular centre {centre:.2f} deg; "
                  f"median/max deviation {np.median(spread):.2f}/{np.max(spread):.2f} deg")
    print("Review rows and videos: pan/roll are joint angles, not gripper Euler yaw; "
          "this does not certify grasp alignment or success.")


if __name__ == "__main__":
    main()
