# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# core/calibration/calibration_stats.py  (ported from the workshop's so101_calibration_stats.py)
"""Per-joint motion-range statistics from a folder of known-good calibration files.

Reads every .json in arms/<arm>/calibration/samples/, computes the mean and std of (range_max - range_min) per
joint, writes arms/<arm>/calibration/stats.json (what check_calibration compares against) and draws a
box-and-whisker plot. Run it again after adding or removing sample files.

Usage:
    s2r calibration-stats
    s2r calibration-stats --exclude follower_arm_1
"""

import argparse
import json
from pathlib import Path

import numpy as np

from core.arms.joint_calibration import FOLLOWER_JOINTS
from core.arms.spec import load_arm_spec


# Motion range of every joint in every sample file: ({joint: [range, ...]}, [file, ...]).
def load_all(calibration_dir, exclude=()):
    files = sorted(p for p in Path(calibration_dir).glob("*.json") if p.stem not in set(exclude))
    if not files:
        raise SystemExit(f"No JSON files found in {calibration_dir} (after exclusions)")
    motion_ranges = {joint: [] for joint in FOLLOWER_JOINTS}
    print(f"Using {len(files)} calibration file(s):")
    for path in files:
        print(f"  {path.name}")
        data = json.loads(path.read_text())
        for joint in FOLLOWER_JOINTS:
            if joint in data:
                motion_ranges[joint].append(abs(data[joint]["range_max"] - data[joint]["range_min"]))
    return motion_ranges, files


def compute_stats(motion_ranges):
    stats = {}
    for joint, values in motion_ranges.items():
        array = np.array(values, dtype=float)
        stats[joint] = {"mean": round(float(array.mean()), 2), "std": round(float(array.std()), 2),
                        "min": int(array.min()), "max": int(array.max()), "n": len(array)}
    return stats


# Box plot of the ranges per joint, one coloured dot per calibration file.
def plot(motion_ranges, files, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([motion_ranges[j] for j in FOLLOWER_JOINTS], tick_labels=range(1, len(FOLLOWER_JOINTS) + 1),
               patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2),
               boxprops=dict(facecolor="lightgrey", alpha=0.6), whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    colors = plt.cm.tab10.colors
    for index in range(len(files)):
        for joint_index, joint in enumerate(FOLLOWER_JOINTS, start=1):
            ax.scatter(joint_index, motion_ranges[joint][index], color=colors[index % len(colors)],
                       edgecolors="black", linewidths=0.5, zorder=3, s=60)
    ax.set_title("Calibration: motion range per joint\n(abs(range_max − range_min), encoder counts)", fontsize=13)
    ax.set_ylabel("Motion range (encoder counts)")
    ax.set_xlabel("Joint index (" + ", ".join(f"{i}={j}" for i, j in enumerate(FOLLOWER_JOINTS, 1)) + ")")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.legend(handles=[patches.Patch(color=colors[i % len(colors)], label=f.stem) for i, f in enumerate(files)],
               title="Arm", loc="upper right", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Plot saved to {output}")


def main(argv=None):
    _, arm_dir = load_arm_spec()
    base = arm_dir / "calibration"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib-dir", default=str(base / "samples"), help="folder of sample calibration JSONs")
    parser.add_argument("--output-json", default=str(base / "stats.json"))
    parser.add_argument("--output-plot", default=str(base / "stats.png"))
    parser.add_argument("--exclude", nargs="*", default=[], metavar="NAME",
                        help="sample file stem(s) to leave out, e.g. --exclude follower_arm_1")
    parser.add_argument("--no-plot", action="store_true", help="only write the stats JSON")
    args = parser.parse_args(argv)

    motion_ranges, files = load_all(args.calib_dir, args.exclude)
    stats = compute_stats(motion_ranges)
    print("\nPer-joint motion range statistics (encoder counts):")
    print(f"  {'Joint':<18} {'Mean':>7} {'Std':>7} {'Min':>6} {'Max':>6} {'N':>4}")
    print("  " + "-" * 52)
    for joint, s in stats.items():
        print(f"  {joint:<18} {s['mean']:>7.1f} {s['std']:>7.1f} {s['min']:>6} {s['max']:>6} {s['n']:>4}")

    Path(args.output_json).write_text(json.dumps(stats, indent=4) + "\n")
    print(f"Stats saved to {args.output_json}")
    if not args.no_plot:
        plot(motion_ranges, files, Path(args.output_plot))


if __name__ == "__main__":
    main()
