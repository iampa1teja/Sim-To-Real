"""Produce Method A and compare old/new simulation joints with paired real recordings."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from sim_to_real_so101.utils.calibration_dataset import relabel_dataset, snapshot, units_to_degrees
from sim_to_real_so101.utils.joint_calibration import FOLLOWER_JOINTS, follower_mapping_from_calibration


def joint_columns(root):
    values = {name: [] for name in ("action", "observation.state")}
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=list(values))
        for name in values:
            values[name].extend(table[name].to_pylist())
    return {name: np.asarray(value, dtype=np.float64).reshape(-1, 6) for name, value in values.items()}


def compare(groups, report):
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for feature in ("action", "observation.state"):
        figure, axes = plt.subplots(6, 2, figsize=(13, 17), constrained_layout=True)
        bins = {}
        for unit in ("LeRobot units", "degrees"):
            population = np.concatenate([data[feature] if unit == "LeRobot units"
                                         else units_to_degrees(data[feature], mapping)
                                         for _, data, mapping in groups], axis=0)
            for index in range(6):
                low, high = population[:, index].min(), population[:, index].max()
                bins[index, unit] = np.linspace(low, high if high > low else low + 1, 51)
        for label, data, mapping in groups:
            raw = data[feature]
            degrees = units_to_degrees(raw, mapping)
            for unit, values in (("LeRobot units", raw), ("degrees", degrees)):
                for index, joint in enumerate(FOLLOWER_JOINTS):
                    array = values[:, index]
                    row = dict(dataset=label, feature=feature, joint=joint, units=unit, count=len(array))
                    row.update(dict(zip(("mean", "min", "max", "p5", "p95"),
                                        [float(array.mean()), float(array.min()), float(array.max()),
                                         float(np.percentile(array, 5)), float(np.percentile(array, 95))]
                                        if len(array) else [None] * 5)))
                    rows.append(row)
                    if len(array):
                        axes[index, int(unit == "degrees")].hist(array, bins=bins[index, unit], density=True,
                                                                 histtype="step", label=label, linewidth=1.3)
        for i, joint in enumerate(FOLLOWER_JOINTS):
            for j, unit in enumerate(("LeRobot units", "degrees")):
                axes[i, j].set_title(f"{joint}: {unit}")
                axes[i, j].set_ylabel("Density")
                axes[i, j].legend(fontsize=8)
        figure.suptitle(f"{feature}: sim USD angles; real encoder angles (gripper frame differs)")
        figure.savefig(report / (feature.replace(".", "_") + "_histograms.png"), dpi=150)
        plt.close(figure)
    with (report / "joint_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (report / "joint_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["Each cell is **mean / min / max / p5 / p95**. All frames are included.", "",
             "Simulation degrees use the USD joint frame. Real degrees use encoder zero 2048;",
             "the real gripper's USD-equivalent angle subtracts 9.209 degrees from its encoder angle.", ""]
    labels = [group[0] for group in groups]
    for feature in ("action", "observation.state"):
        for unit in ("LeRobot units", "degrees"):
            lines += [f"### {feature}: {unit}", "", "| Joint | " + " | ".join(labels) + " |",
                      "|---|" + "---|" * len(labels)]
            for joint in FOLLOWER_JOINTS:
                cells = []
                for label in labels:
                    row = next(r for r in rows if (r["feature"], r["units"], r["joint"], r["dataset"]) ==
                               (feature, unit, joint, label))
                    cells.append(" / ".join(f"{row[k]:.3f}" for k in ("mean", "min", "max", "p5", "p95"))
                                 if row["count"] else "No retained frames")
                lines.append("| " + joint + " | " + " | ".join(cells) + " |")
            lines.append("")
    (report / "joint_comparison.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/pick_place_v1")
    parser.add_argument("--method-a", type=Path, default=ROOT / "datasets/pick_place_v1_calibrated_A")
    parser.add_argument("--method-b", type=Path)
    parser.add_argument("--real", type=Path, default=ROOT / "datasets/pick_place_v1-real")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--relabel", action="store_true")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    old = audit["old_mapping"]
    new = follower_mapping_from_calibration(audit["calibration"], gripper_closed_degrees=audit["usd_limits"]["Jaw"][0])
    if snapshot(args.source) != audit["source_hashes"]:
        raise RuntimeError("Source no longer matches the pre-change hash snapshot")
    if args.relabel:
        manifest = relabel_dataset(args.source, args.method_a, old, new)
        args.report.mkdir(parents=True, exist_ok=True)
        (args.report / "method_a.json").write_text(json.dumps(manifest, indent=2) + "\n")
    groups = [("Old sim", joint_columns(args.source), old), ("Method A", joint_columns(args.method_a), new)]
    if args.method_b:
        groups.append(("Method B", joint_columns(args.method_b), new))
    if args.real.is_dir():
        if snapshot(args.real) != audit["real_hashes"]:
            raise RuntimeError("Real dataset no longer matches its pre-change snapshot")
        groups.append(("Real", joint_columns(args.real), follower_mapping_from_calibration(audit["calibration"])))
    compare(groups, args.report)
    if snapshot(args.source) != audit["source_hashes"]:
        raise RuntimeError("Source changed during comparison")


if __name__ == "__main__":
    main()
