"""Versioned operator references, clearance math and independent pose verdicts."""
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
import tomllib

import numpy as np

from .joint_calibration import FOLLOWER_JOINTS, file_sha256

POSE_SET_PATH = Path(__file__).resolve().parents[3] / "configs/calibration/so101_reference_poses.toml"
BODY_JOINTS = FOLLOWER_JOINTS[:-1]


@dataclass(frozen=True)
class PoseThresholds:
    """The operator's lock rule, recorded with every verdict."""

    pass_deg: float = 1.0
    warn_deg: float = 2.0
    spread_deg: float = 1.0
    min_poses: int = 5

    def __post_init__(self):
        if (not all(np.isfinite(v) and v >= 0 for v in asdict(self).values())
                or self.warn_deg < self.pass_deg or isinstance(self.min_poses, bool)
                or not isinstance(self.min_poses, int) or self.min_poses < 1):
            raise ValueError("Require finite nonnegative pose thresholds, WARN >= PASS, and a positive integer min_poses")


def load_pose_set(joint_order, runtime_limits, path=POSE_SET_PATH):
    """Read the committed set and enforce coverage and distance from limits."""
    path = Path(path)
    data = tomllib.loads(path.read_text())
    if data.get("version") != 1 or data.get("joint_order") != list(joint_order):
        raise ValueError("Unsupported reference pose version or joint_order")
    bounds = np.asarray(runtime_limits, dtype=float)
    if bounds.shape != (len(joint_order), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Expected finite increasing runtime limits")
    poses = [(row["name"], np.asarray(row["degrees"], dtype=float)) for row in data["poses"]]
    if not 7 <= len(poses) <= 9 or len({name for name, _ in poses}) != len(poses):
        raise ValueError("Require all-zero plus 6–8 uniquely named reference poses")
    if any(p.shape != (len(joint_order),) or not np.isfinite(p).all() for _, p in poses):
        raise ValueError("Reference poses must contain one finite angle per joint")
    angles = np.asarray([p for _, p in poses])
    if poses[0][0] != "all-zero" or np.any(angles[0] != 0):
        raise ValueError("All-zero must be first")
    if np.any(angles - bounds[:, 0] < 5) or np.any(bounds[:, 1] - angles < 5):
        raise ValueError("Reference poses must stay at least 5 degrees from every runtime limit")
    coverage = {}
    for i, name in enumerate(joint_order):
        if name == "gripper":
            continue
        coverage[name] = float(np.ptp(angles[:, i]) / np.ptp(bounds[i]))
        if coverage[name] < .6:
            raise ValueError(f"{name}: reference pose coverage {coverage[name]:.1%} is below 60%")
    safety = data["preview"]
    if safety["margin_m"] < .02:
        raise ValueError("Preview clearance margin must be at least 2 cm")
    return poses, {"path": str(path.resolve()), "sha256": file_sha256(path), "version": data["version"],
                   "coverage_fraction": coverage, "preview": safety}


def summarize_poses(measurements, thresholds=None, joint_order=FOLLOWER_JOINTS):
    """Median(real − target) and max–min spread of distinct accepted joint targets."""
    thresholds = thresholds or PoseThresholds()
    joints = {}
    for i, name in enumerate(joint_order):
        offsets, targets = [], set()
        for row in measurements:
            accepted = row.get("joint_accepted", {}).get(name, row.get("accepted")) if name == "gripper" else row.get("accepted")
            if not accepted or row.get("at_limit", {}).get(name, False):
                continue
            target = float(row["target_deg"][i])
            offset = float(row["real_minus_target_deg"][i])
            if not np.isfinite([target, offset]).all():
                raise ValueError("Pose evidence contains nonfinite angles")
            key = round(target, 6)
            if key not in targets:
                offsets.append(offset)
                targets.add(key)
        median = float(np.median(offsets)) if offsets else None
        spread = float(np.ptp(offsets)) if offsets else None
        if len(offsets) < thresholds.min_poses:
            verdict, reason = "FAIL", "INSUFFICIENT"
        elif spread > thresholds.spread_deg:
            verdict, reason = "FAIL", "SPREAD"
        elif abs(median) <= thresholds.pass_deg:
            verdict, reason = "PASS", "WITHIN_PASS"
        elif abs(median) <= thresholds.warn_deg:
            verdict, reason = "WARN", "WITHIN_WARN"
        else:
            verdict, reason = "FAIL", "OFFSET"
        joints[name] = dict(median_offset_deg=median, spread_deg=spread, accepted_poses=len(offsets),
                            verdict=verdict, reason=reason, offsets_deg=offsets)
    body = {name: joints[name] for name in BODY_JOINTS}
    return dict(joints=body, gripper=joints.get("gripper"), thresholds=asdict(thresholds),
                decision="LOCKABLE" if all(j["verdict"] in {"PASS", "WARN"} for j in body.values()) else "NOT_LOCKABLE",
                offset_sign="real minus target", spread_definition="max minus min; no angular wrapping",
                gripper_blocks=False)


def read_pose_log(path, thresholds=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("mode") != "operator_poses" or rows[0].get("type") != "provenance":
        raise ValueError("Expected an operator pose log with a provenance header")
    header = rows[0]
    if header.get("joint_order") != list(FOLLOWER_JOINTS) or not header.get("pose_set", {}).get("sha256"):
        raise ValueError("Pose evidence needs the joint order and versioned pose-set hash")
    thresholds = thresholds or PoseThresholds(**header.get("pose_thresholds", {}))
    measurements = [row for row in rows[1:] if row.get("type") == "pose"]
    return header, measurements, summarize_poses(measurements, thresholds)


def box_vertices(low, high):
    """Eight corners of a conservative local collision bounding box."""
    return np.array(list(product(*zip(low, high))), dtype=float)


def quaternion_matrix(xyzw):
    """PhysX quaternions use xyzw; return a column-vector rotation matrix."""
    x, y, z, w = np.asarray(xyzw, dtype=float) / np.linalg.norm(xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_separation(a, axes_a, b, axes_b):
    """Conservative lower bound using all OBB separating axes (metres).

    Bounding boxes enclose collision meshes. A separating projection gap >= the
    margin proves clearance; an inconclusive/overlapping bound rejects the pose.
    """
    axes = [*axes_a.T, *axes_b.T, *(np.cross(x, y) for x in axes_a.T for y in axes_b.T)]
    gaps = []
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            axis = axis / norm
            pa, pb = a @ axis, b @ axis
            gaps.append(max(float(pa.min() - pb.max()), float(pb.min() - pa.max())))
    return max(gaps)


def check_clearance(boxes, adjacent, table_point, table_normal, *, margin_m=.02, mounted=(), hulls=None):
    """Check table and nonadjacent-link geometry placed by PhysX link poses."""
    if margin_m < .02:
        raise ValueError("Clearance margin must be at least 2 cm")
    normal = np.asarray(table_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    table, pairs, reasons, exempt_table = {}, {}, [], {}
    names = list(boxes)
    for name, (vertices, axes) in boxes.items():
        clearance = float(np.min(((hulls[name] if hulls else vertices) - table_point) @ normal))
        if name in mounted:
            exempt_table[name] = clearance
        else:
            table[name] = clearance
            if clearance < margin_m:
                reasons.append(f"table: {name} clearance {clearance:.4f} m < {margin_m:.3f} m")
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if frozenset((name, other)) in adjacent:
                continue
            gap = box_separation(*boxes[name], *boxes[other])
            if gap < margin_m and hulls is not None:
                gap = convex_clearance(hulls[name], hulls[other])
            pairs[f"{name}/{other}"] = gap
            if gap < margin_m:
                reasons.append(f"self: {name}/{other} clearance bound {gap:.4f} m < {margin_m:.3f} m")
    return dict(status="REJECT" if reasons else "OK", reasons=reasons, margin_m=margin_m,
                table_clearance_m=table, exempt_table_clearance_m=exempt_table, self_clearance_lower_bound_m=pairs,
                method="collision mesh enclosing OBBs transformed by PhysX get_link_transforms; SAT separation lower bound",
                exempt_adjacent_pairs=sorted("/".join(sorted(pair)) for pair in adjacent), mounted_table_exemptions=list(mounted))


def convex_clearance(a, b, iterations=80):
    """A separating-plane lower bound for two convex collision hulls (GJK).

    Each returned distance is a projection gap and therefore never overstates
    clearance. Closest-simplex iteration improves the separating direction.
    """
    from itertools import combinations

    simplex = [a[0] - b[0]]
    closest = simplex[0]
    bound = -float("inf")
    for _ in range(iterations):
        norm = np.linalg.norm(closest)
        if norm < 1e-10:
            return 0.
        direction = closest / norm
        point = a[np.argmin(a @ direction)] - b[np.argmax(b @ direction)]
        gap = float(point @ direction)
        bound = max(bound, gap)
        if norm - bound < 1e-8 or any(np.linalg.norm(point - p) < 1e-10 for p in simplex):
            return max(0., bound)
        simplex.append(point)
        best = None
        for size in range(1, min(4, len(simplex)) + 1):
            for indices in combinations(range(len(simplex)), size):
                vertices = np.asarray([simplex[i] for i in indices])
                matrix = np.block([[vertices @ vertices.T, np.ones((size, 1))],
                                   [np.ones((1, size)), np.zeros((1, 1))]])
                weights = np.linalg.lstsq(matrix, np.r_[np.zeros(size), 1.], rcond=None)[0][:size]
                if np.any(weights < -1e-9):
                    continue
                candidate = weights @ vertices
                squared = candidate @ candidate
                if best is None or squared < best[0]:
                    best = squared, candidate, [simplex[i] for i, weight in zip(indices, weights) if weight > 1e-9]
        _, closest, simplex = best
    return max(0., bound)
