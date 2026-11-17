"""Arm-file kinematics with full-vector FK, geometric Jacobians and bounded IK."""

from __future__ import annotations

import json as _json
import math as _math
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import numpy as _np
from scipy.optimize import least_squares as _least_squares
from scipy.spatial.transform import Rotation as _Rotation

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.generate.errors import ConfigError as _ConfigError, KinematicsError as _KinematicsError, SchemaError as _SchemaError
from core.tasks.base import Pose as _Pose


def _readonly(value):
    result = _np.array(value, dtype=_np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _numbers(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(type(v) in (int, float) and _math.isfinite(v) for v in value))


def _basis(axis):
    return _numbers(axis, 3) and tuple(axis) in ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not _math.isfinite(value):
        raise ValueError("Non-finite numeric value")


def load_kinematics_json(path) -> dict:
    """Read and validate schema 1 without reordering or changing its values."""
    try:
        data = _json.loads(_Path(path).read_text())
        _finite_tree(data)
        if not isinstance(data, dict) or not {"schema_version", "arm", "base_link", "source", "joints"} <= data.keys():
            raise ValueError("Missing kinematics fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Expected schema_version 1")
        if not all(isinstance(data[k], str) for k in ("arm", "base_link")):
            raise ValueError("Arm and base_link must be strings")
        source = data["source"]
        if not isinstance(source, dict) or not all(isinstance(source[k], str) for k in ("usd", "usd_sha256", "git_commit")):
            raise ValueError("Invalid source")
        joints = data["joints"]
        if not isinstance(joints, list) or not joints:
            raise ValueError("joints must be a non-empty list")
        names = set()
        for joint in joints:
            if not isinstance(joint, dict) or not {"name", "type", "parent", "child", "axis", "parent_frame", "child_frame", "limits_rad"} <= joint.keys():
                raise ValueError("Missing joint fields")
            if not all(isinstance(joint[k], str) for k in ("name", "parent", "child")):
                raise ValueError("Joint and link names must be strings")
            if joint["name"] in names or joint["type"] != "revolute" or not _basis(joint["axis"]):
                raise ValueError("Invalid joint name, type or axis")
            names.add(joint["name"])
            for key in ("parent_frame", "child_frame"):
                frame = joint[key]
                if not isinstance(frame, dict) or not _numbers(frame["position"], 3) or not _numbers(frame["quat_wxyz"], 4):
                    raise ValueError("Invalid joint frame")
            limits = joint["limits_rad"]
            if not _numbers(limits, 2) or limits[0] >= limits[1]:
                raise ValueError("Invalid joint limits")
        return data
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise _SchemaError(f"{path}: {exc}") from exc


def _frame(position, quat_wxyz):
    position = _np.asarray(position, dtype=_np.float64)
    quat = _np.asarray(quat_wxyz, dtype=_np.float64)
    if position.shape != (3,) or quat.shape != (4,) or not _np.all(_np.isfinite(position)) or not _np.all(_np.isfinite(quat)):
        raise ValueError("Expected finite position (3,) and quaternion (4,)")
    norm = float(_np.linalg.norm(quat))
    if norm == 0:
        raise ValueError("Quaternion cannot be zero")
    quat = quat / norm
    result = _np.eye(4)
    result[:3, :3] = _Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def _pose_matrix(pose):
    x, y, z, w = pose.quat_xyzw
    return _frame(pose.position, (w, x, y, z))


def _to_pose(matrix):
    quat = _Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return _Pose(tuple(float(v) for v in matrix[:3, 3]), tuple(float(v) for v in quat))


def _skew(v):
    x, y, z = v
    return _np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=_np.float64)


@_dataclass(frozen=True, eq=False)
class JointDef:
    name: str
    parent: str
    child: str
    axis: _np.ndarray
    parent_frame: _np.ndarray
    child_frame: _np.ndarray
    lower: float
    upper: float

    def __post_init__(self):
        for key in ("axis", "parent_frame", "child_frame"):
            object.__setattr__(self, key, _readonly(getattr(self, key)))


@_dataclass(frozen=True)
class IKWeights:
    position: float = 1.0
    axis: float = 0.3
    roll: float = 0.1

    def __post_init__(self):
        if any(not _math.isfinite(value) or value < 0 for value in (self.position, self.axis, self.roll)):
            raise ValueError("IK weights must be finite and nonnegative")


@_dataclass(frozen=True, eq=False)
class IKResult:
    q: _np.ndarray
    success: bool
    position_error_m: float
    orientation_error_rad: float
    axis_error_rad: float
    roll_error_rad: float
    iterations: int

    def __post_init__(self):
        q = _readonly(self.q)
        if q.ndim != 1:
            raise ValueError("IK result q must be a joint vector")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "success", bool(self.success))
        for key in ("position_error_m", "orientation_error_rad", "axis_error_rad", "roll_error_rad"):
            object.__setattr__(self, key, float(getattr(self, key)))
        object.__setattr__(self, "iterations", int(self.iterations))


class KinematicChain:
    @classmethod
    def from_arm(cls, name=None):
        spec, folder = _load_arm_spec(name)
        try:
            kin = spec["kinematics"]
            arguments = dict(q_names=spec["joints"]["names"], chain_joints=spec["expert"]["arm_joints"],
                             ee_link=kin.get("ee_link", spec["links"]["ee"]), tcp_position_m=kin["tcp_position_m"],
                             tcp_quat_wxyz=kin.get("tcp_quat_wxyz", (1, 0, 0, 0)),
                             approach_axis=kin["approach_axis"], closing_axis=kin["closing_axis"])
        except (KeyError, TypeError) as exc:
            raise _ConfigError(f"{folder}/arm.toml: incomplete kinematics configuration") from exc
        return cls.from_dict(load_kinematics_json(folder / kin.get("file", "kinematics.json")), **arguments)

    @classmethod
    def from_dict(cls, data, *, q_names, chain_joints, ee_link, tcp_position_m, tcp_quat_wxyz,
                  approach_axis, closing_axis):
        try:
            self = cls()
            self.q_names, self.chain_joints = tuple(q_names), tuple(chain_joints)
            self._base, self._ee = data["base_link"], ee_link
            rows = list(data["joints"])
            lookup = {row["name"]: row for row in rows}
            if len(lookup) != len(rows) or len(set(self.q_names)) != len(self.q_names):
                raise ValueError("Duplicate joint name")
            if any(j not in lookup or j not in self.q_names for j in self.chain_joints):
                raise ValueError("Chain joint missing from data or q_names")
            children = [row["child"] for row in rows]
            if len(set(children)) != len(children) or self._base in children:
                raise ValueError("A link has multiple parents or the base has a parent")
            known, ordered = {self._base}, []
            while rows:
                progressed = False
                for row in rows[:]:
                    if row["parent"] not in known:
                        continue
                    if not _basis(row["axis"]) or row["type"] != "revolute":
                        raise ValueError("Invalid revolute joint axis")
                    start, end = row["parent_frame"], row["child_frame"]
                    limits = row["limits_rad"]
                    if not _numbers(limits, 2) or limits[0] >= limits[1]:
                        raise ValueError("Invalid joint limits")
                    joint = JointDef(row["name"], row["parent"], row["child"], _np.asarray(row["axis"]),
                                     _frame(start["position"], start["quat_wxyz"]),
                                     _frame(end["position"], end["quat_wxyz"]), float(limits[0]), float(limits[1]))
                    ordered.append(joint)
                    known.add(joint.child)
                    rows.remove(row)
                    progressed = True
                if not progressed:
                    raise ValueError("Cycle or joint disconnected from base")
            if ee_link not in known:
                raise ValueError("Unknown ee_link")
            self._joints = tuple(ordered)
            self._indices = {name: i for i, name in enumerate(self.q_names)}
            self.lower = _readonly([lookup[name]["limits_rad"][0] if name in lookup else -_np.inf for name in self.q_names])
            self.upper = _readonly([lookup[name]["limits_rad"][1] if name in lookup else _np.inf for name in self.q_names])
            self._tcp = _frame(tcp_position_m, tcp_quat_wxyz)
            self._approach, self._closing = _readonly(approach_axis), _readonly(closing_axis)
            if (self._approach.shape != (3,) or self._closing.shape != (3,)
                    or not _np.all(_np.isfinite(self._approach)) or not _np.all(_np.isfinite(self._closing))
                    or abs(_np.linalg.norm(self._approach) - 1) > 1e-6
                    or abs(_np.linalg.norm(self._closing) - 1) > 1e-6 or abs(self._approach @ self._closing) > 1e-6):
                raise ValueError("Approach and closing axes must be orthonormal")
            ancestors = set()
            by_child = {joint.child: joint for joint in ordered}
            link = ee_link
            while link != self._base:
                joint = by_child[link]
                ancestors.add(joint.name)
                link = joint.parent
            self._moving = [(self._indices[j.name], j.name) for j in ordered if j.name in ancestors and j.name in self.chain_joints]
            self._evaluation = [(joint, self._indices.get(joint.name), int(_np.argmax(joint.axis)),
                                 _np.linalg.inv(joint.child_frame)) for joint in ordered]
            return self
        except (ValueError, TypeError, KeyError, IndexError, _np.linalg.LinAlgError) as exc:
            raise _KinematicsError(f"Cannot construct kinematic chain: {exc}") from exc

    def _q(self, q):
        q = _np.asarray(q, dtype=_np.float64)
        if q.shape != (len(self.q_names),) or not _np.all(_np.isfinite(q)):
            raise ValueError("Expected finite full-length joint vector")
        return q

    def _evaluate(self, q):
        poses = {self._base: _np.eye(4)}
        origins, axes = {}, {}
        for joint, index, axis_index, end in self._evaluation:
            start = poses[joint.parent] @ joint.parent_frame
            origins[joint.name], axes[joint.name] = start[:3, 3], start[:3, :3] @ joint.axis
            angle = q[index] if index is not None else 0.0
            c, s = _math.cos(angle), _math.sin(angle)
            rotate = _np.eye(4)
            a, b = (axis_index + 1) % 3, (axis_index + 2) % 3
            rotate[a, a] = rotate[b, b] = c
            rotate[a, b], rotate[b, a] = -s, s
            poses[joint.child] = start @ rotate @ end
        return poses, origins, axes

    def link_poses(self, q) -> dict[str, _np.ndarray]:
        return self._evaluate(self._q(q))[0]

    def _chosen(self, poses, frame):
        if frame == "ee":
            return poses[self._ee]
        if frame == "tcp":
            return poses[self._ee] @ self._tcp
        raise ValueError("frame must be tcp or ee")

    def fk(self, q, frame="tcp") -> _np.ndarray:
        return self._chosen(self.link_poses(q), frame)

    def fk_pose(self, q, frame="tcp") -> _Pose:
        return _to_pose(self.fk(q, frame))

    def tcp_from_ee(self, ee: _Pose) -> _Pose:
        return _to_pose(_pose_matrix(ee) @ self._tcp)

    def _jacobian(self, origin, origins, axes):
        jac = _np.zeros((6, len(self.q_names)))
        for index, name in self._moving:
            jac[:3, index] = _np.cross(axes[name], origin - origins[name])
            jac[3:, index] = axes[name]
        return jac

    def jacobian(self, q, frame="tcp") -> _np.ndarray:
        poses, origins, axes = self._evaluate(self._q(q))
        return self._jacobian(self._chosen(poses, frame)[:3, 3], origins, axes)

    def ik(self, target: _Pose, seed_q, *, weights=None, lower=None, upper=None, max_iters=200,
           tol_pos_m=5e-4, tol_rot_rad=0.02) -> IKResult:
        weights = weights or IKWeights()
        seed = self._q(seed_q).copy()
        target_matrix = _pose_matrix(target)
        if any(value is not None and _np.shape(value) != seed.shape for value in (lower, upper)):
            raise ValueError("IK bounds must be full-length joint vectors")
        lo = self.lower.copy() if lower is None else _np.maximum(_np.asarray(lower, dtype=float), self.lower)
        hi = self.upper.copy() if upper is None else _np.minimum(_np.asarray(upper, dtype=float), self.upper)
        if lo.shape != seed.shape or hi.shape != seed.shape or _np.any(_np.isnan(lo)) or _np.any(_np.isnan(hi)) or _np.any(lo > hi):
            raise ValueError("Invalid or empty IK bounds")
        indices = _np.array([self._indices[name] for name in self.chain_joints], dtype=int)
        free = indices[lo[indices] < hi[indices]]
        q = seed.copy()
        q[indices] = _np.clip(q[indices], lo[indices], hi[indices])
        at = target_matrix[:3, :3] @ self._approach
        ct = target_matrix[:3, :3] @ self._closing

        def evaluate(values, jacobian=False):
            trial = q.copy()
            trial[free] = values
            poses, origins, axes = self._evaluate(trial)
            transform = self._chosen(poses, "tcp")
            a, c = transform[:3, :3] @ self._approach, transform[:3, :3] @ self._closing
            cp = c - float(c @ at) * at
            s, k = float(at @ _np.cross(ct, cp)), float(ct @ cp)
            roll = _math.atan2(s, k)
            residual = _np.r_[weights.position * (transform[:3, 3] - target_matrix[:3, 3]),
                              weights.axis * _np.cross(a, at), weights.roll * roll]
            if not jacobian:
                return residual
            geometric = self._jacobian(transform[:3, 3], origins, axes)[:, free]
            dc = -_skew(c) @ geometric[3:]
            ds = at @ _skew(ct) @ dc
            dk = ct @ dc
            denom = s*s + k*k
            roll_jac = (k*ds - s*dk) / denom if denom > 1e-30 else _np.zeros(len(free))
            jac = _np.vstack((weights.position * geometric[:3],
                              weights.axis * _skew(at) @ _skew(a) @ geometric[3:], weights.roll * roll_jac))
            return residual, jac

        current = q[free].copy()
        clipped_seed = current.copy()
        residual = evaluate(current)
        best, best_cost = current.copy(), float(residual @ residual)
        iterations = 0
        for _ in range(max_iters if len(free) else 0):
            residual, jac = evaluate(current, True)
            iterations += 1
            if float(residual @ residual) < 1e-20:
                break
            step = -jac.T @ _np.linalg.solve(jac @ jac.T + 1e-6 * _np.eye(7), residual)
            scale = float(_np.max(_np.abs(step)))
            if scale > 0.3:
                step *= 0.3 / scale
            candidate = _np.clip(current + step, lo[free], hi[free])
            residual = evaluate(candidate)
            cost = float(residual @ residual)
            if cost < best_cost:
                best, best_cost = candidate.copy(), cost
            if float(_np.linalg.norm(candidate - current)) < 1e-13:
                break
            current = candidate
        if best_cost > 1e-16 and len(free):
            for initial in (best.copy(), clipped_seed):
                result = _least_squares(evaluate, initial, jac=lambda v: evaluate(v, True)[1],
                                        bounds=(lo[free], hi[free]), method="trf", xtol=1e-15, ftol=1e-15,
                                        gtol=1e-15, max_nfev=max_iters)
                iterations += int(result.nfev)
                cost = float(result.fun @ result.fun)
                if cost < best_cost:
                    best, best_cost = result.x.copy(), cost
        q[free] = _np.clip(best, lo[free], hi[free])
        transform = self.fk(q)
        rotation = transform[:3, :3]
        a, c = rotation @ self._approach, rotation @ self._closing
        cp = c - float(c @ at) * at
        position_error = float(_np.linalg.norm(transform[:3, 3] - target_matrix[:3, 3]))
        orientation_error = _math.acos(float(_np.clip((_np.trace(rotation.T @ target_matrix[:3, :3]) - 1) / 2, -1, 1)))
        axis_error = _math.acos(float(_np.clip(a @ at, -1, 1)))
        roll_error = abs(_math.atan2(float(at @ _np.cross(ct, cp)), float(ct @ cp)))
        success = (position_error <= tol_pos_m and (weights.axis == 0 or axis_error <= tol_rot_rad)
                   and (weights.roll == 0 or roll_error <= tol_rot_rad))
        return IKResult(q, success, position_error, orientation_error, axis_error, roll_error, iterations)
