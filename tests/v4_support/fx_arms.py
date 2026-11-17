"""Synthetic arm data for CPU tests: kinematics.json dicts, arm.toml tables, setup blocks and a reference FK.

Everything here is plain data plus numpy. Joint, link and LeRobot names deliberately avoid every
SO-101 literal (contracts §12), so code that hard-codes SO-101 fails on these arms.

* ``fx_arm6``: 6 revolute arm joints + 1 gripper joint; one joint has rotated joint frames and one
  a non-zero ``child_frame`` position, so frame handling is exercised.
* ``fx_arm5``: SO-101-like 5 arm joints + jaw (pan, lift, elbow, flex, roll), upright at q = 0.

WP2 owns the committed fixtures ``tests/fixtures/arms/fx_arm{5,6}/``; it may generate them from
``write_fx_arm`` or hand-write them, and is free to choose different numbers.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .toml_writer import dump_toml

_I = [1.0, 0.0, 0.0, 0.0]
_RX90 = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]


def _joint(name, parent, child, axis, pos0, limits, *, rot0=_I, pos1=(0.0, 0.0, 0.0), rot1=_I):
    return {"name": name, "type": "revolute", "parent": parent, "child": child, "axis": list(axis),
            "parent_frame": {"position": list(pos0), "quat_wxyz": list(rot0)},
            "child_frame": {"position": list(pos1), "quat_wxyz": list(rot1)},
            "limits_rad": list(limits)}


_KINEMATICS = {
    "fx_arm6": {
        "schema_version": 1, "arm": "fx_arm6", "base_link": "fx6_base",
        "source": {"usd": "usd/fx_arm6.usd", "usd_sha256": "0" * 64, "git_commit": "unknown"},
        "joints": [
            _joint("fx6_j1", "fx6_base", "fx6_l1", (0, 0, 1), (0.0, 0.0, 0.05), (-2.8, 2.8)),
            _joint("fx6_j2", "fx6_l1", "fx6_l2", (0, 0, 1), (0.0, 0.0, 0.04), (-1.8, 1.8), rot0=_RX90, rot1=_RX90),
            _joint("fx6_j3", "fx6_l2", "fx6_l3", (0, 1, 0), (0.0, 0.0, 0.12), (-2.4, 2.4)),
            _joint("fx6_j4", "fx6_l3", "fx6_l4", (0, 0, 1), (0.0, 0.0, 0.06), (-2.8, 2.8)),
            _joint("fx6_j5", "fx6_l4", "fx6_l5", (0, 1, 0), (0.0, 0.0, 0.07), (-2.0, 2.0)),
            _joint("fx6_j6", "fx6_l5", "fx6_hand", (0, 0, 1), (0.0, 0.0, 0.04), (-2.8, 2.8), pos1=(0.0, 0.0, -0.01)),
            _joint("fx6_grip", "fx6_hand", "fx6_finger", (1, 0, 0), (0.01, 0.0, 0.03), (0.0, 1.0)),
        ],
    },
    "fx_arm5": {
        "schema_version": 1, "arm": "fx_arm5", "base_link": "fx5_base",
        "source": {"usd": "usd/fx_arm5.usd", "usd_sha256": "0" * 64, "git_commit": "unknown"},
        "joints": [
            _joint("fx5_pan", "fx5_base", "fx5_shoulder", (0, 0, 1), (0.0, 0.0, 0.055), (-2.0, 2.0)),
            _joint("fx5_lift", "fx5_shoulder", "fx5_upper", (0, 1, 0), (0.0, 0.0, 0.035), (-1.8, 1.8)),
            _joint("fx5_elbow", "fx5_upper", "fx5_fore", (0, 1, 0), (0.0, 0.0, 0.13), (-1.7, 2.6)),
            _joint("fx5_flex", "fx5_fore", "fx5_wrist", (0, 1, 0), (0.0, 0.0, 0.14), (-1.8, 1.8)),
            _joint("fx5_roll", "fx5_wrist", "fx5_hand", (0, 0, 1), (0.0, 0.0, 0.04), (-3.0, 3.0)),
            _joint("fx5_grip", "fx5_hand", "fx5_jaw", (1, 0, 0), (0.0, 0.015, 0.02), (0.0, 1.0)),
        ],
    },
}

#: Per-arm names and arm.toml values (contracts §5.1 keys).
_ARMS = {
    "fx_arm6": {
        "lerobot": ["j1", "j2", "j3", "j4", "j5", "j6", "grip"],
        "ee": "fx6_hand", "moving_jaw": "fx6_finger", "gripper_joint": "fx6_grip",
        "home": [0.0, 0.4, 0.9, 0.0, 1.2, 0.0, 0.5],
        "tcp_position_m": [0.0, 0.0, 0.08],
    },
    "fx_arm5": {
        "lerobot": ["pan", "lift", "elbow", "flex", "roll", "grip"],
        "ee": "fx5_hand", "moving_jaw": "fx5_jaw", "gripper_joint": "fx5_grip",
        "home": [0.0, -0.3, 1.6, 1.2, 0.0, 0.5],
        "tcp_position_m": [0.0, 0.0, 0.09],
    },
}

FX_ARMS = tuple(_ARMS)
#: Gripper table shared by both fx arms: angle (rad, strictly increasing) -> gap (m, strictly increasing).
FX_OPENING_TABLE = [[0.0, 0.0], [0.25, 0.012], [0.5, 0.025], [0.75, 0.037], [1.0, 0.048]]


def fx_kinematics(name: str) -> dict:
    """A deep copy of the kinematics.json dict (contracts §5.2, schema_version 1) for ``fx_arm6`` / ``fx_arm5``."""
    return copy.deepcopy(_KINEMATICS[name])


def fx_joint_names(name: str) -> tuple[str, ...]:
    """USD joint names in joint-vector order (``[arm.joints].names``); the gripper joint is last."""
    return tuple(j["name"] for j in _KINEMATICS[name]["joints"])


def fx_lerobot_names(name: str) -> tuple[str, ...]:
    """LeRobot names paired index-by-index with ``fx_joint_names`` (``[arm.lerobot].names``)."""
    return tuple(_ARMS[name]["lerobot"])


def fx_limits(name: str) -> tuple[np.ndarray, np.ndarray]:
    """USD joint limits ``(lower, upper)`` in rad, float64 ``(J,)``, joint-vector order."""
    rows = _KINEMATICS[name]["joints"]
    return (np.array([r["limits_rad"][0] for r in rows], dtype=np.float64),
            np.array([r["limits_rad"][1] for r in rows], dtype=np.float64))


def fx_home_q(name: str) -> np.ndarray:
    """The ``[arm.initial_state.joint_positions]`` home pose as float64 ``(J,)`` rad (inside the limits)."""
    return np.array(_ARMS[name]["home"], dtype=np.float64)


def fx_arm_table(name: str) -> dict:
    """The ``[arm]`` table of a synthetic arm.toml, including the v4 tables of contracts §5.1.

    Contains ``name``, ``usd_path``, ``links`` (base, ee, moving_jaw, moving_jaw_mesh, fixed_jaw_mesh),
    ``joints.names``, ``initial_state`` (+ ``joint_positions``), ``lerobot`` (names, use_degrees = false,
    gripper_joint, roll_joint), ``kinematics``, ``gripper`` and ``expert``. Values are illustrative.
    """
    arm, kin = _ARMS[name], _KINEMATICS[name]
    names = list(fx_joint_names(name))
    arm_joints = [n for n in names if n != arm["gripper_joint"]]
    prefix = name.replace("_arm", "")
    return {
        "name": name.upper(),
        "usd_path": f"usd/{name}.usd",
        "links": {"base": kin["base_link"], "ee": arm["ee"], "fixed_jaw_mesh": f"{prefix}_fixed_mesh",
                  "moving_jaw": arm["moving_jaw"], "moving_jaw_mesh": f"{prefix}_moving_mesh"},
        "joints": {"names": names},
        "initial_state": {"position": [0.0, 0.0, 0.0], "rotation_euler_degrees": [0.0, 0.0, 0.0],
                          "joint_positions": dict(zip(names, arm["home"]))},
        "lerobot": {"follower_type": f"{name}_follower", "leader_type": f"{name}_leader", "use_degrees": False,
                    "roll_joint": arm["lerobot"][-2], "gripper_joint": arm["lerobot"][-1],
                    "names": list(arm["lerobot"])},
        "kinematics": {"file": "kinematics.json", "ee_link": arm["ee"], "tcp_position_m": list(arm["tcp_position_m"]),
                       "tcp_quat_wxyz": [1.0, 0.0, 0.0, 0.0], "approach_axis": [0.0, 0.0, 1.0],
                       "closing_axis": [1.0, 0.0, 0.0]},
        "gripper": {"joint": arm["gripper_joint"], "open_rad": 1.0, "closed_rad": 0.0,
                    "opening_table": [list(row) for row in FX_OPENING_TABLE], "max_object_width_m": 0.045},
        "expert": {"arm_joints": arm_joints, "max_joint_speed_rad_s": 1.5, "max_gripper_speed_rad_s": 3.0,
                   "max_ee_speed_m_s": 0.15, "ik_position_weight": 1.0, "ik_axis_weight": 0.3,
                   "ik_roll_weight": 0.1, "ik_tol_pos_m": 0.0005, "ik_tol_rot_rad": 0.02},
    }


def fx_setup_block(name: str, *, margin: float = 0.95) -> dict:
    """``{"sim_joint_mapping": ..., "runtime_joint_limits": ...}`` for ``setup["arms"][name]`` (contracts §5.4).

    Keys are LeRobot names, values ``{"joint_min", "joint_max"}`` in degrees. The mapping spans the
    USD limits; the runtime limits are the USD range shrunk symmetrically to ``margin`` of its width.
    """
    lower, upper = fx_limits(name)
    mapping, runtime = {}, {}
    for key, lo, hi in zip(fx_lerobot_names(name), np.degrees(lower), np.degrees(upper)):
        mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * margin
        mapping[key] = {"joint_min": float(lo), "joint_max": float(hi)}
        runtime[key] = {"joint_min": float(mid - half), "joint_max": float(mid + half)}
    return {"sim_joint_mapping": mapping, "runtime_joint_limits": runtime}


def fx_setup(arms: Sequence[str] = FX_ARMS, *, base: Mapping | None = None) -> dict:
    """A real_setup.json-like dict whose ``"arms"`` block covers ``arms`` (contracts §13 ``fx_scene``).

    ``base`` (e.g. the parsed repo ``real_setup.json``) is deep-copied and the ``arms`` block added.
    """
    setup = copy.deepcopy(dict(base)) if base is not None else {}
    setup.setdefault("arms", {})
    for name in arms:
        setup["arms"][name] = fx_setup_block(name)
    return setup


def write_arm(arms_dir: str | Path, name: str, arm_table: Mapping, *, kinematics: Mapping | None = None,
              files: Mapping[str, str | bytes] | None = None) -> Path:
    """Write ``arms_dir/<name>/arm.toml`` (``[arm]`` = ``arm_table``) and optionally ``kinematics.json`` and extra files.

    ``files`` maps relative paths to text or bytes. Returns the arm folder. ``load_arm_spec(name)``
    reads it once ``S2R_ARMS_DIR`` points at ``arms_dir`` (see ``temp_env`` / ``sandbox``).
    """
    folder = Path(arms_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "arm.toml").write_text(dump_toml({"arm": dict(arm_table)}))
    if kinematics is not None:
        (folder / "kinematics.json").write_text(json.dumps(kinematics, indent=2) + "\n")
    for rel, content in (files or {}).items():
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return folder


def write_fx_arm(arms_dir: str | Path, name: str = "fx_arm6") -> Path:
    """Write the complete synthetic arm folder (arm.toml + kinematics.json) for ``fx_arm6`` / ``fx_arm5``."""
    return write_arm(arms_dir, name, fx_arm_table(name), kinematics=fx_kinematics(name))


# ----------------------------------------------------------------------------------------------
# Reference kinematics (independent of src/core; formula of contracts §5.2)


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix of a (normalised) WXYZ quaternion."""
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64) / np.linalg.norm(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """WXYZ quaternion (``w >= 0``) of a 3x3 rotation matrix (or the rotation block of a 4x4)."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def frame_matrix(position: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform ``F`` from a position and a WXYZ quaternion."""
    out = np.eye(4)
    out[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """4x4 rotation ``R(axis, angle)`` (Rodrigues) about a unit axis through the origin."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
    return out


def reference_link_poses(kinematics: Mapping, q: Sequence[float], q_names: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Every link's 4x4 pose in the base frame: ``T_child = T_parent · F(parent_frame) · R(axis, q) · F(child_frame)⁻¹``.

    ``q`` is indexed by ``q_names`` (default: the kinematics joint order). Joints are resolved in
    topological order from ``base_link``; ``ValueError`` if the chain is not connected.
    """
    q_names = list(q_names) if q_names is not None else [j["name"] for j in kinematics["joints"]]
    values = dict(zip(q_names, np.asarray(q, dtype=np.float64)))
    poses = {kinematics["base_link"]: np.eye(4)}
    pending = list(kinematics["joints"])
    while pending:
        ready = [j for j in pending if j["parent"] in poses]
        if not ready:
            raise ValueError(f"Unconnected joints: {[j['name'] for j in pending]}")
        for joint in ready:
            parent = frame_matrix(joint["parent_frame"]["position"], joint["parent_frame"]["quat_wxyz"])
            child = frame_matrix(joint["child_frame"]["position"], joint["child_frame"]["quat_wxyz"])
            poses[joint["child"]] = (poses[joint["parent"]] @ parent @ axis_rotation(joint["axis"], values.get(joint["name"], 0.0))
                                     @ np.linalg.inv(child))
            pending.remove(joint)
    return poses


def reference_fk(kinematics: Mapping, q: Sequence[float], link: str, *, q_names: Sequence[str] | None = None,
                 tcp_position_m: Sequence[float] | None = None, tcp_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4 pose of ``link`` in the base frame; with ``tcp_position_m`` it returns ``link ∘ F(tcp)`` (the TCP)."""
    pose = reference_link_poses(kinematics, q, q_names)[link]
    if tcp_position_m is not None:
        pose = pose @ frame_matrix(tcp_position_m, tcp_quat_wxyz)
    return pose


def fx_reference_tcp(name: str, q: Sequence[float]) -> np.ndarray:
    """4x4 TCP pose of fx arm ``name`` at full-length joint vector ``q`` (ee link + ``tcp_position_m``)."""
    arm = _ARMS[name]
    return reference_fk(_KINEMATICS[name], q, arm["ee"], tcp_position_m=arm["tcp_position_m"])


def random_q(name: str, rng: np.random.Generator, *, margin: float = 0.05) -> np.ndarray:
    """A uniform random in-limit joint vector (``margin`` rad away from each limit), float64 ``(J,)``."""
    lower, upper = fx_limits(name)
    return rng.uniform(lower + margin, upper - margin)
