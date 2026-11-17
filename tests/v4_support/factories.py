"""Small, valid instances of the WP0 protocol types (contracts §3) for any package's tests.

``core.generate.protocols`` and ``core.tasks.base`` are imported lazily inside each function, so this
module imports fine before WP0 exists. ``make_*`` give one deterministic, valid instance (override
any field by keyword); ``random_*(rng)`` give varied valid instances for property loops over
``np.random.default_rng(k)``. Every float is a Python ``float`` and every int a Python ``int``.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

#: The 13 WP0 dataclass names, in contracts §3.2 order.
DATACLASS_NAMES = ("ObjectState", "PerturbationSpec", "NoiseSpec", "EpisodeSpec", "StageProgress", "ExpertObs",
                   "ExpertAction", "FrameLabels", "StepRecord", "Segment", "RecoveryInfo", "Fingerprints",
                   "EpisodeResult")


def _p():
    from core.generate import protocols
    return protocols


def hex_digest(text: Any) -> str:
    """Lower-case hex SHA-256 of ``str(text)`` (a valid fingerprint value)."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    """XYZW quaternion of a rotation by ``yaw`` rad about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def make_pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0):
    """``core.tasks.base.Pose`` at (x, y, z) with yaw about +z (XYZW in memory)."""
    from core.tasks.base import Pose
    return Pose((float(x), float(y), float(z)), yaw_quat_xyzw(float(yaw)))


def make_object_state(x: float = 0.2, y: float = 0.0, z: float = 0.01, *, yaw: float = 0.0, active: bool = True,
                      color: str | None = None):
    """``ObjectState`` at (x, y, z, yaw)."""
    return _p().ObjectState(pose=make_pose(x, y, z, yaw), active=active, color=color)


def make_perturbation(type: str = "grasp_offset", group=None, onset_stage: str | None = None, delay_steps: int = 0,
                      params: dict | None = None):
    """``PerturbationSpec``; ``group`` defaults to ``PerturbationGroup.EXPERT``, ``params`` to ``{"xy_m": 0.01, "dir_rad": 1.0}``."""
    p = _p()
    return p.PerturbationSpec(type=type, group=group if group is not None else p.PerturbationGroup.EXPERT,
                              onset_stage=onset_stage, delay_steps=delay_steps,
                              params=params if params is not None else {"xy_m": 0.01, "dir_rad": 1.0})


def make_noise(theta: float = 0.15, sigma_rad: float = 0.01, max_abs_rad: float = 0.02):
    """``NoiseSpec`` with the §8.5 default values."""
    return _p().NoiseSpec(theta=theta, sigma_rad=sigma_rad, max_abs_rad=max_abs_rad)


def make_episode_spec(kind: Any = "success", **overrides):
    """A valid ``EpisodeSpec`` of ``kind`` (``Kind`` or its value); keyword overrides replace fields.

    Defaults: task ``"fx_pick"``, arm ``"fx_arm5"``, scene ``"fx_scene"``, objects ``cube_0`` (active,
    red), ``cube_1`` (inactive, parked) and ``box`` (active); roles target→cube_0, container→box;
    goal ``{}``; 6-joint ``initial_q``; a group-A ``grasp_offset`` perturbation for failure/recovery
    and ``None`` for success (the §3.2 invariants hold).
    """
    p = _p()
    kind = p.Kind(kind)
    fields = dict(
        task="fx_pick", arm="fx_arm5", scene="fx_scene", kind=kind, attempt_index=0, seed=12345,
        objects={"cube_0": make_object_state(0.2, 0.05, 0.01, color="red"),
                 "cube_1": make_object_state(0.1, 0.0, -2.0, active=False),
                 "box": make_object_state(0.25, -0.05, 0.0)},
        roles={"target": "cube_0", "container": "box"}, goal={},
        instruction="Pick up the red cube and place it in the box",
        initial_q=(0.0, -0.3, 1.6, 1.2, 0.0, 0.5),
        perturbation=None if kind is p.Kind.SUCCESS else make_perturbation(),
    )
    fields.update(overrides)
    return p.EpisodeSpec(**fields)


def make_stage_progress(frame: int = 0, stage_index: int = -1, furthest_index: int = -1, satisfied: dict | None = None,
                        regressed_stage: str | None = None):
    """``StageProgress``; ``satisfied`` defaults to ``{"reached": False, "success": False}``."""
    return _p().StageProgress(frame=frame, stage_index=stage_index, furthest_index=furthest_index,
                              satisfied=satisfied if satisfied is not None else {"reached": False, "success": False},
                              regressed_stage=regressed_stage)


def make_expert_obs(q=None, frame: int = 0, objects: dict | None = None, signals: dict | None = None, *,
                    gripper_gap_m: float = 0.03):
    """``ExpertObs`` with ``q`` (default 6 zeros), ee at (0.2, 0, 0.15), tcp 9 cm lower, and objects
    ``cube_0`` (active) and ``cube_1`` (inactive). ``signals`` omitted → the class default."""
    kwargs = dict(frame=frame, q=np.zeros(6) if q is None else q, ee=make_pose(0.2, 0.0, 0.15),
                  tcp=make_pose(0.2, 0.0, 0.06), gripper_gap_m=gripper_gap_m,
                  objects=objects if objects is not None else {
                      "cube_0": make_object_state(0.2, 0.05, 0.01, color="red"),
                      "cube_1": make_object_state(0.1, 0.0, -2.0, active=False)})
    if signals is not None:
        kwargs["signals"] = signals
    return _p().ExpertObs(**kwargs)


def make_expert_action(q_target=None, phase: str = "approach", done: bool = False, injected: bool = False,
                       reentered: bool = False, resumed_stage: str | None = None):
    """``ExpertAction`` (``q_target`` default 6 zeros)."""
    return _p().ExpertAction(q_target=np.zeros(6) if q_target is None else q_target, phase=phase, done=done,
                             injected=injected, reentered=reentered, resumed_stage=resumed_stage)


def make_frame_labels(stage: int = 0, imitate: bool = True, ee_pose_wxyz=None):
    """``FrameLabels``; ``ee_pose_wxyz`` default ``[0.2, 0, 0.15, 1, 0, 0, 0]`` (disk order)."""
    return _p().FrameLabels(stage=stage, imitate=imitate,
                            ee_pose_wxyz=np.array([0.2, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0]) if ee_pose_wxyz is None else ee_pose_wxyz)


def make_step_record(frame: int = 0, J: int = 6, **overrides):
    """``StepRecord`` with zero raw vectors of length ``J`` and one active object ``cube_0``."""
    fields = dict(frame=frame, state_raw=np.zeros(J), action_raw=np.zeros(J), labels=make_frame_labels(),
                  phase="approach", injected=False, reentered=False, objects={"cube_0": make_pose(0.2, 0.05, 0.01)})
    fields.update(overrides)
    return _p().StepRecord(**fields)


def make_segment(label: Any = "nominal", start: int = 0, end: int = 10, phase: str | None = None):
    """``Segment`` (``label`` is a ``SegmentLabel`` or its value)."""
    p = _p()
    return p.Segment(label=p.SegmentLabel(label), start=start, end=end, phase=phase)


def make_recovery_info(detected_frame: int = 10, resumed_stage: str = "grasped", retries: int = 1,
                       back_on_plan_frame: int = 20):
    """``RecoveryInfo``."""
    return _p().RecoveryInfo(detected_frame=detected_frame, resumed_stage=resumed_stage, retries=retries,
                             back_on_plan_frame=back_on_plan_frame)


def make_fingerprints(tag: Any = "x", *, skipped: bool = False):
    """``Fingerprints`` with hex digests derived from ``tag``; ``skipped=True`` leaves traj/appearance/episode None."""
    return _p().Fingerprints(scenario_key=hex_digest(f"s{tag}"), spec_fp=hex_digest(f"p{tag}"),
                             traj_fp=None if skipped else hex_digest(f"t{tag}"),
                             appearance_fp=None if skipped else hex_digest(f"a{tag}"),
                             episode_fp=None if skipped else hex_digest(f"e{tag}"))


def make_episode_result(kind: Any = "success", status: Any = "accepted", num_frames: int = 5, **overrides):
    """A self-consistent ``EpisodeResult``: per-frame tuples of length ``num_frames``, one nominal segment
    covering ``[0, num_frames)``, reason ``None`` for accepted, ``expert_failed`` for rejected and
    ``scenario_seen`` (with no frames) for skipped."""
    p = _p()
    kind, status = p.Kind(kind), p.AttemptStatus(status)
    n = 0 if status is p.AttemptStatus.SKIPPED else num_frames
    reason = {p.AttemptStatus.ACCEPTED: None, p.AttemptStatus.REJECTED: p.RejectionReason.EXPERT_FAILED,
              p.AttemptStatus.SKIPPED: p.SkipReason.SCENARIO_SEEN}[status]
    accepted = status is p.AttemptStatus.ACCEPTED
    fields = dict(
        task="fx_pick", kind=kind, attempt_index=0, seed=12345, status=status, reason=reason,
        episode_index=0 if accepted else None, num_frames=n,
        success=None if n == 0 else kind is p.Kind.SUCCESS,
        failure_mode=None if (n == 0 or kind is p.Kind.SUCCESS) else "missed_grasp",
        furthest_stage=None if n == 0 else "reached",
        stage=tuple(min(t, 1) - 1 for t in range(n)), imitate=(True,) * n,
        segments=(make_segment("nominal", 0, n),) if n else (),
        perturbation_onset_frame=None, perturbation_applied=False, recovery=None,
        fingerprints=make_fingerprints(0, skipped=n == 0),
    )
    fields.update(overrides)
    return p.EpisodeResult(**fields)


# ---------------------------------------------------------------------------------------------
# Random valid instances (property tests)


def _f(rng, lo=-1.0, hi=1.0) -> float:
    return float(rng.uniform(lo, hi))


def _name(rng, prefix="n") -> str:
    return f"{prefix}{int(rng.integers(0, 1000))}"


def _json_value(rng, depth=0):
    choice = int(rng.integers(0, 7 if depth < 2 else 4))
    if choice == 0:
        return _f(rng, -10, 10)
    if choice == 1:
        return int(rng.integers(-1000, 1000))
    if choice == 2:
        return _name(rng, "s")
    if choice == 3:
        return bool(rng.integers(0, 2))
    if choice == 4:
        return [_f(rng) for _ in range(int(rng.integers(0, 4)))]
    if choice == 5:
        return [[_f(rng) for _ in range(3)] for _ in range(int(rng.integers(1, 3)))]
    return {_name(rng, "k"): _json_value(rng, depth + 1) for _ in range(int(rng.integers(1, 3)))}


def random_pose(rng):
    """Random ``Pose`` with a random unit XYZW quaternion."""
    from core.tasks.base import Pose
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return Pose(tuple(_f(rng, -0.5, 0.5) for _ in range(3)), tuple(float(v) for v in quat))


def random_object_state(rng):
    """Random ``ObjectState`` (random activity and optional colour)."""
    return _p().ObjectState(pose=random_pose(rng), active=bool(rng.integers(0, 2)),
                            color=None if rng.random() < 0.3 else _name(rng, "c"))


def random_perturbation(rng):
    """Random ``PerturbationSpec`` whose params mix float, int, str and tuple values."""
    p = _p()
    params = {"xy_m": _f(rng, 0, 0.05), "dir_rad": _f(rng, 0, 2 * math.pi), "steps": int(rng.integers(1, 50)),
              "object": "@target", "range": (_f(rng), _f(rng))}
    keep = [k for k in params if rng.random() < 0.7]
    return p.PerturbationSpec(type=_name(rng, "t"), group=p.PerturbationGroup(rng.choice(["expert", "world"])),
                              onset_stage=None if rng.random() < 0.3 else _name(rng, "st"),
                              delay_steps=int(rng.integers(0, 40)), params={k: params[k] for k in keep})


def random_noise(rng):
    """Random ``NoiseSpec``."""
    return _p().NoiseSpec(theta=_f(rng, 0, 1), sigma_rad=_f(rng, 0, 0.05), max_abs_rad=_f(rng, 0, 0.1))


def random_episode_spec(rng):
    """Random valid ``EpisodeSpec`` of a random kind (seed up to 2**64 − 1, nested JSON goal)."""
    p = _p()
    kind = p.Kind(rng.choice(["success", "failure", "recovery"]))
    names = [f"obj_{i}" for i in range(int(rng.integers(1, 5)))]
    objects = {n: p.ObjectState(pose=random_pose(rng), active=(i == 0) or bool(rng.integers(0, 2)),
                                color=None if rng.random() < 0.3 else _name(rng, "c")) for i, n in enumerate(names)}
    active = [n for n in names if objects[n].active]
    roles = {"target": active[0]}
    if len(active) > 1 and rng.random() < 0.5:
        roles["container"] = active[-1]
    J = int(rng.integers(5, 8))
    return p.EpisodeSpec(
        task=_name(rng, "task"), arm=_name(rng, "arm"), scene=_name(rng, "scene"), kind=kind,
        attempt_index=int(rng.integers(0, 10**6)), seed=int(rng.integers(0, 2**64, dtype=np.uint64)),
        objects=objects, roles=roles, goal={_name(rng, "g"): _json_value(rng) for _ in range(int(rng.integers(0, 4)))},
        instruction=f"do {_name(rng)}", initial_q=tuple(_f(rng, -2, 2) for _ in range(J)),
        held_object=None if rng.random() < 0.7 else active[0],
        perturbation=None if kind is p.Kind.SUCCESS else random_perturbation(rng),
        noise=random_noise(rng) if kind is p.Kind.SUCCESS and rng.random() < 0.5 else None,
        max_steps=int(rng.integers(1, 2000)), fps=int(rng.choice([15, 30, 60])))


def random_stage_progress(rng):
    """Random ``StageProgress``."""
    stages = [f"s{i}" for i in range(int(rng.integers(1, 5)))]
    return _p().StageProgress(frame=int(rng.integers(0, 600)), stage_index=int(rng.integers(-1, len(stages))),
                              furthest_index=int(rng.integers(-1, len(stages))),
                              satisfied={s: bool(rng.integers(0, 2)) for s in stages},
                              regressed_stage=None if rng.random() < 0.6 else stages[0])


def random_expert_obs(rng):
    """Random ``ExpertObs`` (random J, objects and signals)."""
    J = int(rng.integers(5, 8))
    return _p().ExpertObs(frame=int(rng.integers(0, 600)), q=rng.uniform(-2, 2, J), ee=random_pose(rng),
                          tcp=random_pose(rng), gripper_gap_m=_f(rng, 0, 0.05),
                          objects={_name(rng, "o"): random_object_state(rng) for _ in range(int(rng.integers(0, 4)))},
                          signals={_name(rng, "sig"): bool(rng.integers(0, 2)) for _ in range(int(rng.integers(0, 3)))})


def random_expert_action(rng):
    """Random ``ExpertAction``."""
    return _p().ExpertAction(q_target=rng.uniform(-2, 2, int(rng.integers(5, 8))), phase=_name(rng, "ph"),
                             done=bool(rng.integers(0, 2)), injected=bool(rng.integers(0, 2)),
                             reentered=bool(rng.integers(0, 2)),
                             resumed_stage=None if rng.random() < 0.5 else _name(rng, "st"))


def random_frame_labels(rng):
    """Random ``FrameLabels`` (ee pose with a unit WXYZ quaternion)."""
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return _p().FrameLabels(stage=int(rng.integers(-1, 6)), imitate=bool(rng.integers(0, 2)),
                            ee_pose_wxyz=np.concatenate([rng.uniform(-0.5, 0.5, 3), quat]))


def random_step_record(rng):
    """Random ``StepRecord``."""
    J = int(rng.integers(5, 8))
    return _p().StepRecord(frame=int(rng.integers(0, 600)), state_raw=rng.uniform(-100, 100, J),
                           action_raw=rng.uniform(-100, 100, J), labels=random_frame_labels(rng),
                           phase=_name(rng, "ph"), injected=bool(rng.integers(0, 2)), reentered=bool(rng.integers(0, 2)),
                           objects={_name(rng, "o"): random_pose(rng) for _ in range(int(rng.integers(0, 4)))})


def random_segment(rng):
    """Random ``Segment`` with ``start < end``."""
    p = _p()
    start = int(rng.integers(0, 100))
    return p.Segment(label=p.SegmentLabel(rng.choice([m.value for m in p.SegmentLabel])), start=start,
                     end=start + int(rng.integers(1, 100)), phase=None if rng.random() < 0.3 else _name(rng, "ph"))


def random_recovery_info(rng):
    """Random ``RecoveryInfo``."""
    detected = int(rng.integers(0, 300))
    return _p().RecoveryInfo(detected_frame=detected, resumed_stage=_name(rng, "st"), retries=int(rng.integers(1, 3)),
                             back_on_plan_frame=detected + int(rng.integers(0, 300)))


def random_fingerprints(rng):
    """Random ``Fingerprints`` (optional digests are sometimes ``None``)."""
    def maybe():
        return None if rng.random() < 0.3 else hex_digest(rng.random())
    return _p().Fingerprints(scenario_key=hex_digest(rng.random()), spec_fp=hex_digest(rng.random()),
                             traj_fp=maybe(), appearance_fp=maybe(), episode_fp=maybe())


def random_episode_result(rng):
    """Random ``EpisodeResult`` (any status/reason combination, consistent per-frame lengths)."""
    p = _p()
    status = p.AttemptStatus(rng.choice([m.value for m in p.AttemptStatus]))
    reason = {p.AttemptStatus.ACCEPTED: None,
              p.AttemptStatus.REJECTED: p.RejectionReason(rng.choice([m.value for m in p.RejectionReason])),
              p.AttemptStatus.SKIPPED: p.SkipReason(rng.choice([m.value for m in p.SkipReason]))}[status]
    n = int(rng.integers(1, 40))
    cut = int(rng.integers(1, n + 1))
    segments = tuple(s for s in (p.Segment(p.SegmentLabel.NOMINAL, 0, cut, "a"),
                                 p.Segment(p.SegmentLabel.FAILURE, cut, n, None)) if s.start < s.end)
    return p.EpisodeResult(
        task=_name(rng, "task"), kind=p.Kind(rng.choice(["success", "failure", "recovery"])),
        attempt_index=int(rng.integers(0, 10**6)), seed=int(rng.integers(0, 2**64, dtype=np.uint64)), status=status,
        reason=reason, episode_index=int(rng.integers(0, 100)) if status is p.AttemptStatus.ACCEPTED else None,
        num_frames=n, success=None if rng.random() < 0.2 else bool(rng.integers(0, 2)),
        failure_mode=None if rng.random() < 0.5 else _name(rng, "fm"),
        furthest_stage=None if rng.random() < 0.3 else _name(rng, "st"),
        stage=tuple(int(v) for v in rng.integers(-1, 5, n)), imitate=tuple(bool(v) for v in rng.integers(0, 2, n)),
        segments=segments, perturbation_onset_frame=None if rng.random() < 0.5 else int(rng.integers(0, n)),
        perturbation_applied=bool(rng.integers(0, 2)),
        recovery=None if rng.random() < 0.5 else random_recovery_info(rng), fingerprints=random_fingerprints(rng))


_RANDOM = {"ObjectState": random_object_state, "PerturbationSpec": random_perturbation, "NoiseSpec": random_noise,
           "EpisodeSpec": random_episode_spec, "StageProgress": random_stage_progress, "ExpertObs": random_expert_obs,
           "ExpertAction": random_expert_action, "FrameLabels": random_frame_labels, "StepRecord": random_step_record,
           "Segment": random_segment, "RecoveryInfo": random_recovery_info, "Fingerprints": random_fingerprints,
           "EpisodeResult": random_episode_result}


def random_instance(name: str, rng):
    """Random valid instance of the WP0 dataclass called ``name`` (one of ``DATACLASS_NAMES``)."""
    return _RANDOM[name](rng)
