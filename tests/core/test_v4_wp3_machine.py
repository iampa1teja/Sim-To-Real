"""WP3: ScriptedExpert state machine — output rules, determinism, closed-loop vs blind, re-entry (contracts §3.3, §6.2).

These tests drive the expert with synthetic observations: ``q`` of the next observation is exactly the
previous ``q_target`` (perfect tracking, TCP from ``chain.fk``), objects never attach, and the
``StageProgress`` fed to ``act`` is chosen by each test. The stand-in with attaching objects lives in
``test_v4_wp3_episodes.py``.

Decisions pinned here (see docs/v4/prompts/WP3.md):
* Step lifecycle: a step emits its first action on the frame it starts; from the next frame on, its
  termination condition is evaluated on the frame's observation *before* acting; a step that has emitted
  ``budget`` actions ends (timeout). On the frame a step ends the next step starts and emits that frame's
  action. Every step therefore emits between 1 and ``budget`` actions.
* Re-entry happens on the detection frame; that frame's action already belongs to the re-entered step
  (``reentered=True``, ``resumed_stage`` set, ``phase`` = the re-entered step's name); both flags are
  False/None on every other frame. ``reentries`` records ``(obs.frame, resumed_stage)``.
* Step index generalisation: the "first step tagged resumed_stage" is the first step whose stage is at
  or after ``resumed_stage`` in stage order.
* Detection rule 2 is not applied to ``approach`` steps (their goal is a pre-grasp offset by design).
* After ``max_retries`` re-entries no detection rule fires any more (references stay closed-loop).
* The gripper speed limit is ``max_gripper_speed_rad_s`` whatever the gripper entry of the speed array.
"""

import math
import unittest

import numpy as np

from core.arms.gripper import Gripper
from core.arms.kinematics import KinematicChain
from core.expert.machine import ScriptedExpert
from core.expert.motion import MotionLimits
from core.expert.program import parse_program
from core.generate.protocols import (EpisodeSpec, Expert, ExpertAction, ExpertObs, Kind, ObjectState,
                                     PerturbationGroup, PerturbationSpec, StageProgress)
from core.tasks.base import Pose
from core.tasks.loader import load_task
from v4_support import clear_task_cache, copy_fixture, sandbox, temp_env
from v4_support.fx_arms import FX_ARMS, fx_arm_table, fx_home_q, fx_joint_names, fx_kinematics, fx_limits
from v4_support.tasks import STUB_USDA, write_task

STAGES = ("reached", "grasped", "lifted", "placed", "success")
FPS = 30
TABLE_Z = -0.05     # fx arms reach a down-pointing TCP for z in about [-0.06, 0.10] m at r in [0.10, 0.20] m


# ---------------------------------------------------------------------------------------------------
# helpers (private to this file)

def tool_task_table():
    """GenericTask with a cube, a second cube, a container and four goal kinds; stage checks are signals."""
    return {
        "variants": {"teleop": "env_cfg:apply_teleop", "generate": "env_cfg:apply_generate"},
        "objects": {"cube": {"size": [0.02, 0.02, 0.02]}, "cube_b": {"size": [0.02, 0.02, 0.02]},
                    "box": {"usd": "usd/box.usda", "origin": "bottom", "size_outer": [0.05, 0.05, 0.035],
                            "inner_half_size": [0.02, 0.02], "floor_z": 0.002}},
        "roles": {"target": "cube", "container": "box"},
        "start": {"goal": {
            "point": {"type": "point", "region": {"xy_min": [0.12, -0.05], "xy_max": [0.18, 0.05]}, "z_m": [0.0, 0.05]},
            "joint": {"type": "joint", "joints": "arm"},
            "gap": {"type": "gap", "gap_m": [0.01, 0.03]},
            "area": {"type": "region", "center_region": {"xy_min": [0.14, 0.0], "xy_max": [0.16, 0.02]},
                     "half_size_m": [0.01, 0.01]}}},
        "stages": {"order": list(STAGES)},
        "checks": {s: {"check": "signal", "name": s} for s in STAGES},
        "failures": {"rules": [{"name": "failed"}]},
    }


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def make_chain(arm):
    table = fx_arm_table(arm)
    kin = table["kinematics"]
    return KinematicChain.from_dict(fx_kinematics(arm), q_names=fx_joint_names(arm),
                                    chain_joints=table["expert"]["arm_joints"], ee_link=kin["ee_link"],
                                    tcp_position_m=kin["tcp_position_m"], tcp_quat_wxyz=kin["tcp_quat_wxyz"],
                                    approach_axis=kin["approach_axis"], closing_axis=kin["closing_axis"])


def make_gripper(arm):
    return Gripper.from_table(fx_arm_table(arm)["gripper"], fx_joint_names(arm))


def expert_limits(arm):
    lo, hi = fx_limits(arm)
    return lo + 0.05, hi - 0.05


def motion_limits(arm, *, ee=0.15):
    """Distinct per-joint speeds; the gripper entry of the array is deliberately huge (it must be ignored)."""
    J = len(fx_joint_names(arm))
    speeds = np.linspace(1.0, 2.2, J)
    speeds[-1] = 99.0
    return MotionLimits(max_joint_speed_rad_s=speeds, max_gripper_speed_rad_s=2.0, max_ee_speed_m_s=ee)


def max_steps_for(arm, limits):
    out = np.array(limits.max_joint_speed_rad_s, dtype=float) / FPS
    out[make_gripper(arm).index] = limits.max_gripper_speed_rad_s / FPS
    return out


def tool_objects(cube=(0.15, 0.03, 0.0), cube_b=(0.12, -0.08, 0.0), box=(0.17, -0.06, 0.0)):
    return {"cube": ObjectState(Pose((cube[0], cube[1], TABLE_Z + 0.01), yaw_quat(cube[2])), True, "red"),
            "cube_b": ObjectState(Pose((cube_b[0], cube_b[1], TABLE_Z + 0.01), yaw_quat(cube_b[2])), True, "blue"),
            "box": ObjectState(Pose((box[0], box[1], TABLE_Z), yaw_quat(box[2])), True, None)}


def world_perturbation():
    """A group-B perturbation: the expert itself ignores it (only failure/recovery need *some* perturbation)."""
    return PerturbationSpec(type="object_shift", group=PerturbationGroup.WORLD, onset_stage="reached",
                            params={"xy_m": 0.03, "dir_rad": 0.0})


def make_spec(arm, kind="success", *, objects=None, roles=None, goal=None, perturbation=None, q0=None, seed=11,
              task="fx_wp3_tool"):
    kind = Kind(kind)
    if perturbation is None and kind is not Kind.SUCCESS:
        perturbation = world_perturbation()
    return EpisodeSpec(task=task, arm=arm, scene="fx_scene", kind=kind, attempt_index=0, seed=seed,
                       objects=tool_objects() if objects is None else objects,
                       roles={"target": "cube", "container": "box"} if roles is None else roles,
                       goal={"point": [0.15, 0.02, 0.02]} if goal is None else goal, instruction="test",
                       initial_q=tuple(float(v) for v in (fx_home_q(arm) if q0 is None else q0)),
                       perturbation=perturbation)


def progress(frame, stage_index=-1, regressed=None, furthest=None):
    return StageProgress(frame=frame, stage_index=stage_index,
                         furthest_index=stage_index if furthest is None else furthest,
                         satisfied={s: i <= stage_index for i, s in enumerate(STAGES)}, regressed_stage=regressed)


class Tracker:
    """Perfect tracking: the next observation's q is the previous q_target (no object physics)."""

    def __init__(self, arm, spec):
        self.chain, self.gripper = make_chain(arm), make_gripper(arm)
        self.objects = dict(spec.objects)
        self.q = np.array(spec.initial_q, dtype=float)
        self.frame = 0

    def obs(self):
        return ExpertObs(frame=self.frame, q=self.q.copy(), ee=self.chain.fk_pose(self.q, "ee"),
                         tcp=self.chain.fk_pose(self.q, "tcp"),
                         gripper_gap_m=self.gripper.gap_from_angle(self.q[self.gripper.index]), objects=self.objects)

    def step(self, q_target):
        self.q = np.array(q_target, dtype=float)
        self.frame += 1
        return self.obs()

    def move(self, name, xyz, yaw=0.0):
        old = self.objects[name]
        self.objects[name] = ObjectState(Pose(tuple(float(v) for v in xyz), yaw_quat(yaw)), old.active, old.color)


def run(expert, spec, arm, frames, prog_fn=None, *, on_frame=None):
    """Reset and act for ``frames`` frames; returns (actions, observations used for each action)."""
    env = Tracker(arm, spec)
    obs = env.obs()
    expert.reset(spec, obs)
    actions, seen = [], []
    for t in range(frames):
        if on_frame is not None:
            on_frame(t, env)
            obs = env.obs()
        act = expert.act(obs, prog_fn(t) if prog_fn else progress(t))
        actions.append(act)
        seen.append(obs)
        obs = env.step(act.q_target)
    return actions, seen


class _ToolTaskCase(unittest.TestCase):
    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()
        write_task(self.box.tasks, "fx_wp3_tool", tool_task_table(), files={"usd/box.usda": STUB_USDA})
        clear_task_cache("fx_wp3_tool")
        self.task = load_task("fx_wp3_tool")

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def program(self, steps, arm="fx_arm5", **top):
        return parse_program({"steps": steps, **top}, task=self.task, arm_spec=fx_arm_table(arm),
                             joint_limits=expert_limits(arm))

    def expert(self, steps, arm="fx_arm5", *, limits=None, stages=STAGES, **top):
        lo, hi = expert_limits(arm)
        return ScriptedExpert(self.program(steps, arm, **top), make_chain(arm), make_gripper(arm), lower=lo, upper=hi,
                              limits=limits or motion_limits(arm), fps=FPS, stages=stages)


# ---------------------------------------------------------------------------------------------------

class ProtocolTests(_ToolTaskCase):
    """WP3 §3.3 Expert protocol."""

    def test_is_expert(self):
        """§3.3 ScriptedExpert satisfies the runtime-checkable Expert protocol; phase/done are properties."""
        ex = self.expert([{"skill": "wait", "steps": 2}])
        self.assertIsInstance(ex, Expert)
        self.assertEqual(ex.reentries, [])

    def test_act_before_reset(self):
        """§3.3 calling act before reset raises RuntimeError."""
        ex = self.expert([{"skill": "wait", "steps": 2}])
        spec = make_spec("fx_arm5")
        with self.assertRaises(RuntimeError):
            ex.act(Tracker("fx_arm5", spec).obs(), progress(0))

    def test_bad_construction(self):
        """§1.6 lower/upper not (J,), lower > upper, or a speed array of the wrong length -> ValueError."""
        arm = "fx_arm5"
        prog = self.program([{"skill": "wait", "steps": 2}])
        lo, hi = expert_limits(arm)
        chain, grip = make_chain(arm), make_gripper(arm)
        bad = {
            "short_lower": dict(lower=lo[:-1], upper=hi, limits=motion_limits(arm)),
            "inverted": dict(lower=hi, upper=lo, limits=motion_limits(arm)),
            "speed_len": dict(lower=lo, upper=hi, limits=MotionLimits(np.ones(3), 1.0, 0.1)),
        }
        for label, kw in bad.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    ScriptedExpert(prog, chain, grip, fps=FPS, stages=STAGES, **kw)

    def test_reset_wrong_q_length(self):
        """§1.6 an observation whose q is not (J,) -> ValueError at reset."""
        ex = self.expert([{"skill": "wait", "steps": 2}], arm="fx_arm5")
        spec6 = make_spec("fx_arm6")
        with self.assertRaises(ValueError):
            ex.reset(make_spec("fx_arm5"), Tracker("fx_arm6", spec6).obs())

    def test_action_fields(self):
        """§3.2 ExpertAction per frame: q_target (J,) float64 finite rad; phase non-empty; flags are bools."""
        ex = self.expert([{"skill": "reach", "target": "@goal.point"}, {"skill": "wait", "steps": 3}])
        actions, _ = run(ex, make_spec("fx_arm5"), "fx_arm5", 20)
        for a in actions:
            self.assertIsInstance(a, ExpertAction)
            self.assertEqual(a.q_target.shape, (6,))
            self.assertEqual(a.q_target.dtype, np.float64)
            self.assertTrue(np.all(np.isfinite(a.q_target)))
            self.assertIsInstance(a.phase, str)
            self.assertTrue(a.phase)
            self.assertEqual(ex.phase, a.phase)
            self.assertEqual(ex.done, a.done)
            self.assertIs(type(a.injected), bool)
            self.assertIs(a.reentered, False)
            self.assertIsNone(a.resumed_stage)


class OutputRuleTests(_ToolTaskCase):
    """WP3 §6.2 output rules on random starts, both fx arms (property test)."""

    def _random_program(self, arm, rng):
        names = fx_joint_names(arm)
        lo, hi = expert_limits(arm)
        arm_joints = fx_arm_table(arm)["expert"]["arm_joints"]
        qj = {str(j): float(rng.uniform(lo[names.index(j)] + 0.1, hi[names.index(j)] - 0.1))
              for j in rng.choice(arm_joints, 2, replace=False)}
        sweep_j = str(rng.choice(arm_joints))
        i = names.index(sweep_j)
        point = [float(rng.uniform(0.11, 0.19)), float(rng.uniform(-0.08, 0.08)), float(rng.uniform(-0.03, 0.08))]
        return [{"skill": "move_joint", "q": qj, "budget": 40},
                {"skill": "set_gripper", "gap_m": float(rng.uniform(0.0, 0.048))},
                {"skill": "reach", "target": point, "budget": 60},
                {"skill": "approach", "target": "@target", "yaw": "object", "budget": 40},
                {"skill": "sweep", "joint": sweep_j, "budget": 30,
                 "angles": [float(rng.uniform(lo[i] + 0.1, hi[i] - 0.1)) for _ in range(2)]},
                {"skill": "wait", "steps": 2}]

    def test_limits_speed_shape(self):
        """§6.2 every action: (J,) finite rad within [lower, upper]; |Δq| ≤ max_speed/fps + 1e-9 per joint (gripper:
        max_gripper_speed_rad_s), with q_target[−1] := obs.q at reset; phase non-empty."""
        for arm in FX_ARMS:
            lo, hi = expert_limits(arm)
            lim = motion_limits(arm)
            dmax = max_steps_for(arm, lim)
            for k in range(5):
                rng = np.random.default_rng(1000 * (arm == "fx_arm6") + k)
                q0 = rng.uniform(lo + 0.02, hi - 0.02)
                cube = (float(rng.uniform(0.12, 0.18)), float(rng.uniform(-0.06, 0.06)), float(rng.uniform(-1, 1)))
                spec = make_spec(arm, str(rng.choice(["success", "failure", "recovery"])), q0=q0,
                                 objects=tool_objects(cube=cube))
                ex = self.expert(self._random_program(arm, rng), arm, limits=lim)
                with self.subTest(arm=arm, seed=k):
                    actions, seen = run(ex, spec, arm, 160, lambda t: progress(t, int(rng.integers(-1, 5))))
                    prev = np.array(spec.initial_q)
                    for t, a in enumerate(actions):
                        q = a.q_target
                        self.assertEqual(q.shape, (len(lo),))
                        self.assertTrue(np.all(np.isfinite(q)), t)
                        self.assertTrue(np.all(q >= lo) and np.all(q <= hi), t)
                        self.assertTrue(np.all(np.abs(q - prev) <= dmax + 1e-9), (t, np.abs(q - prev) - dmax))
                        self.assertTrue(a.phase)
                        prev = q

    def test_gripper_speed_uses_gripper_limit(self):
        """§6.2 the gripper moves at max_gripper_speed_rad_s/fps per frame while opening fully."""
        arm = "fx_arm6"
        lim = motion_limits(arm)
        ex = self.expert([{"skill": "set_gripper", "gap_m": "closed"}, {"skill": "set_gripper", "gap_m": "open"}],
                         arm, limits=lim)
        q0 = fx_home_q(arm)
        actions, _ = run(ex, make_spec(arm, q0=q0), arm, 40)
        g = make_gripper(arm).index
        steps = np.abs(np.diff([q0[g]] + [a.q_target[g] for a in actions]))
        self.assertAlmostEqual(float(steps.max()), lim.max_gripper_speed_rad_s / FPS, places=9)

    def test_cartesian_goal_speed(self):
        """§6.2 the Cartesian goal advances ≤ max_ee_speed_m_s/fps per step: with perfect tracking, a reach that
        starts already pointing down moves the commanded TCP at most that far per frame (small IK slack)."""
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                a, b = [0.12, -0.04, 0.0], [0.19, 0.05, 0.05]
                ex = self.expert([{"skill": "reach", "target": a}], arm)
                actions, _ = run(ex, make_spec(arm), arm, 200)
                self.assertTrue(actions[-1].done)
                start = actions[-1].q_target
                ee = 0.06
                ex = self.expert([{"skill": "reach", "target": b}], arm, limits=motion_limits(arm, ee=ee))
                actions, _ = run(ex, make_spec(arm, q0=start), arm, 200)
                self.assertTrue(actions[-1].done)
                chain = make_chain(arm)
                tcp = [chain.fk(start)[:3, 3]] + [chain.fk(x.q_target)[:3, 3] for x in actions]
                moves = np.linalg.norm(np.diff(tcp, axis=0), axis=1)
                self.assertLessEqual(float(moves.max()), ee / FPS * 1.1 + 2e-4)
                self.assertLess(np.linalg.norm(tcp[-1] - np.array(b)), 0.005 + 1e-9)


class DeterminismTests(_ToolTaskCase):
    """WP3 §6.2 identical observations give identical outputs; no hidden global state."""

    STEPS = [{"skill": "move_joint", "q": "home"}, {"skill": "set_gripper", "gap_m": "open"},
             {"skill": "approach", "target": "@target", "stage": "reached"},
             {"skill": "reach", "target": "@goal.point", "stage": "grasped"}, {"skill": "wait", "steps": 3}]

    @staticmethod
    def _same(a, b):
        return (np.array_equal(a.q_target, b.q_target) and a.phase == b.phase and a.done == b.done
                and a.injected == b.injected and a.reentered == b.reentered and a.resumed_stage == b.resumed_stage)

    def test_replay_same_observations(self):
        """§6.2 a second instance fed the first one's observation sequence reproduces every action bit for bit."""
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                spec = make_spec(arm, "recovery")
                prog_fn = lambda t: progress(t, 0 if t < 40 else -1, "reached" if t == 40 else None)  # noqa: E731
                actions, seen = run(self.expert(self.STEPS, arm), spec, arm, 120, prog_fn)
                other = self.expert(self.STEPS, arm)
                other.reset(spec, seen[0])
                for t, obs in enumerate(seen):
                    self.assertTrue(self._same(actions[t], other.act(obs, prog_fn(t))), t)

    def test_reset_starts_fresh(self):
        """§3.3 reset fully re-initialises: a reused instance gives the same actions as a new one."""
        arm = "fx_arm5"
        spec = make_spec(arm, "recovery")
        prog_fn = lambda t: progress(t, -1, "grasped" if t in (5, 30) else None)  # noqa: E731
        ex = self.expert(self.STEPS, arm)
        first, _ = run(ex, spec, arm, 80, prog_fn)
        reentries = list(ex.reentries)
        second, _ = run(ex, spec, arm, 80, prog_fn)
        self.assertEqual(ex.reentries, reentries)
        self.assertTrue(all(self._same(a, b) for a, b in zip(first, second)))

    def test_instances_do_not_interact(self):
        """§6.2 two experts stepped alternately give exactly what each gives alone."""
        arm = "fx_arm6"
        spec_a = make_spec(arm, "recovery", goal={"point": [0.14, -0.03, 0.03]})
        spec_b = make_spec(arm, "failure", goal={"point": [0.18, 0.05, 0.0]}, objects=tool_objects(cube=(0.13, -0.05, 0.4)))
        fn_a = lambda t: progress(t, -1, "reached" if t == 25 else None)  # noqa: E731
        fn_b = lambda t: progress(t, 1)  # noqa: E731
        alone_a, _ = run(self.expert(self.STEPS, arm), spec_a, arm, 70, fn_a)
        alone_b, _ = run(self.expert(self.STEPS, arm), spec_b, arm, 70, fn_b)
        ea, eb = self.expert(self.STEPS, arm), self.expert(self.STEPS, arm)
        ta, tb = Tracker(arm, spec_a), Tracker(arm, spec_b)
        oa, ob = ta.obs(), tb.obs()
        ea.reset(spec_a, oa)
        eb.reset(spec_b, ob)
        for t in range(70):
            a = ea.act(oa, fn_a(t))
            b = eb.act(ob, fn_b(t))
            self.assertTrue(self._same(a, alone_a[t]), t)
            self.assertTrue(self._same(b, alone_b[t]), t)
            oa, ob = ta.step(a.q_target), tb.step(b.q_target)


class ClosedLoopBlindTests(_ToolTaskCase):
    """WP3 §6.2 modes: closed_loop re-reads object poses every step; blind resolves them once at reset."""

    STEPS = [{"skill": "reach", "target": "@target", "offset_m": [0.0, 0.0, 0.03]}, {"skill": "wait", "steps": 3}]

    def _run_with_move(self, arm, kind, prog_fn=None):
        moved = (0.18, -0.03, TABLE_Z + 0.01)

        def mover(t, env):
            if t == 10:
                env.move("cube", moved)
        ex = self.expert(self.STEPS, arm)
        actions, seen = run(ex, make_spec(arm, kind), arm, 150, prog_fn, on_frame=mover)
        return ex, actions, seen, np.array(moved)

    def test_closed_loop_target_follows(self):
        """§6.2 closed_loop: the object is moved mid-reach and the TCP ends at the new object pose + offset."""
        for arm in FX_ARMS:
            for kind in ("success", "recovery"):
                with self.subTest(arm=arm, kind=kind):
                    ex, actions, seen, moved = self._run_with_move(arm, kind)
                    self.assertTrue(actions[-1].done)
                    tcp = make_chain(arm).fk(actions[-1].q_target)[:3, 3]
                    self.assertLess(np.linalg.norm(tcp - (moved + [0, 0, 0.03])), 0.005 + 1e-9)

    def test_blind_never_follows_or_reenters(self):
        """§6.2 blind (kind failure): the reach ends at the reset pose of the object; regressions are ignored."""
        old = np.array(tool_objects()["cube"].pose.position)
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                ex, actions, seen, moved = self._run_with_move(arm, "failure",
                                                               lambda t: progress(t, -1, "reached"))
                self.assertTrue(actions[-1].done)
                tcp = make_chain(arm).fk(actions[-1].q_target)[:3, 3]
                self.assertLess(np.linalg.norm(tcp - (old + [0, 0, 0.03])), 0.005 + 1e-9)
                self.assertEqual(ex.reentries, [])
                self.assertFalse(any(a.reentered for a in actions))

    def test_blind_rule2_never_fires(self):
        """§6.2 blind: a stage-tagged step that ends with its stage unsatisfied just advances."""
        arm = "fx_arm5"
        ex = self.expert([{"skill": "reach", "target": "@goal.point", "stage": "reached"},
                          {"skill": "wait", "steps": 3, "stage": "success"}], arm)
        actions, _ = run(ex, make_spec(arm, "failure"), arm, 150)
        self.assertTrue(actions[-1].done)
        self.assertEqual(ex.reentries, [])


class ReentryTests(_ToolTaskCase):
    """WP3 §6.2 detection rules 1 and 2, the re-entry step index, max_retries."""

    def _fx_color(self):
        copy_fixture("v4_wp1/tasks/fx_color", self.box.root)
        with temp_env(S2R_TASKS_DIR=self.box.root / "v4_wp1" / "tasks"):
            clear_task_cache("fx_color")
            task = load_task("fx_color")
        clear_task_cache("fx_color")
        return task

    def _color_spec(self, arm):
        objs = {f"cube_{i}": ObjectState(Pose((0.15, 0.06, TABLE_Z + 0.01)) if i == 0 else Pose((0.1 * i, 0.0, -2.0)),
                                         i == 0, "red" if i == 0 else None) for i in range(6)}
        objs["small_box"] = ObjectState(Pose((0.17, -0.07, TABLE_Z)), True, None)
        return EpisodeSpec(task="fx_color", arm=arm, scene="fx_scene", kind=Kind.RECOVERY, attempt_index=0, seed=3,
                           objects=objs, roles={"target": "cube_0", "container": "small_box"}, goal={},
                           instruction="x", initial_q=tuple(fx_home_q(arm)), perturbation=world_perturbation())

    def test_reentry_table_fx_color(self):
        """§6.2 resumed_stage = stages[stage_index + 1]; step = 1 + last stage-tagged step before the first step
        tagged resumed_stage (0 if none); reported on the detection frame, whose action starts that step."""
        task = self._fx_color()
        arm = "fx_arm5"
        stages = task.stages
        prog = parse_program(task.spec.raw["expert"], task=task, arm_spec=fx_arm_table(arm),
                             joint_limits=expert_limits(arm))
        table = {-1: ("reached", "move_joint"), 0: ("grasped", "grasp"), 1: ("lifted", "lift"),
                 2: ("over_box", "transport"), 3: ("placed", "place"), 4: ("success", "retreat")}
        for k, (resumed, phase) in table.items():
            with self.subTest(stage_index=k):
                lo, hi = expert_limits(arm)
                ex = ScriptedExpert(prog, make_chain(arm), make_gripper(arm), lower=lo, upper=hi,
                                    limits=motion_limits(arm), fps=FPS, stages=stages)
                spec = self._color_spec(arm)
                env = Tracker(arm, spec)
                obs = env.obs()
                ex.reset(spec, obs)

                def prog_(t, idx=-1, reg=None):
                    return StageProgress(frame=t, stage_index=idx, furthest_index=max(idx, 0),
                                         satisfied={s: i <= idx for i, s in enumerate(stages)}, regressed_stage=reg)
                a0 = ex.act(obs, prog_(0))
                self.assertEqual(a0.phase, "move_joint")
                obs = env.step(a0.q_target)
                # the regressed stage named by the tracker does not choose the resumed stage; stage_index does
                a1 = ex.act(obs, prog_(1, k, stages[-1]))
                self.assertTrue(a1.reentered)
                self.assertEqual(a1.resumed_stage, resumed)
                self.assertEqual(a1.phase, phase)
                self.assertEqual(ex.phase, phase)
                self.assertEqual(ex.reentries, [(1, resumed)])
                obs = env.step(a1.q_target)
                a2 = ex.act(obs, prog_(2, k))
                self.assertFalse(a2.reentered)
                self.assertIsNone(a2.resumed_stage)
                self.assertEqual(ex.reentries, [(1, resumed)])

    def test_reentry_untagged_stage(self):
        """§6.2 (generalised): with no step tagged resumed_stage, the first step tagged a later stage stands in."""
        steps = [{"skill": "move_joint", "q": "home"}, {"skill": "reach", "target": "@goal.point", "stage": "reached"},
                 {"skill": "lift", "name": "up", "stage": "lifted"}, {"skill": "set_gripper", "gap_m": "open",
                                                                     "name": "open"},
                 {"skill": "wait", "steps": 3, "stage": "success"}]
        cases = {-1: ("reached", "move_joint"), 0: ("grasped", "up"), 1: ("lifted", "up"), 2: ("placed", "open"),
                 3: ("success", "open")}
        for k, (resumed, phase) in cases.items():
            with self.subTest(stage_index=k):
                ex = self.expert(steps)
                actions, _ = run(ex, make_spec("fx_arm5", "recovery"), "fx_arm5", 3,
                                 lambda t: progress(t, k if t else -1, "reached" if t == 1 else None))
                self.assertEqual((actions[1].resumed_stage, actions[1].phase), (resumed, phase))
                self.assertEqual(ex.reentries, [(1, resumed)])

    def test_rule2_on_termination(self):
        """§6.2 rule 2: a step tagged s ends while s is unsatisfied -> re-entry on exactly that frame; the
        frame before, the step had not met its condition yet."""
        arm = "fx_arm5"
        goal = np.array([0.15, 0.02, 0.02])
        steps = [{"skill": "move_joint", "q": "home"}, {"skill": "reach", "target": "@goal.point", "stage": "reached"},
                 {"skill": "wait", "steps": 3, "stage": "success"}]
        ex = self.expert(steps, arm)
        actions, seen = run(ex, make_spec(arm, "recovery"), arm, 200)
        self.assertGreaterEqual(len(ex.reentries), 1)
        t, resumed = ex.reentries[0]
        self.assertEqual(resumed, "reached")
        self.assertTrue(actions[t].reentered)
        self.assertEqual(actions[t].phase, "move_joint")
        self.assertEqual(actions[t - 1].phase, "reach")
        tcp = lambda o: np.array(o.tcp.position)  # noqa: E731
        self.assertLessEqual(np.linalg.norm(tcp(seen[t]) - goal), 0.005)
        self.assertGreater(np.linalg.norm(tcp(seen[t - 1]) - goal), 0.005)

    def test_rule2_satisfied_advances(self):
        """§6.2 a step whose stage is satisfied when it ends just advances; done after the last step."""
        arm = "fx_arm6"
        steps = [{"skill": "reach", "target": "@goal.point", "stage": "reached"},
                 {"skill": "wait", "steps": 3, "stage": "success"}]
        ex = self.expert(steps, arm)
        actions, _ = run(ex, make_spec(arm, "recovery"), arm, 200, lambda t: progress(t, len(STAGES) - 1))
        self.assertEqual(ex.reentries, [])
        self.assertTrue(actions[-1].done)
        self.assertEqual([a.phase for a in actions if a.phase == "wait" and not a.done], ["wait"] * 3)

    def test_rule2_on_budget_timeout(self):
        """§6.2 rule 2 also fires when the tagged step runs out of budget; an untagged step just advances."""
        arm = "fx_arm5"
        far = [0.6, 0.0, 0.5]
        for tagged in (True, False):
            with self.subTest(tagged=tagged):
                step = {"skill": "reach", "target": far, "budget": 25}
                if tagged:
                    step["stage"] = "reached"
                ex = self.expert([{"skill": "wait", "steps": 2}, step, {"skill": "wait", "steps": 2, "name": "after"}],
                                 arm)
                actions, _ = run(ex, make_spec(arm, "recovery"), arm, 40)
                self.assertEqual([a.phase for a in actions[2:27]], ["reach"] * 25)
                if tagged:
                    self.assertTrue(actions[27].reentered)
                    self.assertEqual(actions[27].phase, "wait")
                    self.assertEqual(ex.reentries[0], (27, "reached"))
                else:
                    self.assertEqual(actions[27].phase, "after")
                    self.assertEqual(ex.reentries, [])

    def test_approach_not_checked(self):
        """§6.2 decision: rule 2 does not apply to approach (pre-grasp offset); the next tagged step is checked."""
        arm = "fx_arm5"
        steps = [{"skill": "approach", "target": "@target", "stage": "reached"},
                 {"skill": "wait", "steps": 3, "stage": "grasped"}]
        ex = self.expert(steps, arm)
        actions, _ = run(ex, make_spec(arm, "recovery"), arm, 200)
        waits = [i for i, a in enumerate(actions) if a.phase == "wait"]
        self.assertTrue(waits)
        self.assertFalse(any(a.reentered for a in actions[:waits[0] + 1]))
        self.assertEqual(ex.reentries[0], (waits[0] + 3, "reached"))
        self.assertEqual(actions[waits[0] + 3].phase, "approach")

    def test_max_retries(self):
        """§6.2 at most max_retries re-entries; then detection stops (default max_retries = 2)."""
        arm = "fx_arm5"
        steps = [{"skill": "wait", "steps": 4, "stage": "reached"}, {"skill": "wait", "steps": 4, "name": "w2"}]
        for retries in (None, 0, 1, 3):
            with self.subTest(max_retries=retries):
                top = {} if retries is None else {"max_retries": retries}
                ex = self.expert(steps, arm, **top)
                actions, _ = run(ex, make_spec(arm, "recovery"), arm, 60, lambda t: progress(t, -1, "reached"))
                n = 2 if retries is None else retries
                self.assertEqual(len(ex.reentries), n)
                self.assertEqual([t for t, a in enumerate(actions) if a.reentered], list(range(n)))
                self.assertTrue(actions[-1].done)

    def test_done_holds_last_target(self):
        """§6.2 done after the last step terminates (or overruns its budget); then the last target is held."""
        arm = "fx_arm6"
        for steps in ([{"skill": "wait", "steps": 3}],
                      [{"skill": "reach", "target": [0.6, 0.0, 0.5], "budget": 5, "name": "far"}]):
            with self.subTest(steps=steps):
                ex = self.expert(steps, arm)
                actions, _ = run(ex, make_spec(arm), arm, 12)
                n = steps[0].get("steps", steps[0].get("budget"))
                self.assertEqual([a.done for a in actions], [False] * n + [True] * (12 - n))
                for a in actions[n:]:
                    np.testing.assert_array_equal(a.q_target, actions[n - 1].q_target)
                    self.assertEqual(a.phase, steps[0].get("name", steps[0]["skill"]))
                self.assertTrue(ex.done)


if __name__ == "__main__":
    unittest.main()
