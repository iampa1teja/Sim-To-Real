"""WP3: the expert on a kinematic stand-in — skill termination and budgets, the contract example programs,
group-A perturbations (contracts §6.1, §6.2, §7.1, §4.6, §11 dynamics in simplified form).

``_StandIn`` below is private to this file and is *not* FakeEnv (WP7): joints track ``q_target`` exactly
(clipped to the USD limits); an object attaches when the jaws close (gap decreasing, gap ≤ w + 0.004)
around its centre and then moves rigidly with the TCP; it detaches when the gap opens past w + 0.002
and drops onto the table or a container floor; a TCP inside an object's footprint (+1 cm) at object
height pushes it. The table is at z = −0.05 m in the base frame because the fx arms reach a
down-pointing TCP only for z ≈ −0.06…0.10 m.

Decisions pinned here (see docs/v4/prompts/WP3.md):
* grasp: up (to max(start z, grasp z + h)) → over → down → close → dwell; in closed-loop mode a grasp that
  was started by a re-entry commands the gripper open while it moves; place / transport / lift / retreat
  keep the TCP orientation they start with.
* Group A (§7.1): only the *first execution* of the triggering step is perturbed; ``injected`` is True on
  exactly the window frames; ``wrong_target`` binds ``sorted(candidates)[rng.integers(len)]`` with
  ``rng = rng_for(spec.seed, "expert")``, candidates = the other active members of the target's set;
  ``stall`` starts on the first frame with ``progress.furthest_index ≥ index(onset_stage)``, plus
  ``delay_steps``, and holds ``q_target`` for ``steps`` frames.
"""

import math
import shutil
import unittest

import numpy as np

from core.arms.gripper import Gripper
from core.arms.kinematics import KinematicChain
from core.arms.spec import load_arm_spec
from core.expert.machine import ScriptedExpert
from core.expert.motion import MotionLimits, expert_joint_limits
from core.expert.program import parse_program
from core.generate.protocols import (EpisodeSpec, ExpertObs, Kind, ObjectState, PerturbationGroup,
                                     PerturbationSpec)
from core.generate.seeds import rng_for
from core.tasks.base import CheckContext, Pose
from core.tasks.loader import load_task
from v4_support import FIXTURES_DIR, clear_task_cache, copy_fixture, sandbox
from v4_support.fx_arms import FX_ARMS, fx_home_q
from v4_support.tasks import STUB_USDA, write_task

TABLE_Z = -0.05
STAGES = ("reached", "grasped", "lifted", "placed", "success")


# ---------------------------------------------------------------------------------------------------
# kinematic stand-in (private; FakeEnv is WP7's)

def _R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _quat(m):
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    return tuple(float(v) for v in q)


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


class _StandIn:
    """Perfect joint tracking with attach-on-close, drop-on-open and a point pusher (see module docstring)."""

    def __init__(self, chain, gripper, objects, *, approach_axis):
        self.chain, self.gripper, self.specs = chain, gripper, objects
        self.a_t = np.asarray(approach_axis, float)
        self.attached_log = []

    def dims(self, n):
        s = self.specs[n]
        return tuple(s.size) if s.size is not None else tuple(s.params.get("size_outer", (0.0, 0.0, 0.0)))

    def bottom_origin(self, n):
        s = self.specs[n]
        return s.size is None and s.params.get("origin", "center") == "bottom"

    def container(self, n):
        return "inner_half_size" in self.specs[n].params

    def centre(self, n):
        p = np.array(self.pose[n][0])
        if self.bottom_origin(n):
            p[2] += self.dims(n)[2] / 2
        return p

    def reset(self, spec):
        self.frame = 0
        self.q = np.array(spec.initial_q, dtype=float)
        self.active = {n: s.active for n, s in spec.objects.items()}
        self.pose = {n: (np.array(s.pose.position, float), tuple(s.pose.quat_xyzw)) for n, s in spec.objects.items()}
        self.colors = {n: s.color for n, s in spec.objects.items()}
        self.attached = None
        self._kin()
        return self.observe()

    def _kin(self):
        self.T_tcp = self.chain.fk(self.q, "tcp")
        self.T_ee = self.chain.fk(self.q, "ee")
        self.gap = self.gripper.gap_from_angle(self.q[self.gripper.index])

    def _T_obj(self, n):
        p, q = self.pose[n]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = _R(q), p
        return T

    def _support_z(self, n, xy):
        z = TABLE_Z
        for m, act in self.active.items():
            if not act or m == n or not self.container(m):
                continue
            p, q = self.pose[m]
            yaw = _yaw(q)
            d = xy - p[:2]
            local = np.array([math.cos(yaw) * d[0] + math.sin(yaw) * d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]])
            if np.all(np.abs(local) <= np.asarray(self.specs[m].params["inner_half_size"])):
                z = max(z, p[2] + float(self.specs[m].params.get("floor_z", 0.0)))
        return z

    def step(self, q_target):
        q_target = np.asarray(q_target, float)
        assert q_target.shape == self.q.shape and np.all(np.isfinite(q_target))
        prev_gap, prev_tcp = self.gap, self.T_tcp[:3, 3].copy()
        g = self.gripper.index
        q = np.clip(q_target, self.chain.lower, self.chain.upper)
        if self.attached is not None:
            q[g] = max(q[g], self.gripper.angle_from_gap(min(self.dims(self.attached)[:2])))
        self.q = q
        self._kin()
        if self.attached is not None:
            T = self.T_tcp @ self.rel
            self.pose[self.attached] = (T[:3, 3].copy(), _quat(T[:3, :3]))
        if self.attached is None and self.gap < prev_gap - 1e-12:
            best = None
            for n, act in self.active.items():
                if not act or self.container(n):
                    continue
                w, h = min(self.dims(n)[:2]), self.dims(n)[2]
                if self.gap > w + 0.004:
                    continue
                d = self.T_tcp[:3, :3].T @ (self.centre(n) - self.T_tcp[:3, 3])
                along = float(d @ self.a_t)
                if np.linalg.norm(d - along * self.a_t) <= w / 2 and abs(along) <= h / 2:
                    if best is None or np.linalg.norm(d) < best[0]:
                        best = (float(np.linalg.norm(d)), n)
            if best is not None:
                n = best[1]
                self.q[g] = self.gripper.angle_from_gap(min(self.dims(n)[:2]))
                self._kin()
                self.attached = n
                self.attached_log.append((self.frame + 1, n))
                self.rel = np.linalg.inv(self.T_tcp) @ self._T_obj(n)
        elif self.attached is not None and self.gap > min(self.dims(self.attached)[:2]) + 0.002:
            n, self.attached = self.attached, None
            p, q = self.pose[n]
            z = self._support_z(n, p[:2]) + (0.0 if self.bottom_origin(n) else self.dims(n)[2] / 2)
            self.pose[n] = (np.array([p[0], p[1], z]), yaw_quat(_yaw(q)))
        tcp = self.T_tcp[:3, 3]
        for n, act in self.active.items():
            if not act or n == self.attached or self.container(n):
                continue
            p, q = self.pose[n]
            yaw = _yaw(q)
            dx, dy, h = self.dims(n)
            c = self.centre(n)
            d = tcp[:2] - p[:2]
            local = np.array([math.cos(yaw) * d[0] + math.sin(yaw) * d[1], -math.sin(yaw) * d[0] + math.cos(yaw) * d[1]])
            if abs(local[0]) <= dx / 2 + 0.01 and abs(local[1]) <= dy / 2 + 0.01 and c[2] - h / 2 <= tcp[2] <= c[2] + h / 2:
                moved = p.copy()
                moved[:2] += tcp[:2] - prev_tcp[:2]
                self.pose[n] = (moved, q)
        self.frame += 1
        return self.observe()

    def observe(self):
        objects = {n: ObjectState(pose=Pose(tuple(float(v) for v in p), tuple(float(v) for v in q)),
                                  active=self.active[n], color=self.colors[n]) for n, (p, q) in self.pose.items()}
        tcp = Pose(tuple(float(v) for v in self.T_tcp[:3, 3]), _quat(self.T_tcp[:3, :3]))
        ee = Pose(tuple(float(v) for v in self.T_ee[:3, 3]), _quat(self.T_ee[:3, :3]))
        return ExpertObs(frame=self.frame, q=self.q.copy(), ee=ee, tcp=tcp, gripper_gap_m=float(self.gap),
                         objects=objects, signals={"holding": self.attached is not None})


# ---------------------------------------------------------------------------------------------------
# episode driver (what WP7's loop does, reduced to the parts the expert sees)

class Episode:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def runs(self, phase):
        """[(start_frame, length)] of consecutive non-done actions with this phase."""
        out = []
        for t, a in enumerate(self.actions):
            if a.phase == phase and not a.done:
                if out and out[-1][0] + out[-1][1] == t:
                    out[-1][1] += 1
                else:
                    out.append([t, 1])
        return [tuple(r) for r in out]

    def injected(self):
        return [t for t, a in enumerate(self.actions) if a.injected]


def build_expert(task, arm, steps=None, **top):
    table = dict(task.spec.raw["expert"]) if steps is None else {"steps": steps, **top}
    spec_arm, _ = load_arm_spec(arm)
    lo, hi = expert_joint_limits(arm)
    prog = parse_program(table, task=task, arm_spec=spec_arm, joint_limits=(lo, hi))
    ex = ScriptedExpert(prog, KinematicChain.from_arm(arm), Gripper.from_arm(arm), lower=lo, upper=hi,
                        limits=MotionLimits.from_arm(arm), fps=30, stages=task.stages)
    return ex, prog


def run_episode(task, arm, spec, steps=None, *, frames=None, **top):
    ex, prog = build_expert(task, arm, steps, **top)
    env = _StandIn(KinematicChain.from_arm(arm), Gripper.from_arm(arm), task.spec.objects,
                   approach_axis=prog.approach_axis)
    obs = env.reset(spec)
    ex.reset(spec, obs)
    tracker = task.make_tracker()
    ctx = CheckContext.from_spec(spec, task.spec)
    names = KinematicChain.from_arm(arm).q_names
    actions, observations, progresses, checks_log = [], [], [], []
    settle = 0
    for t in range(frames or prog.max_steps):
        checks = task.observe_checks(obs.step_state(names), ctx)
        prog_t = tracker.update(checks)
        act = ex.act(obs, prog_t)
        actions.append(act)
        observations.append(obs)
        progresses.append(prog_t)
        checks_log.append(checks)
        if act.done:
            settle += 1
            if settle >= prog.settle_steps and frames is None:
                break
        obs = env.step(act.q_target)
    final_checks = task.observe_checks(obs.step_state(names), ctx)
    outcome = task.outcome(tracker, final_checks, obs.step_state(names))
    return Episode(expert=ex, program=prog, env=env, actions=actions, obs=observations, progress=progresses,
                   checks=checks_log, tracker=tracker, outcome=outcome, final=obs)


def color_objects(*, n_active=2):
    pos = {0: (0.15, 0.06, 0.3), 1: (0.12, 0.13, 0.0), 2: (0.11, -0.12, 0.5)}
    objs = {}
    for i in range(6):
        if i < n_active:
            x, y, yaw = pos[i]
            objs[f"cube_{i}"] = ObjectState(Pose((x, y, TABLE_Z + 0.01), yaw_quat(yaw)), True, ("red", "blue", "green")[i])
        else:
            objs[f"cube_{i}"] = ObjectState(Pose((0.1 * i, 0.0, -2.0)), False, None)
    objs["small_box"] = ObjectState(Pose((0.17, -0.07, TABLE_Z), yaw_quat(0.2)), True, None)
    return objs


def color_spec(arm, kind="success", perturbation=None, *, n_active=2, seed=5):
    kind = Kind(kind)
    if perturbation is None and kind is not Kind.SUCCESS:
        perturbation = PerturbationSpec(type="object_shift", group=PerturbationGroup.WORLD, onset_stage="reached",
                                        params={"xy_m": 0.03, "dir_rad": 0.0})
    return EpisodeSpec(task="fx_color", arm=arm, scene="fx_scene", kind=kind, attempt_index=0, seed=seed,
                       objects=color_objects(n_active=n_active), roles={"target": "cube_0", "container": "small_box"},
                       goal={}, instruction="Pick up the red cube and place it in the box",
                       initial_q=tuple(float(v) for v in fx_home_q(arm)), perturbation=perturbation)


def reach_spec(arm, kind="success", perturbation=None, point=(0.15, 0.05, 0.03)):
    kind = Kind(kind)
    if perturbation is None and kind is not Kind.SUCCESS:
        perturbation = PerturbationSpec(type="object_shift", group=PerturbationGroup.WORLD, onset_stage="reached",
                                        params={})
    return EpisodeSpec(task="fx_reach", arm=arm, scene="fx_scene", kind=kind, attempt_index=0, seed=9, objects={},
                       roles={}, goal={"point": list(point)}, instruction="Move the gripper to the point",
                       initial_q=tuple(float(v) for v in fx_home_q(arm)), perturbation=perturbation)


def pert(type_, params=None, onset=None, delay=0):
    return PerturbationSpec(type=type_, group=PerturbationGroup.EXPERT, onset_stage=onset, delay_steps=delay,
                            params=params or {})


class _Base(unittest.TestCase):
    def setUp(self):
        self._box = sandbox()
        self.box = self._box.__enter__()
        for arm in FX_ARMS:
            copy_fixture(f"arms/{arm}", self.box.root)
        copy_fixture("scenes/fx_scene", self.box.root)
        for name in ("fx_reach", "fx_color"):
            shutil.copytree(FIXTURES_DIR / "v4_wp1" / "tasks" / name, self.box.tasks / name,
                            ignore=shutil.ignore_patterns("__pycache__"))
        clear_task_cache()

    def tearDown(self):
        self._box.__exit__(None, None, None)

    def task(self, name):
        return load_task(name)


# ---------------------------------------------------------------------------------------------------

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


class SkillTerminationTests(_Base):
    """WP3 §6.1 / §6.2 every skill terminates within its budget on a kinematic stand-in that tracks targets
    exactly; a skill that cannot finish ends after exactly ``budget`` actions (timeout) and the next step
    (or done) follows."""

    def setUp(self):
        super().setUp()
        write_task(self.box.tasks, "fx_wp3_tool", tool_task_table(), files={"usd/box.usda": STUB_USDA})
        clear_task_cache("fx_wp3_tool")
        self.tool = load_task("fx_wp3_tool")

    def spec(self, arm, *, goal=None, cube=(0.15, 0.03, 0.3)):
        objs = {"cube": ObjectState(Pose((cube[0], cube[1], TABLE_Z + 0.01), yaw_quat(cube[2])), True, "red"),
                "cube_b": ObjectState(Pose((0.11, -0.12, TABLE_Z + 0.01)), True, "blue"),
                "box": ObjectState(Pose((0.17, -0.06, TABLE_Z)), True, None)}
        g = {"point": [0.16, -0.02, 0.04], "joint": {"joint": None, "angles": []}, "gap": 0.03,
             "area": {"xy_min": [0.14, -0.03], "xy_max": [0.16, -0.01]}}
        g.update(goal or {})
        return EpisodeSpec(task="fx_wp3_tool", arm=arm, scene="fx_scene", kind=Kind.SUCCESS, attempt_index=0, seed=1,
                           objects=objs, roles={"target": "cube", "container": "box"}, goal=g, instruction="x",
                           initial_q=tuple(float(v) for v in fx_home_q(arm)))

    def assert_completed(self, ep, name, budget):
        runs = ep.runs(name)
        self.assertEqual(len(runs), 1, (name, runs))
        start, n = runs[0]
        self.assertLess(n, budget, f"{name} used its whole budget ({n})")
        return start, n

    def test_pick_place_chain(self):
        """§6.1 approach/grasp/lift/transport/place/release/retreat each finish before their budget and leave the
        state their termination rule names; approach aligns the closing axis with the object yaw (mod π/2)."""
        steps = [{"skill": "set_gripper", "gap_m": "open"}, {"skill": "approach", "target": "@target"},
                 {"skill": "grasp", "target": "@target"}, {"skill": "lift"},
                 {"skill": "transport", "target": "@container"}, {"skill": "place", "target": "@container"},
                 {"skill": "release"}, {"skill": "retreat"}]
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                ep = run_episode(self.tool, arm, self.spec(arm), steps)
                self.assertTrue(ep.actions[-1].done)
                budgets = {"set_gripper": 60, "approach": 200, "grasp": 150, "lift": 120, "transport": 250,
                           "place": 150, "release": 60, "retreat": 120}
                ends = {}
                for name, budget in budgets.items():
                    start, n = self.assert_completed(ep, name, budget)
                    ends[name] = (start, start + n)      # obs[start + n] is the observation that ended the step
                tcp = lambda t: np.array(ep.obs[t].tcp.position)  # noqa: E731
                cube0 = np.array(self.spec(arm).objects["cube"].pose.position)
                box = np.array(self.spec(arm).objects["box"].pose.position)
                self.assertLess(abs(ep.obs[ends["set_gripper"][1]].gripper_gap_m - 0.048), 0.002 + 1e-9)
                t = ends["approach"][1]
                self.assertLess(np.linalg.norm(tcp(t) - (cube0 + [0, 0, 0.06])), 0.005 + 1e-9)
                c = _R(ep.obs[t].tcp.quat_xyzw) @ np.asarray(ep.program.closing_axis)
                diff = (math.atan2(c[1], c[0]) - 0.3) % (math.pi / 2)
                self.assertLess(min(diff, math.pi / 2 - diff), 0.02)
                self.assertTrue(ep.obs[ends["grasp"][1]].signals["holding"])
                self.assertEqual(ep.env.attached_log[0][1], "cube")
                t0, t1 = ends["lift"]
                self.assertLess(abs(tcp(t1)[2] - tcp(t0)[2] - 0.08), 0.005 + 1e-9)
                t = ends["transport"][1]
                self.assertLess(np.linalg.norm(tcp(t) - (box[:2].tolist() + [box[2] + 0.035 + 0.06])), 0.005 + 1e-9)
                t = ends["place"][1]
                self.assertLess(np.linalg.norm(tcp(t) - (box[:2].tolist() + [box[2] + 0.035 + 0.01 + 0.005])),
                                0.003 + 1e-9)
                t = ends["release"][1]
                self.assertGreaterEqual(ep.obs[t].gripper_gap_m, 0.048 - 0.002)
                self.assertFalse(ep.obs[t].signals["holding"])
                cube = np.array(ep.obs[t].objects["cube"].pose.position)
                self.assertLess(np.linalg.norm(cube[:2] - box[:2]), 0.01)
                self.assertAlmostEqual(cube[2], box[2] + 0.002 + 0.01, places=6)     # dropped onto the box floor
                t0, t1 = ends["retreat"]
                self.assertLess(abs(tcp(t1)[2] - tcp(t0)[2] - 0.08), 0.005 + 1e-9)

    def test_joint_gripper_reach_wait(self):
        """§6.1 move_joint, set_gripper (target and blocked), reach (point and @goal), sweep, wait, touch."""
        for arm in FX_ARMS:
            names = KinematicChain.from_arm(arm).q_names
            j0, j1 = names[0], names[1]
            steps = [{"skill": "move_joint", "q": {j0: 0.4, j1: 0.2}, "name": "mj"},
                     {"skill": "move_joint", "q": "home"},
                     {"skill": "set_gripper", "gap_m": "@goal.gap", "name": "gap"},
                     {"skill": "reach", "target": "@goal.point"},
                     {"skill": "sweep", "joint": "@goal.joint", "angles": "@goal.joint.angles"},
                     {"skill": "touch", "target": "@target"},
                     {"skill": "wait", "steps": 7}]
            goal = {"joint": {"joint": names[-2], "angles": [0.3, -0.2, 0.1]}}
            with self.subTest(arm=arm):
                ep = run_episode(self.tool, arm, self.spec(arm, goal=goal), steps)
                self.assertTrue(ep.actions[-1].done)
                s, n = self.assert_completed(ep, "mj", 150)
                q_end = ep.obs[s + n].q
                self.assertLessEqual(max(abs(q_end[0] - 0.4), abs(q_end[1] - 0.2)), 0.01)
                s, n = self.assert_completed(ep, "gap", 60)
                self.assertLessEqual(abs(ep.obs[s + n].gripper_gap_m - 0.03), 0.002)
                s, n = self.assert_completed(ep, "reach", 200)
                self.assertLessEqual(np.linalg.norm(np.array(ep.obs[s + n].tcp.position) - [0.16, -0.02, 0.04]), 0.005)
                s, n = self.assert_completed(ep, "sweep", 300)
                jq = [o.q[len(names) - 2] for o in ep.obs[s:s + n + 1]]
                hits = [next(i for i, v in enumerate(jq) if abs(v - a) <= 0.01) for a in (0.3, -0.2)]
                self.assertLess(hits[0], hits[1])
                self.assertLessEqual(abs(jq[-1] - 0.1), 0.01)
                s, n = self.assert_completed(ep, "touch", 200)
                top = np.array(self.spec(arm).objects["cube"].pose.position) + [0, 0, 0.01]
                self.assertLessEqual(np.linalg.norm(np.array(ep.obs[s + n].tcp.position) - top), 0.004)
                self.assertEqual(ep.runs("wait"), [(ep.runs("wait")[0][0], 7)])

    def test_set_gripper_blocked(self):
        """§6.1 set_gripper ends when the gap changed < 1e-4 m for 5 steps (jaws blocked by a held object)."""
        arm = "fx_arm5"
        steps = [{"skill": "set_gripper", "gap_m": "open"}, {"skill": "approach", "target": "@target"},
                 {"skill": "grasp", "target": "@target", "close_gap_m": 0.012},
                 {"skill": "set_gripper", "gap_m": "closed", "name": "squeeze"}]
        ep = run_episode(self.tool, arm, self.spec(arm), steps)
        s, n = self.assert_completed(ep, "squeeze", 60)
        self.assertLessEqual(n, 7)
        self.assertAlmostEqual(ep.obs[s + n].gripper_gap_m, 0.02, places=6)
        self.assertTrue(ep.actions[-1].done)

    def test_push(self):
        """§6.1 push ends as soon as the target object is inside the goal region."""
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                steps = [{"skill": "push", "target": "@target", "goal": "@goal.area"}]
                ep = run_episode(self.tool, arm, self.spec(arm, cube=(0.15, 0.03, 0.0)), steps)
                s, n = self.assert_completed(ep, "push", 300)
                p = ep.obs[s + n].objects["cube"].pose.position
                self.assertTrue(0.14 <= p[0] <= 0.16 and -0.03 <= p[1] <= -0.01, p)
                p_before = ep.obs[s + n - 1].objects["cube"].pose.position
                self.assertFalse(0.14 <= p_before[0] <= 0.16 and -0.03 <= p_before[1] <= -0.01)

    def test_timeouts(self):
        """§6.2 a step that cannot finish emits exactly ``budget`` actions, then the next step starts."""
        arm = "fx_arm6"
        names = KinematicChain.from_arm(arm).q_names
        cases = {
            "move_joint": {"skill": "move_joint", "q": {names[0]: 1.0}, "budget": 3},
            "set_gripper": {"skill": "set_gripper", "gap_m": "closed", "budget": 2},
            "reach": {"skill": "reach", "target": [0.6, 0.0, 0.4], "budget": 6},
            "approach": {"skill": "approach", "target": "@target", "budget": 4},
            "grasp": {"skill": "grasp", "target": "@target", "budget": 5},
            "lift": {"skill": "lift", "height_m": 0.5, "budget": 4},
            "transport": {"skill": "transport", "target": "@container", "budget": 3},
            "place": {"skill": "place", "target": "@container", "budget": 3},
            "release": {"skill": "release", "budget": 2},
            "retreat": {"skill": "retreat", "dz_m": 0.5, "budget": 3},
            "push": {"skill": "push", "target": "@target", "goal": "@goal.area", "budget": 4},
            "touch": {"skill": "touch", "target": "@target", "budget": 3},
            "sweep": {"skill": "sweep", "joint": names[0], "angles": [1.0, -1.0], "budget": 5},
            "wait": {"skill": "wait", "steps": 9, "budget": 4},
        }
        for skill, step in cases.items():
            with self.subTest(skill=skill):
                steps = [{"skill": "set_gripper", "gap_m": "closed", "budget": 1, "name": "pre"}, step,
                         {"skill": "wait", "steps": 2, "name": "post"}]
                ep = run_episode(self.tool, arm, self.spec(arm), steps, frames=step["budget"] + 6)
                self.assertEqual(ep.runs(skill), [(1, step["budget"])])
                self.assertEqual(ep.actions[1 + step["budget"]].phase, "post")
                done = [t for t, a in enumerate(ep.actions) if a.done]
                self.assertEqual(done[0], 1 + step["budget"] + 2)


class ContractProgramEpisodeTests(_Base):
    """WP3 §4.6 / §6 the contract example programs complete all stages on the stand-in, both arms."""

    def _check_complete(self, ep, task):
        self.assertTrue(ep.actions[-1].done)
        self.assertEqual(ep.expert.reentries, [])
        self.assertEqual(ep.tracker.furthest_stage, task.stages[-1])
        self.assertTrue(ep.outcome.success, ep.outcome)
        idx = [p.stage_index for p in ep.progress]
        self.assertEqual(idx, sorted(idx), "stages must be reached in order without regressions")
        self.assertEqual(idx[-1], len(task.stages) - 1)
        self.assertLess(len(ep.actions), ep.program.max_steps)

    def test_fx_reach(self):
        """§4.6 prim_reach_point program: closed-loop (success) and blind (failure kind) both complete."""
        task = self.task("fx_reach")
        for arm in FX_ARMS:
            for point in ((0.15, 0.05, 0.03), (0.13, -0.06, 0.07)):
                for kind in ("success", "failure"):
                    with self.subTest(arm=arm, point=point, kind=kind):
                        ep = run_episode(task, arm, reach_spec(arm, kind, point=point))
                        self._check_complete(ep, task)
                        self.assertEqual([r[1] for r in ep.runs("wait")], [10])

    def test_fx_color(self):
        """§4.6 pick_place_color_cube program: closed-loop (success) and blind (failure kind) both complete."""
        task = self.task("fx_color")
        for arm in FX_ARMS:
            for kind in ("success", "failure"):
                with self.subTest(arm=arm, kind=kind):
                    ep = run_episode(task, arm, color_spec(arm, kind))
                    self._check_complete(ep, task)
                    self.assertEqual([a.phase for a in ep.actions if not a.done][0], "move_joint")
                    order = []
                    for a in ep.actions:
                        if not order or order[-1] != a.phase:
                            order.append(a.phase)
                    self.assertEqual(order, [s.name for s in ep.program.steps])
                    self.assertFalse(any(a.injected for a in ep.actions))


class GroupAPerturbationTests(_Base):
    """WP3 §7.1 group-A perturbations executed inside ScriptedExpert: windows (``injected``) and effects; blind
    (failure) keeps the error, closed loop (recovery) re-enters and succeeds."""

    def setUp(self):
        super().setUp()
        self.color_task = self.task("fx_color")
        self.arm = "fx_arm5"

    def color(self, kind, p, **kw):
        return run_episode(self.color_task, self.arm, color_spec(self.arm, kind, p, **kw))

    def assert_recovers(self, ep, onset):
        self.assertTrue(ep.outcome.success, ep.outcome)
        self.assertGreaterEqual(len(ep.expert.reentries), 1)
        self.assertGreaterEqual(ep.expert.reentries[0][0], onset)
        self.assertEqual([t for t, a in enumerate(ep.actions) if a.reentered], [t for t, _ in ep.expert.reentries])

    def test_grasp_offset(self):
        """grasp_offset: injected = the first grasp execution; its grasp point is shifted by xy_m·(cos, sin)."""
        p = pert("grasp_offset", {"xy_m": 0.015, "dir_rad": 0.5})
        ep = self.color("failure", p)
        (s, n), = ep.runs("grasp")
        self.assertEqual(ep.injected(), list(range(s, s + n)))
        self.assertFalse(ep.outcome.success)
        cube = np.array(color_objects()["cube_0"].pose.position)
        tcp = np.array(ep.obs[s + n - 1].tcp.position)
        self.assertLess(np.linalg.norm(tcp[:2] - cube[:2] - 0.015 * np.array([math.cos(0.5), math.sin(0.5)])), 0.005)
        ep = self.color("recovery", p)
        s = ep.runs("grasp")[0][0]
        first = ep.expert.reentries[0][0]          # the re-entered grasp follows directly and is not perturbed
        self.assertEqual(ep.actions[first].phase, "grasp")
        self.assertEqual(ep.injected(), list(range(s, first)))
        self.assert_recovers(ep, s)

    def test_partial_close(self):
        """partial_close: the first grasp closes only to w + extra_gap_m and never holds the cube (blind)."""
        p = pert("partial_close", {"extra_gap_m": 0.008})
        ep = self.color("failure", p)
        (s, n), = ep.runs("grasp")
        self.assertEqual(ep.injected(), list(range(s, s + n)))
        self.assertAlmostEqual(min(o.gripper_gap_m for o in ep.obs[s:s + n + 1]), 0.028, delta=0.002 + 1e-9)
        self.assertEqual(ep.env.attached_log, [])
        self.assertFalse(ep.outcome.success)
        self.assert_recovers(self.color("recovery", p), s)

    def test_early_close(self):
        """early_close: from (first approach start + delay_steps) to the approach end the gripper closes."""
        p = pert("early_close", {}, delay=5)
        ep = self.color("failure", p)
        (s, n), = ep.runs("approach")
        self.assertEqual(ep.injected(), list(range(s + 5, s + n)))
        self.assertLess(ep.obs[s + n].gripper_gap_m, 0.003)
        self.assertFalse(ep.outcome.success)
        self.assert_recovers(self.color("recovery", p), s + 5)

    def test_early_release(self):
        """early_release: from (first transport start + delay_steps) until the gap is open (or the step ends)."""
        p = pert("early_release", {}, delay=4)
        ep = self.color("failure", p)
        (s, n), = ep.runs("transport")
        inj = ep.injected()
        self.assertTrue(inj)
        self.assertEqual(inj, list(range(s + 4, inj[-1] + 1)))
        end = inj[-1] + 1
        self.assertGreaterEqual(ep.obs[end].gripper_gap_m, 0.048 - 0.002)
        self.assertTrue(all(o.gripper_gap_m < 0.048 - 0.002 for o in ep.obs[s + 4:end]))
        self.assertFalse(ep.obs[end].signals["holding"])
        self.assertFalse(ep.outcome.success)
        self.assert_recovers(self.color("recovery", p), s + 4)

    def test_place_offset(self):
        """place_offset: injected over the first place and the release that follows; the cube lands outside."""
        p = pert("place_offset", {"xy_m": 0.04, "dir_rad": 1.0})
        ep = self.color("failure", p)
        (s, n), = ep.runs("place")
        (rs, rn), = ep.runs("release")
        self.assertEqual(rs, s + n)
        self.assertEqual(ep.injected(), list(range(s, rs + rn)))
        box = np.array(color_objects()["small_box"].pose.position)
        tcp = np.array(ep.obs[s + n].tcp.position)
        self.assertLess(np.linalg.norm(tcp[:2] - box[:2] - 0.04 * np.array([math.cos(1.0), math.sin(1.0)])), 0.003 + 1e-9)
        self.assertFalse(ep.outcome.success)
        self.assert_recovers(self.color("recovery", p), s)

    def test_wrong_target(self):
        """wrong_target: the expert binds a distractor drawn with rng_for(seed, "expert") over the sorted other
        active members of the target's set; injected from frame 0 until the first re-entry (or the end)."""
        p = pert("wrong_target")
        for seed in (5, 6, 7, 8):
            with self.subTest(seed=seed):
                expected = ["cube_1", "cube_2"][int(rng_for(seed, "expert").integers(2))]
                ep = self.color("failure", p, n_active=3, seed=seed)
                self.assertEqual(ep.injected(), list(range(len(ep.actions))))
                self.assertEqual(ep.env.attached_log[0][1], expected)
                self.assertFalse(ep.outcome.success)
        ep = self.color("recovery", p, n_active=3, seed=5)
        first = ep.expert.reentries[0][0]
        self.assertEqual(ep.injected(), list(range(first)))
        self.assert_recovers(ep, 0)

    def test_wrong_target_not_applicable(self):
        """wrong_target without another active member of the target's set is never applied (no injected frame)."""
        ep = self.color("failure", pert("wrong_target"), n_active=1)
        self.assertEqual(ep.injected(), [])
        self.assertTrue(ep.outcome.success)

    def test_target_offset_reach(self):
        """target_offset: the first execution of the first stage-tagged step with a target (reach) aims at
        goal + (xy_m·(cos, sin), z_m); blind stops there, closed loop re-enters and reaches the goal."""
        task = self.task("fx_reach")
        p = pert("target_offset", {"xy_m": 0.04, "dir_rad": 2.0, "z_m": 0.01})
        for arm in FX_ARMS:
            with self.subTest(arm=arm):
                ep = run_episode(task, arm, reach_spec(arm, "failure", p))
                (s, n), = ep.runs("reach")
                self.assertEqual(ep.injected(), list(range(s, s + n)))
                aim = np.array([0.15, 0.05, 0.03]) + [0.04 * math.cos(2.0), 0.04 * math.sin(2.0), 0.01]
                self.assertLessEqual(np.linalg.norm(np.array(ep.obs[s + n].tcp.position) - aim), 0.005)
                self.assertFalse(ep.outcome.success)
                ep = run_episode(task, arm, reach_spec(arm, "recovery", p))
                s, n = ep.runs("reach")[0]
                self.assertEqual(ep.injected(), list(range(s, s + n)))
                self.assert_recovers(ep, s)

    def test_stall(self):
        """stall: from the first frame with furthest_index ≥ index(onset_stage), + delay_steps, q_target is held
        for ``steps`` frames (injected exactly there); nothing else happens in that window."""
        task = self.task("fx_reach")
        arm = "fx_arm6"
        p = pert("stall", {"steps": 15}, onset="reached", delay=3)
        ep = run_episode(task, arm, reach_spec(arm, "failure", p))
        latched = next(t for t, pr in enumerate(ep.progress) if pr.furthest_index >= 0)
        self.assertEqual(ep.injected(), list(range(latched + 3, latched + 18)))
        held = ep.actions[latched + 2].q_target
        for a in ep.actions[latched + 3:latched + 18]:
            np.testing.assert_array_equal(a.q_target, held)

    def test_world_perturbation_not_injected(self):
        """§7.1 group B is the WorldPerturber's job: the expert never flags injected frames for it."""
        p = PerturbationSpec(type="container_shift", group=PerturbationGroup.WORLD, onset_stage="lifted",
                             params={"xy_m": 0.03, "dir_rad": 1.0})
        ep = self.color("recovery", p)
        self.assertEqual(ep.injected(), [])
        self.assertTrue(ep.outcome.success)


if __name__ == "__main__":
    unittest.main()
