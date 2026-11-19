"""The ordered colour roles, outcomes, and recording pipeline for stack_two_cubes."""

from collections import Counter
from itertools import permutations
import re
import unittest

from core.generate.fake_env import FakeEnv
from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
from core.generate.plan import attempt_seed
from core.generate.protocols import AttemptStatus, Kind
from core.tasks.base import CheckContext, Pose, StepState
from core.tasks.loader import load_task
from v4_support import harness as H, sandbox
from v4_support.stubs import ListWriter


ARM = "fx_arm5l"
TASK = "stack_two_cubes"
COLORS = {"red", "blue", "yellow"}


class StackColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        box = cls.enterClassContext(sandbox())
        H.prepare_sandbox(box, (TASK,))
        cls.work = box.root
        cls.task = load_task(TASK)
        cls.rig = H.Rig(ARM)
        cls.sample = staticmethod(make_spec_fn(cls.task, ARM, scene=H.SCENE))

    def spec(self, index):
        return self.sample(Kind.SUCCESS, index, attempt_seed(0, TASK, Kind.SUCCESS, index))

    def test_pairs_counts_and_instructions(self):
        pairs = Counter()
        counts = set()
        for index in range(300):
            spec = self.spec(index)
            target, base = (spec.roles[role] for role in ("target", "base"))
            active = {name: state.color for name, state in spec.objects.items() if state.active}
            pair = active[target], active[base]
            with self.subTest(index=index):
                self.assertNotEqual(target, base)
                self.assertEqual(len(set(active.values())), len(active))
                self.assertLessEqual(set(active.values()), COLORS)
                self.assertEqual(re.findall(r"\b(?:red|blue|yellow)\b", spec.instruction), list(pair))
            counts.add(len(active))
            pairs[pair] += 1
        self.assertEqual(counts, {2, 3})
        self.assertEqual(set(pairs), set(permutations(COLORS, 2)))
        self.assertGreaterEqual(min(pairs.values()), 30, pairs)

    def test_repeatable_specs(self):
        for index in range(12):
            with self.subTest(index=index):
                self.assertEqual(self.spec(index), self.spec(index))

    def test_wrong_cube_in_either_position(self):
        spec = next(spec for spec in map(self.spec, range(300)) if sum(obj.active for obj in spec.objects.values()) == 3)
        ctx = CheckContext.from_spec(spec, self.task.spec)
        target, base = (spec.roles[role] for role in ("target", "base"))
        distractor, = ctx.active - {target, base}
        far_tcp = Pose((0, 0, 0.4))
        for top, support, wrong in ((distractor, base, True), (target, distractor, True), (target, base, False)):
            with self.subTest(top=top, support=support):
                poses = dict(ctx.start)
                x, y, z = poses[support].position
                poses[top] = Pose((x, y, z + (ctx.objects[top].size[2] + ctx.objects[support].size[2]) / 2))
                state = StepState(poses, far_tcp, 0.05, {})
                checks = self.task.observe_checks(state, ctx)
                self.assertEqual(checks["wrong_object_stacked"], wrong)
                self.assertEqual(checks["on_base"], not wrong)
                tracker = self.task.make_tracker()
                tracker.update(checks)
                self.assertEqual(self.task.outcome(tracker, checks, state).failure_mode == "wrong_object_stacked", wrong)

    def test_nominal_expert_success(self):
        factory = make_expert_factory(self.task, ARM)
        accepted = 0
        for index in range(50):
            spec = self.spec(index)
            env = FakeEnv(self.task, self.rig.chain, self.rig.gripper)
            result, _ = run_episode(spec, self.task, env, factory(spec), ListWriter(), self.rig.mapping)
            accepted += result.status is AttemptStatus.ACCEPTED
        self.assertGreaterEqual(accepted, 45, f"{accepted}/50")

    def test_no_duplicates(self):
        result = H.run_no_duplicates_suite(self, (TASK,), work=self.work / "stack-colors-gate", arm=ARM,
                                           seed=0, max_attempt_factor=10.0)
        self.assertEqual(set(result), {TASK})
