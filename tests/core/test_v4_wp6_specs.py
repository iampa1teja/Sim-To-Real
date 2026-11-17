"""WP6: ``core.generate.specs.build_spec`` — EpisodeSpec construction from the sub-seed streams
(contracts §7.3, §1.4 sub-seeds, §3.2 EpisodeSpec invariants, §4.5, §8.5 NoiseConfig, docs/v4/prompts/WP6.md)."""

import types
import unittest

import numpy as np

from core.generate.errors import SamplingError
from core.generate.perturbations import draw_perturbation
from core.generate.protocols import EpisodeSpec, Kind, NoiseSpec
from core.generate.seeds import rng_for
from core.tasks.base import load_task_spec
from core.tasks.generic import GenericTask
from core.generate.specs import build_spec
from v4_support import assert_frozen, sandbox
from v4_support.tasks import STUB_USDA, contract_color_task_table, contract_reach_task_table, write_task

HOME = (0.0, -0.3, 1.6, 1.2, 0.0, 0.5)
NAMES = ("j1", "j2", "j3", "j4", "j5", "jaw")


def noise_cfg(fraction=0.5, theta=0.15, sigma_rad=0.01, max_abs_rad=0.02):
    """Duck-typed ``config.NoiseConfig`` (§8.5): WP6 may not import WP5, it reads the four attributes."""
    return types.SimpleNamespace(fraction=fraction, theta=theta, sigma_rad=sigma_rad, max_abs_rad=max_abs_rad)


def load_tasks(tables):
    out = {}
    with sandbox() as box:
        for name, table in tables.items():
            write_task(box.tasks, name, table, files={"usd/Small-Box-Physics.usda": STUB_USDA})
            out[name] = GenericTask(load_task_spec(name))
    return out


def spec(task, kind, index=0, seed=7, **kw):
    kw.setdefault("arm", "fx_arm5")
    kw.setdefault("scene", "fx_scene")
    kw.setdefault("home_q", HOME)
    kw.setdefault("joint_names", NAMES)
    return build_spec(task, kind, index, seed, **kw)


class BuildSpecTests(unittest.TestCase):
    """WP6 §7.3 ``build_spec(task, kind, attempt_index, seed, *, arm, scene, home_q, reach_fn=None, noise=None,
    weights_override=None, fps=30, joint_names=None, joint_limits=None, arm_joints=None)``."""

    @classmethod
    def setUpClass(cls):
        reach = contract_reach_task_table()
        reach["expert"]["max_steps"] = 250
        small = contract_color_task_table()
        small["start"]["count"] = [1, 2]
        small["holdout"]["counts"] = []
        tasks = load_tasks({"fx6_spec_color": contract_color_task_table(), "fx6_spec_reach": reach,
                            "fx6_spec_small": small})
        cls.color, cls.reach, cls.small = tasks["fx6_spec_color"], tasks["fx6_spec_reach"], tasks["fx6_spec_small"]

    def test_fields_and_streams(self):
        """Start from ``rng_for(seed, "start")`` with the instruction from ``"instruction"``, the perturbation from
        ``"perturbation"`` (with ``start=`` the sampled start); identity fields copied; frozen spec."""
        for task in (self.color, self.reach):
            for kind in Kind:
                for seed in (0, 7, 2**64 - 1):
                    s = spec(task, kind, index=4, seed=seed, fps=15)
                    start = task.sample_episode_start(rng_for(seed, "start"), home_q=HOME,
                                                      instruction_rng=rng_for(seed, "instruction"),
                                                      joint_names=NAMES)
                    with self.subTest(task=task.name, kind=kind.value, seed=seed):
                        self.assertIsInstance(s, EpisodeSpec)
                        assert_frozen(s)
                        self.assertEqual((s.task, s.arm, s.scene, s.kind, s.attempt_index, s.seed, s.fps),
                                         (task.name, "fx_arm5", "fx_scene", kind, 4, seed, 15))
                        self.assertEqual(dict(s.objects), dict(start.objects))
                        self.assertEqual(dict(s.roles), dict(start.roles))
                        self.assertEqual(s.goal, start.goal)
                        self.assertEqual(s.instruction, start.instruction)
                        self.assertEqual(s.held_object, start.held_object)
                        self.assertEqual(s.initial_q, HOME)
                        self.assertEqual(s.perturbation, draw_perturbation(task, kind, rng_for(seed, "perturbation"),
                                                                           start=start))

    def test_kind_invariants(self):
        """§3.2: success ⇔ no perturbation, for many seeds and every kind; failure/recovery never get noise."""
        for seed in range(60):
            for kind in Kind:
                s = spec(self.color, kind, seed=seed, noise=noise_cfg(fraction=1.0))
                with self.subTest(seed=seed, kind=kind.value):
                    self.assertEqual(s.perturbation is None, kind is Kind.SUCCESS)
                    if kind is not Kind.SUCCESS:
                        self.assertIsNone(s.noise)

    def test_noise_decision(self):
        """§7.3 noise: ``rng_for(seed, "noise").random() < fraction`` → ``NoiseSpec(theta, sigma_rad, max_abs_rad)``;
        fraction 0 → never, 1 → always, ``noise=None`` → never."""
        cfg = noise_cfg(fraction=0.5, theta=0.2, sigma_rad=0.03, max_abs_rad=0.04)
        hits = 0
        for seed in range(100):
            s = spec(self.color, Kind.SUCCESS, seed=seed, noise=cfg)
            expected = rng_for(seed, "noise").random() < 0.5
            hits += expected
            with self.subTest(seed=seed):
                if expected:
                    self.assertEqual(s.noise, NoiseSpec(theta=0.2, sigma_rad=0.03, max_abs_rad=0.04))
                else:
                    self.assertIsNone(s.noise)
                self.assertIsNone(spec(self.color, Kind.SUCCESS, seed=seed, noise=noise_cfg(fraction=0.0)).noise)
                self.assertIsNotNone(spec(self.color, Kind.SUCCESS, seed=seed, noise=noise_cfg(fraction=1.0)).noise)
                self.assertIsNone(spec(self.color, Kind.SUCCESS, seed=seed).noise)
        self.assertTrue(30 < hits < 70)

    def test_streams_are_independent(self):
        """§1.4 adding noise draws does not change the start or the perturbation (separate sub-seeds)."""
        for seed in range(20):
            a = spec(self.color, Kind.SUCCESS, seed=seed, noise=noise_cfg(fraction=1.0))
            b = spec(self.color, Kind.SUCCESS, seed=seed)
            self.assertEqual(dict(a.objects), dict(b.objects))
            self.assertEqual(a.instruction, b.instruction)
            f1 = spec(self.color, Kind.FAILURE, seed=seed)
            self.assertEqual(dict(f1.objects), dict(b.objects), "the start does not depend on the kind")

    def test_deterministic(self):
        for seed in range(20):
            for kind in Kind:
                a = spec(self.color, kind, seed=seed, noise=noise_cfg())
                b = spec(self.color, kind, seed=seed, noise=noise_cfg())
                self.assertEqual(a.to_json(), b.to_json())

    def test_max_steps_and_defaults(self):
        """``max_steps`` = ``[task.expert].max_steps`` (600 when absent, §6.1); fps default 30."""
        self.assertEqual(spec(self.color, Kind.SUCCESS).max_steps, 600)
        self.assertEqual(spec(self.reach, Kind.FAILURE).max_steps, 250)
        self.assertEqual(spec(self.reach, Kind.FAILURE).fps, 30)

    def test_weights_override_is_passed(self):
        only_place = {"grasp_offset": 0, "early_release": 0, "wrong_target": 0, "object_drop": 0}
        for seed in range(15):
            self.assertEqual(spec(self.color, Kind.FAILURE, seed=seed, weights_override=only_place).perturbation.type,
                             "place_offset")

    def test_wrong_target_follows_the_start(self):
        """Episodes whose start has a single active cube never carry wrong_target; two cubes may."""
        seen = {1: set(), 2: set()}
        for seed in range(250):
            s = spec(self.small, Kind.FAILURE, seed=seed)
            active = sum(1 for n, o in s.objects.items() if o.active and n.startswith("cube_"))
            seen[active].add(s.perturbation.type)
        self.assertNotIn("wrong_target", seen[1])
        self.assertIn("wrong_target", seen[2])

    def test_sampling_error_propagates(self):
        """§7.3 'raises SamplingError' when the start cannot be sampled (here reach_fn always false)."""
        with self.assertRaises(SamplingError):
            spec(self.color, Kind.SUCCESS, reach_fn=lambda pose: False)


if __name__ == "__main__":
    unittest.main()
