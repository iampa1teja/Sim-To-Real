"""WP0: ``core.generate.seeds`` (contracts §1.4, §2, §7.3)."""

import hashlib
import unittest

import numpy as np

from core.generate.seeds import derive_seed, rng_for


def _reference(*parts):
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big")


class DeriveSeedTests(unittest.TestCase):
    """WP0 §1.4 seeds."""

    def test_golden_values(self):
        """§1.4 derive_seed golden values."""
        self.assertEqual(derive_seed(0, "prim_reach_point", "success", 0), 11062146242513595739)
        self.assertEqual(derive_seed(1984, 3, "red"), 7801958555027104258)

    def test_matches_formula_for_random_parts(self):
        """§1.4 derive_seed = first 8 big-endian bytes of sha256(':'.join(str(p)))."""
        for k in range(5):
            rng = np.random.default_rng(k)
            for _ in range(40):
                parts = []
                for _ in range(int(rng.integers(1, 6))):
                    choice = int(rng.integers(0, 3))
                    parts.append(int(rng.integers(0, 2**63)) if choice == 0 else
                                 f"p{int(rng.integers(0, 10**6))}" if choice == 1 else int(rng.integers(-50, 50)))
                with self.subTest(k=k, parts=parts):
                    self.assertEqual(derive_seed(*parts), _reference(*parts))

    def test_returns_python_int_in_uint64_range(self):
        """§1.3/§1.4 seeds are Python int in [0, 2**64)."""
        for parts in [(0,), ("a", "b"), (2**64 - 1, "x", 7), ("",)]:
            with self.subTest(parts=parts):
                value = derive_seed(*parts)
                self.assertIs(type(value), int)
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, 2**64)

    def test_equals_dr_variants_episode_seed(self):
        """§1.4 dr_variants.episode_seed(seed, src, color) == derive_seed(seed, src, color)."""
        from v4_support import REPO_ROOT, temp_env
        with temp_env(S2R_SCENE_DIR=REPO_ROOT / "scenes" / "rpd_meeting_room"):   # dr_variants reads the setup at import
            from sim.generate.dr_variants import episode_seed
        for seed, src, color in [(1984, 3, "red"), (0, 0, "blue"), (7, 123, "green")]:
            with self.subTest(seed=seed, src=src, color=color):
                self.assertEqual(derive_seed(seed, src, color), episode_seed(seed, src, color))

    def test_attempt_and_sub_seed_composition_is_order_sensitive(self):
        """§1.4 parts are positional: a different order gives a different seed."""
        self.assertNotEqual(derive_seed(1, 2), derive_seed(2, 1))
        attempt = derive_seed(0, "fx_task", "failure", 5)
        purposes = {p: derive_seed(attempt, p) for p in ("start", "instruction", "perturbation", "noise", "expert")}
        self.assertEqual(len(set(purposes.values())), 5)


class RngForTests(unittest.TestCase):
    """WP0 §7.3 rng_for."""

    def test_rng_for_is_default_rng_of_derived_seed(self):
        """§7.3 rng_for(seed, purpose) = np.random.default_rng(derive_seed(seed, purpose))."""
        for seed in (0, 1, 2**64 - 1, 11062146242513595739):
            for purpose in ("start", "instruction", "perturbation", "noise", "expert"):
                with self.subTest(seed=seed, purpose=purpose):
                    rng = rng_for(seed, purpose)
                    self.assertIsInstance(rng, np.random.Generator)
                    expected = np.random.default_rng(derive_seed(seed, purpose))
                    np.testing.assert_array_equal(rng.random(8), expected.random(8))
                    self.assertEqual(int(rng.integers(0, 2**62)), int(expected.integers(0, 2**62)))

    def test_purposes_are_independent_streams(self):
        """§1.4 each purpose has its own Generator; draws on one never change another."""
        a = rng_for(42, "start").random(4)
        noise = rng_for(42, "noise")
        noise.random(1000)
        self.assertFalse(np.array_equal(a, rng_for(42, "noise").random(4)))
        np.testing.assert_array_equal(rng_for(42, "start").random(4), a)

    def test_does_not_touch_global_numpy_state(self):
        """§1.4 no global np.random use."""
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        rng_for(5, "start").random(3)
        derive_seed(5, "start")
        np.testing.assert_array_equal(np.random.get_state()[1], before)


if __name__ == "__main__":
    unittest.main()
