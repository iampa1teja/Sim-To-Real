"""WP5: ``core.generate.config`` — configs/generate/*.yaml (contracts §8.5, §1.5, §1.6)."""

import dataclasses
import tempfile
import types
import unittest
from pathlib import Path

import yaml

from core.generate.config import DRConfig, GenerateConfig, NoiseConfig, Quotas, load_generate_config
from core.generate.errors import ConfigError
from core.utils.hashing import json_sha256
from v4_support import FIXTURES_DIR, assert_frozen

EXAMPLE = FIXTURES_DIR / "v4_wp5" / "contract_example.yaml"
RAW = yaml.safe_load(EXAMPLE.read_text())


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="wp5cfg_")
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, data, name="cfg.yaml"):
        path = self.dir / name
        path.write_text(data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False))
        return path

    def load(self, data, overrides=None):
        return load_generate_config(self.write(data), overrides)


def example(**changes):
    d = dict(RAW)
    d.update(changes)
    return d


def without(*keys):
    return {k: v for k, v in RAW.items() if k not in keys}


class DataclassTests(unittest.TestCase):
    """WP5 §8.5 dataclass shapes and defaults."""

    def test_fields_and_defaults(self):
        """§8.5 field names/order and defaults of Quotas, NoiseConfig, DRConfig, GenerateConfig."""
        self.assertEqual([f.name for f in dataclasses.fields(Quotas)], ["success", "failure", "recovery"])
        self.assertEqual(NoiseConfig(), NoiseConfig(fraction=0.5, theta=0.15, sigma_rad=0.01, max_abs_rad=0.02))
        self.assertEqual([f.name for f in dataclasses.fields(NoiseConfig)], ["fraction", "theta", "sigma_rad", "max_abs_rad"])
        self.assertEqual(DRConfig(), DRConfig(copies=2, terms=("light", "robot_color"), recolor=True, seed=0))
        self.assertEqual([f.name for f in dataclasses.fields(DRConfig)], ["copies", "terms", "recolor", "seed",
                                                                   "camera_jitter"])  # camera_jitter: WP14 §16.7
        self.assertEqual([f.name for f in dataclasses.fields(GenerateConfig)],
                         ["arm", "scene", "tasks", "master_seed", "quotas", "shard", "dst_root", "repo_id", "fps",
                          "max_attempt_factor", "max_consecutive_skips", "scenario_unique_across_kinds", "noise", "dr",
                          "perturbation_weights", "cameras", "sha256"])
        for cls in (Quotas, NoiseConfig, DRConfig, GenerateConfig):
            self.assertTrue(cls.__dataclass_params__.frozen, cls)


class ContractExampleTests(Tmp):
    """WP5 §8.5 the contract example parses to the documented values."""

    def test_example_values(self):
        """§8.5 load_generate_config(contract example) gives the example's values with resolved types."""
        cfg = load_generate_config(EXAMPLE)
        self.assertIsInstance(cfg, GenerateConfig)
        self.assertEqual(cfg.arm, RAW["arm"])
        self.assertEqual(cfg.scene, "rpd_meeting_room")
        self.assertEqual(cfg.tasks, ("pick_place_color_cube",))
        self.assertEqual(cfg.master_seed, 0)
        self.assertEqual(dict(cfg.quotas), {"pick_place_color_cube": Quotas(4000, 1000, 1000)})
        self.assertIsInstance(cfg.quotas, types.MappingProxyType)
        self.assertEqual(cfg.shard, (0, 1))
        self.assertEqual(cfg.dst_root, Path("/data/v4"))
        self.assertIsInstance(cfg.dst_root, Path)
        self.assertEqual(cfg.repo_id, "local/{task}")
        self.assertEqual((cfg.fps, cfg.max_attempt_factor, cfg.max_consecutive_skips), (30, 3.0, 1000))
        self.assertIs(type(cfg.max_attempt_factor), float)
        self.assertIs(cfg.scenario_unique_across_kinds, True)
        self.assertEqual(cfg.noise, NoiseConfig(0.5, 0.15, 0.01, 0.02))
        self.assertEqual(cfg.dr, DRConfig(2, ("light", "robot_color"), True, 0))
        self.assertIsNone(cfg.perturbation_weights)
        self.assertIs(cfg.cameras, True)
        self.assertEqual(cfg.dataset_root("pick_place_color_cube"), Path("/data/v4/pick_place_color_cube"))
        self.assertEqual(cfg.dataset_repo_id("pick_place_color_cube"), "local/pick_place_color_cube")
        assert_frozen(cfg)

    def test_to_json_and_sha256(self):
        """§8.5 to_json() is the resolved config; sha256 = json_sha256(to_json() without shard and dst_root)."""
        cfg = load_generate_config(EXAMPLE)
        resolved = {
            "arm": RAW["arm"], "scene": "rpd_meeting_room", "tasks": ["pick_place_color_cube"], "master_seed": 0,
            "quotas": {"pick_place_color_cube": {"success": 4000, "failure": 1000, "recovery": 1000}},
            "shard": [0, 1], "dst_root": "/data/v4", "repo_id": "local/{task}", "fps": 30, "max_attempt_factor": 3.0,
            "max_consecutive_skips": 1000, "scenario_unique_across_kinds": True,
            "noise": {"fraction": 0.5, "theta": 0.15, "sigma_rad": 0.01, "max_abs_rad": 0.02},
            "dr": {"copies": 2, "terms": ["light", "robot_color"], "recolor": True, "seed": 0},
            "perturbation_weights": None, "cameras": True}
        self.assertEqual(cfg.to_json(), resolved)
        del resolved["shard"], resolved["dst_root"]
        self.assertEqual(cfg.sha256, json_sha256(resolved))

    def test_defaults_for_optional_keys(self):
        """§8.5 only arm, scene, tasks, master_seed, quotas, dst_root and repo_id are required; the rest default
        to the example's values (per_task_quotas {}, shard 0/1, fps 30, factor 3.0, skips 1000, unique true,
        noise/dr dataclass defaults, weights null, cameras true)."""
        minimal = {k: RAW[k] for k in ("arm", "scene", "tasks", "master_seed", "quotas", "dst_root", "repo_id")}
        cfg = self.load(minimal)
        self.assertEqual(cfg.sha256, load_generate_config(EXAMPLE).sha256)
        self.assertEqual(cfg.shard, (0, 1))
        for key in ("arm", "scene", "tasks", "master_seed", "quotas", "dst_root", "repo_id"):
            with self.subTest(missing=key), self.assertRaises(ConfigError):
                self.load(without(key))

    def test_partial_noise_and_dr(self):
        """§8.5 noise/dr tables may give a subset of keys; the rest are the dataclass defaults."""
        cfg = self.load(example(noise={"fraction": 0.25}, dr={"copies": 5}))
        self.assertEqual(cfg.noise, NoiseConfig(fraction=0.25))
        self.assertEqual(cfg.dr, DRConfig(copies=5))


class SemanticsTests(Tmp):
    """WP5 §8.5 quotas per task, shards, overrides, sha256 identity."""

    def test_per_task_quotas(self):
        """§8.5 quotas apply to every task; per_task_quotas override per task (missing keys inherit)."""
        cfg = self.load(example(tasks=["a", "b", "c"], per_task_quotas={"b": {"success": 7}, "c": {"success": 1, "failure": 2,
                                                                                                    "recovery": 3}}))
        self.assertEqual(dict(cfg.quotas), {"a": Quotas(4000, 1000, 1000), "b": Quotas(7, 1000, 1000), "c": Quotas(1, 2, 3)})
        self.assertEqual(cfg.dataset_root("b"), Path("/data/v4/b"))
        self.assertEqual(cfg.dataset_repo_id("c"), "local/c")
        with self.assertRaises(ValueError):
            cfg.dataset_root("zzz")

    def test_shard_strings(self):
        """§8.5 shard 'i/n' requires 0 <= i < n."""
        for text, expected in (("0/1", (0, 1)), ("3/4", (3, 4)), ("0/16", (0, 16))):
            with self.subTest(shard=text):
                self.assertEqual(self.load(example(shard=text)).shard, expected)
        for text in ("1/1", "4/4", "-1/2", "0/0", "1", "a/b", "1/2/3", ""):
            with self.subTest(shard=text), self.assertRaises(ConfigError):
                self.load(example(shard=text))

    def test_overrides(self):
        """§8.5 overrides replace top-level keys before validation; None values are ignored; unknown keys and
        bad values raise ConfigError."""
        path = self.write(RAW)
        cfg = load_generate_config(path, {"shard": "2/3", "dst_root": "/tmp/x", "master_seed": 9, "tasks": ["t1"]})
        self.assertEqual((cfg.shard, cfg.dst_root, cfg.master_seed, cfg.tasks), ((2, 3), Path("/tmp/x"), 9, ("t1",)))
        self.assertEqual(load_generate_config(path, {"shard": None, "master_seed": None}), load_generate_config(path))
        with self.assertRaises(ConfigError):
            load_generate_config(path, {"bogus": 1})
        with self.assertRaises(ConfigError):
            load_generate_config(path, {"shard": "3/3"})

    def test_sha256_ignores_shard_and_dst_root_only(self):
        """§8.5 sha256 excludes shard and dst_root (shards of one run agree) and covers every other setting."""
        base = self.load(RAW).sha256
        self.assertEqual(self.load(example(shard="1/2", dst_root="/elsewhere")).sha256, base)
        self.assertEqual(self.load(example(max_attempt_factor=3)).sha256, base)
        self.assertEqual(self.load(example(per_task_quotas={"pick_place_color_cube": {"success": 4000}})).sha256, base)
        reordered = {k: RAW[k] for k in reversed(list(RAW))}
        self.assertEqual(self.load(reordered).sha256, base)
        changed = {"master_seed": 1, "quotas": {"success": 4001, "failure": 1000, "recovery": 1000}, "fps": 15,
                   "max_attempt_factor": 2.5, "max_consecutive_skips": 10, "scenario_unique_across_kinds": False,
                   "noise": {"fraction": 0.4}, "dr": {"seed": 3}, "perturbation_weights": {"grasp_offset": 2.0},
                   "cameras": False, "repo_id": "org/{task}-v2", "arm": "fx_arm6", "scene": "fx_scene",
                   "tasks": ["pick_place_color_cube", "other"]}
        for key, value in changed.items():
            with self.subTest(changed=key):
                self.assertNotEqual(self.load(example(**{key: value})).sha256, base)

    def test_perturbation_weights(self):
        """§8.5 perturbation_weights: null or a {type: weight >= 0} mapping, stored read-only."""
        cfg = self.load(example(perturbation_weights={"grasp_offset": 2, "stall": 0.5}))
        self.assertEqual(dict(cfg.perturbation_weights), {"grasp_offset": 2.0, "stall": 0.5})
        self.assertIsInstance(cfg.perturbation_weights, types.MappingProxyType)
        assert_frozen(cfg)


class RejectionTests(Tmp):
    """WP5 §8.5/§1.5 invalid configs raise ConfigError."""

    def test_unknown_keys(self):
        """§8.5 unknown keys raise ConfigError, at the top level and inside quotas/noise/dr/per_task_quotas."""
        cases = {"top": example(extra=1), "quotas": example(quotas={**RAW["quotas"], "nominal": 1}),
                 "noise": example(noise={"fraction": 0.5, "sigma": 0.1}), "dr": example(dr={"copies": 1, "camera": 1}),
                 "per_task": example(per_task_quotas={"pick_place_color_cube": {"succes": 1}}),
                 "holdout_in_yaml": example(holdout={"colors": ["red"]})}
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaises(ConfigError):
                self.load(data)

    def test_bad_values(self):
        """§8.5/§1.5 bad types and ranges raise ConfigError (which is also a ValueError)."""
        cases = {
            "tasks_str": example(tasks="pick_place_color_cube"), "tasks_empty": example(tasks=[]),
            "tasks_dup": example(tasks=["a", "a"]), "seed_neg": example(master_seed=-1),
            "seed_big": example(master_seed=2**64), "seed_str": example(master_seed="0"),
            "seed_bool": example(master_seed=True), "quota_neg": example(quotas={"success": -1, "failure": 0, "recovery": 0}),
            "quota_float": example(quotas={"success": 1.5, "failure": 0, "recovery": 0}),
            "quota_missing": example(quotas={"success": 1, "failure": 0}),
            "per_task_unknown_task": example(per_task_quotas={"nope": {"success": 1}}),
            "fps_zero": example(fps=0), "fps_float": example(fps=30.5), "factor_lt_1": example(max_attempt_factor=0.5),
            "factor_str": example(max_attempt_factor="3"), "skips_zero": example(max_consecutive_skips=0),
            "unique_str": example(scenario_unique_across_kinds="yes please"),
            "noise_fraction": example(noise={"fraction": 1.5}), "noise_sigma": example(noise={"sigma_rad": -0.1}),
            "dr_copies": example(dr={"copies": -1}), "dr_term": example(dr={"terms": ["light", "camera_wrist"]}),
            "dr_term_dup": example(dr={"terms": ["light", "light"]}), "dr_recolor": example(dr={"recolor": "no thanks"}),
            "weights_neg": example(perturbation_weights={"grasp_offset": -1}),
            "weights_list": example(perturbation_weights=["grasp_offset"]),
            "cameras": example(cameras="sometimes"), "repo_two_tasks": example(tasks=["a", "b"], repo_id="local/fixed"),
            "repo_placeholder": example(repo_id="local/{name}"), "arm_empty": example(arm=""),
            "dst_root_int": example(dst_root=5),
        }
        for name, data in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(ConfigError) as ctx:
                    self.load(data)
                self.assertIsInstance(ctx.exception, ValueError)

    def test_not_a_mapping_or_bad_yaml(self):
        """§1.5 a YAML list, scalar or syntax error is a ConfigError."""
        for text in ("- a\n- b\n", "42\n", "arm: [unclosed\n"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
