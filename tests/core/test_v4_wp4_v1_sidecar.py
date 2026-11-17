"""WP4: v1 sidecars (<root>/pick_place_meta/) through v1_to_v2 and load_episode (contracts §9.1).

The fixture ``tests/fixtures/v4_wp4/v1_root/pick_place_meta/episode_00000{0,1}.json`` was written by the
pre-WP4 ``sim.recording.episode_metadata.save_cube_trajectory`` (see
``tests/sim/test_v4_wp4_episode_metadata.py``, which pins its bytes). The expected v2 form is built here
from the §9.1 "v1 → v2 mapping" list, independently of the implementation.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from core.datasets.task_meta import list_sidecars, load_episode, v1_to_v2, validate_sidecar, write_sidecar
from core.generate.errors import SchemaError
from v4_support import FIXTURES_DIR, copy_fixture

FIXTURE = "v4_wp4/v1_root"
V1_DIR = FIXTURES_DIR / FIXTURE / "pick_place_meta"


def read_v1(index):
    return json.loads((V1_DIR / f"episode_{index:06d}.json").read_text())


def expected_v2(v1, *, task, arm, scene):
    """§9.1 v1 → v2 mapping, written out key by key."""
    n = v1["num_frames"]
    return {
        "schema_version": 2,
        "episode_index": v1["episode_index"], "num_frames": n, "fps": v1["fps"],
        "task": task, "arm": arm, "scene": scene,
        "reference_frame": v1["reference_frame"], "units": "m", "angle_units": "rad",
        "joint_names": None,
        "objects": {"blue_cube": {"set": None, "size_m": v1["cube_size_m"], "color": "blue", "active": True,
                                  "role": "target", "position": v1["position"],
                                  "orientation_wxyz": v1["orientation_wxyz"]}},
        "ee_pose": None, "instruction": None,
        "roles": {"target": "blue_cube", "container": "white_box"},
        "kind": "success", "success": None, "failure_mode": None, "furthest_stage": None,
        "stages": [], "stage": None, "imitate": [True] * n, "phases": [],
        "segments": [{"label": "nominal", "start": 0, "end": n, "phase": None}],
        "perturbation": None, "recovery": None, "noise": None, "attempt": None, "fingerprints": None,
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "teleop", "git_commit": None}, "legacy_v1": True,
    }


class FixtureTests(unittest.TestCase):
    """WP4 §13 the committed v1 fixture has the v1 writer's shape."""

    def test_fixture_shape(self):
        """§9.1 v1 keys, frame "robot base link (base)", per-frame lengths equal num_frames."""
        for index, n in ((0, 12), (1, 8)):
            v1 = read_v1(index)
            with self.subTest(index=index):
                self.assertEqual(list(v1), ["episode_index", "reference_frame", "units", "fps", "num_frames",
                                            "cube_size_m", "position", "orientation_wxyz"])
                self.assertEqual((v1["episode_index"], v1["num_frames"], v1["fps"]), (index, n, 30))
                self.assertEqual(v1["reference_frame"], "robot base link (base)")
                self.assertEqual((len(v1["position"]), len(v1["orientation_wxyz"])), (n, n))


class V1ToV2Tests(unittest.TestCase):
    """WP4 §9.1 v1_to_v2 normalises a v1 payload."""

    def test_fixture_maps_to_documented_form(self):
        """§9.1 v1 → v2 mapping of the committed fixture equals the documented form and validates."""
        for index in (0, 1):
            v1 = read_v1(index)
            for task, arm, scene in (("pick_place_cube", None, None), ("fx_pick", "fx_arm5", "fx_scene")):
                with self.subTest(index=index, task=task, arm=arm):
                    out = v1_to_v2(v1, task=task, arm=arm, scene=scene)
                    self.assertEqual(out, expected_v2(v1, task=task, arm=arm, scene=scene))
                    self.assertEqual(list(out)[:1], ["schema_version"])
                    validate_sidecar(out, num_frames=v1["num_frames"], fps=30)

    def test_does_not_alias_or_mutate_input(self):
        """§9.1 the input is not mutated and the output shares no lists with it."""
        v1 = read_v1(0)
        before = copy.deepcopy(v1)
        out = v1_to_v2(v1, task="fx_pick", arm=None, scene=None)
        self.assertEqual(v1, before)
        out["objects"]["blue_cube"]["position"][0][0] = 99.0
        out["objects"]["blue_cube"]["size_m"][0] = 99.0
        self.assertEqual(v1, before)

    def test_strict_v1(self):
        """§9.1/§14 v1 readers stay strict: frame must be "robot base link (base)"; missing keys or bad data → SchemaError."""
        v1 = read_v1(1)
        bad = {
            "other base": dict(v1, reference_frame="robot base link (fx5_base)"),
            "units": dict(v1, units="cm"),
            "short position": dict(v1, position=v1["position"][:-1]),
            "orientation width": dict(v1, orientation_wxyz=[row[:3] for row in v1["orientation_wxyz"]]),
            "nan": dict(v1, position=[[float("nan"), 0.0, 0.0]] + v1["position"][1:]),
            "non-unit": dict(v1, orientation_wxyz=[[2.0, 0.0, 0.0, 0.0]] + v1["orientation_wxyz"][1:]),
            "cube size": dict(v1, cube_size_m=[0.025, 0.025]),
            "zero frames": dict(v1, num_frames=0, position=[], orientation_wxyz=[]),
        }
        for key in v1:
            bad[f"missing {key}"] = {k: v for k, v in v1.items() if k != key}
        for name, payload in bad.items():
            with self.subTest(case=name), self.assertRaises(SchemaError):
                v1_to_v2(payload, task="fx_pick", arm=None, scene=None)

    def test_unknown_v1_keys_ignored(self):
        """§9.1 decision: extra keys in a v1 payload are ignored (only the 8 writer keys are mapped)."""
        v1 = dict(read_v1(0), comment="hand edited")
        self.assertEqual(v1_to_v2(v1, task="t", arm=None, scene=None),
                         expected_v2(read_v1(0), task="t", arm=None, scene=None))


class LoadEpisodeTests(unittest.TestCase):
    """WP4 §9.1 load_episode: task_meta/ first, else pick_place_meta/ via v1_to_v2."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = copy_fixture(FIXTURE, self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_falls_back_to_v1(self):
        """§9.1 with only pick_place_meta/, load_episode returns the mapped v2 form (default task pick_place_cube)."""
        for index in (0, 1):
            with self.subTest(index=index):
                self.assertEqual(load_episode(self.root, index),
                                 expected_v2(read_v1(index), task="pick_place_cube", arm=None, scene=None))
                self.assertEqual(load_episode(str(self.root), index, task="fx_pick", arm="fx_arm5", scene="fx_scene"),
                                 expected_v2(read_v1(index), task="fx_pick", arm="fx_arm5", scene="fx_scene"))
        self.assertEqual(list_sidecars(self.root), [], "v1 files are not v2 sidecars")

    def test_v2_wins(self):
        """§9.1 when task_meta/ has the index it is returned (v1 ignored, task/arm/scene arguments unused)."""
        v2 = expected_v2(read_v1(0), task="fx_pick", arm="fx_arm5", scene="fx_scene")
        v2["legacy_v1"] = False
        v2["generator"] = {"name": "scripted_expert", "git_commit": None}
        v2["fingerprints"] = {k: "a" * 64 for k in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp",
                                                    "episode_fp")}
        v2["instruction"] = "pick it"
        write_sidecar(self.root, v2)
        self.assertEqual(load_episode(self.root, 0, task="other", arm="x", scene="y"), v2)
        self.assertEqual(load_episode(self.root, 1),
                         expected_v2(read_v1(1), task="pick_place_cube", arm=None, scene=None))

    def test_missing_everywhere(self):
        """§9.1 an index in neither folder raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_episode(self.root, 2)
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(FileNotFoundError):
            load_episode(empty, 0)

    def test_v1_index_mismatch_and_corrupt(self):
        """§9.1 a v1 file whose episode_index differs from its name, or that is not JSON, raises SchemaError."""
        v1_dir = Path(self.root) / "pick_place_meta"
        (v1_dir / "episode_000005.json").write_text(json.dumps(read_v1(0)))
        with self.assertRaises(SchemaError):
            load_episode(self.root, 5)
        (v1_dir / "episode_000006.json").write_text('{"episode_index": 6, ')
        with self.assertRaises(SchemaError):
            load_episode(self.root, 6)


if __name__ == "__main__":
    unittest.main()
