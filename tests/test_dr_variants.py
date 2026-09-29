"""Step 2 - utils/dr_variants.py: pure-Python DR planning (no Isaac, no LeRobot).

CPU only: python -m unittest tests.test_dr_variants -v

Contract under test
- COLORS = {"blue": <real_setup.json object_materials.blue_cube.color>, "red": (0.8, 0.05, 0.05)}
- DR_TERMS = ("light", "robot_color", "camera_external", "camera_wrist")
- DEFAULT_DR = ("light", "robot_color")      (cameras are calibrated to the real setup: never by default)
- DRRanges(exposure, robot_colors, camera_pos, camera_rot, focal_length)   dataclass
- DRDraw(exposure, robot_color, camera_external, camera_wrist_focal)       dataclass, fields None when off;
      camera_external = {"pos": {x,y,z}, "rot": {roll,pitch,yaw}, "focal": float}
- PlannedEpisode(dst_index, src_index, color, dr_seed, task_text, draw=None)
- parse_csv(text, allowed) -> list[str]            (trims; rejects empty / unknown / duplicates)
- episode_seed(seed, src_index, color) -> int      (stable across processes: no built-in hash())
- draw_dr(terms, ranges, rng: random.Random) -> DRDraw
- task_text(color, lang_blue, lang_red) -> str
- plan_episodes(src_indices, colors, terms, ranges, seed, include_originals, lang_blue, lang_red)
      originals first (color "blue", draw None), then for each source: one episode per colour, in
      `colors` order, each with its own seed + draw; dst_index contiguous from 0
- build_manifest(plan, args, git_commit) -> dict   (JSON-serialisable):
      {"git_commit", "args", "episodes": [asdict(PlannedEpisode) ...],
       "counts": {"total", "originals", "blue", "red"}}   # blue/red count GENERATED episodes only
- write_json_atomic(path, data)                    (temp file + os.replace; no temp left behind)
"""
import dataclasses
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from _fakes import REAL_SETUP_JSON, block_imports, load_module  # noqa: E402

REL = "source/sim_to_real_so101/utils/dr_variants.py"
D = None
LANG_BLUE = "Pick up the blue cube and place it in the white box"
LANG_RED = "Pick up the red cube and place it in the white box"


def setUpModule():
    global D
    # Must import with no simulator / robot stack available.
    with block_imports("pxr", "isaaclab", "isaacsim", "omni", "lerobot"):
        D = load_module("dr_variants_under_test", REL)


def ranges():
    return D.DRRanges(
        exposure=(-1.0, 1.0),
        robot_colors={"teal": (0.0, 0.8, 0.5), "black": (0.08, 0.08, 0.08)},
        camera_pos={"x": (-0.01, 0.01), "y": (-0.02, 0.0), "z": (0.0, 0.005)},
        camera_rot={"roll": (-0.02, 0.02), "pitch": (0.0, 0.0), "yaw": (-0.05, 0.05)},
        focal_length=(12.0, 15.0),
    )


def plan(**kw):
    args = dict(src_indices=list(range(48)), colors=["blue", "red"], terms=list(D.DEFAULT_DR),
                ranges=ranges(), seed=0, include_originals=False, lang_blue=LANG_BLUE, lang_red=LANG_RED)
    args.update(kw)
    return D.plan_episodes(**args)


class ConstantsTests(unittest.TestCase):
    def test_colors(self):
        blue = json.loads(REAL_SETUP_JSON.read_text())["object_materials"]["blue_cube"]["color"]
        self.assertEqual(tuple(D.COLORS["blue"]), tuple(blue))
        self.assertEqual(tuple(D.COLORS["red"]), (0.8, 0.05, 0.05))

    def test_terms(self):
        self.assertEqual(tuple(D.DR_TERMS), ("light", "robot_color", "camera_external", "camera_wrist"))
        self.assertEqual(tuple(D.DEFAULT_DR), ("light", "robot_color"))
        for camera_term in ("camera_external", "camera_wrist"):
            self.assertNotIn(camera_term, D.DEFAULT_DR, "camera views stay fixed to the calibrated real setup")

    def test_default_draw_leaves_cameras_untouched(self):
        draw = D.draw_dr(D.DEFAULT_DR, ranges(), random.Random(0))
        self.assertIsNone(draw.camera_external)
        self.assertIsNone(draw.camera_wrist_focal)


class ParseTests(unittest.TestCase):
    def test_parse_csv(self):
        self.assertEqual(D.parse_csv(" blue , red ", D.COLORS), ["blue", "red"])
        for bad in ("", "blue,,red", "green", "blue,blue"):
            with self.assertRaises(ValueError, msg=bad):
                D.parse_csv(bad, D.COLORS)

    def test_task_text(self):
        self.assertEqual(D.task_text("blue", LANG_BLUE, LANG_RED), LANG_BLUE)
        self.assertEqual(D.task_text("red", LANG_BLUE, LANG_RED), LANG_RED)
        with self.assertRaises(ValueError):
            D.task_text("green", LANG_BLUE, LANG_RED)


class SeedTests(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        self.assertEqual(D.episode_seed(0, 5, "red"), D.episode_seed(0, 5, "red"))
        seeds = {D.episode_seed(0, i, c) for i in range(48) for c in ("blue", "red")}
        self.assertEqual(len(seeds), 96)
        self.assertNotEqual(D.episode_seed(0, 5, "red"), D.episode_seed(1, 5, "red"))

    def test_stable_across_processes(self):
        code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('m',sys.argv[1]);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(m.episode_seed(7,3,'red'))")
        path = str(Path(__file__).resolve().parents[1] / REL)
        outputs = {subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True, check=True,
                                  env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip() for h in (1, 2, 3)}
        self.assertEqual(outputs, {str(D.episode_seed(7, 3, "red"))},
                         "episode_seed must not depend on Python's randomised hash()")


class DrawTests(unittest.TestCase):
    def test_in_range_and_off_terms_none(self):
        rng, r = random.Random(0), ranges()
        for _ in range(200):
            draw = D.draw_dr(("light", "robot_color", "camera_external"), r, rng)
            self.assertTrue(r.exposure[0] <= draw.exposure <= r.exposure[1])
            self.assertIn(draw.robot_color, r.robot_colors)
            cam = draw.camera_external
            self.assertEqual(set(cam), {"pos", "rot", "focal"})
            for axis, (lo, hi) in r.camera_pos.items():
                self.assertTrue(lo <= cam["pos"][axis] <= hi)
            for axis, (lo, hi) in r.camera_rot.items():
                self.assertTrue(lo <= cam["rot"][axis] <= hi)
            self.assertTrue(r.focal_length[0] <= cam["focal"] <= r.focal_length[1])
            self.assertIsNone(draw.camera_wrist_focal)

    def test_only_selected_terms(self):
        draw = D.draw_dr(["light"], ranges(), random.Random(1))
        self.assertIsNotNone(draw.exposure)
        self.assertIsNone(draw.robot_color)
        self.assertIsNone(draw.camera_external)
        wrist = D.draw_dr(["camera_wrist"], ranges(), random.Random(1))
        self.assertTrue(12.0 <= wrist.camera_wrist_focal <= 15.0)
        self.assertIsNone(wrist.exposure)

    def test_unknown_term(self):
        with self.assertRaises(ValueError):
            D.draw_dr(["fog"], ranges(), random.Random(0))

    def test_same_rng_seed_same_draw(self):
        a = D.draw_dr(D.DEFAULT_DR, ranges(), random.Random(42))
        b = D.draw_dr(D.DEFAULT_DR, ranges(), random.Random(42))
        self.assertEqual(dataclasses.asdict(a), dataclasses.asdict(b))


class PlanTests(unittest.TestCase):
    def test_48_sources_give_96_blue_red(self):
        episodes = plan()
        self.assertEqual(len(episodes), 96)
        self.assertEqual([e.dst_index for e in episodes], list(range(96)))
        self.assertEqual([(e.src_index, e.color) for e in episodes[:4]],
                         [(0, "blue"), (0, "red"), (1, "blue"), (1, "red")])
        self.assertEqual(sum(e.color == "red" for e in episodes), 48)
        for e in episodes:
            self.assertIsNotNone(e.draw, "every generated episode is randomized")
            self.assertEqual(e.task_text, LANG_RED if e.color == "red" else LANG_BLUE)
            self.assertEqual(e.dr_seed, D.episode_seed(0, e.src_index, e.color))

    def test_include_originals_first(self):
        episodes = plan(include_originals=True)
        self.assertEqual(len(episodes), 144)
        originals, generated = episodes[:48], episodes[48:]
        self.assertEqual([e.src_index for e in originals], list(range(48)))
        for e in originals:
            self.assertEqual(e.color, "blue")
            self.assertIsNone(e.draw)
        self.assertEqual(generated[0].dst_index, 48)
        self.assertTrue(all(e.draw is not None for e in generated))

    def test_excluded_sources_are_just_absent(self):
        src = [i for i in range(50) if i not in (0, 6)]
        episodes = plan(src_indices=src)
        self.assertEqual(len(episodes), 96)
        self.assertNotIn(0, {e.src_index for e in episodes})
        self.assertNotIn(6, {e.src_index for e in episodes})

    def test_reproducible_and_seed_sensitive(self):
        as_dicts = lambda eps: [dataclasses.asdict(e) for e in eps]  # noqa: E731
        self.assertEqual(as_dicts(plan(seed=3)), as_dicts(plan(seed=3)))
        self.assertNotEqual(as_dicts(plan(seed=3)), as_dicts(plan(seed=4)))

    def test_blue_and_red_of_same_source_get_different_draws(self):
        episodes = plan()
        self.assertNotEqual(dataclasses.asdict(episodes[0].draw), dataclasses.asdict(episodes[1].draw))

    def test_blue_only(self):
        episodes = plan(colors=["blue"])
        self.assertEqual(len(episodes), 48)
        self.assertEqual({e.color for e in episodes}, {"blue"})


class ManifestTests(unittest.TestCase):
    def test_manifest_round_trips(self):
        episodes = plan(include_originals=True)
        manifest = D.build_manifest(episodes, {"seed": 0, "colors": "blue,red"}, "abc1234")
        loaded = json.loads(json.dumps(manifest, allow_nan=False))
        self.assertEqual(loaded["git_commit"], "abc1234")
        self.assertEqual(loaded["args"]["seed"], 0)
        self.assertEqual(len(loaded["episodes"]), 144)
        row = loaded["episodes"][48]
        self.assertTrue(set(row) >= {"dst_index", "src_index", "color", "dr_seed", "task_text", "draw"})
        self.assertIsNone(loaded["episodes"][0]["draw"])
        self.assertEqual(loaded["counts"], {"total": 144, "originals": 48, "blue": 48, "red": 48})

    def test_write_json_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta" / "dr_replay_manifest.json"
            path.parent.mkdir()
            D.write_json_atomic(path, {"a": 1})
            D.write_json_atomic(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 2})
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name], "no temp files left")


if __name__ == "__main__":
    unittest.main()
