"""WP11: DR re-render v2 — planning over v2 sidecars, the copies' sidecars and manifest, the Isaac-free dry run, the
new SourceEpisode fields, and v1 sources behaving exactly as today (contracts §12 WP11, §8.6, §9.1, §15 WP4/WP5).

Decisions pinned here (docs/v4/prompts/WP11.md):
* ``plan_episodes_v2(sources, *, task_spec, cfg, exposure_range, robot_colors, include_originals)`` returns
  ``PlannedEpisodeV2(dst_index, src_index, draw, copy_index)``: the originals first (``draw=None``,
  ``copy_index=None``) when ``include_originals``, then for each source in the given order ``cfg.copies``
  copies (``copy_index`` 0..copies−1) drawn with ``draw_appearance(parent, …, dr_seed=cfg.seed,
  copy_index=c, taken={parent appearance_fp} ∪ earlier siblings)``; ``dst_index`` runs 0..len−1. A legacy / v1
  source (``legacy_v1`` or no fingerprints) raises ``ValueError`` (v1 sources use ``plan_episodes``).
* ``robot_color_names()`` = the keys of ``ROBOT_COLORS`` in ``sim/envs/mdp/resets.py``, in order, read without
  importing Isaac; ``exposure_range(task_spec)`` = ``[task.dr].room_light_exposure_range`` or (−3.0, 1.0).
* ``sidecar_v2(planned, parent, *, source_repo_id)``: an original is the parent with ``episode_index`` replaced;
  a copy is ``dr_copy_sidecar(parent, draw, episode_index=dst, source_repo_id=…)``; both are validated.
* ``build_manifest_v2(plan, args, git_commit)`` = ``{"git_commit", "args", "episodes": [{"dst_index",
  "src_index", "copy_index", "draw": {"colors", "instruction", "light_exposure", "robot_color", "seed",
  "appearance_fp"} | null}], "counts": {"total", "originals", "copies"}}``.
* ``dr_replay`` takes the v2 path when the source has ``task_meta/``; new arguments ``--copies`` (default 2) and
  ``--no_recolor``; ``--dry_run`` prints one JSON object per planned episode (``dst``, ``src``, ``copy``,
  ``robot_color``, ``light``, ``instruction``) and a summary line ``"<n> episodes: <m> originals, <k> copies"``,
  without importing Isaac or LeRobot.
* ``load_source(root, repo_id, exclude, dataset=None, *, task="pick_place_cube", arm=None, scene=None)`` reads each
  episode through ``load_episode``; ``SourceEpisode`` gains ``objects`` ({name: (N, 7) x y z qw qx qy qz}),
  ``colors``, ``roles`` and ``meta`` (the v2 sidecar); ``poses_base`` is the target role's poses and
  ``cube_size`` its ``size_m``. For v1 sources ``poses_base`` and ``cube_size`` are unchanged.
"""

import ast
import contextlib
import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.datasets.audit import audit_dataset
from core.datasets.frames import LiteDatasetWriter, read_frames
from core.datasets.task_meta import list_sidecars, load_episode, read_sidecar, validate_sidecar, write_sidecar
from core.generate.appearance import draw_appearance, dr_copy_sidecar
from core.generate.config import DRConfig
from core.generate.protocols import FrameLabels
from core.tasks.base import load_task_spec
from sim.generate.dr_variants import (DRRanges, build_manifest, build_manifest_v2, exposure_range, plan_episodes,
                                      plan_episodes_v2, robot_color_names, sidecar_v2)
from v4_support import FIXTURES_DIR, REPO_ROOT, SRC_DIR, TESTS_DIR, copy_repo_folder, sandbox
from v4_support import harness as H

GOLDEN = FIXTURES_DIR / "v4_wp11" / "plan_v1_golden.json"
V1_ROOT = FIXTURES_DIR / "v4_wp4" / "v1_root"
COLORS = ("orange", "beige", "teal", "white", "black")


class V1RegressionTests(unittest.TestCase):
    """WP11 §12: plan_episodes / build_manifest for v1 sources are unchanged (golden from the v3 code)."""

    def test_golden(self):
        g = json.loads(GOLDEN.read_text())
        ranges = DRRanges(exposure=tuple(g["exposure"]), robot_colors={k: tuple(v) for k, v in g["robot_colors"]},
                          camera_pos={}, camera_rot={}, focal_length=None)
        for case in g["cases"]:
            a = case["args"]
            with self.subTest(include=a["include_originals"], terms=a["terms"]):
                plan = plan_episodes(a["src_indices"], a["colors"], a["terms"], ranges, a["seed"], a["include_originals"],
                                     a["lang_blue"], a["lang_red"])
                self.assertEqual([dataclasses.asdict(p) for p in plan], case["plan"])
                self.assertEqual(json.loads(json.dumps(build_manifest(plan, {"note": "golden"}, "abc123"))),
                                 case["manifest"])


class Datasets(unittest.TestCase):
    TASKS = ("fx7_color", "fx7_grip")

    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls.box = cls._stack.enter_context(sandbox())
        H.prepare_sandbox(cls.box, cls.TASKS)
        rig = H.Rig(H.DEFAULT_ARM)
        cls.color = H.run_shard(H.write_config(cls.box.root / "c.yaml", tasks=["fx7_color"], dst_root=cls.box.root / "c",
                                               quotas={"success": 2, "failure": 2, "recovery": 1}), "fx7_color", rig=rig)[0]
        cls.grip = H.run_shard(H.write_config(cls.box.root / "g.yaml", tasks=["fx7_grip"], dst_root=cls.box.root / "g",
                                              quotas={"success": 2, "failure": 1, "recovery": 1}), "fx7_grip", rig=rig)[0]

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()

    def sources(self, root):
        return [read_sidecar(root, i) for i in list_sidecars(root)]

    def plan(self, root, *, copies=2, include_originals=True, recolor=True, seed=0, terms=("light", "robot_color")):
        spec = load_task_spec(self.sources(root)[0]["task"])
        cfg = DRConfig(copies=copies, terms=tuple(terms), recolor=recolor, seed=seed)
        return spec, plan_episodes_v2(self.sources(root), task_spec=spec, cfg=cfg, exposure_range=(-3.0, 1.0),
                                      robot_colors=COLORS, include_originals=include_originals)


class PlanV2Tests(Datasets):
    """WP11 plan_episodes_v2: layout, determinism, exact draws, distinct appearance, hold-outs."""

    def test_layout(self):
        n = len(self.sources(self.color))
        for copies in (0, 1, 3):
            for inc in (True, False):
                _, plan = self.plan(self.color, copies=copies, include_originals=inc)
                with self.subTest(copies=copies, include=inc):
                    self.assertEqual([p.dst_index for p in plan], list(range(len(plan))))
                    self.assertEqual(len(plan), n * copies + (n if inc else 0))
                    head = plan[:n] if inc else []
                    self.assertTrue(all(p.draw is None and p.copy_index is None for p in head))
                    self.assertEqual([p.src_index for p in head], list(range(n)) if inc else [])
                    tail = plan[n if inc else 0:]
                    self.assertEqual([(p.src_index, p.copy_index) for p in tail],
                                     [(i, c) for i in range(n) for c in range(copies)])

    def test_deterministic_and_exact(self):
        spec, plan = self.plan(self.color, copies=3)
        self.assertEqual(plan, self.plan(self.color, copies=3)[1])
        sources = {sc["episode_index"]: sc for sc in self.sources(self.color)}
        taken = {}
        for p in plan:
            if p.draw is None:
                continue
            parent = sources[p.src_index]
            t = taken.setdefault(p.src_index, {parent["fingerprints"]["appearance_fp"]})
            expected = draw_appearance(parent, task_spec=spec, terms=("light", "robot_color"), exposure_range=(-3.0, 1.0),
                                       robot_colors=COLORS, recolor=True, dr_seed=0, copy_index=p.copy_index,
                                       taken=set(t))
            self.assertEqual(p.draw, expected)
            t.add(expected.appearance_fp)
        self.assertNotEqual(plan, self.plan(self.color, copies=3, seed=1)[1])

    def test_appearance_rules(self):
        for root in (self.color, self.grip):
            spec, plan = self.plan(root, copies=3)
            sources = {sc["episode_index"]: sc for sc in self.sources(root)}
            seen = {}
            for p in plan:
                if p.draw is None:
                    continue
                parent = sources[p.src_index]
                s = seen.setdefault(p.src_index, {parent["fingerprints"]["appearance_fp"]})
                with self.subTest(root=root.name, dst=p.dst_index):
                    self.assertNotIn(p.draw.appearance_fp, s)
                    s.add(p.draw.appearance_fp)
                    self.assertIn(p.draw.robot_color, COLORS)
                    self.assertTrue(-3.0 <= p.draw.light_exposure <= 1.0)
                    colors = [c for c in p.draw.colors.values() if c is not None]
                    self.assertFalse(set(colors) & set(spec.holdout.colors))
                    target = parent["roles"].get("target")
                    if target in p.draw.colors and p.draw.colors[target] is not None:
                        self.assertEqual(colors.count(p.draw.colors[target]), 1)

    def test_no_recolor_keeps_colours(self):
        _, plan = self.plan(self.color, recolor=False)
        sources = {sc["episode_index"]: sc for sc in self.sources(self.color)}
        for p in plan:
            if p.draw is not None:
                parent = sources[p.src_index]
                self.assertEqual(dict(p.draw.colors), {n: o["color"] for n, o in parent["objects"].items()})

    def test_legacy_source_refused(self):
        legacy = load_episode(V1_ROOT, 0)
        spec = load_task_spec("fx7_color")
        with self.assertRaises(ValueError):
            plan_episodes_v2([legacy], task_spec=spec, cfg=DRConfig(), exposure_range=(-3.0, 1.0),
                             robot_colors=COLORS, include_originals=False)

    def test_helpers(self):
        tree = ast.parse((SRC_DIR / "sim" / "envs" / "mdp" / "resets.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ROBOT_COLORS" for t in n.targets))
        self.assertEqual(tuple(robot_color_names()), tuple(ast.literal_eval(node.value)))
        self.assertEqual(exposure_range(load_task_spec("fx7_color")), (-3.0, 1.0))
        spec = SimpleNamespace(raw={"dr": {"room_light_exposure_range": [-1.5, 0.5]}})
        self.assertEqual(exposure_range(spec), (-1.5, 0.5))


class SidecarAndManifestTests(Datasets):
    """WP11 sidecar_v2 and build_manifest_v2; a DR dataset built from the plan passes the WP7 audit."""

    def test_sidecars(self):
        """Planned over sources 1.. so an original's dst_index differs from its source index."""
        kept = self.sources(self.color)[1:]
        spec = load_task_spec("fx7_color")
        plan = plan_episodes_v2(kept, task_spec=spec, cfg=DRConfig(), exposure_range=(-3.0, 1.0), robot_colors=COLORS,
                                include_originals=True)
        self.assertEqual(plan[0].src_index, 1)
        sources = {sc["episode_index"]: sc for sc in self.sources(self.color)}
        for p in plan:
            parent = sources[p.src_index]
            out = sidecar_v2(p, parent, source_repo_id="local/fx7_color")
            with self.subTest(dst=p.dst_index):
                validate_sidecar(out)
                self.assertEqual(out["episode_index"], p.dst_index)
                if p.draw is None:
                    self.assertEqual({**out, "episode_index": parent["episode_index"]}, parent)
                else:
                    self.assertEqual(out, dr_copy_sidecar(parent, p.draw, episode_index=p.dst_index,
                                                          source_repo_id="local/fx7_color"))
        self.assertEqual(sources[0], read_sidecar(self.color, 0))          # parents not mutated

    def test_dataset_from_plan_passes_audit(self):
        for root in (self.color, self.grip):
            _, plan = self.plan(root)
            sources = {sc["episode_index"]: sc for sc in self.sources(root)}
            frames = read_frames(root)
            info = json.loads((root / "meta" / "info.json").read_text())
            dst = self.box.root / f"dr_{root.name}"
            w = LiteDatasetWriter(dst, fps=30, joint_names=tuple(info["features"]["action"]["names"]))
            for p in plan:
                sc = sidecar_v2(p, sources[p.src_index], source_repo_id="local/x")
                f = frames[p.src_index]
                for t in range(sc["num_frames"]):
                    w.add_frame(f["action"][t], f["observation.state"][t], {}, sc["instruction"],
                                FrameLabels(int(f["label.stage"][t]), bool(f["label.imitate"][t]),
                                            np.asarray(f["label.ee_pose"][t], np.float64)))
                write_sidecar(dst, sc)
                w.save_episode()
            report = audit_dataset(dst, source_root=root)
            with self.subTest(root=root.name):
                self.assertTrue(report.passed, [c for c in report.checks if not c["passed"]])

    def test_manifest(self):
        _, plan = self.plan(self.color, copies=2)
        m = build_manifest_v2(plan, {"seed": 0}, "abc")
        self.assertEqual(set(m), {"git_commit", "args", "episodes", "counts"})
        n = len(self.sources(self.color))
        self.assertEqual(m["counts"], {"total": 3 * n, "originals": n, "copies": 2 * n})
        for p, e in zip(plan, m["episodes"]):
            self.assertEqual((e["dst_index"], e["src_index"], e["copy_index"]), (p.dst_index, p.src_index, p.copy_index))
            if p.draw is None:
                self.assertIsNone(e["draw"])
            else:
                self.assertEqual(e["draw"], {"colors": dict(p.draw.colors), "instruction": p.draw.instruction,
                                             "light_exposure": p.draw.light_exposure, "robot_color": p.draw.robot_color,
                                             "seed": p.draw.seed, "appearance_fp": p.draw.appearance_fp})
        json.dumps(m)


class DryRunTests(Datasets):
    """WP11 dr_replay --dry_run on a v2 source: plan lines without Isaac or LeRobot."""

    def dry_run(self, *extra, root=None):
        root = root or self.color
        code = ("import importlib.abc, sys\n"
                "class B(importlib.abc.MetaPathFinder):\n"
                "    def find_spec(self, name, path=None, target=None):\n"
                "        if name.split('.')[0] in {'isaaclab','isaacsim','omni','carb','pxr','warp','lerobot','gymnasium'}:\n"
                "            raise ImportError(name)\n"
                "sys.meta_path.insert(0, B())\n"
                "from sim.generate import dr_replay\n"
                "rc = dr_replay.main(sys.argv[1:])\n"
                "sys.exit(rc or 0)\n")
        argv = ["--src_root", str(root), "--src_repo_id", "local/src", "--dst_root", str(self.box.root / "never"),
                "--dst_repo_id", "local/dst", "--dry_run", "--arm", "so101", "--scene", "rpd_meeting_room",
                "--task", "pick_place_cube", *extra]
        env = {k: v for k, v in os.environ.items() if not k.startswith("S2R_")}
        env.update(PYTHONPATH=f"{SRC_DIR}:{TESTS_DIR}", S2R_TASKS_DIR=str(self.box.tasks))
        out = subprocess.run([sys.executable, "-c", code, *argv], capture_output=True, text=True, env=env, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        return out.stdout.strip().splitlines()

    def test_plan_lines(self):
        lines = self.dry_run("--include_originals", "--copies", "2", "--seed", "3")
        spec, plan = self.plan(self.color, copies=2, seed=3)
        rows = [json.loads(x) for x in lines[:-1]]
        self.assertEqual(len(rows), len(plan))
        for r, p in zip(rows, plan):
            self.assertEqual(set(r), {"dst", "src", "copy", "robot_color", "light", "instruction"})
            self.assertEqual((r["dst"], r["src"], r["copy"]), (p.dst_index, p.src_index, p.copy_index))
            if p.draw is not None:
                self.assertEqual(r["robot_color"], p.draw.robot_color)
                self.assertEqual(r["instruction"], p.draw.instruction)
        self.assertNotIn(None, [r["robot_color"] for r in rows if r["copy"] is not None])
        n = len(self.sources(self.color))
        self.assertEqual(lines[-1], f"{3 * n} episodes: {n} originals, {2 * n} copies")
        self.assertFalse((self.box.root / "never").exists())

    def test_options(self):
        n = len(self.sources(self.color))
        lines = self.dry_run("--copies", "1", "--exclude_episodes", "0", "--no_recolor")
        self.assertEqual(lines[-1], f"{n - 1} episodes: 0 originals, {n - 1} copies")
        self.assertNotIn(0, [json.loads(x)["src"] for x in lines[:-1]])
        lines = self.dry_run("--copies", "2", "--limit", "3")
        self.assertEqual(len(lines) - 1, 3)


class StubDataset:
    """The parts of LeRobotDataset that load_source reads (meta.episodes rows and hf_dataset columns)."""

    def __init__(self, lengths, J, tasks, rng):
        rows, start, self.actions, self.states = [], 0, [], []
        for i, n in enumerate(lengths):
            rows.append({"episode_index": i, "dataset_from_index": start, "dataset_to_index": start + n,
                         "tasks": [tasks[i]]})
            start += n
        self.meta = SimpleNamespace(episodes=rows)
        self.hf_dataset = {"action": rng.uniform(-100, 100, (start, J)).astype(np.float32),
                           "observation.state": rng.uniform(-100, 100, (start, J)).astype(np.float32)}


class SourceTests(Datasets):
    """WP11 load_source: v2 fields, v1 sources unchanged."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        copy_repo_folder("arms", "so101", cls.box.root)        # replay_source reads the default arm's joints

    def test_v1_source_unchanged(self):
        from sim.recording.replay_source import load_sidecar, load_source, sidecar_cube_size
        from core.arms.joint_calibration import FOLLOWER_JOINTS
        lengths = [read_sidecar_v1(V1_ROOT, i)["num_frames"] for i in (0, 1)]
        ds = StubDataset(lengths, len(FOLLOWER_JOINTS), ["blue task", "blue task"], np.random.default_rng(0))
        eps = load_source(str(V1_ROOT), "local/x", set(), dataset=ds)
        for e in eps:
            with self.subTest(episode=e.index):
                np.testing.assert_array_equal(e.poses_base, load_sidecar(str(V1_ROOT), e.index))
                self.assertEqual(e.cube_size, sidecar_cube_size(str(V1_ROOT), e.index))
                self.assertEqual(set(e.objects), {"blue_cube"})
                np.testing.assert_array_equal(e.objects["blue_cube"], e.poses_base)
                self.assertEqual(e.colors, {"blue_cube": "blue"})
                self.assertEqual(e.roles, {"target": "blue_cube", "container": "white_box"})
                self.assertEqual(e.meta, load_episode(V1_ROOT, e.index))

    def test_v2_source(self):
        from sim.recording.replay_source import load_source
        from core.arms.joint_calibration import FOLLOWER_JOINTS
        sides = self.sources(self.color)
        ds = StubDataset([s["num_frames"] for s in sides], len(FOLLOWER_JOINTS), [s["instruction"] for s in sides],
                         np.random.default_rng(1))
        eps = load_source(str(self.color), "local/x", {1}, dataset=ds, task="fx7_color", arm="fx_arm5", scene="fx_scene")
        self.assertEqual([e.index for e in eps], [i for i in range(len(sides)) if i != 1])
        for e in eps:
            sc = sides[e.index]
            with self.subTest(episode=e.index):
                self.assertEqual(e.meta, sc)
                self.assertEqual(set(e.objects), set(sc["objects"]))
                for name, o in sc["objects"].items():
                    np.testing.assert_array_equal(e.objects[name][:, :3], o["position"])
                    np.testing.assert_array_equal(e.objects[name][:, 3:], o["orientation_wxyz"])
                target = sc["roles"]["target"]
                np.testing.assert_array_equal(e.poses_base, e.objects[target])
                self.assertEqual(e.cube_size, tuple(sc["objects"][target]["size_m"]))
                self.assertEqual(e.colors, {n: o["color"] for n, o in sc["objects"].items()})
                self.assertEqual(e.roles, sc["roles"])


def read_sidecar_v1(root, index):
    return json.loads((Path(root) / "pick_place_meta" / f"episode_{index:06d}.json").read_text())


if __name__ == "__main__":
    unittest.main()
