"""WP5: ``core.generate.appearance`` — DR appearance draws (contracts §8.6, §4.5 step 2, §8.2, §9.1, D13, D22)."""

import copy
import dataclasses
import json
import types
import unittest

import numpy as np

from core.generate.appearance import AppearanceDraw, draw_appearance, dr_copy_sidecar
from core.generate.errors import GenerationError
from core.generate.fingerprint import appearance_fp, episode_fp
from core.generate.seeds import derive_seed
from v4_support import assert_frozen
from v4_support import factories as fx

PALETTE = {"red": (0.8, 0.05, 0.05), "green": (0.05, 0.6, 0.1), "blue": (0.05, 0.1, 0.8), "yellow": (0.9, 0.8, 0.05),
           "orange": (0.95, 0.4, 0.02), "purple": (0.4, 0.1, 0.6), "pink": (0.95, 0.45, 0.65)}
TEMPLATE = "Pick up the {target.color} cube and place it in the box"
ROBOT_COLORS = ("black", "white", "orange_robot")


def task_spec(*, palette=PALETTE, holdout_colors=("purple",), templates=(TEMPLATE,), distinct=True):
    """Stand-in for the WP1 ``TaskSpec`` with the attributes draw_appearance reads (§4.1): ``palette``,
    ``holdout.colors``, ``instruction_templates``, ``roles`` and ``raw["start"]["distinct_colors"]``."""
    return types.SimpleNamespace(
        name="fx_color", palette=dict(palette),
        holdout=types.SimpleNamespace(colors=tuple(holdout_colors), counts=(4,)),
        instruction_templates=tuple(templates), roles={"target": "cubes", "container": "small_box"},
        raw={"start": {"distinct_colors": distinct, "count": [2, 6]}})


def parent_sidecar(colors=("red", "green", "blue"), *, target=0, slots=None, tag="p", reverse=False, episode_index=3):
    """A v2 sidecar (§9.1) of an original episode with cube slots ``slots`` coloured ``colors`` and a box."""
    slots = list(range(len(colors))) if slots is None else list(slots)
    n = 2
    objects = {}
    for slot, color in zip(slots, colors):
        name = f"cube_{slot}"
        objects[name] = {"set": "cubes", "size_m": [0.02] * 3, "color": color, "active": True,
                         "role": "target" if slot == slots[target] else None,
                         "position": [[0.1 + 0.03 * slot, 0.0, 0.01]] * n, "orientation_wxyz": [[1.0, 0.0, 0.0, 0.0]] * n}
    objects["small_box"] = {"set": None, "size_m": [0.05, 0.05, 0.035], "color": None, "active": True,
                            "role": "container", "position": [[0.25, 0.0, 0.0]] * n,
                            "orientation_wxyz": [[1.0, 0.0, 0.0, 0.0]] * n}
    if reverse:
        objects = dict(reversed(list(objects.items())))
    roles = {"target": f"cube_{slots[target]}", "container": "small_box"}
    instruction = TEMPLATE.format(target=types.SimpleNamespace(color=colors[target]))
    traj = fx.hex_digest(f"traj-{tag}")
    app = appearance_fp({k: v["color"] for k, v in objects.items()}, instruction, None, None)
    return {
        "schema_version": 2, "episode_index": episode_index, "num_frames": n, "fps": 30, "task": "fx_color",
        "arm": "fx_arm5", "scene": "fx_scene", "reference_frame": "robot base link (base)", "units": "m",
        "angle_units": "rad", "joint_names": ["j1", "j2", "j3", "j4", "j5", "jaw"], "objects": objects,
        "ee_pose": {"position": [[0.2, 0.0, 0.1]] * n, "orientation_wxyz": [[1.0, 0.0, 0.0, 0.0]] * n},
        "instruction": instruction, "roles": roles, "kind": "success", "success": True, "failure_mode": None,
        "furthest_stage": "success", "stages": ["reached", "success"], "stage": [0, 1], "imitate": [True, True],
        "phases": [{"name": "approach", "start": 0, "end": n}],
        "segments": [{"label": "nominal", "start": 0, "end": n, "phase": "approach"}], "perturbation": None,
        "recovery": None, "noise": None,
        "attempt": {"index": 11, "seed": 2**64 - 3, "shard": [0, 1], "spec_fp": fx.hex_digest(f"spec-{tag}")},
        "fingerprints": {"scenario_key": fx.hex_digest(f"scen-{tag}"), "spec_fp": fx.hex_digest(f"spec-{tag}"),
                         "traj_fp": traj, "appearance_fp": app, "episode_fp": episode_fp(traj, app)},
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "scripted_expert", "git_commit": "abc123"}, "legacy_v1": False,
    }


def draw(parent, *, spec=None, terms=("light", "robot_color"), recolor=True, dr_seed=0, copy_index=0, taken=None,
         exposure_range=(0.5, 1.5), robot_colors=ROBOT_COLORS, max_redraws=100):
    return draw_appearance(parent, task_spec=spec or task_spec(), terms=terms, exposure_range=exposure_range,
                           robot_colors=robot_colors, recolor=recolor, dr_seed=dr_seed, copy_index=copy_index,
                           taken=set() if taken is None else taken, max_redraws=max_redraws)


def cube_colors(parent, d):
    return {n: c for n, c in d.colors.items() if parent["objects"][n]["set"] == "cubes"}


class DrawTests(unittest.TestCase):
    """WP5 §8.6 draw_appearance."""

    def test_type(self):
        """§8.6 AppearanceDraw fields; instances are deeply frozen (colors read-only)."""
        self.assertEqual([f.name for f in dataclasses.fields(AppearanceDraw)],
                         ["colors", "instruction", "light_exposure", "robot_color", "seed", "appearance_fp"])
        d = draw(parent_sidecar())
        assert_frozen(d)
        self.assertIsInstance(d.colors, types.MappingProxyType)

    def test_deterministic(self):
        """§8.6 equal inputs give equal draws (pure; global RNG state irrelevant)."""
        parent = parent_sidecar()
        a = draw(parent, copy_index=4, dr_seed=9)
        np.random.seed(0)
        np.random.random(5)
        b = draw(copy.deepcopy(parent), copy_index=4, dr_seed=9)
        self.assertEqual(a, b)

    def test_parent_dict_order_irrelevant(self):
        """§8.6 a parent read back with a different key order (canonical JSON sorts keys) gives the same draw."""
        for k in range(5):
            rng = np.random.default_rng(k)
            colors = tuple(rng.choice(["red", "green", "blue", "yellow", "orange", "pink"], 4, replace=False))
            a = parent_sidecar(colors, target=int(rng.integers(4)), slots=(5, 1, 3, 0), tag=str(k))
            b = json.loads(json.dumps(a, sort_keys=True))
            c = copy.deepcopy(a)
            c["objects"] = dict(reversed(list(c["objects"].items())))
            with self.subTest(k=k):
                self.assertEqual(draw(a, copy_index=k), draw(b, copy_index=k))
                self.assertEqual(draw(a, copy_index=k), draw(c, copy_index=k))

    def test_seed_formula_and_redraw(self):
        """§8.6 seed = derive_seed(dr_seed, traj_fp, copy_index, r); r = 0 unless taken, then the next r."""
        parent = parent_sidecar()
        traj = parent["fingerprints"]["traj_fp"]
        first = draw(parent, dr_seed=5, copy_index=2)
        self.assertEqual(first.seed, derive_seed(5, traj, 2, 0))
        second = draw(parent, dr_seed=5, copy_index=2, taken={first.appearance_fp})
        self.assertEqual(second.seed, derive_seed(5, traj, 2, 1))
        self.assertNotEqual(second.appearance_fp, first.appearance_fp)

    def test_fingerprint_matches_fields(self):
        """§8.6/§8.2 appearance_fp = appearance_fp(colors, instruction, light_exposure, robot_color); colors covers
        exactly the parent's active objects (null for uncoloured ones)."""
        parent = parent_sidecar()
        for i in range(10):
            d = draw(parent, copy_index=i)
            with self.subTest(i=i):
                self.assertEqual(d.appearance_fp, appearance_fp(dict(d.colors), d.instruction, d.light_exposure,
                                                                d.robot_color))
                self.assertEqual(set(d.colors), set(parent["objects"]))
                self.assertIsNone(d.colors["small_box"])

    def test_siblings_differ_and_never_equal_parent(self):
        """§8.6 copies of one parent have pairwise distinct appearance_fp, none equal to the parent's; inputs
        (dr_seed, copy_index) change the draw; taken is not mutated."""
        parent = parent_sidecar()
        taken = {parent["fingerprints"]["appearance_fp"]}
        fps = []
        for i in range(25):
            snapshot = set(taken)
            d = draw(parent, copy_index=i, taken=taken)
            self.assertEqual(taken, snapshot)
            fps.append(d.appearance_fp)
            taken.add(d.appearance_fp)
        self.assertEqual(len(set(fps)), 25)
        self.assertNotIn(parent["fingerprints"]["appearance_fp"], fps)
        self.assertNotEqual(draw(parent, dr_seed=1).seed, draw(parent, dr_seed=2).seed)

    def test_parent_excluded_even_if_not_in_taken(self):
        """§8.6/D13 a draw never reproduces the parent's appearance, even with an empty taken set; a space with no
        new appearance left raises GenerationError."""
        spec = task_spec(palette={"red": (1, 0, 0), "blue": (0, 0, 1), "purple": (1, 0, 1)}, holdout_colors=("purple",))
        parent = parent_sidecar(("red",))
        for i in range(20):
            d = draw(parent, spec=spec, terms=(), copy_index=i)
            with self.subTest(i=i):
                self.assertEqual(d.colors["cube_0"], "blue")
                self.assertNotEqual(d.appearance_fp, parent["fingerprints"]["appearance_fp"])
        only = draw(parent, spec=spec, terms=())
        with self.assertRaises(GenerationError):
            draw(parent, spec=spec, terms=(), copy_index=1, taken={only.appearance_fp})

    def test_no_effective_change_raises(self):
        """D13 a copy without recolour or DR terms equals its parent and cannot be drawn: GenerationError."""
        with self.assertRaises(GenerationError):
            draw(parent_sidecar(), terms=(), recolor=False, max_redraws=5)

    def test_target_unique_and_holdout_never_drawn(self):
        """§8.6/§4.5 step 2 recolour: target colour unique among the active set members, distinct colours when
        distinct_colors, never a hold-out colour; objects outside the set keep their colour."""
        parent = parent_sidecar(("red", "green", "blue", "yellow", "orange"), target=2, slots=(0, 2, 3, 4, 5))
        seen = set()
        for i in range(200):
            d = draw(parent, copy_index=i)
            cubes = cube_colors(parent, d)
            target = cubes[parent["roles"]["target"]]
            seen.update(cubes.values())
            with self.subTest(i=i):
                self.assertNotIn("purple", cubes.values())
                self.assertEqual(sorted(cubes), sorted(n for n in parent["objects"] if n.startswith("cube_")))
                self.assertEqual(list(cubes.values()).count(target), 1)
                self.assertEqual(len(set(cubes.values())), len(cubes))
                self.assertIsNone(d.colors["small_box"])
                self.assertIn(target, PALETTE)
        self.assertEqual(seen, set(PALETTE) - {"purple"})

    def test_non_distinct_colors(self):
        """§4.5 step 2 with distinct_colors = false: distractors may repeat, the target colour stays unique."""
        spec = task_spec(palette={"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "purple": (1, 0, 1)},
                         distinct=False)
        parent = parent_sidecar(("red", "green", "blue", "green", "blue"))
        repeated = False
        for i in range(60):
            d = draw(parent, spec=spec, copy_index=i)
            cubes = cube_colors(parent, d)
            target = cubes[parent["roles"]["target"]]
            with self.subTest(i=i):
                self.assertEqual(list(cubes.values()).count(target), 1)
                self.assertNotIn("purple", cubes.values())
            repeated |= len(set(cubes.values())) < len(cubes)
        self.assertTrue(repeated)

    def test_holdout_property_random(self):
        """§4.6 [task.holdout] colours are never drawn by DR, for random palettes, hold-outs and parents."""
        names = ["c%d" % i for i in range(9)]
        for k in range(20):
            rng = np.random.default_rng(k)
            hold = tuple(rng.choice(names, int(rng.integers(1, 3)), replace=False))
            allowed = [c for c in names if c not in hold]
            n = int(rng.integers(1, len(allowed) + 1))
            colors = tuple(rng.choice(allowed, n, replace=False))
            spec = task_spec(palette={c: (0.1, 0.2, 0.3) for c in names}, holdout_colors=hold)
            parent = parent_sidecar(colors, target=int(rng.integers(n)), tag=f"h{k}")
            for i in range(10):
                d = draw(parent, spec=spec, copy_index=i)
                with self.subTest(k=k, i=i):
                    self.assertFalse(set(d.colors.values()) & set(hold))

    def test_recolor_false_keeps_colours(self):
        """§8.6 recolor = false keeps every colour of the parent; light/robot colour still vary."""
        parent = parent_sidecar()
        for i in range(10):
            d = draw(parent, recolor=False, copy_index=i)
            with self.subTest(i=i):
                self.assertEqual(dict(d.colors), {n: o["color"] for n, o in parent["objects"].items()})

    def test_instruction_rerendered(self):
        """§8.6 the instruction is re-rendered from the template with the new colours; with several templates it is
        one of them."""
        parent = parent_sidecar()
        for i in range(20):
            d = draw(parent, copy_index=i)
            target = d.colors[parent["roles"]["target"]]
            with self.subTest(i=i):
                self.assertEqual(d.instruction, f"Pick up the {target} cube and place it in the box")
        spec = task_spec(templates=(TEMPLATE, "Put the {target.color} one in the box", "Grab {target.color}"))
        forms = set()
        for i in range(40):
            d = draw(parent, spec=spec, copy_index=i)
            target = d.colors[parent["roles"]["target"]]
            options = {t.replace("{target.color}", target) for t in spec.instruction_templates}
            with self.subTest(i=i):
                self.assertIn(d.instruction, options)
            forms.add(d.instruction.split(" ")[0])
        self.assertGreater(len(forms), 1)

    def test_terms(self):
        """§8.6/§8.5 'light' draws an exposure inside exposure_range, 'robot_color' one of robot_colors; absent
        terms give None; unknown terms raise ValueError."""
        parent = parent_sidecar()
        for terms in ((), ("light",), ("robot_color",), ("light", "robot_color")):
            for i in range(15):
                d = draw(parent, terms=terms, copy_index=i, exposure_range=(0.7, 0.9))
                with self.subTest(terms=terms, i=i):
                    if "light" in terms:
                        self.assertTrue(0.7 <= d.light_exposure <= 0.9)
                    else:
                        self.assertIsNone(d.light_exposure)
                    if "robot_color" in terms:
                        self.assertIn(d.robot_color, ROBOT_COLORS)
                    else:
                        self.assertIsNone(d.robot_color)
        with self.assertRaises(ValueError):
            draw(parent, terms=("camera_wrist",))


class CopySidecarTests(unittest.TestCase):
    """WP5 §8.6 dr_copy_sidecar and §9.1 source / appearance keys."""

    def test_copy_fields(self):
        """§8.6 the copy keeps traj_fp and scenario_key, sets source, recomputes appearance_fp and episode_fp,
        applies colours/instruction/appearance; everything per-frame is unchanged; the parent is not mutated."""
        parent = parent_sidecar()
        frozen_parent = copy.deepcopy(parent)
        d = draw(parent, copy_index=1)
        out = dr_copy_sidecar(parent, d, episode_index=42, source_repo_id="local/fx_color")
        self.assertEqual(parent, frozen_parent)
        pf = parent["fingerprints"]
        self.assertEqual(out["episode_index"], 42)
        self.assertEqual(out["fingerprints"], {"scenario_key": pf["scenario_key"], "spec_fp": pf["spec_fp"],
                                               "traj_fp": pf["traj_fp"], "appearance_fp": d.appearance_fp,
                                               "episode_fp": episode_fp(pf["traj_fp"], d.appearance_fp)})
        self.assertEqual(out["source"], {"repo_id": "local/fx_color", "episode_index": 3, "traj_fp": pf["traj_fp"],
                                         "episode_fp": pf["episode_fp"]})
        self.assertEqual(out["appearance"], {"light_exposure": d.light_exposure, "robot_color": d.robot_color})
        self.assertEqual(out["instruction"], d.instruction)
        self.assertEqual({n: o["color"] for n, o in out["objects"].items()}, dict(d.colors))
        self.assertEqual(out["generator"]["name"], "dr_replay")
        self.assertIs(out["legacy_v1"], False)
        self.assertNotEqual(out["fingerprints"]["episode_fp"], pf["episode_fp"])
        for key in ("num_frames", "fps", "task", "arm", "scene", "roles", "kind", "success", "stage", "imitate",
                    "segments", "phases", "ee_pose", "attempt", "stages", "perturbation"):
            with self.subTest(key=key):
                self.assertEqual(out[key], parent[key])
        for name, obj in out["objects"].items():
            with self.subTest(obj=name):
                self.assertEqual({k: v for k, v in obj.items() if k != "color"},
                                 {k: v for k, v in parent["objects"][name].items() if k != "color"})
        self.assertEqual(json.loads(json.dumps(out)), out)

    def test_episode_fps_unique_over_copies(self):
        """§8.6 + no-duplicates: parent plus N copies have N + 1 distinct episode_fp values."""
        parent = parent_sidecar()
        taken = {parent["fingerprints"]["appearance_fp"]}
        fps = {parent["fingerprints"]["episode_fp"]}
        for i in range(8):
            d = draw(parent, copy_index=i, taken=taken)
            taken.add(d.appearance_fp)
            fps.add(dr_copy_sidecar(parent, d, episode_index=10 + i, source_repo_id="r")["fingerprints"]["episode_fp"])
        self.assertEqual(len(fps), 9)


if __name__ == "__main__":
    unittest.main()
