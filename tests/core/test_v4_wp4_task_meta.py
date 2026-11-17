"""WP4: sidecar v2 writer, reader, validator, naming and listing (contracts §9.1, §9.3).

The sidecars here are built by ``make_sidecar`` from the WP0 factories (``v4_support.factories``) so
the embedded ``perturbation`` / ``recovery`` / ``noise`` / ``fingerprints`` objects are exactly what
``PerturbationSpec.to_json()`` etc. produce.
"""

import ast
import copy
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import core.datasets
from core.datasets import task_meta
from core.datasets.task_meta import (LABEL_FEATURES, SIDECAR_V2_DIR, list_sidecars, load_episode, read_sidecar,
                                     sidecar_path, v1_to_v2, validate_sidecar, write_sidecar)
from core.generate.errors import SchemaError
from v4_support import REPO_ROOT
from v4_support.factories import (hex_digest, make_episode_spec, make_fingerprints, make_noise, make_perturbation,
                                  make_recovery_info)

PUBLIC = {"SIDECAR_V2_DIR", "LABEL_FEATURES", "sidecar_path", "write_sidecar", "read_sidecar", "load_episode",
          "v1_to_v2", "validate_sidecar", "list_sidecars"}
TOP_KEYS = ("schema_version", "episode_index", "num_frames", "fps", "task", "arm", "scene", "reference_frame",
            "units", "angle_units", "joint_names", "objects", "ee_pose", "instruction", "roles", "kind", "success",
            "failure_mode", "furthest_stage", "stages", "stage", "imitate", "phases", "segments", "perturbation",
            "recovery", "noise", "attempt", "fingerprints", "appearance", "source", "generator", "legacy_v1")
STAGES = ["reached", "grasped", "lifted", "placed"]
JOINTS = ["fx5_pan", "fx5_lift", "fx5_elbow", "fx5_flex", "fx5_roll", "fx5_grip"]
FORBIDDEN_ROOTS = {"isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch", "random", "yaml",
                   "scipy", "huggingface_hub", "fsspec", "sim"}
DELETE = object()


# ---------------------------------------------------------------------------------------------
# Builders (test-local helpers)


def _unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _rows(values, rounded):
    return [[round(float(v), 6) if rounded else float(v) for v in row] for row in values]


def _runs(names, n):
    """Run-length phases [{name, start, end}] from boundaries."""
    out, start = [], 0
    for name, end in names:
        if end > start:
            out.append({"name": name, "start": start, "end": end})
            start = end
    assert start == n
    return out


def _phase_at(phases, t):
    return next(p["name"] for p in phases if p["start"] <= t < p["end"])


def make_sidecar(kind="success", *, n=40, index=0, rng=None, rounded=True, fps=30):
    """A valid v2 sidecar (§9.1) of ``kind`` with ``n`` frames.

    Layout: objects ``cube_0`` (target, red, set "cubes") and ``box`` (container); phases approach /
    grasp / lift; failure: nominal, perturbation (5 frames), failure; recovery: nominal, perturbation,
    recovery, nominal (§7.2); imitate false on the 3 injected frames after onset. ``rng`` varies the
    poses (unit quaternions); ``rounded`` rounds poses to 6 decimals as ``write_sidecar`` stores them.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    spec = make_episode_spec(kind)
    phases = _runs([("approach", n // 3), ("grasp", 2 * n // 3), ("lift", n)], n)
    onset = n // 4
    if kind == "success":
        bounds = [("nominal", n)]
    elif kind == "failure":
        bounds = [("nominal", onset), ("perturbation", onset + 5), ("failure", n)]
    else:
        detected, back = onset + 4, onset + 12
        bounds = [("nominal", onset), ("perturbation", detected), ("recovery", back), ("nominal", n)]
    segments, start = [], 0
    for label, end in bounds:
        segments.append({"label": label, "start": start, "end": end, "phase": _phase_at(phases, start)})
        start = end
    imitate = [not (kind != "success" and onset <= t < onset + 3) for t in range(n)]
    stage = [min(len(STAGES) - 1, t * len(STAGES) // n) if t else -1 for t in range(n)]
    if kind == "recovery":
        for t in range(onset + 4, onset + 8):
            stage[t] = 0

    def track(base):
        pos = np.asarray(base) + rng.uniform(-0.01, 0.01, (n, 3))
        return {"position": _rows(pos, rounded), "orientation_wxyz": _rows(_unit_quats(rng, n), rounded)}

    objects = {
        "cube_0": {"set": "cubes", "size_m": [0.02, 0.02, 0.02], "color": "red", "active": True, "role": "target",
                   **track((0.2, 0.05, 0.01))},
        "box": {"set": None, "size_m": [0.12, 0.1, 0.05], "color": None, "active": True, "role": "container",
                **track((0.25, -0.05, 0.0))},
    }
    success = {"success": True, "failure": False, "recovery": True}[kind]
    perturbation = None
    if spec.perturbation is not None:
        perturbation = dict(spec.perturbation.to_json(), onset_frame=onset, applied=True)
    payload = {
        "schema_version": 2, "episode_index": index, "num_frames": n, "fps": fps,
        "task": spec.task, "arm": spec.arm, "scene": spec.scene, "reference_frame": "robot base link (fx5_base)",
        "units": "m", "angle_units": "rad", "joint_names": list(JOINTS), "objects": objects,
        "ee_pose": track((0.2, 0.0, 0.15)), "instruction": spec.instruction, "roles": dict(spec.roles),
        "kind": kind, "success": success, "failure_mode": None if success else "missed_grasp",
        "furthest_stage": "placed" if success else "grasped", "stages": list(STAGES), "stage": stage,
        "imitate": imitate, "phases": phases, "segments": segments, "perturbation": perturbation,
        "recovery": make_recovery_info(detected_frame=onset + 4, back_on_plan_frame=onset + 12).to_json()
        if kind == "recovery" else None,
        "noise": make_noise().to_json() if kind == "success" else None,
        "attempt": {"index": 3 + index, "seed": 2 ** 64 - 5, "shard": [1, 4], "spec_fp": hex_digest(f"spec{index}")},
        "fingerprints": make_fingerprints(f"{kind}{index}").to_json(),
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "scripted_expert", "git_commit": "0123abcd"}, "legacy_v1": False,
    }
    return payload


def make_dr_copy(index=1, rng=None):
    """A valid DR-copy sidecar: generator dr_replay, a source block and appearance values."""
    payload = make_sidecar("failure", index=index, rng=rng)
    payload["source"] = {"repo_id": "local/fx_source", "episode_index": 0, "traj_fp": hex_digest("t"),
                         "episode_fp": hex_digest("e")}
    payload["appearance"] = {"light_exposure": 0.75, "robot_color": "black"}
    payload["generator"] = {"name": "dr_replay", "git_commit": None}
    payload["objects"]["cube_0"]["color"] = "green"
    return payload


def rounded_copy(payload):
    """``payload`` with object and ee positions/quaternions rounded to 6 decimals (§9.1)."""
    out = copy.deepcopy(payload)
    tracks = list(out["objects"].values()) + ([out["ee_pose"]] if out["ee_pose"] is not None else [])
    for track in tracks:
        for key in ("position", "orientation_wxyz"):
            track[key] = [[round(v, 6) for v in row] for row in track[key]]
    return out


def mutated(payload, path, value):
    """Deep copy of ``payload`` with ``path`` (tuple of keys/indices) set to ``value`` (DELETE removes it)."""
    out = copy.deepcopy(payload)
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


def dir_files(root):
    directory = Path(root) / "task_meta"
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


# ---------------------------------------------------------------------------------------------


class ModuleSurfaceTests(unittest.TestCase):
    """WP4 §2/§1.6 public names and imports of core.datasets.task_meta."""

    def test_public_names(self):
        """§2 task_meta defines exactly the contract's public names (helpers are private)."""
        defined = {name for name, value in vars(task_meta).items()
                   if not name.startswith("_") and not inspect.ismodule(value)
                   and getattr(value, "__module__", task_meta.__name__) == task_meta.__name__}
        self.assertEqual(defined, PUBLIC)

    def test_signatures(self):
        """§9.1 parameter names and keyword-only flags."""
        expect = {
            "sidecar_path": ["root", "index"], "write_sidecar": ["root", "payload"], "read_sidecar": ["root", "index"],
            "load_episode": ["root", "index", "task", "arm", "scene"], "v1_to_v2": ["v1", "task", "arm", "scene"],
            "validate_sidecar": ["payload", "num_frames", "fps"], "list_sidecars": ["root"],
        }
        for name, params in expect.items():
            with self.subTest(name=name):
                sig = inspect.signature(getattr(task_meta, name))
                self.assertEqual(list(sig.parameters), params)
        load = inspect.signature(load_episode).parameters
        self.assertEqual((load["task"].default, load["arm"].default, load["scene"].default),
                         ("pick_place_cube", None, None))
        for name in ("task", "arm", "scene"):
            self.assertEqual(load[name].kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertEqual(inspect.signature(v1_to_v2).parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        val = inspect.signature(validate_sidecar).parameters
        self.assertEqual((val["num_frames"].default, val["fps"].default), (None, None))
        self.assertEqual(val["num_frames"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_no_forbidden_imports(self):
        """§1.6 task_meta imports only stdlib, numpy, pyarrow and core (no Isaac, torch, lerobot, sim)."""
        tree = ast.parse(Path(task_meta.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            with self.subTest(line=node.lineno):
                self.assertFalse(roots & FORBIDDEN_ROOTS, roots)
                self.assertTrue(roots <= set(sys.stdlib_module_names) | {"numpy", "pyarrow", "core"}, roots)

    def test_import_pulls_no_heavy_modules(self):
        """§1.6 importing core.datasets.task_meta loads no Isaac, torch, lerobot or sim modules."""
        roots = sorted(FORBIDDEN_ROOTS - {"random"})
        code = ("import sys; import core.datasets.task_meta; "
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {roots!r}))")
        src = Path(core.datasets.__file__).resolve().parents[2]
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_label_features_golden(self):
        """§9.3 LABEL_FEATURES: stage and imitate int64 (1,), ee_pose float32 (7,) in WXYZ disk order."""
        self.assertEqual(LABEL_FEATURES, {
            "label.stage": {"dtype": "int64", "shape": (1,), "names": ["stage"]},
            "label.imitate": {"dtype": "int64", "shape": (1,), "names": ["imitate"]},
            "label.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        })
        self.assertEqual(list(LABEL_FEATURES), ["label.stage", "label.imitate", "label.ee_pose"])
        for value in LABEL_FEATURES.values():
            self.assertIs(type(value["shape"]), tuple)


class NamingTests(unittest.TestCase):
    """WP4 §9.1 file naming: <root>/task_meta/episode_NNNNNN.json."""

    def test_dir_constant(self):
        """§9.1 SIDECAR_V2_DIR is "task_meta"."""
        self.assertEqual(SIDECAR_V2_DIR, "task_meta")

    def test_sidecar_path(self):
        """§9.1 six-digit zero padding; wider indices are not truncated; str or Path root; returns Path."""
        cases = {0: "episode_000000.json", 7: "episode_000007.json", 42: "episode_000042.json",
                 999999: "episode_999999.json", 1234567: "episode_1234567.json"}
        with tempfile.TemporaryDirectory() as tmp:
            for index, name in cases.items():
                for root in (tmp, Path(tmp)):
                    with self.subTest(index=index, root=type(root).__name__):
                        path = sidecar_path(root, index)
                        self.assertIsInstance(path, Path)
                        self.assertEqual(path, Path(tmp) / "task_meta" / name)
            self.assertEqual(sidecar_path(tmp, np.int64(5)), Path(tmp) / "task_meta" / "episode_000005.json")
            self.assertFalse((Path(tmp) / "task_meta").exists(), "sidecar_path must not create folders")

    def test_sidecar_path_rejects_bad_index(self):
        """§1.5 negative index → ValueError; bool, float or str → TypeError."""
        with self.assertRaises(ValueError):
            sidecar_path("/x", -1)
        for bad in (True, 1.0, "3"):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                sidecar_path("/x", bad)

    def test_write_returns_path_from_payload_index(self):
        """§9.1 write_sidecar returns sidecar_path(root, payload["episode_index"])."""
        with tempfile.TemporaryDirectory() as tmp:
            for index in (0, 17):
                with self.subTest(index=index):
                    path = write_sidecar(tmp, make_sidecar(index=index))
                    self.assertEqual(path, sidecar_path(tmp, index))
                    self.assertTrue(path.is_file())


class RoundTripTests(unittest.TestCase):
    """WP4 §9.1 write_sidecar → read_sidecar returns the payload."""

    def test_round_trip_each_kind(self):
        """§9.1 success, failure and recovery sidecars round-trip to an equal dict."""
        with tempfile.TemporaryDirectory() as tmp:
            for index, kind in enumerate(("success", "failure", "recovery")):
                with self.subTest(kind=kind):
                    payload = make_sidecar(kind, index=index)
                    validate_sidecar(payload)
                    write_sidecar(tmp, payload)
                    self.assertEqual(read_sidecar(tmp, index), payload)
                    self.assertEqual(load_episode(tmp, index), payload)

    def test_round_trip_property(self):
        """§9.1 random valid sidecars (varied N, kind, fps, poses) round-trip exactly."""
        with tempfile.TemporaryDirectory() as tmp:
            for k in range(24):
                rng = np.random.default_rng(k)
                kind = ("success", "failure", "recovery")[k % 3]
                n = int(rng.integers(20, 120))
                payload = make_sidecar(kind, n=n, index=k, rng=rng, fps=int(rng.choice([15, 30, 60])))
                with self.subTest(k=k, kind=kind, n=n):
                    write_sidecar(tmp, payload)
                    back = read_sidecar(tmp, k)
                    self.assertEqual(back, payload)
                    self.assertEqual(json.dumps(back, sort_keys=True), json.dumps(payload, sort_keys=True))

    def test_round_trip_dr_copy_and_legacy(self):
        """§9.1 a DR copy (source, appearance) and a legacy_v1 sidecar round-trip."""
        legacy = make_sidecar("success", index=2)
        legacy.update(legacy_v1=True, fingerprints=None, attempt=None, task=None, arm=None, scene=None,
                      stage=None, ee_pose=None, phases=[], stages=[], success=None, noise=None)
        with tempfile.TemporaryDirectory() as tmp:
            for payload in (make_dr_copy(index=1), legacy):
                with self.subTest(index=payload["episode_index"]):
                    write_sidecar(tmp, payload)
                    self.assertEqual(read_sidecar(tmp, payload["episode_index"]), payload)

    def test_poses_rounded_to_6_decimals(self):
        """§9.1 positions and quaternions are stored rounded to 6 decimals; other floats are kept."""
        payload = make_sidecar("recovery", index=3, rng=np.random.default_rng(5), rounded=False)
        payload["objects"]["cube_0"]["size_m"] = [0.0201234567, 0.02, 0.02]
        payload["noise"] = None
        before = copy.deepcopy(payload)
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, payload)
            back = read_sidecar(tmp, 3)
        self.assertEqual(payload, before, "write_sidecar must not mutate its argument")
        self.assertEqual(back, rounded_copy(before))
        self.assertEqual(back["objects"]["cube_0"]["size_m"], [0.0201234567, 0.02, 0.02])
        self.assertNotEqual(back, before)

    def test_read_returns_independent_plain_json(self):
        """§9.1 read_sidecar returns plain JSON types (dict/list/str/int/float/bool/None)."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("failure"))
            back = read_sidecar(tmp, 0)

        def walk(value):
            if type(value) is dict:
                for item in value.values():
                    walk(item)
            elif type(value) is list:
                for item in value:
                    walk(item)
            else:
                self.assertIn(type(value), (str, int, float, bool, type(None)))
        walk(back)
        self.assertIs(type(back["attempt"]["seed"]), int)
        self.assertEqual(back["attempt"]["seed"], 2 ** 64 - 5)

    def test_file_is_strict_json(self):
        """§9.1 the file is UTF-8 JSON without NaN/Infinity tokens and parses to the stored dict."""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_sidecar(tmp, make_sidecar("success"))
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text), make_sidecar("success"))


class ReadErrorTests(unittest.TestCase):
    """WP4 §9.1 read_sidecar is v2 only and validates."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "task_meta").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def put(self, index, text):
        (self.root / "task_meta" / f"episode_{index:06d}.json").write_text(text)

    def test_missing_file(self):
        """§9.1 a missing sidecar raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_sidecar(self.root, 0)

    def test_rejects_bad_content(self):
        """§9.1 not JSON, NaN tokens, v1 content, a payload of another index → SchemaError."""
        v1 = {"episode_index": 0, "reference_frame": "robot base link (base)", "units": "m", "fps": 30,
              "num_frames": 1, "cube_size_m": [0.02] * 3, "position": [[0, 0, 0]], "orientation_wxyz": [[1, 0, 0, 0]]}
        nan_payload = mutated(make_sidecar(), ("ee_pose", "position", 0, 0), float("nan"))
        cases = {"truncated": json.dumps(make_sidecar())[:-20], "nan": json.dumps(nan_payload),
                 "v1": json.dumps(v1), "other index": json.dumps(make_sidecar(index=1)), "list": "[]"}
        for name, text in cases.items():
            with self.subTest(case=name):
                self.put(0, text)
                with self.assertRaises(SchemaError):
                    read_sidecar(self.root, 0)


class AtomicWriteTests(unittest.TestCase):
    """WP4 §9.1/§8.4 write_sidecar: validate, then temp file + fsync + os.replace."""

    def test_fsync_failure_leaves_no_file(self):
        """§9.1 a failure after the temp file is written (fsync) leaves neither target nor temp file."""
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("os.fsync", side_effect=OSError("io error")):
                with self.assertRaises(OSError):
                    write_sidecar(tmp, make_sidecar())
            self.assertEqual(dir_files(tmp), [])
            self.assertEqual(list_sidecars(tmp), [])

    def test_replace_failure_keeps_previous_version(self):
        """§9.1 if the rename fails, the existing sidecar is intact and no temp file remains."""
        with tempfile.TemporaryDirectory() as tmp:
            first = make_sidecar("success", n=30)
            write_sidecar(tmp, first)
            before = sidecar_path(tmp, 0).read_bytes()
            with mock.patch("os.replace", side_effect=OSError("rename failed")):
                with self.assertRaises(OSError):
                    write_sidecar(tmp, make_sidecar("failure", n=50))
            self.assertEqual(sidecar_path(tmp, 0).read_bytes(), before)
            self.assertEqual(dir_files(tmp), ["episode_000000.json"])
            self.assertEqual(read_sidecar(tmp, 0), first)

    def test_temp_file_in_same_dir_fsynced_then_replaced(self):
        """§9.1 the temp file lives in task_meta/, is fsynced before os.replace and is not a sidecar name."""
        events = []
        real_fsync, real_replace = os.fsync, os.replace

        def fsync(fd):
            events.append(("fsync",))
            return real_fsync(fd)

        def replace(src, dst):
            events.append(("replace", Path(src), Path(dst)))
            return real_replace(src, dst)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("os.fsync", side_effect=fsync), mock.patch("os.replace", side_effect=replace):
                path = write_sidecar(tmp, make_sidecar(index=4))
            replaces = [e for e in events if e[0] == "replace"]
            self.assertEqual(len(replaces), 1)
            _, src, dst = replaces[0]
            self.assertEqual(dst, path)
            self.assertEqual(src.parent, dst.parent)
            self.assertNotEqual(src, dst)
            self.assertNotRegex(src.name, r"^episode_\d+\.json$")
            self.assertIn(("fsync",), events[:events.index(replaces[0])])
            self.assertEqual(dir_files(tmp), ["episode_000004.json"])

    def test_invalid_payload_writes_nothing(self):
        """§9.1 write_sidecar validates first: an invalid payload raises SchemaError and writes no file."""
        bad = mutated(make_sidecar(), ("imitate",), [True])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SchemaError):
                write_sidecar(tmp, bad)
            self.assertEqual(dir_files(tmp), [])
            write_sidecar(tmp, make_sidecar())
            with self.assertRaises(SchemaError):
                write_sidecar(tmp, mutated(make_sidecar(), ("kind",), "partial"))
            self.assertEqual(read_sidecar(tmp, 0), make_sidecar())

    def test_overwrite(self):
        """§9.1 writing an existing index replaces it."""
        with tempfile.TemporaryDirectory() as tmp:
            write_sidecar(tmp, make_sidecar("success", n=30))
            second = make_sidecar("recovery", n=60)
            write_sidecar(tmp, second)
            self.assertEqual(read_sidecar(tmp, 0), second)
            self.assertEqual(dir_files(tmp), ["episode_000000.json"])


class ListingTests(unittest.TestCase):
    """WP4 §9.1 list_sidecars and in-order iteration."""

    def test_empty(self):
        """§9.1 a root without task_meta/ (or an empty one) lists nothing."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(list_sidecars(tmp), [])
            (Path(tmp) / "task_meta").mkdir()
            self.assertEqual(list_sidecars(Path(tmp)), [])

    def test_sorted_and_iterable(self):
        """§9.1 indices are returned sorted as ints, whatever the write order; reading them in order works."""
        order = [5, 0, 12, 3, 1000000, 7]
        with tempfile.TemporaryDirectory() as tmp:
            for index in order:
                write_sidecar(tmp, make_sidecar(("success", "failure", "recovery")[index % 3], index=index))
            listed = list_sidecars(tmp)
            self.assertEqual(listed, sorted(order))
            self.assertTrue(all(type(i) is int for i in listed))
            self.assertEqual([read_sidecar(tmp, i)["episode_index"] for i in listed], sorted(order))
            self.assertEqual([load_episode(tmp, i)["kind"] for i in listed],
                             [("success", "failure", "recovery")[i % 3] for i in sorted(order)])

    def test_ignores_other_files(self):
        """§9.1 temp files, wrong names, other extensions, folders and v1 sidecars are not listed."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_sidecar(root, make_sidecar(index=2))
            directory = root / "task_meta"
            for name in ("episode_7.json", "episode_000008.json.tmp", ".episode_000009.json.tmp", "tmpx1y2.tmp",
                         "episodes.parquet", "episode_000010.JSON", "episode_0000011.json", "episode_00001a.json",
                         "notes.json"):
                (directory / name).write_text("{}")
            (directory / "episode_000013.json").mkdir()
            (root / "pick_place_meta").mkdir()
            (root / "pick_place_meta" / "episode_000000.json").write_text("{}")
            self.assertEqual(list_sidecars(root), [2])


class ValidatorStructureTests(unittest.TestCase):
    """WP4 §9.1 validator: keys, types, schema version, enums."""

    def assert_rejects(self, payload, **kwargs):
        with self.assertRaises(SchemaError):
            validate_sidecar(payload, **kwargs)

    def test_valid_examples_pass(self):
        """§9.1 the builder's sidecars validate and validation does not mutate."""
        for kind in ("success", "failure", "recovery"):
            payload = make_sidecar(kind)
            before = copy.deepcopy(payload)
            with self.subTest(kind=kind):
                self.assertIsNone(validate_sidecar(payload))
                self.assertIsNone(validate_sidecar(payload, num_frames=40, fps=30))
                self.assertEqual(payload, before)
        validate_sidecar(make_dr_copy())

    def test_exact_top_level_keys(self):
        """§9.1 every key is required and unknown keys are rejected."""
        base = make_sidecar("recovery")
        self.assertEqual(set(base), set(TOP_KEYS))
        for key in TOP_KEYS:
            with self.subTest(missing=key):
                self.assert_rejects(mutated(base, (key,), DELETE))
        for key in ("schema", "notes", "episode", "label"):
            with self.subTest(unknown=key):
                self.assert_rejects(mutated(base, (key,), None))
        self.assert_rejects([base])
        self.assert_rejects(None)

    def test_exact_nested_keys(self):
        """§9.1 nested objects have exactly their listed keys (missing or unknown → SchemaError)."""
        base = make_sidecar("recovery")
        base_dr = make_dr_copy()
        cases = [
            (base, ("objects", "cube_0"), "size_m"), (base, ("objects", "box"), "role"),
            (base, ("ee_pose",), "orientation_wxyz"), (base, ("phases", 0), "name"), (base, ("segments", 1), "phase"),
            (base, ("perturbation",), "onset_frame"), (base, ("perturbation",), "applied"),
            (base, ("perturbation",), "group"), (base, ("recovery",), "retries"), (base, ("attempt",), "shard"),
            (base, ("fingerprints",), "traj_fp"), (base, ("appearance",), "robot_color"),
            (base, ("generator",), "git_commit"), (base_dr, ("source",), "repo_id"),
            (make_sidecar("success"), ("noise",), "theta"),
        ]
        for payload, parent, key in cases:
            with self.subTest(missing=parent + (key,)):
                self.assert_rejects(mutated(payload, parent + (key,), DELETE))
            with self.subTest(unknown=parent + ("extra",)):
                self.assert_rejects(mutated(payload, parent + ("extra",), 1))

    def test_schema_version(self):
        """§9.1 schema_version must be the JSON integer 2."""
        for value in (1, 3, "2", 2.0, True, None):
            with self.subTest(value=value):
                self.assert_rejects(mutated(make_sidecar(), ("schema_version",), value))

    def test_unknown_enum_values(self):
        """§9.1/§3.1 kind, segment label, perturbation group and generator name are closed sets."""
        base = make_sidecar("failure")
        cases = [(("kind",), "partial"), (("kind",), "SUCCESS"), (("segments", 1, "label"), "noise"),
                 (("perturbation", "group"), "both"), (("generator", "name"), "human")]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.assert_rejects(mutated(base, path, value))
        for name in ("scripted_expert", "teleop", "dr_replay"):
            with self.subTest(generator=name):
                validate_sidecar(mutated(base, ("generator", "name"), name))

    def test_wrong_types(self):
        """§9.1 field types: ints are JSON ints (not bool/float), bools are bools, strings are strings."""
        base = make_sidecar("recovery")
        cases = [
            (("episode_index",), 0.0), (("episode_index",), -1), (("episode_index",), True), (("num_frames",), 40.0),
            (("fps",), "30"), (("fps",), 0), (("task",), None), (("arm",), 5), (("scene",), None),
            (("units",), "mm"), (("angle_units",), "deg"), (("joint_names",), "fx5_pan"), (("joint_names",), [1, 2]),
            (("instruction",), 3), (("roles",), [["target", "cube_0"]]), (("roles", "target"), None),
            (("success",), 1), (("failure_mode",), 0), (("furthest_stage",), ["placed"]), (("stages",), "reached"),
            (("stages",), [1]), (("stage", 3), 1.0), (("stage", 3), True), (("imitate", 2), 1),
            (("imitate", 2), None), (("legacy_v1",), 0), (("objects",), []), (("objects", "cube_0", "active"), 1),
            (("objects", "cube_0", "active"), False), (("objects", "cube_0", "color"), 3),
            (("objects", "cube_0", "size_m"), [0.02, 0.02]), (("objects", "cube_0", "size_m"), [0.02, 0.02, "a"]),
            (("objects", "cube_0", "position", 0, 1), "0.1"), (("ee_pose",), [[0] * 7] * 40),
            (("perturbation", "applied"), "yes"), (("perturbation", "onset_frame"), 40),
            (("perturbation", "onset_frame"), 1.5), (("attempt", "seed"), 2 ** 64), (("attempt", "seed"), -1),
            (("attempt", "seed"), 12.0), (("attempt", "seed"), "12"), (("attempt", "shard"), [4, 4]),
            (("attempt", "shard"), [0]), (("attempt", "index"), -2), (("attempt", "spec_fp"), "ABC"),
            (("fingerprints", "traj_fp"), "f" * 63), (("fingerprints", "spec_fp"), "F" * 64),
            (("appearance", "light_exposure"), "bright"), (("appearance", "robot_color"), 1),
            (("generator", "git_commit"), 7), (("segments", 0, "phase"), 3), (("phases", 0, "name"), None),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.assert_rejects(mutated(base, path, value))

    def test_reference_frame(self):
        """§1.1/§9.1 v2 accepts "robot base link (<base>)" for any base name, nothing else."""
        for frame in ("robot base link (base)", "robot base link (fx6_base)", "robot base link (base_link)"):
            with self.subTest(frame=frame):
                validate_sidecar(mutated(make_sidecar(), ("reference_frame",), frame))
        for frame in ("world", "robot base link", "robot base link ()", "Robot base link (base)", None):
            with self.subTest(frame=frame):
                self.assert_rejects(mutated(make_sidecar(), ("reference_frame",), frame))

    def test_nullable_fields(self):
        """§9.1 joint_names, ee_pose, instruction, stage, noise, attempt, source may be null."""
        base = make_sidecar("failure")
        for key in ("joint_names", "ee_pose", "instruction", "stage", "noise", "attempt", "source", "success",
                    "failure_mode", "furthest_stage"):
            with self.subTest(key=key):
                payload = mutated(base, (key,), None)
                if key == "failure_mode":
                    payload["success"] = True
                validate_sidecar(payload)
        validate_sidecar(mutated(base, ("objects",), {}))
        validate_sidecar(mutated(base, ("roles",), {}))
        validate_sidecar(mutated(base, ("perturbation", "onset_frame"), None))


class ValidatorValueTests(unittest.TestCase):
    """WP4 §9.1 validator: per-frame lengths, finiteness, quaternions, segments, cross-field rules."""

    def assert_rejects(self, payload, **kwargs):
        with self.assertRaises(SchemaError):
            validate_sidecar(payload, **kwargs)

    def test_non_finite_floats(self):
        """§9.1 NaN or ±inf anywhere is rejected."""
        base = make_sidecar("success")
        paths = [("objects", "cube_0", "position", 5, 2), ("objects", "box", "orientation_wxyz", 0, 0),
                 ("ee_pose", "position", 39, 0), ("ee_pose", "orientation_wxyz", 7, 3),
                 ("objects", "cube_0", "size_m", 1), ("noise", "sigma_rad"), ("appearance", "light_exposure")]
        for path in paths:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(path=path, value=value):
                    self.assert_rejects(mutated(base, path, value))
        failure = make_sidecar("failure")
        with self.subTest(path="perturbation.params"):
            self.assert_rejects(mutated(failure, ("perturbation", "params", "xy_m"), float("nan")))

    def test_per_frame_lengths(self):
        """§9.1 every per-frame array has num_frames rows of the right width."""
        base = make_sidecar("recovery", n=40)
        for path in (("objects", "cube_0", "position"), ("objects", "box", "orientation_wxyz"),
                     ("ee_pose", "position"), ("ee_pose", "orientation_wxyz"), ("stage",), ("imitate",)):
            node = base
            for key in path:
                node = node[key]
            for rows in (node[:-1], node + node[-1:], []):
                with self.subTest(path=path, length=len(rows)):
                    self.assert_rejects(mutated(base, path, rows))
        for path, row in ((("objects", "cube_0", "position", 3), [0.1, 0.2]),
                          (("objects", "cube_0", "orientation_wxyz", 3), [1.0, 0.0, 0.0]),
                          (("ee_pose", "position", 0), [0.1, 0.2, 0.3, 0.4]),
                          (("ee_pose", "orientation_wxyz", 0), 1.0)):
            with self.subTest(path=path, row=row):
                self.assert_rejects(mutated(base, path, row))
        self.assert_rejects(mutated(base, ("num_frames",), 41))
        self.assert_rejects(mutated(base, ("num_frames",), 0))

    def test_quaternion_norm(self):
        """§9.1 quaternion norms within 1e-3 of 1."""
        base = make_sidecar()
        for value, ok in (([1.0, 0.0, 0.0, 0.0], True), ([1.0009, 0.0, 0.0, 0.0], True),
                          ([0.0, 0.0, 0.0, -0.9992], True), ([1.002, 0.0, 0.0, 0.0], False),
                          ([0.5, 0.5, 0.5, 0.0], False), ([0.0, 0.0, 0.0, 0.0], False)):
            for path in (("objects", "cube_0", "orientation_wxyz", 4), ("ee_pose", "orientation_wxyz", 4)):
                with self.subTest(value=value, path=path):
                    payload = mutated(base, path, value)
                    if ok:
                        validate_sidecar(payload)
                    else:
                        self.assert_rejects(payload)

    def test_stage_range(self):
        """§9.1 stage values are in [-1, len(stages))."""
        base = make_sidecar()
        for value, ok in ((-1, True), (0, True), (3, True), (4, False), (-2, False)):
            with self.subTest(value=value):
                payload = mutated(base, ("stage", 10), value)
                if ok:
                    validate_sidecar(payload)
                else:
                    self.assert_rejects(payload)
        self.assert_rejects(mutated(base, ("stages",), []))
        legacy_like = mutated(mutated(base, ("stages",), []), ("stage",), None)
        validate_sidecar(legacy_like)

    def run_cover_cases(self, key, make_item):
        n = 40
        base = make_sidecar("recovery", n=n)
        cases = {
            "gap": [make_item(0, 10), make_item(12, n)],
            "overlap": [make_item(0, 12), make_item(10, n)],
            "late start": [make_item(1, n)],
            "short": [make_item(0, 20), make_item(20, n - 1)],
            "long": [make_item(0, 20), make_item(20, n + 1)],
            "empty item": [make_item(0, 20), make_item(20, 20), make_item(20, n)],
            "reversed item": [make_item(0, 20), make_item(25, 20), make_item(20, n)],
            "out of order": [make_item(20, n), make_item(0, 20)],
            "float bound": [make_item(0, 20.0), make_item(20.0, n)],
        }
        for name, items in cases.items():
            with self.subTest(key=key, case=name):
                self.assert_rejects(mutated(base, (key,), items))
        validate_sidecar(mutated(base, (key,), [make_item(0, 7), make_item(7, 8), make_item(8, n)]))
        validate_sidecar(mutated(base, (key,), [make_item(0, n)]))

    def test_segments_cover(self):
        """§9.1/§7.2 segments are non-empty, in order, contiguous, non-overlapping and cover [0, N)."""
        self.run_cover_cases("segments", lambda s, e: {"label": "nominal", "start": s, "end": e, "phase": None})
        self.assert_rejects(mutated(make_sidecar(), ("segments",), []))

    def test_phases_cover(self):
        """§9.1 phases are run-length, contiguous and cover [0, N); [] (no phase information) is allowed."""
        self.run_cover_cases("phases", lambda s, e: {"name": "grasp", "start": s, "end": e})
        validate_sidecar(mutated(make_sidecar(), ("phases",), []))

    def test_kind_rules(self):
        """§9.1 success ⇒ perturbation null; recovery ⇒ recovery non-null."""
        failure = make_sidecar("failure")
        success = make_sidecar("success")
        recovery = make_sidecar("recovery")
        self.assert_rejects(mutated(success, ("perturbation",), failure["perturbation"]))
        self.assert_rejects(mutated(recovery, ("recovery",), None))
        validate_sidecar(mutated(failure, ("recovery",), recovery["recovery"]))
        validate_sidecar(mutated(failure, ("perturbation",), None))

    def test_success_failure_mode_rule(self):
        """§9.1 success == (failure_mode is None) unless success is null."""
        base = make_sidecar("failure")
        for success, mode, ok in ((True, None, True), (False, "missed_grasp", True), (True, "missed_grasp", False),
                                  (False, None, False), (None, None, True), (None, "missed_grasp", True)):
            with self.subTest(success=success, mode=mode):
                payload = mutated(mutated(base, ("success",), success), ("failure_mode",), mode)
                if ok:
                    validate_sidecar(payload)
                else:
                    self.assert_rejects(payload)

    def test_fingerprints_null_only_if_legacy(self):
        """§9.1 a fingerprint is null only if legacy_v1."""
        base = make_sidecar("success")
        self.assert_rejects(mutated(base, ("fingerprints",), None))
        for key in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp"):
            with self.subTest(key=key):
                self.assert_rejects(mutated(base, ("fingerprints", key), None))
                legacy = mutated(mutated(base, ("fingerprints", key), None), ("legacy_v1",), True)
                validate_sidecar(legacy)
        validate_sidecar(mutated(mutated(base, ("fingerprints",), None), ("legacy_v1",), True))

    def test_legacy_allows_null_task_arm_scene(self):
        """§9.1 task, arm, scene are null only when legacy_v1."""
        base = mutated(make_sidecar(), ("legacy_v1",), True)
        for key in ("task", "arm", "scene"):
            with self.subTest(key=key):
                validate_sidecar(mutated(base, (key,), None))
                self.assert_rejects(mutated(make_sidecar(), (key,), None))

    def test_num_frames_and_fps_arguments(self):
        """§9.1 num_frames and fps must match the arguments when given."""
        payload = make_sidecar(n=40, fps=30)
        validate_sidecar(payload, num_frames=40)
        validate_sidecar(payload, fps=30)
        self.assert_rejects(payload, num_frames=39)
        self.assert_rejects(payload, fps=15)
        self.assert_rejects(payload, num_frames=40, fps=60)

    def test_random_corruptions_property(self):
        """§9.1 one random corruption of a random valid sidecar is always rejected."""
        for k in range(30):
            rng = np.random.default_rng(100 + k)
            kind = ("success", "failure", "recovery")[k % 3]
            n = int(rng.integers(20, 60))
            base = make_sidecar(kind, n=n, rng=rng)
            obj = str(rng.choice(["cube_0", "box"]))
            t = int(rng.integers(0, n))
            corruptions = [
                (("objects", obj, "position", t, int(rng.integers(0, 3))), float("nan")),
                (("objects", obj, "position"), base["objects"][obj]["position"][:t]),
                (("ee_pose", "orientation_wxyz", t), [2.0, 0.0, 0.0, 0.0]),
                (("stage", t), len(STAGES) + int(rng.integers(0, 3))),
                (("imitate",), base["imitate"] + [True]),
                (("segments", 0, "end"), base["segments"][0]["end"] - 1 if base["segments"][0]["end"] > 1 else n + 1),
                (("num_frames",), n + int(rng.integers(1, 5))),
            ]
            path, value = corruptions[int(rng.integers(0, len(corruptions)))]
            with self.subTest(k=k, path=path):
                validate_sidecar(base)
                self.assert_rejects(mutated(base, path, value))


if __name__ == "__main__":
    unittest.main()
