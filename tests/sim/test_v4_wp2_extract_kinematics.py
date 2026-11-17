"""WP2: sim/generate/extract_kinematics.py — the pure kinematics.json builder (contracts §5.2, §1.6, §12).

Only the Isaac-free part is tested: ``build_kinematics`` and that the module imports without Isaac / pxr.
The USD reading in ``main()`` is container-only.
"""

import ast
import copy
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from core.arms.kinematics import KinematicChain, load_kinematics_json
from core.generate.errors import KinematicsError
from sim.generate.extract_kinematics import build_kinematics, main  # noqa: F401
from v4_support import SRC_DIR
from v4_support.fx_arms import FX_ARMS, fx_arm_table, fx_joint_names, fx_kinematics, fx_reference_tcp, random_q

ISAAC_MODULES = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch")
MODULE_PATH = SRC_DIR / "sim" / "generate" / "extract_kinematics.py"
_LETTER = {(1, 0, 0): "X", (0, 1, 0): "Y", (0, 0, 1): "Z"}


def usd_rows(name):
    """build_kinematics input rows (§5.2) equivalent to the fx arm's kinematics.json."""
    rows = []
    for j in fx_kinematics(name)["joints"]:
        rows.append({"name": j["name"], "parent": j["parent"], "child": j["child"],
                     "axis": _LETTER[tuple(int(v) for v in j["axis"])],
                     "pos0": list(j["parent_frame"]["position"]), "rot0_wxyz": list(j["parent_frame"]["quat_wxyz"]),
                     "pos1": list(j["child_frame"]["position"]), "rot1_wxyz": list(j["child_frame"]["quat_wxyz"]),
                     "lower_deg": math.degrees(j["limits_rad"][0]), "upper_deg": math.degrees(j["limits_rad"][1])})
    return rows


def build(name, rows=None):
    data = fx_kinematics(name)
    return build_kinematics(usd_rows(name) if rows is None else rows, data["base_link"], data["arm"], data["source"])


def assert_close_tree(test, got, want, path="$"):
    if isinstance(want, dict):
        test.assertIsInstance(got, dict, path)
        test.assertEqual(set(got), set(want), path)
        for key in want:
            assert_close_tree(test, got[key], want[key], f"{path}.{key}")
    elif isinstance(want, list):
        test.assertIsInstance(got, list, path)
        test.assertEqual(len(got), len(want), path)
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close_tree(test, g, w, f"{path}[{i}]")
    elif isinstance(want, (int, float)) and not isinstance(want, bool):
        test.assertIsInstance(got, (int, float), path)
        test.assertNotIsInstance(got, bool, path)
        test.assertAlmostEqual(float(got), float(want), delta=1e-12, msg=path)
    else:
        test.assertEqual(got, want, path)


class BuildKinematicsTests(unittest.TestCase):
    """WP2 §5.2 build_kinematics(joints, base_link, arm, source) -> kinematics.json dict (pure)."""

    def test_output_matches_schema_document(self):
        """§5.2 USD rows (axis letter, degrees) become the schema_version 1 document (unit axis vectors, radians)."""
        for name in FX_ARMS:
            with self.subTest(arm=name):
                assert_close_tree(self, build(name), fx_kinematics(name))

    def test_output_is_plain_json_and_loads(self):
        """§5.2 the output is plain JSON (no NaN) and passes load_kinematics_json."""
        for name in FX_ARMS:
            with self.subTest(arm=name), tempfile.TemporaryDirectory() as tmp:
                out = build(name)
                text = json.dumps(out, allow_nan=False)
                path = Path(tmp) / "kinematics.json"
                path.write_text(text)
                self.assertEqual(load_kinematics_json(path), json.loads(text))

    def test_fk_of_built_document_matches_reference(self):
        """§5.2 FK of the built document equals the reference FK (the formula of check_grasp_stability)."""
        for name in FX_ARMS:
            table = fx_arm_table(name)
            kin = table["kinematics"]
            chain = KinematicChain.from_dict(build(name), q_names=fx_joint_names(name),
                                             chain_joints=table["expert"]["arm_joints"], ee_link=kin["ee_link"],
                                             tcp_position_m=kin["tcp_position_m"], tcp_quat_wxyz=kin["tcp_quat_wxyz"],
                                             approach_axis=kin["approach_axis"], closing_axis=kin["closing_axis"])
            rng = np.random.default_rng(0)
            for k in range(20):
                q = random_q(name, rng)
                with self.subTest(arm=name, k=k):
                    np.testing.assert_allclose(chain.fk(q), fx_reference_tcp(name, q), rtol=0, atol=1e-9)

    def test_joints_are_topologically_ordered(self):
        """§5.2 joints are listed parent-before-child (USD traversal order is arbitrary); chain order is kept."""
        for name in FX_ARMS:
            self.assertEqual([j["name"] for j in build(name)["joints"]], list(fx_joint_names(name)))
            for seed in range(5):
                rows = usd_rows(name)
                random.Random(seed).shuffle(rows)
                with self.subTest(arm=name, seed=seed):
                    out = build(name, rows)
                    known = {out["base_link"]}
                    for joint in out["joints"]:
                        self.assertIn(joint["parent"], known)
                        known.add(joint["child"])
                    by_name = {j["name"]: j for j in out["joints"]}
                    self.assertEqual(set(by_name), set(fx_joint_names(name)))
                    for want in fx_kinematics(name)["joints"]:
                        assert_close_tree(self, by_name[want["name"]], want)

    def test_pure(self):
        """§5.2 build_kinematics does not mutate its inputs."""
        rows = usd_rows("fx_arm6")
        before = copy.deepcopy(rows)
        source = dict(fx_kinematics("fx_arm6")["source"])
        build_kinematics(rows, "fx6_base", "fx_arm6", source)
        self.assertEqual(rows, before)
        self.assertEqual(source, fx_kinematics("fx_arm6")["source"])

    def test_invalid_rows_raise_kinematics_error(self):
        """§1.5 a bad axis letter or an unconnected chain raises KinematicsError."""
        bad_axis = usd_rows("fx_arm5")
        bad_axis[1]["axis"] = "W"
        unconnected = usd_rows("fx_arm5")
        unconnected[2]["parent"] = "fx5_nowhere"
        for label, rows in {"bad_axis": bad_axis, "unconnected": unconnected}.items():
            with self.subTest(case=label):
                with self.assertRaises(KinematicsError):
                    build_kinematics(rows, "fx5_base", "fx_arm5", fx_kinematics("fx_arm5")["source"])


class ModuleHygieneTests(unittest.TestCase):
    """WP2 §1.6 extract_kinematics keeps Isaac / pxr imports inside functions."""

    def test_no_module_level_isaac_imports(self):
        """§1.6 no top-level import of isaaclab / isaacsim / omni / carb / pxr / warp / lerobot / torch."""
        tree = ast.parse(MODULE_PATH.read_text(), feature_version=(3, 11))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for module in names:
                self.assertNotIn(module.split(".")[0], ISAAC_MODULES, module)
        self.assertIn("main", [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])
        self.assertIn("build_kinematics", [n.name for n in tree.body if isinstance(n, ast.FunctionDef)])

    def test_import_in_clean_interpreter(self):
        """§1.6 importing the module in a fresh interpreter loads no Isaac, pxr or torch module."""
        code = textwrap.dedent("""
            import sys
            import sim.generate.extract_kinematics
            bad = sorted(m for m in sys.modules if m.split(".")[0] in %r)
            assert not bad, bad
            print("ok")
        """ % (ISAAC_MODULES,))
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
