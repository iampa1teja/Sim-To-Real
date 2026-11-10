"""CPU regressions for concrete USD paths, without importing the simulator.

Load the pure path helper from its source AST so these tests need neither Kit
nor fake Isaac Lab modules installed in the host Python environment.
"""

import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


def _load_path_helper():
    source = Path(__file__).resolve().parents[2] / "src/sim/envs/mdp/_common.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_env_prim_path")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_env_prim_path"]


_env_prim_path = _load_path_helper()


class EnvPrimPathTests(unittest.TestCase):
    def resolve(self, path, index=0, env_paths=None):
        scene = SimpleNamespace()
        if env_paths is not None:
            scene.env_prim_paths = env_paths
        env = SimpleNamespace(scene=scene)
        asset = SimpleNamespace(cfg=SimpleNamespace(prim_path=path))
        return _env_prim_path(env, asset, index)

    def test_both_expanded_regexes_resolve_to_concrete_usd_path(self):
        for namespace in ("env_.*", "env_[^/]+"):
            with self.subTest(namespace=namespace):
                self.assertEqual(
                    self.resolve(f"/World/envs/{namespace}/Robot/gripper", env_paths=["/World/envs/env_0"]),
                    "/World/envs/env_0/Robot/gripper",
                )

    def test_scene_namespace_is_authoritative_for_each_environment(self):
        paths = ["/Sim/instances/arm_a", "/Sim/instances/arm_z"]
        for namespace in ("env_.*", "env_[^/]+", "{ENV_REGEX_NS}"):
            path = f"/World/envs/{namespace}/CUBE" if namespace != "{ENV_REGEX_NS}" else namespace + "/CUBE"
            for index in (0, 1):
                with self.subTest(namespace=namespace, index=index):
                    self.assertEqual(self.resolve(path, index, paths), paths[index] + "/CUBE")

    def test_old_and_new_regexes_have_fallback_without_scene_paths(self):
        for namespace in ("env_.*", "env_[^/]+"):
            with self.subTest(namespace=namespace):
                self.assertEqual(self.resolve(f"/World/envs/{namespace}/Robot", index=2), "/World/envs/env_2/Robot")

    def test_concrete_global_path_is_preserved(self):
        self.assertEqual(self.resolve("/World/Lighting/Sun", env_paths=["/World/envs/env_0"]), "/World/Lighting/Sun")

    def test_only_complete_environment_regex_segments_are_replaced(self):
        self.assertEqual(
            self.resolve("/World/envs/env_.*_preview/Robot", env_paths=["/World/envs/env_0"]),
            "/World/envs/env_.*_preview/Robot",
        )


if __name__ == "__main__":
    unittest.main()
