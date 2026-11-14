"""Cached material inputs still update every reset and belong to one stage."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    from pxr import Gf, Usd, UsdGeom, UsdShade
except ImportError:
    Usd = None


@unittest.skipUnless(Usd is not None, "Requires USD from s2r-sim")
class ColorCacheTests(unittest.TestCase):
    def make_stage(self):
        stage = Usd.Stage.CreateInMemory()
        cube = UsdGeom.Cube.Define(stage, "/Cube")
        material = UsdShade.Material.Define(stage, "/Looks/Cube")
        shader = UsdShade.Shader.Define(stage, "/Looks/Cube/Shader")
        from pxr import Sdf
        shader.CreateIdAttr("UsdPreviewSurface")
        color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        color.Set((0, 0, 1))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
        return stage, color

    def test_repeated_resets_reuse_inputs_and_new_stage_resolves_its_own(self):
        stage, color = self.make_stage()
        source = Path(__file__).resolve().parents[2] / "workspace/tasks/pick_place_cube/mdp.py"
        function = next(n for n in ast.parse(source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == "set_cube_color")
        env = SimpleNamespace(num_envs=1, scene={"blue_cube": object()})
        namespace = {"Gf": Gf, "Usd": Usd, "UsdShade": UsdShade,
                     "get_current_stage": lambda: stage,
                     "env_indices": lambda env, ids: np.arange(env.num_envs)[ids] if isinstance(ids, slice) else ids,
                     "_env_prim_path": lambda *_: "/Cube"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        set_color = namespace["set_cube_color"]
        set_color(env, slice(None), (1, 0, 0), "red")
        self.assertEqual(tuple(color.Get()), (1, 0, 0))
        with patch.object(Usd, "PrimRange", side_effect=AssertionError("cached resets must not traverse")):
            set_color(env, np.array([0]), (0, 1, 0), "green")
        self.assertEqual(tuple(color.Get()), (0, 1, 0))
        self.assertEqual(env._pick_place_cube_color, ["green"])
        old_stage = stage  # Keep the original stage alive while inspecting its shader.
        stage, new_color = self.make_stage()
        set_color(env, np.array([0]), (1, 0, 0), "red")
        self.assertIsNot(stage, old_stage)
        self.assertEqual(tuple(new_color.Get()), (1, 0, 0))
        self.assertEqual(tuple(color.Get()), (0, 1, 0))
