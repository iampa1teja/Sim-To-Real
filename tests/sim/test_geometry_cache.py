"""Static geometry caching preserves measurements across environments and stages."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None


@unittest.skipUnless(Usd is not None, "Requires USD from s2r-sim")
class GeometryCacheTests(unittest.TestCase):
    def setUp(self):
        from sim.workspace.scene import geometry
        self.geometry = geometry
        self.anchor = SimpleNamespace(prim_name="Surface", position=(0, 0, 0), up=(0, 0, 1))

    def table(self, stage, path, height):
        root = UsdGeom.Xform.Define(stage, path)
        root.AddTranslateOp().Set((0, 0, height))
        mesh = UsdGeom.Mesh.Define(stage, path + "/Surface")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
        return root.GetPrim()

    def test_each_environment_and_stage_has_its_own_plane(self):
        stages = [Usd.Stage.CreateInMemory(), Usd.Stage.CreateInMemory()]
        for stage in stages:
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        roots = [(stages[0], self.table(stages[0], "/env0", 1)),
                 (stages[0], self.table(stages[0], "/env1", 2)),
                 (stages[1], self.table(stages[1], "/env0", 3))]
        with patch.object(self.geometry, "_measure_table_plane", wraps=self.geometry._measure_table_plane) as fit:
            for _ in range(3):
                for height, (stage, root) in enumerate(roots, 1):
                    plane = self.geometry.table_plane(stage, root, self.anchor)
                    np.testing.assert_allclose(plane, [0, 0, height], atol=1e-12)
                    plane[:] = 100  # A caller must not mutate the cached measurement.
            self.assertEqual(fit.call_count, 3)

    def test_explicit_invalidation_remeasures_edited_geometry(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = self.table(stage, "/env0", 1)
        self.geometry.table_plane(stage, root, self.anchor)
        UsdGeom.Xform(root).GetOrderedXformOps()[0].Set((0, 0, 2))
        self.geometry.clear_geometry_cache(stage)
        np.testing.assert_allclose(self.geometry.table_plane(stage, root, self.anchor), [0, 0, 2], atol=1e-12)

    def test_jaw_vertices_cached_without_caching_object_size_or_clearance(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/Robot/gripper")
        mesh = UsdGeom.Mesh.Define(stage, "/Robot/gripper/visuals/finger/Mesh")
        mesh.CreatePointsAttr([(-0.01, -0.02, -0.1), (0.01, 0.02, -0.1),
                               (0.03, -0.03, -0.06), (0.03, 0.03, -0.06)])
        with patch.object(self.geometry, "_measure_jaw_points", wraps=self.geometry._measure_jaw_points) as read:
            small = self.geometry.gripper_tip_local_offset(stage, 0.005, 0.003, "/Robot/gripper", "finger")
            large = self.geometry.gripper_tip_local_offset(stage, 0.03, 0.001, "/Robot/gripper", "finger")
            self.assertEqual(read.call_count, 1)
        np.testing.assert_allclose(small, [0.018, 0, -0.095], atol=1e-8)
        np.testing.assert_allclose(large, [0.061, 0, -0.07], atol=1e-8)
