"""Generic collision promotion must not require an arm-specific geometry hook."""
from dataclasses import replace
import importlib.util
import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


@unittest.skipUnless(Usd is not None and importlib.util.find_spec("isaaclab"), "Requires s2r-sim USD")
class ArmHooksTests(unittest.TestCase):
    def test_arm_without_utils_only_promotes_its_own_meshes(self):
        from utils.arms.spec import load_arm_spec
        from sim.scene.contact_materials import enable_collision_meshes

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/DifferentArm").GetPrim()
        collisions = UsdGeom.Xform.Define(stage, "/DifferentArm/tool/contact_geometry").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collisions).CreateCollisionEnabledAttr(True)
        mesh = UsdGeom.Mesh.Define(stage, "/DifferentArm/tool/contact_geometry/tetrahedron")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
        mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        arm = replace(load_arm_spec("so101")[0], utils=None)

        authored = enable_collision_meshes(root, arm)

        self.assertEqual(authored, (str(mesh.GetPath()),))
        self.assertTrue(UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get())
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get(), "convexHull")
        self.assertFalse(collisions.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(enable_collision_meshes(root, arm), ())
        self.assertFalse(any(prim.GetName().startswith("contact_pad") for prim in Usd.PrimRange(root)))
