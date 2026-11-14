"""Actual SO101 USD pad/contact-support regressions; never starts Kit or a GPU."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import unittest

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


ASSET = Path(__file__).resolve().parents[2] / "workspace/arms/so101/usd/SO-ARM101-USD.usd"


@unittest.skipUnless(Usd is not None and importlib.util.find_spec("isaaclab"),
                     "Run inside s2r-sim with s2r-entrypoint python")
class GripperCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from sim.scene.contact_materials import enable_collision_meshes
        from utils.arms.spec import load_arm_spec, load_arm_utils
        apply_primitive_pads = load_arm_utils(load_arm_spec("so101")[0]).apply_primitive_pads

        cls.digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
        cls.stage = Usd.Stage.Open(str(ASSET))
        cls.stage.SetEditTarget(cls.stage.GetSessionLayer())
        cls.root = cls.stage.GetDefaultPrim()
        enable_collision_meshes(cls.root)
        cls.source = {}
        for link, mesh in (("gripper", "wrist_roll_follower_so101_v1"), ("jaw", "moving_jaw_so101_v1")):
            path = cls.root.GetPath().AppendPath(f"{link}/collisions/{mesh}/finger_contact")
            cls.source[link] = cls.points(cls.stage.GetPrimAtPath(path), link)
        cls.original = cls.immutable_attributes()
        cls.replaced, cls.created = apply_primitive_pads(cls.root)
        if not cls.created:
            # Once production activates the same helper, enabling collisions has
            # already authored these parts; the explicit second call is a no-op.
            cls.created = tuple(str(prim.GetPath()) for prim in Usd.PrimRange(cls.root)
                                if prim.GetName().startswith(("contact_pad_", "contact_nose",
                                                              "contact_back_", "contact_upper")))
            cls.replaced = tuple(str(cls.root.GetPath().AppendPath(f"{link}/collisions/{mesh}/finger_contact"))
                                 for link, mesh in (("gripper", "wrist_roll_follower_so101_v1"),
                                                    ("jaw", "moving_jaw_so101_v1")))
        cls.after = {}
        for link in cls.source:
            body = cls.root.GetPath().AppendChild(link)
            cls.after[link] = np.vstack([cls.points(cls.stage.GetPrimAtPath(path), link)
                                         for path in cls.created if cls.stage.GetPrimAtPath(path).GetPath().HasPrefix(body)])

    @classmethod
    def points(cls, prim, link):
        import numpy as np

        body = cls.stage.GetPrimAtPath(cls.root.GetPath().AppendChild(link))
        transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, body)
        if prim.IsA(UsdGeom.Cube):
            points = np.asarray(list(itertools.product((-.5, .5), repeat=3))) * UsdGeom.Cube(prim).GetSizeAttr().Get()
        else:
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        return np.asarray([transform.Transform(Gf.Vec3d(*map(float, point))) for point in points])

    @classmethod
    def immutable_attributes(cls):
        return {str(prim.GetPath()): {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()}
                for prim in Usd.PrimRange(cls.root, Usd.TraverseInstanceProxies())
                if prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.MassAPI)
                or "/visuals/" in str(prim.GetPath())}

    def test_four_analytic_contact_faces_match_cad_steps(self):
        boxes = [self.stage.GetPrimAtPath(path) for path in self.created
                 if self.stage.GetPrimAtPath(path).IsA(UsdGeom.Cube)]
        self.assertEqual(len(boxes), 4)
        expected = {("gripper", 0): -.0079, ("gripper", 1): -.0099,
                    ("jaw", 0): -.0123, ("jaw", 1): -.0103}
        for box in boxes:
            fit = json.loads(box.GetCustomDataByKey("s2rPadFit"))
            index = int(box.GetName().rsplit("_", 1)[1])
            self.assertLess(abs(fit["cad_plane_x_m"] - expected[fit["link_frame"], index]), .00005)
            self.assertLess(fit["maximum_cad_plane_residual_mm"], .05)
            self.assertLess(fit["contact_plane_offset_error_mm"], .00001)
            self.assertAlmostEqual(fit["continuous_triangle_coverage_ratio"], 1., places=6)
            self.assertLess(fit["box_dimensions_m"][fit["width_axis"]], .012)
            self.assertTrue(UsdPhysics.CollisionAPI(box).GetCollisionEnabledAttr().Get())

    def test_recorded_table_support_is_preserved(self):
        import numpy as np

        # Table normals expressed in the corresponding body frame at recorded
        # real/sim episode1 frames180,188,198. No recording files are needed.
        directions = {
            "gripper": [( .0045746564, -.0209787020, .9999414313),
                        (-.0070669294, .0347807120, .9995420252),
                        (-.0067475669, .0335018236, .9995879139),
                        (-.0072319313, .0365884699, .9994763041)],
            "jaw": [(.3125079915, .9498645678, .0209751634),
                    (.3198795624, .9470010669, -.0347842531),
                    (.3022906659, .9531654082, .0209751634),
                    (.3015633694, .9530376825, -.0335053647),
                    (.0967982318, .9952557683, .0209751634),
                    (.1347528278, .9903769987, -.0365920108)],
        }
        for link, normals in directions.items():
            for normal in normals:
                before = float((self.source[link] @ np.asarray(normal)).min())
                after = float((self.after[link] @ np.asarray(normal)).min())
                self.assertLess(abs(after - before), .00005, (link, normal, before, after))

    def test_original_visuals_masses_and_joints_remain_intact(self):
        from utils.arms.spec import load_arm_spec, load_arm_utils
        apply_primitive_pads = load_arm_utils(load_arm_spec("so101")[0]).apply_primitive_pads

        self.assertEqual(self.original, self.immutable_attributes())
        self.assertEqual(hashlib.sha256(ASSET.read_bytes()).hexdigest(), self.digest)
        self.assertEqual(len(self.created), 12)
        self.assertEqual(len(self.replaced), 2)
        self.assertEqual(apply_primitive_pads(self.root), ((), ()))
        for path in self.replaced:
            self.assertFalse(UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath(path)).GetCollisionEnabledAttr().Get())


if __name__ == "__main__":
    unittest.main()
