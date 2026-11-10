"""Real USD regressions for the SO101 asset's shared-CAD camera override."""

import copy
import hashlib
from pathlib import Path
import unittest

try:
    from pxr import Gf, Sdf, Usd, UsdPhysics
except ImportError:
    Usd = None


ASSETS = Path(__file__).resolve().parents[2] / "arms/so101/usd"
BASE_SHA256 = "1b4132acb64952fd952a0afa2eb4e6c805bac1df9dc80e01feda9ba3a28f277d"


@unittest.skipUnless(Usd is not None, "Run inside s2r-sim for real USD composition checks")
class ArmAssetVariantsTests(unittest.TestCase):
    def test_shared_cad_and_complete_override_match_original_assets(self):
        # A full resolved comparison against the original NO-CAMERA binary covered
        # 4,016 prim/property paths before replacement. The unchanged base bytes
        # plus this complete, exact list of sparse differences preserve that proof
        # without repeatedly traversing millions of CAD mesh elements in tests.
        base_path = ASSETS / "SO-ARM101-USD.usd"
        self.assertEqual(hashlib.sha256(base_path.read_bytes()).hexdigest(), BASE_SHA256)
        base = Sdf.Layer.FindOrOpen(str(base_path))
        variant = ASSETS / "SO-ARM101-USD-NO-CAMERA.usda"
        self.assertLess(variant.stat().st_size, 10_000)
        layer = Sdf.Layer.FindOrOpen(str(variant))
        self.assertEqual(list(layer.subLayerPaths), [base_path.name])

        expected_metadata = copy.deepcopy(base.customLayerData)
        perspective = expected_metadata["cameraSettings"]["Perspective"]
        perspective["position"] = Gf.Vec3d(0.2906734279202774, 0.4138578789280124, 0.421891178137979)
        perspective["target"] = Gf.Vec3d(0.20291794906801094, -0.005953009897695316, 0.20401197513471492)
        self.assertEqual(layer.customLayerData, expected_metadata)
        for key in set(base.pseudoRoot.ListInfoKeys()) | set(layer.pseudoRoot.ListInfoKeys()):
            if key not in {"customLayerData", "subLayers", "subLayerOffsets"}:
                self.assertEqual(layer.pseudoRoot.GetInfo(key), base.pseudoRoot.GetInfo(key), key)

        render = "/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0"
        expected_values = {
            render + ".omni:rtx:material:db:nonVisualMaterialCSV:enabled": False,
            render + ".omni:rtx:post:dlss:execMode": "performance",
            render + ".resolution": Gf.Vec2i(2025, 1080),
            "/so101_new_calib/gripper.xformOp:orient": Gf.Quatd(
                0.5000019669532776, -0.49999964237213135, -0.5000036358833313, -0.4999951124191284),
            "/so101_new_calib/gripper.xformOp:translate": Gf.Vec3d(
                0.02061525173485279, -0.2774733006954193, 0.2668520510196686),
            "/so101_new_calib/jaw.xformOp:orient": Gf.Quatd(
                0.7071079611778259, 0.0000029206275939941406, -0.7071058750152588, 0.0000073015689849853516),
            "/so101_new_calib/jaw.xformOp:translate": Gf.Vec3d(
                0.03941557556390762, -0.3008729815483093, 0.2870521545410156),
            "/so101_new_calib/joints/Wrist_Roll.physics:localRot1": Gf.Quatf(1, 0, 0, 0),
        }
        expected_active = {"/visuals/gripper/camera_mount", "/colliders/gripper/camera_mount"}
        properties, inactive = set(), set()
        paths = []
        layer.Traverse("/", paths.append)
        for path in paths:
            if path == Sdf.Path.absoluteRootPath:
                continue
            spec = layer.GetObjectAtPath(path)
            if isinstance(spec, Sdf.PrimSpec):
                self.assertEqual(spec.specifier, Sdf.SpecifierOver)
                keys = set(spec.ListInfoKeys())
                if spec.HasInfo("active"):
                    self.assertFalse(spec.active)
                    inactive.add(str(path))
                    keys.remove("active")
                self.assertEqual(keys, {"specifier"}, str(path))
            else:
                self.assertIsInstance(spec, Sdf.AttributeSpec)
                properties.add(str(path))
                self.assertIn(str(path), expected_values)
                self.assertEqual(spec.default, expected_values[str(path)])
                original = base.GetAttributeAtPath(path)
                if original:
                    for key in set(spec.ListInfoKeys()) | set(original.ListInfoKeys()):
                        if key != "default":
                            self.assertEqual(spec.GetInfo(key), original.GetInfo(key), str(path) + ":" + key)
                else:
                    self.assertFalse(spec.custom)
                    self.assertEqual(spec.variability, Sdf.VariabilityVarying)
                    expected_type = Sdf.ValueTypeNames.Bool if isinstance(spec.default, bool) else Sdf.ValueTypeNames.Token
                    self.assertEqual(spec.typeName, expected_type)
        self.assertEqual(properties, set(expected_values))
        self.assertEqual(inactive, expected_active)

    def test_no_camera_composes_under_spawn_reference(self):
        # Spawn imports only the default prim. Its visual/collision instances
        # reference sibling roots in the asset, including the disabled camera.
        stage = Usd.Stage.CreateInMemory()
        robot = stage.DefinePrim("/World/Robot")
        robot.GetReferences().AddReference(str(ASSETS / "SO-ARM101-USD-NO-CAMERA.usda"))
        for branch in ("visuals", "collisions"):
            camera = stage.GetPrimAtPath(f"/World/Robot/gripper/{branch}/camera_mount")
            self.assertTrue(camera.IsValid())
            self.assertFalse(camera.IsActive())
        roll = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Robot/joints/Wrist_Roll"))
        self.assertEqual(roll.GetLocalRot1Attr().Get(), Gf.Quatf(1, 0, 0, 0))
        jaw = stage.GetPrimAtPath("/World/Robot/jaw")
        self.assertEqual(jaw.GetAttribute("xformOp:translate").Get(), Gf.Vec3d(
            0.03941557556390762, -0.3008729815483093, 0.2870521545410156))
        joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
        self.assertEqual(len(joints), 6)
