"""Run with s2r-entrypoint python: real USD collision/material checks without starting Kit."""

import hashlib
import importlib.util
from pathlib import Path
import unittest

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None

if Usd is not None:
    from sim.scene.contact_materials import (
        GRASP_FRICTION, bind_contact_material, contact_material, contact_properties, enable_collision_meshes,
    )


REPO = Path(__file__).resolve().parents[2]
ARM = REPO / "workspace/arms/so101/usd/SO-ARM101-USD.usd"
ROOM = REPO / "workspace/scenes/rpd_meeting_room/usd/digital_twin_room.usda"


@unittest.skipUnless(Usd is not None and importlib.util.find_spec("isaaclab"),
                     "Run inside the existing s2r-sim container with s2r-entrypoint python")
class ContactMaterialsTests(unittest.TestCase):
    def stage(self, asset):
        stage = Usd.Stage.Open(str(asset))
        stage.SetEditTarget(stage.GetSessionLayer())
        return stage

    def geometry(self, root):
        cache = UsdGeom.XformCache()
        return {
            str(prim.GetPath()): (
                prim.GetTypeName(),
                str(cache.GetLocalToWorldTransform(prim)),
                str(UsdGeom.Mesh(prim).GetPointsAttr().Get()) if prim.IsA(UsdGeom.Mesh) else None,
            )
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        }

    def joints(self, root):
        return {
            str(prim.GetPath()): (
                {attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()},
                {rel.GetName(): [str(path) for path in rel.GetTargets()] for rel in prim.GetRelationships()},
                prim.GetAppliedSchemas(),
            )
            for prim in Usd.PrimRange(root) if prim.IsA(UsdPhysics.Joint)
        }

    def physics_binding(self, prim):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        if not material:
            return None
        api = UsdPhysics.MaterialAPI(material.GetPrim())
        return (str(material.GetPath()), api.GetStaticFrictionAttr().Get(),
                api.GetDynamicFrictionAttr().Get(), api.GetRestitutionAttr().Get())

    def assert_contact_properties(self, prim):
        self.assertEqual(prim.GetAttribute("mjc:condim").Get(), 6)
        self.assertEqual(tuple(prim.GetAttribute("mjc:solref").Get()), (0.004, 1.0))
        self.assertEqual(tuple(prim.GetAttribute("mjc:solimp").Get()), (0.999, 0.999, 0.001, 0.5, 2.0))

    def test_arm_collision_promotion_preserves_geometry_mount_and_visuals(self):
        digest = hashlib.sha256(ARM.read_bytes()).hexdigest()
        stage = self.stage(ARM)
        root = stage.GetDefaultPrim()
        geometry, joints = self.geometry(root), self.joints(root)
        visuals = {str(prim.GetPath()): prim.IsInstance()
                   for prim in Usd.PrimRange(root) if prim.GetName() == "visuals"}
        containers = [prim for prim in Usd.PrimRange(root)
                      if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim)]
        self.assertEqual(len(containers), 7)
        self.assertEqual(sum(visuals.values()), 7)

        promoted = enable_collision_meshes(root)

        self.assertEqual(len(promoted), 30)
        self.assertEqual(enable_collision_meshes(root), ())
        for path in promoted:
            mesh = stage.GetPrimAtPath(path)
            self.assertTrue(mesh.IsA(UsdGeom.Gprim))
            self.assertTrue(UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get())
            if mesh.IsA(UsdGeom.Mesh):
                self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "convexHull")
            else:
                self.assertTrue(mesh.IsA(UsdGeom.Cube))
        for container in containers:
            self.assertFalse(container.IsInstance())
            self.assertFalse(container.HasAPI(UsdPhysics.CollisionAPI))
            self.assertFalse(container.HasAPI(UsdPhysics.MeshCollisionAPI))
            self.assertNotIn("PhysxMeshMergeCollisionAPI", container.GetMetadata("apiSchemas").GetAppliedItems())
        after = self.geometry(root)
        self.assertEqual({path: after[path] for path in geometry}, geometry)
        additions = set(after) - set(geometry)
        self.assertEqual(len(additions), 16)
        self.assertEqual({Path(path).name for path in additions}, {
            "finger_contact", "body_contact", "contact_pad_0", "contact_pad_1",
            "contact_nose", "contact_back_0", "contact_back_1", "contact_upper",
        })
        for path in ("gripper/collisions/wrist_roll_follower_so101_v1/mesh",
                     "jaw/collisions/moving_jaw_so101_v1/mesh"):
            source = stage.GetPrimAtPath(root.GetPath().AppendPath(path))
            self.assertFalse(UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get())
        from pxr import Gf
        import numpy as np
        gripper = stage.GetPrimAtPath(root.GetPath().AppendChild("gripper"))
        distal = stage.GetPrimAtPath(root.GetPath().AppendPath(
            "gripper/collisions/wrist_roll_follower_so101_v1/finger_contact"))
        transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(distal, gripper)
        points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, point)))
                             for point in UsdGeom.Mesh(distal).GetPointsAttr().Get()])
        self.assertAlmostEqual(float(points[:, 0].max()), -0.0079, places=6)
        self.assertAlmostEqual(float(points[:, 2].max()), -0.04, places=6)
        self.assertEqual(self.joints(root), joints)
        self.assertEqual(
            {str(prim.GetPath()): prim.IsInstance()
             for prim in Usd.PrimRange(root) if prim.GetName() == "visuals"}, visuals)
        self.assertFalse(any(prim.HasAPI(UsdPhysics.CollisionAPI)
                             for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                             if "/visuals/" in str(prim.GetPath())))
        self.assertEqual(hashlib.sha256(ARM.read_bytes()).hexdigest(), digest)

    def test_moving_finger_hull_leaves_the_cad_opening_clear(self):
        from pxr import Gf
        import numpy as np
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation

        stage = self.stage(ARM)
        root = stage.GetDefaultPrim()
        enable_collision_meshes(root)
        jaw = stage.GetPrimAtPath(root.GetPath().AppendChild("jaw"))
        prefix = root.GetPath().AppendPath("jaw/collisions/moving_jaw_so101_v1")

        def points(name):
            mesh = stage.GetPrimAtPath(prefix.AppendChild(name))
            transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh, jaw)
            return np.asarray([transform.Transform(Gf.Vec3d(*map(float, point)))
                               for point in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])

        distal, proximal, source = points("finger_contact"), points("body_contact"), points("mesh")
        self.assertAlmostEqual(float(distal[:, 1].max()), -0.04, places=6)
        self.assertAlmostEqual(float(proximal[:, 1].min()), -0.04, places=6)
        # Both pieces retain the CAD envelope on their side of the cut.
        for axis in (0, 2):
            self.assertAlmostEqual(float(np.vstack((distal, proximal))[:, axis].min()),
                                   float(source[:, axis].min()), places=6)
            self.assertAlmostEqual(float(np.vstack((distal, proximal))[:, axis].max()),
                                   float(source[:, axis].max()), places=6)

        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(root.GetPath().AppendPath("joints/Jaw")))

        def rotation(value):
            return Rotation.from_quat([*value.GetImaginary(), value.GetReal()]).as_matrix()

        r = (rotation(joint.GetLocalRot0Attr().Get()) @ Rotation.from_rotvec([0, 0, 0.05]).as_matrix()
             @ rotation(joint.GetLocalRot1Attr().Get()).T)
        t = np.asarray(joint.GetLocalPos0Attr().Get()) - r @ np.asarray(joint.GetLocalPos1Attr().Get())

        def inner_x(vertices):
            planes = ConvexHull(vertices @ r.T + t).equations
            planes = planes[planes[:, 0] < -1e-6]
            return float((-(planes[:, 2] * -0.085 + planes[:, 3]) / planes[:, 0]).max())

        # The old whole-part hull protruded into the opening by 1.01 mm here.
        self.assertAlmostEqual(inner_x(distal), 0.01232772, places=6)
        self.assertGreater(inner_x(distal) - inner_x(source), 0.0009)

    def test_disabled_collision_container_stays_disabled(self):
        stage = self.stage(ARM)
        root = stage.GetDefaultPrim()
        container = stage.GetPrimAtPath(root.GetPath().AppendPath("jaw/collisions"))
        UsdPhysics.CollisionAPI(container).GetCollisionEnabledAttr().Set(False)

        promoted = enable_collision_meshes(root)

        jaw_meshes = [stage.GetPrimAtPath(path) for path in promoted if "/jaw/" in path]
        self.assertEqual(len(jaw_meshes), 1)
        self.assertFalse(UsdPhysics.CollisionAPI(jaw_meshes[0]).GetCollisionEnabledAttr().Get())
        selected = bind_contact_material(root, lambda prim: "/jaw/" in str(prim.GetPath()))
        self.assertEqual(selected, ())
        self.assertFalse(stage.GetPrimAtPath(root.GetPath().AppendChild("ContactMaterial")))

    def test_material_binding_changes_only_jaws_and_tabletop(self):
        for asset, expected_count in ((ARM, 16), (ROOM, 49)):
            with self.subTest(asset=asset.name):
                digest = hashlib.sha256(asset.read_bytes()).hexdigest()
                stage = self.stage(asset)
                root = stage.GetDefaultPrim()
                if asset == ARM:
                    enable_collision_meshes(root)
                    jaw_paths = tuple(root.GetPath().AppendChild(name) for name in ("gripper", "jaw"))
                    predicate = lambda prim: any(prim.GetPath().HasPrefix(path) for path in jaw_paths)
                else:
                    predicate = lambda prim: "tabletop" in str(prim.GetPath()).lower()
                unselected = {str(prim.GetPath()): self.physics_binding(prim)
                              for prim in Usd.PrimRange(root)
                              if prim.HasAPI(UsdPhysics.CollisionAPI) and not predicate(prim)}
                original_contact_properties = {
                    path: (str(stage.GetPrimAtPath(path).GetAttribute("mjc:condim").Get()),
                           str(stage.GetPrimAtPath(path).GetAttribute("mjc:solimp").Get()))
                    for path in unselected
                }
                self.assertTrue(unselected)

                friction = GRASP_FRICTION if asset == ARM else 0.5
                selected = bind_contact_material(root, predicate, friction=friction)

                self.assertEqual(len(selected), expected_count)
                for path in selected:
                    prim = stage.GetPrimAtPath(path)
                    self.assertEqual(self.physics_binding(prim)[1:], (friction, friction, 0.0))
                    self.assert_contact_properties(prim)
                self.assertEqual({path: self.physics_binding(stage.GetPrimAtPath(path)) for path in unselected},
                                 unselected)
                self.assertEqual({
                    path: (str(stage.GetPrimAtPath(path).GetAttribute("mjc:condim").Get()),
                           str(stage.GetPrimAtPath(path).GetAttribute("mjc:solimp").Get()))
                    for path in unselected
                }, original_contact_properties)
                if asset == ROOM:
                    self.assertTrue(any(binding and abs(binding[2] - 0.6) < 1e-6
                                        for binding in unselected.values()))
                self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), digest)

    def test_cube_fragments_author_frictional_contact_before_import(self):
        from isaaclab.sim.schemas.schemas import apply_collision_properties
        from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material_from_fragments
        from isaaclab.sim.utils import bind_physics_material

        stage = Usd.Stage.CreateInMemory()
        cube = UsdGeom.Cube.Define(stage, "/Cube").GetPrim()
        apply_collision_properties("/Cube", contact_properties(), create_if_missing=True, stage=stage)
        spawn_physics_material_from_fragments("/ContactMaterial", contact_material(GRASP_FRICTION), stage=stage)
        bind_physics_material("/Cube", "/ContactMaterial", stage=stage)

        self.assertTrue(UsdPhysics.CollisionAPI(cube))
        self.assert_contact_properties(cube)
        self.assertEqual(self.physics_binding(cube)[1:], (GRASP_FRICTION, GRASP_FRICTION, 0.0))


if __name__ == "__main__":
    unittest.main()
