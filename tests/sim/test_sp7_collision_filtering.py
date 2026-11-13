"""Mechanical adjacency filters must preserve distant gripper/base collision."""
import pytest


def test_sp7_filters_welded_and_joint_adjacent_parts_only():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom, UsdPhysics
    from core.arms.spec import load_arm_spec, load_arm_utils

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
    names = ['base', 'base_housing', 'upper', 'upper_housing', 'forearm', 'gripper']
    bodies = {}
    for name in names:
        prim = UsdGeom.Xform.Define(stage, '/Robot/' + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        bodies[name] = prim
    for i, (a, b, fixed) in enumerate([
        ('base', 'base_housing', True), ('base_housing', 'upper', False),
        ('upper', 'upper_housing', True), ('upper_housing', 'forearm', False),
        ('forearm', 'gripper', False),
    ]):
        cls = UsdPhysics.FixedJoint if fixed else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, f'/Robot/joint_{i}')
        joint.CreateBody0Rel().SetTargets([bodies[a].GetPath()])
        joint.CreateBody1Rel().SetTargets([bodies[b].GetPath()])
    load_arm_utils(load_arm_spec('sp7')[0]).prepare_collisions(root)

    def filtered(a, b):
        return bodies[b].GetPath() in UsdPhysics.FilteredPairsAPI(bodies[a]).GetFilteredPairsRel().GetTargets()

    assert filtered('base', 'base_housing')
    assert filtered('base', 'upper_housing')
    assert filtered('upper_housing', 'forearm')
    assert not filtered('base', 'forearm')
    assert not filtered('base', 'gripper')
    assert not filtered('upper_housing', 'gripper')


def test_sp7_guides_do_not_disable_jaw_or_base_contacts():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    from core.arms.spec import load_arm_spec, load_arm_utils

    arm = load_arm_spec('sp7')[0]
    stage = Usd.Stage.Open(str(arm.usd_path))
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    load_arm_utils(arm).prepare_collisions(root)
    fingers = [root.GetChild(n) for n in ('gripper_left_link', 'gripper_right_link')]
    guides = [p.GetChild('s2r_guide_collision') for p in fingers]
    jaws = [p.GetChild('s2r_jaw_collision') for p in fingers]
    assert all(guides + jaws)
    for prim in guides + jaws:
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    assert guides[1].GetPath() in UsdPhysics.FilteredPairsAPI(guides[0]).GetFilteredPairsRel().GetTargets()
    assert jaws[1].GetPath() not in UsdPhysics.FilteredPairsAPI(jaws[0]).GetFilteredPairsRel().GetTargets()
    for finger in fingers:
        assert root.GetChild('base_link').GetPath() not in UsdPhysics.FilteredPairsAPI(finger).GetFilteredPairsRel().GetTargets()
    assert root.GetChild('base_link').GetAttribute('newton:selfCollisionEnabled').Get() is True
    assert arm.spawn['articulation_props']['self_collision_enabled'] is True
    for name in ('link5', 'link7'):
        body = root.GetChild(name)
        active = [p for p in body.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)
                  and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        assert len(active) > 1
        assert all(p.GetName().startswith('s2r_wrist_collision_') for p in active)
