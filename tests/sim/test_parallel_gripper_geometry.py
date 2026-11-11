"""Converted collision measurements match the fixture's construction dimensions."""
import numpy as np
import pytest


def test_measured_fixture_gap_and_tcp_within_one_millimetre(imported_urdf_arm):
    if imported_urdf_arm['name']!='seven_parallel':
        pytest.skip('Parallel geometry applies to the seven-axis fixture')
    measured=imported_urdf_arm['report']['measurements']
    assert abs(measured['closed_gap_m'])<.001
    assert abs(measured['open_gap_m']-.09)<.001
    np.testing.assert_allclose(measured['position_offset_m'],[0.,0.,.07],atol=.001,rtol=0)
    assert set(measured['fingertip_points'])=={'Finger_L','Finger_R'}
    for points in measured['fingertip_points'].values():
        assert abs(np.asarray(points)[:,2].max()-.08)<.001


def test_collision_vertex_cache_keeps_live_finger_transforms_independent():
    pytest.importorskip('pxr')
    from pxr import Usd,UsdGeom,UsdPhysics
    from sim.scene.geometry import parallel_fingertip_points,clear_geometry_cache
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/Robot/Tool')
    for name,x in [('Left',.004),('Right',-.004)]:
        UsdGeom.Xform.Define(stage,'/Robot/'+name).AddTranslateOp().Set((x,0,.01))
        mesh=UsdGeom.Mesh.Define(stage,'/Robot/'+name+'/collider')
        mesh.CreatePointsAttr([(a,b,z) for a in (-.004,.004) for b in (-.01,.01) for z in (0.,.07)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    first=parallel_fingertip_points(stage,'/Robot','Tool',('Left','Right'),[0,0,1])
    np.testing.assert_allclose(first['Left'][:,2],.08,atol=1e-8)
    moved={name:np.eye(4) for name in ('Left','Right')}
    moved['Left'][:3,3]=[.049,0,.01];moved['Right'][:3,3]=[-.049,0,.01]
    opened=parallel_fingertip_points(stage,'/Robot','Tool',('Left','Right'),[0,0,1],body_to_ee=moved)
    assert opened['Left'][:,0].min()-opened['Right'][:,0].max()==pytest.approx(.09,abs=1e-8)
    clear_geometry_cache(stage)
