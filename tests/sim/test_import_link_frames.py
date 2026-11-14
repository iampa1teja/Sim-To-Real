"""Nested visual meshes must not replace rigid-link frames during import."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_camera_measurement_ignores_same_named_nested_visuals():
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from sim.workspace.arms.urdf_import import measure_cameras

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    tool = UsdGeom.Xform.Define(stage, '/Robot/tool')
    tool.AddTranslateOp().Set((1., 2., 3.))
    camera = UsdGeom.Xform.Define(stage, '/Robot/camera')
    camera.AddTranslateOp().Set((1.01, 2.02, 3.03))
    for parent in (tool, camera):
        visual = UsdGeom.Mesh.Define(stage, str(parent.GetPath()) + '/' + parent.GetPrim().GetName())
        visual.AddTranslateOp().Set((5., 6., 7.))
    model = SimpleNamespace(links={'tool': None, 'camera': None}, joints={
        'camera_mount': SimpleNamespace(child='camera', parent='tool', type='fixed')})
    measured = measure_cameras(stage, model, SimpleNamespace(camera={'camera': 'wrist_cam'}))
    assert measured['wrist_cam']['parent_link'] == 'tool'
    np.testing.assert_allclose(measured['wrist_cam']['position'], [.01, .02, .03], atol=1e-8)
    np.testing.assert_allclose(measured['wrist_cam']['euler_degrees'], [0., 0., 0.], atol=1e-8)
