"""Calibrated USD optics and generic focal DR without starting Kit."""
import ast
from dataclasses import asdict, is_dataclass
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom


def helpers():
    path = Path(__file__).resolve().parents[2] / "src/sim/workspace/scene/cameras.py"
    names = {"apply_opencv_intrinsics", "quaternion_tuple", "euler_xyz_degrees_to_xyzw",
             "_values", "camera_rotation", "sync_camera_intrinsics"}
    tree = ast.parse(path.read_text())
    namespace = dict(Gf=Gf, Sdf=Sdf, asdict=asdict, is_dataclass=is_dataclass, math=math)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_opencv_calibration_keeps_measured_intrinsics_and_half_pixel_centre():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    helpers().apply_opencv_intrinsics(camera, dict(fx=606, fy=605, cx=330.25, cy=253.25,
                                                 width=640, height=480, coeffs=[0, 0, 0, 0, 0]))
    prefix = "omni:lensdistortion:opencvPinhole:"
    for key, expected in dict(fx=606, fy=605, cx=330.75, cy=253.75).items():
        assert camera.GetAttribute(prefix + key).Get() == expected
    assert camera.GetAttribute(prefix + "imageSize").Get() == Gf.Vec2i(640, 480)
    assert camera.GetAttribute("calibration:pixelCenterOffset").Get() == .5
    with pytest.raises(ValueError, match="inverse Brown-Conrady"):
        helpers().apply_opencv_intrinsics(camera, dict(fx=10, fy=10, cx=5, cy=5, width=10, height=10,
                                                     coeffs=[.1], model="distortion.inverse_brown_conrady"))


def test_camera_rotation_matches_frozen_euler_composition_and_generic_target():
    angles = (37.89536558302176, 1.179900948119167, -176.32444041612325)
    expected = (Gf.Rotation(Gf.Vec3d(1, 0, 0), angles[0])
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), angles[1])
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), angles[2])).GetQuat()
    module = helpers()
    assert module.camera_rotation(dict(euler_degrees=angles)) == (*expected.GetImaginary(), expected.GetReal())
    quaternion = module.camera_rotation(dict(position=(2, -2, 2), target=(0, 0, 0), up=(0, 0, 1)))
    rotation = Gf.Rotation(Gf.Quatd(quaternion[3], Gf.Vec3d(*quaternion[:3])))
    forward = rotation.TransformDir(Gf.Vec3d(0, 0, -1))
    np.testing.assert_allclose(forward, np.array([-2, 2, -2]) / np.sqrt(12), atol=1e-12)
    with pytest.raises(ValueError, match="non-degenerate"):
        module.camera_rotation(dict(position=(0, 0, 0), target=(0, 0, 1), up=(0, 0, 1)))


def test_focal_sync_uses_declared_cameras_and_selected_environments(monkeypatch):
    module = helpers()
    stage = Usd.Stage.CreateInMemory()
    paths = [f"/World/envs/env_{i}/arbitrary_sensor" for i in range(2)]
    cameras = []
    for path in paths:
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(40)
        module.apply_opencv_intrinsics(camera.GetPrim(), dict(fx=500, fy=510, cx=300, cy=200,
                                                             width=640, height=480))
        cameras.append(camera.GetPrim())
    sim_module = ModuleType("isaaclab.sim")
    sim_module.find_matching_prims = lambda path: cameras
    isaac = ModuleType("isaaclab")
    isaac.sim = sim_module
    resets = ModuleType("sim.workspace.envs.mdp._common")
    resets.env_indices = lambda env, ids: SimpleNamespace(tolist=lambda: list(ids))
    monkeypatch.setitem(sys.modules, "isaaclab", isaac)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    monkeypatch.setitem(sys.modules, "sim.workspace.envs.mdp._common", resets)
    sensor = SimpleNamespace(cfg=SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/arbitrary_sensor", spawn=SimpleNamespace(focal_length=20,
                                                                        intrinsics=dict(fx=500, fy=510))))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"unrelated_name": sensor, "other": SimpleNamespace()},
                                                env_prim_paths=[f"/World/envs/env_{i}" for i in range(2)]))
    module.sync_camera_intrinsics(env, [1])
    prefix = "omni:lensdistortion:opencvPinhole:"
    assert cameras[0].GetAttribute(prefix + "fx").Get() == 500
    assert cameras[1].GetAttribute(prefix + "fx").Get() == 1000
    assert cameras[1].GetAttribute(prefix + "fy").Get() == 1020
    assert cameras[1].GetAttribute(prefix + "cx").Get() == 300.5
    assert cameras[1].GetAttribute(prefix + "cy").Get() == 200.5
