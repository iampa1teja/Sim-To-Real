"""The optional Molmo view must preserve the original scene and camera poses."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from sim_to_real_so101.utils.molmo_camera import add_molmo_overhead_camera, TABLE_EDGE_IN_BASE


def environment():
    camera = SimpleNamespace(
        prim_path="{ENV_REGEX_NS}/realsense_camera_rgb", height=480, width=640,
        data_types=["rgb", "instance_id_segmentation_fast"],
        offset=SimpleNamespace(pos=(.65, .01, 1.3), rot=(1., 0., 0., 0.), convention="opengl"),
        spawn=SimpleNamespace(func="original_spawn", intrinsics={"fx": 606., "fy": 605.},
                              response={"gain": [1.3, 1.2, 1.3]}),
    )
    term = SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name="camera_realsense_rgb"),
                                   "data_type": "rgb", "normalize": False})
    return SimpleNamespace(
        scene=SimpleNamespace(camera_realsense_rgb=camera, camera_wrist_cam=deepcopy(camera),
                              robot=SimpleNamespace(init_state=SimpleNamespace(
                                  pos=(.7, -.6, .76), rot=(1., 0., 0., 0.)))),
        observations=SimpleNamespace(visual=SimpleNamespace(rgb_realsense_rgb=term,
                                                           rgb_wrist_cam=deepcopy(term))),
    )


def test_extra_sensor_preserves_existing_camera_and_observation_instances():
    pytest.importorskip("pxr.Gf")
    cfg = environment()
    original, wrist = cfg.scene.camera_realsense_rgb, cfg.scene.camera_wrist_cam
    before = deepcopy(cfg)
    original_term = cfg.observations.visual.rgb_realsense_rgb
    metadata = add_molmo_overhead_camera(cfg)
    assert cfg.scene.camera_realsense_rgb is original
    assert cfg.scene.camera_wrist_cam is wrist
    assert cfg.observations.visual.rgb_realsense_rgb is original_term
    assert original == before.scene.camera_realsense_rgb
    assert wrist == before.scene.camera_wrist_cam
    assert original_term == before.observations.visual.rgb_realsense_rgb
    added = cfg.scene.camera_molmo_room
    assert added is not original and added.offset is not original.offset
    assert added.spawn is not original.spawn
    assert added.spawn.intrinsics == original.spawn.intrinsics
    assert added.spawn.response == original.spawn.response
    assert added.data_types == ["rgb"]
    assert metadata["sensor"] == "molmo_room"
    assert metadata["width"] == 640 and metadata["height"] == 480
    assert cfg.observations.visual.rgb_molmo_room.params["sensor_cfg"].name == "camera_molmo_room"
    assert original_term.params["sensor_cfg"].name == "camera_realsense_rgb"
    untouched = environment()
    assert not hasattr(untouched.scene, "camera_molmo_room")
    assert not hasattr(untouched.observations.visual, "rgb_molmo_room")


def test_camera_is_above_behind_rotated_base_and_looks_at_workspace():
    Gf = pytest.importorskip("pxr.Gf")
    cfg = environment()
    cfg.scene.robot.init_state.rot = (math.sqrt(.5), 0., 0., math.sqrt(.5))
    metadata = add_molmo_overhead_camera(cfg)
    base = np.array(cfg.scene.robot.init_state.pos)
    eye, target = np.array(metadata["position"]), np.array(metadata["target"])
    # Base +Y points along world -X after this 90-degree base yaw.
    assert eye[0] < base[0] and eye[2] > base[2] + .5
    assert target[0] > base[0]
    q = metadata["orientation_wxyz"]
    camera_rotation = Gf.Rotation(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    forward = np.array(camera_rotation.TransformDir(Gf.Vec3d(0, 0, -1)))
    np.testing.assert_allclose(forward, (target - eye) / np.linalg.norm(target - eye), atol=1e-12)


def test_existing_extra_sensor_is_not_overwritten():
    pytest.importorskip("pxr.Gf")
    cfg = environment()
    add_molmo_overhead_camera(cfg)
    existing = cfg.scene.camera_molmo_room
    with pytest.raises(ValueError, match="already exists"):
        add_molmo_overhead_camera(cfg)
    assert cfg.scene.camera_molmo_room is existing


def test_actual_table_edge_is_horizontal_without_inverting_the_view():
    Gf = pytest.importorskip("pxr.Gf")
    cfg = environment()
    metadata = add_molmo_overhead_camera(cfg)
    q = metadata["orientation_wxyz"]
    camera_rotation = Gf.Rotation(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    eye = Gf.Vec3d(*metadata["position"])
    base = Gf.Vec3d(*cfg.scene.robot.init_state.pos)
    points = [camera_rotation.GetInverse().TransformDir(base + Gf.Vec3d(*point) - eye)
              for point in TABLE_EDGE_IN_BASE]
    normalized = [(point[0] / -point[2], point[1] / -point[2]) for point in points]
    assert normalized[1][0] > normalized[0][0]
    assert normalized[0][1] == pytest.approx(normalized[1][1], abs=1e-12)
    assert camera_rotation.TransformDir(Gf.Vec3d(0, 1, 0))[2] > 0
    assert metadata["alignment"] == "table_front_edge"
