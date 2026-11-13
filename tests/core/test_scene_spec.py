"""Declarative scene/mount contracts preserve fitted evidence without arm coupling."""
from pathlib import Path

import pytest

from core.arms.spec import load_arm_spec
from core.scenes import load_mount_spec, load_scene_spec
from core.utils.hashing import json_sha256


ROOT = Path(__file__).resolve().parents[2]


def test_meeting_scene_owns_room_and_camera_data_but_not_arm_mapping(monkeypatch):
    monkeypatch.delenv("S2R_SCENE_DIR", raising=False)
    scene = load_scene_spec("rpd_meeting_room")
    assert scene.room_usd == ROOT / "scenes/rpd_meeting_room/usd/digital_twin_room.usda"
    assert scene.room_root == "Room"
    assert scene.hook == "scene.py:customize"
    assert scene.viewer.camera == "realsense_rgb"
    assert scene.cameras["realsense_rgb"].depth_sensor_name == "realsense_depth"
    assert scene.cameras["realsense_rgb"].metadata["display_name"] == "realsense_camera-rgb"
    assert scene.physics == dict(use_mujoco_contacts=False, cone="elliptic", impratio=1000.0,
                                 tolerance=1e-8, num_substeps=4, collision_decimation=1)
    assert scene.renderer == dict(antialiasing_mode="DLAA", enable_ambient_occlusion=True,
                                  enable_global_illumination=True, enable_dlssg=False,
                                  ambient_light_intensity=0.3875,
                                  carb_settings={"/rtx/post/histogram/enabled": False})
    assert not hasattr(scene, "joint_mapping")
    assert not hasattr(scene, "robot")


def test_fitted_scene_blocks_and_refined_camera_match_content_hashes(monkeypatch):
    # Material/lighting hashes retain the pre-split fit. The camera hash includes
    # the Nov 11 camera-only pose refinement and its provenance string.
    monkeypatch.delenv("S2R_SCENE_DIR", raising=False)
    scene = load_scene_spec("rpd_meeting_room")
    expected = {
        "appearance": "71d6e38d2c15837874bccdc683ecd2320958e743b21ded1a49ef1c4ced9a8880",
        "object_materials": "a2bdc6e8af70818b0ff63e0509bd73970ea8069f5208f147d7f112e3f5072bb2",
    }
    for key, digest in expected.items():
        assert json_sha256(scene.appearance[key]) == digest
    assert json_sha256(scene.appearance["provenance"]["lighting"]) == "936a3d3d307f2d8b34dbafd5dc50995e85b9c3440c8bf8ad6ca90161f2ddc169"
    camera = scene.cameras["realsense_rgb"]
    raw = {name: getattr(camera, name) for name in ("position", "target", "focal_length", "horizontal_aperture",
           "vertical_aperture", "width", "height", "euler_degrees", "calibration")}
    raw["orientation_convention"] = camera.metadata["orientation_convention"]
    raw["response"] = scene.appearance["camera_response"]["realsense_rgb"]
    assert json_sha256(raw) == "8e7231d9986e6ddbbebf9b86ab616d9f16a4e51ad6d3605ced0aa2f713350957"
    assert json_sha256(dict(eye=scene.viewer.eye, lookat=scene.viewer.lookat)) == "826cd10120486dc1819b4f2945237292b69b5bfb4a49453c83cfe0f1d68580e2"


def test_mount_reconstructs_original_robot_measurement_exactly(monkeypatch):
    monkeypatch.delenv("S2R_SCENE_DIR", raising=False)
    arm, _ = load_arm_spec("so101")
    scene = load_scene_spec("rpd_meeting_room")
    mount = load_mount_spec(scene, arm)
    anchor = scene.anchors[mount.anchor]
    raw = dict(position=anchor.position, yaw_degrees=anchor.yaw_degrees, table_normal=anchor.up,
               mount_translation_correction_m=mount.translation_correction_m,
               mount_translation_source=mount.source,
               joint_positions={joint.usd: mount.start_joint_positions[joint.name] for joint in arm.joints})
    assert json_sha256(raw) == "b718a088f8893b5b7ef8707043d8e21e8403cb8e87141cbaf4b6b3bf5dd8e00b"
    import json
    original = json.loads((arm.directory / "calibration/runtime_limits_source.json").read_text())
    assert json_sha256(original) == "a2b77133710c6fb897a73da1381c752ca16604ec6945c4ccbde1a065ca0590a8"


def test_missing_mount_is_optional_and_does_not_create_a_file():
    scene = load_scene_spec("blank_table")
    mount = load_mount_spec(scene, "unmeasured_arm")
    assert not mount.exists and mount.path is None
    assert mount.start_joint_positions == {}
    assert mount.translation_correction_m == (0.0, 0.0, 0.0)
    assert not (scene.directory / "mounts/unmeasured_arm.toml").exists()


@pytest.fixture
def minimal_scene(tmp_path, monkeypatch):
    directory = tmp_path / "scenes/example"
    directory.mkdir(parents=True)
    (directory / "room.usda").write_text('#usda 1.0\n(defaultPrim="Room")\ndef Xform "Room" {}\n')
    (directory / "scene.toml").write_text('[scene]\nname="example"\nroom_usd="room.usda"\n')
    monkeypatch.setenv("S2R_SCENES_DIR", str(directory.parent))
    monkeypatch.delenv("S2R_SCENE_DIR", raising=False)
    monkeypatch.delenv("S2R_ROOM_USD", raising=False)
    return directory


def test_room_override_is_read_at_call_time_and_old_variable_is_ignored(minimal_scene, monkeypatch, tmp_path):
    monkeypatch.setenv('ROOM_USD_PATH', str(tmp_path / 'obsolete-does-not-exist.usda'))
    original = load_scene_spec('example')
    assert original.room_usd == minimal_scene / 'room.usda'
    override = tmp_path / 'replacement.usda'
    override.write_text('#usda 1.0\n(defaultPrim="Room")\ndef Xform "Room" {}\n')
    monkeypatch.setenv('S2R_ROOM_USD', str(override))
    assert load_scene_spec('example').room_usd == override
    assert original.room_usd == minimal_scene / 'room.usda'


def test_relative_room_override_uses_process_working_directory(minimal_scene, monkeypatch, tmp_path):
    override = tmp_path / 'replacement.usda'
    override.write_text('#usda 1.0\n(defaultPrim="Room")\ndef Xform "Room" {}\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('S2R_ROOM_USD', 'replacement.usda')
    assert load_scene_spec('example').room_usd == override


def test_missing_room_override_fails_with_its_environment_variable(minimal_scene, monkeypatch, tmp_path):
    monkeypatch.setenv('S2R_ROOM_USD', str(tmp_path / 'missing.usda'))
    with pytest.raises(ValueError, match='S2R_ROOM_USD file not found'):
        load_scene_spec('example')


def test_scene_selection_reads_environment_at_call_time(minimal_scene, monkeypatch):
    monkeypatch.setenv("S2R_SCENE", "example")
    scene = load_scene_spec()
    assert scene.name == "example"
    assert scene.appearance == {} and scene.cameras == {} and scene.lighting == {}
    monkeypatch.setenv("S2R_SCENE", "unknown")
    with pytest.raises(ValueError, match="unknown"):
        load_scene_spec()


@pytest.mark.parametrize("suffix,error", [
    ('[scene.anchors.mount]\nup=[0,0,0]\n', "nonzero"),
    ('[scene.anchors.table]\nup=[0,0,1]\nright=[0,0,2]\n', "parallel"),
    ('[scene.viewer]\ncamera="missing"\n', "not a fixed scene camera"),
    ('[scene.anchors.workspace]\nxy_min=[1,0]\nxy_max=[0,1]\n', "increasing"),
])
def test_invalid_scene_contracts_have_clear_errors(minimal_scene, suffix, error):
    path = minimal_scene / "scene.toml"
    path.write_text(path.read_text() + suffix)
    with pytest.raises(ValueError, match=error):
        load_scene_spec("example")


def test_missing_referenced_asset_and_bad_mount_fail_clearly(minimal_scene):
    path = minimal_scene / "scene.toml"
    original = path.read_text()
    path.write_text(original.replace("room.usda", "missing.usda"))
    with pytest.raises(ValueError, match="file not found"):
        load_scene_spec("example")
    path.write_text(original)
    scene = load_scene_spec("example")
    mounts = minimal_scene / "mounts"
    mounts.mkdir()
    (mounts / "new_arm.toml").write_text('[mount]\nanchor="missing"\n')
    with pytest.raises(ValueError, match="anchor.*does not exist"):
        load_mount_spec(scene, "new_arm")


@pytest.mark.parametrize("section", ["anchors", "cameras", "lighting", "viewer", "physics", "renderer", "metadata"])
def test_non_table_sections_fail_as_schema_errors(minimal_scene, section):
    path = minimal_scene / "scene.toml"
    path.write_text(path.read_text() + f'{section}=[]\n')
    with pytest.raises(ValueError, match="table/object"):
        load_scene_spec("example")


def test_scene_template_loads_after_naming_it_and_supplying_its_declared_asset(tmp_path, monkeypatch):
    import shutil
    directory = tmp_path / "scenes/new_scene"
    shutil.copytree(ROOT / "scenes/_template", directory)
    (directory / "usd/room.usda").write_text('#usda 1.0\n(defaultPrim="Room")\ndef Xform "Room" {}\n')
    monkeypatch.setenv("S2R_SCENES_DIR", str(directory.parent))
    monkeypatch.delenv("S2R_SCENE_DIR", raising=False)
    scene = load_scene_spec("new_scene")
    assert scene.appearance == {}
    assert scene.hook is None
    assert not load_mount_spec(scene, "so101").exists
    assert scene.cameras["room"].data_types == ("rgb",)


def test_immutable_specs_survive_deepcopy_and_replace_without_shared_fitted_data():
    from copy import deepcopy
    from dataclasses import FrozenInstanceError, replace, asdict
    scene = load_scene_spec("rpd_meeting_room")
    clone = deepcopy(scene)
    assert clone == scene
    clone.appearance["appearance"]["robot_beige"][0] = -1
    assert scene.appearance["appearance"]["robot_beige"][0] == 0.63
    renamed = replace(scene, description="Independent test scene")
    assert renamed.description != scene.description
    assert asdict(scene)["anchors"]["mount"]["position"] == scene.anchors["mount"].position
    with pytest.raises(FrozenInstanceError):
        scene.name = "mutated"
