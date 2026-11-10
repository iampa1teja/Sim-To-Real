"""Declarative scenes retain calibrated layout and bind their declared collider groups."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("isaaclab")
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from core.scenes import load_scene_spec
from core.arms.spec import load_arm_spec
from core.tasks.loader import load_task, resolve
from sim.envs import scene_builder


def compose(scene_name, task=True, variant="teleop"):
    cfg = scene_builder.build_scene_cfg(scene_name)
    if task:
        loaded = load_task("pick_place_cube")
        resolve(loaded.spec.entry_point(variant))(cfg, loaded)
    return cfg


def test_room_builder_preserves_calibrated_names_and_exact_float32_box_pose():
    cfg = compose("rpd_meeting_room")
    assert cfg.scene.camera_realsense_rgb.prim_path == "{ENV_REGEX_NS}/realsense_camera_rgb"
    assert cfg.scene.realsense_depth.prim_path == "{ENV_REGEX_NS}/realsense_camera_depth"
    assert cfg.scene.camera_wrist_cam.prim_path == "{ENV_REGEX_NS}/Robot/gripper/wrist_cam"
    np.testing.assert_array_equal(np.asarray(cfg.scene.white_box.init_state.pos, dtype=np.float32),
                                  np.asarray([.9440419640831723, -.5939824362907724, .7951637516104304], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(cfg.scene.white_box.init_state.rot, dtype=np.float32),
                                  np.asarray([-.0009490871253246191, -.009222361160757035,
                                              .9991571638848087, -.039987612405775474], dtype=np.float32))
    assert cfg.sim.physics.num_substeps == 4
    assert cfg.sim.physics.solver_cfg.use_mujoco_contacts is False
    scene_builder.drop_unused_camera_outputs(cfg)
    assert cfg.scene.realsense_depth is None
    assert cfg.scene.camera_realsense_rgb.data_types == ["rgb"]
    assert cfg.scene.camera_wrist_cam.data_types == ["rgb"]
    assert cfg.observations.visual.depth_realsense_rgb is None
    assert cfg.observations.visual.instance_id_seg_wrist_cam is None


def test_blank_table_without_mount_file_uses_declared_anchor_and_default_arm_joints():
    cfg = compose("blank_table")
    assert cfg.scene_spec.name == "blank_table"
    assert cfg.scene.robot.init_state.pos == (0., 0., .75)
    assert cfg.scene.robot.init_state.joint_pos == cfg.arm_spec.initial_joint_positions
    assert -.6 < cfg.scene.white_box.init_state.pos[0] < .6
    assert -.4 < cfg.scene.white_box.init_state.pos[1] < .4
    assert cfg.scene.white_box.init_state.pos[2] == pytest.approx(.751)
    assert cfg.scene.camera_wrist_cam is not None


@pytest.mark.parametrize("scene", ["rpd_meeting_room", "blank_table"])
@pytest.mark.parametrize("variant", ["eval", "eval_dr"])
def test_task_variants_compose_with_declarative_scene(scene, variant):
    cfg = compose(scene, variant=variant)
    assert cfg.terminations.success is not None
    if variant == "eval_dr":
        from sim.scene.appearance import randomize_scene_light_exposure
        assert cfg.events.eval_room_light.func is randomize_scene_light_exposure


def test_task_arm_base_frame_uses_the_selected_robot_mount_pose():
    cfg = compose("blank_table", task=False)
    cfg.scene.robot.init_state.pos = (1., 2., 3.)
    cfg.scene.robot.init_state.rot = (0., 0., 2 ** -.5, 2 ** -.5)
    task = load_task("pick_place_cube")
    box = task.spec.objects["white_box"]
    box = replace(box, position=(.2, 0., 0.), params={**box.params, "frame": "arm_base"})
    spec = replace(task.spec, objects={**task.spec.objects, "white_box": box})
    resolve(spec.entry_point("teleop"))(cfg, SimpleNamespace(name=task.name, spec=spec))
    np.testing.assert_allclose(cfg.scene.white_box.init_state.pos, (1., 2.2, 3.), atol=1e-14)
    np.testing.assert_allclose(cfg.scene.white_box.init_state.rot, cfg.scene.robot.init_state.rot, atol=1e-14)


def test_explicit_arm_snapshot_survives_config_copy_replace_and_export(monkeypatch):
    arm = load_arm_spec("so101")[0]
    # A different default pose and no optional wrist camera must stay with this
    # composition, even when runtime selection names another arm.
    joints = tuple(replace(joint, initial=joint.initial + .001) for joint in arm.joints)
    arm = replace(arm, joints=joints, cameras={})
    monkeypatch.setenv("S2R_ARM", "arm_that_is_not_installed")
    cfg = scene_builder.build_scene_cfg("blank_table", arm)
    for candidate in (cfg, cfg.copy(), cfg.replace()):
        assert candidate.arm_spec == arm
        assert candidate.scene.robot.spawn.arm_spec == arm
        assert candidate.scene.robot.init_state.joint_pos == arm.initial_joint_positions
        assert not hasattr(candidate.scene, "camera_wrist_cam")
        assert candidate.scene_spec.name == "blank_table"
        assert candidate.scene.room.spawn.scene_spec == candidate.scene_spec
        exported = candidate.to_dict()
        assert exported["arm_spec"]["name"] == arm.name
        assert exported["scene_spec"]["room_usd"] == str(candidate.scene_spec.room_usd)
        assert exported["scene"]["robot"]["spawn"]["arm_spec"]["joints"][0]["initial"] == arm.joints[0].initial
        assert exported["events"]["configure_scene_updates"]["params"]["scene_spec"]["name"] == "blank_table"


@pytest.mark.parametrize("name,expected", [("rpd_meeting_room", 49), ("blank_table", 1)])
def test_declared_contact_group_matches_physical_colliders_in_asset(monkeypatch, name, expected):
    from sim.scene.contact_materials import bind_contact_material
    spec = load_scene_spec(name)
    stage = Usd.Stage.Open(str(spec.room_usd))
    stage.SetEditTarget(stage.GetSessionLayer())
    root = stage.GetDefaultPrim()
    bound = []
    monkeypatch.setattr(scene_builder, "spawn_from_usd", lambda *_args, **_kwargs: root)

    def bind(*args, **kwargs):
        selected = bind_contact_material(*args, **kwargs)
        bound.extend(selected)
        return selected

    monkeypatch.setattr(scene_builder, "bind_contact_material", bind)
    scene_builder.spawn_scene_room(str(root.GetPath()), SimpleNamespace(scene_spec=spec))
    assert len(bound) == expected
    for path in bound:
        prim = stage.GetPrimAtPath(path)
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == .5
