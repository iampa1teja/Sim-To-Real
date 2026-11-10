"""Declared lights and fitted arm appearance work across arbitrary scene roots."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdLux

from sim.scene.appearance import apply_robot_appearance, apply_scene_lighting, set_scene_light_exposure


def test_light_exposure_dr_changes_only_marked_lights_in_selected_environment(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    lighting = {"root": "Effects/Lights", "replace": True, "lights": {
        "Main": dict(type="rect", width=2, height=1, intensity=123, color=[1, .9, .8], exposure_dr=True),
        "Fill": dict(type="dome", intensity=50, color=[.8, .9, 1]),
    }}
    for index in range(2):
        root = f"/World/envs/env_{index}/Chamber/Content"
        stage.DefinePrim(root, "Xform")
        UsdLux.DomeLight.Define(stage, root + "/OldImportedLight")
        apply_scene_lighting(stage, root, lighting)
        assert not stage.GetPrimAtPath(root + "/OldImportedLight")
    resets = ModuleType("sim.envs.mdp._common")
    resets.env_indices = lambda env, ids: SimpleNamespace(tolist=lambda: list(ids))
    resets._env_prim_path = lambda env, room, index: f"/World/envs/env_{index}/Chamber"
    monkeypatch.setitem(sys.modules, "sim.envs.mdp._common", resets)
    env = SimpleNamespace(cfg=SimpleNamespace(scene_spec=SimpleNamespace(room_root="Content", lighting=lighting)),
                          scene={"room": object()}, sim=SimpleNamespace(stage=stage))
    set_scene_light_exposure(env, [1], 2.5)
    for index in range(2):
        root = f"/World/envs/env_{index}/Chamber/Content/Effects/Lights"
        main = UsdLux.RectLight(stage.GetPrimAtPath(root + "/Main"))
        fill = UsdLux.DomeLight(stage.GetPrimAtPath(root + "/Fill"))
        assert main.GetExposureAttr().Get() == (2.5 if index == 1 else 0)
        assert fill.GetExposureAttr().Get() == 0
        assert main.GetIntensityAttr().Get() == 123
    with pytest.raises(ValueError, match="finite"):
        set_scene_light_exposure(env, [0], float("nan"))


def test_empty_appearance_does_not_require_fitted_arm_materials(monkeypatch):
    def unavailable(arm):
        raise AssertionError("An empty scene look must not load a fitted arm hook")
    monkeypatch.setattr("core.arms.spec.load_arm_utils", unavailable)
    apply_robot_appearance(None, "/Robot", object(), {})
    apply_robot_appearance(None, "/Robot", object(), {"appearance": {}})


def test_scene_look_is_explicit_and_arm_material_declarations_are_preserved(monkeypatch):
    calls = []
    monkeypatch.setattr("core.arms.spec.load_arm_utils",
                        lambda arm: SimpleNamespace(apply_appearance=lambda *args: calls.append(args)))
    arm = SimpleNamespace(appearance={"body_material": "Looks/Body"})
    look = {"appearance": {"paint": [1, 0, 0]}}
    apply_robot_appearance("stage", "/CustomRobot", arm, look)
    assert calls == [("stage", "/CustomRobot", {"paint": [1, 0, 0], "body_material": "Looks/Body"})]
    assert look == {"appearance": {"paint": [1, 0, 0]}}
