"""The generic task env plan for pick_place_cube, per variant, against a hand-written expectation.

The expectation is what the legacy pick_place_cube env_cfg.py built (kept in
tests/fixtures/legacy_tasks/pick_place_cube/), with the intended renames of the port: the generic
``eval_object_color`` event, ``{"task": name}`` term parameters and the generic subtask/termination
cfg classes.
"""
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest

from core.tasks.loader import load_task

ENVS = Path(__file__).resolve().parents[2] / "src" / "sim" / "workspace" / "envs"
TASK = "pick_place_cube"


@contextmanager
def task_env_module():
    """Import task_env without the package __init__ (gym registration), restoring sys.modules after."""
    names = ("sim.workspace.envs", "sim.workspace.envs.task_env", "sim.workspace.envs.frame")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    package = types.ModuleType("sim.workspace.envs")
    package.__path__ = [str(ENVS)]
    sys.modules["sim.workspace.envs"] = package
    for name in names[1:]:
        sys.modules.pop(name, None)
    try:
        yield importlib.import_module("sim.workspace.envs.task_env")
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.fixture
def task_env():
    with task_env_module() as module:
        yield module


def _arm(bodies):
    return SimpleNamespace(gripper=SimpleNamespace(contact_bodies=bodies))


SCENE_MATERIALS = SimpleNamespace(appearance={"object_materials": {
    "blue_cube": {"color": [0.005, 0.07, 0.26], "roughness": 0.62},
    "white_box": {"color": [0.95, 0.95, 0.95], "roughness": 0.7},
}})
SCENE_PLAIN = SimpleNamespace(appearance={})


def _expected(task_env, variant, scene, bodies):
    spec = load_task(TASK).spec
    S, O, T, E = task_env.SensorPlan, task_env.ObjectPlan, task_env.TermPlan, task_env.Entity
    materials = scene.appearance.get("object_materials", {})
    cube_color = tuple(materials.get("blue_cube", {}).get("color", (0., 0., 1.)))
    box_color = tuple(materials.get("white_box", {}).get("color", (1., 1., 1.)))
    scene_items = (
        S("contact_grasp", "{ENV_REGEX_NS}/Robot/" + (bodies[0] if len(bodies) == 1 else "(" + "|".join(bodies) + ")"),
          ("{ENV_REGEX_NS}/CUBE",)),
        O("white_box", "{ENV_REGEX_NS}/WhiteBox", spec.objects["white_box"].usd, None, None, box_color,
          materials.get("white_box", {}).get("roughness", .5), False,
          (-0.247171785265209, -0.042505977873273204, 0.000999999999999829), (0., 0., 0.), "table"),
        O("blue_cube", "{ENV_REGEX_NS}/CUBE", None, (.02, .02, .02), .008, cube_color,
          materials.get("blue_cube", {}).get("roughness", .5), True, (0., 0., .05), (0., 0., 0.), "world"),
    )
    env = "sim.workspace.envs.task_env"
    tracking = T("reset_tracking", f"{env}:reset_task_tracking", {"task": TASK}, mode="reset")
    if variant == "teleop":
        events = (T("reset_from_starts", "sim.workspace.envs.mdp:park_object", {"asset_cfg": E("blue_cube")}, mode="reset"),
                  tracking)
    else:
        events = (T("reset_from_starts", "sim.workspace.envs.mdp.starts:reset_from_starts",
                    {"task": TASK, "starts_dir": None, "mode": "cycle", "random_fraction": 0.0, "seed": None,
                     "reset_robot": True}, mode="reset"),
                  tracking,
                  T("check_eval_order", "sim.workspace.envs.mdp.starts:assert_reset_event_order", mode="startup"))
    if variant == "eval_dr":
        events += (T("eval_room_light", "sim.workspace.scene.appearance:randomize_scene_light_exposure",
                     {"exposure_range": (-3.0, 1.0)}, mode="reset"),
                   T("eval_robot_color", "sim.workspace.envs.mdp:randomize_robot_color", mode="reset"),
                   T("eval_object_color", "sim.workspace.envs.mdp.object_color:randomize_object_color",
                     {"colors": {"blue": cube_color, "red": (.8, .05, .05)}, "object_name": "blue_cube"}, mode="reset"))
    subtask_terms = (T("cube_grasped", f"{env}:stage_observation", {"task": TASK, "stage": "grasped"}),
                     T("cube_placed", f"{env}:stage_observation", {"task": TASK, "stage": "placed"}))
    terminations = () if variant == "teleop" else (
        T("time_out", "sim.workspace.envs.mdp:time_out", time_out=True),
        T("success", f"{env}:task_success", {"task": TASK}, time_out=False))
    return task_env.TaskCfgPlan(scene=scene_items, events=events, subtask_terms=subtask_terms, terminations=terminations,
                                decimation=4, episode_steps=None if variant == "teleop" else 450,
                                drop_unused_cameras=variant != "teleop")


@pytest.mark.parametrize("variant", ["teleop", "eval", "eval_dr"])
@pytest.mark.parametrize("scene", [SCENE_MATERIALS, SCENE_PLAIN], ids=["scene_materials", "no_materials"])
@pytest.mark.parametrize("bodies", [("jaw",), ("gripper_right_link", "gripper_left_link")])
def test_plan_matches_the_legacy_env_cfg(task_env, variant, scene, bodies):
    spec = load_task(TASK).spec
    plan = task_env.plan_task_cfg(spec, variant, _arm(bodies), scene)
    assert plan == _expected(task_env, variant, scene, bodies)


def test_frozen_names(task_env):
    """Gym ids, reset event order, the success termination, prims and the contact sensor stay fixed."""
    task = load_task(TASK)
    assert [task.gym_id(variant) for variant in task.spec.all_variants()] == [
        "S2R-pick_place_cube-teleop", "S2R-pick_place_cube-eval", "S2R-pick_place_cube-eval_dr"]
    plan = task_env.plan_task_cfg(task.spec, "eval_dr", _arm(("jaw",)), SCENE_PLAIN)
    names = [event.name for event in plan.events]
    assert names[:2] == ["reset_from_starts", "reset_tracking"]
    assert [term.name for term in plan.terminations] == ["time_out", "success"]
    assert [item.name for item in plan.scene] == ["contact_grasp", "white_box", "blue_cube"]


def test_plan_refuses_arms_without_contact_bodies(task_env):
    spec = load_task(TASK).spec
    with pytest.raises(ValueError, match="contact_bodies"):
        task_env.plan_task_cfg(spec, "eval", SimpleNamespace(gripper=None), SCENE_PLAIN)
    with pytest.raises(KeyError, match="no variant"):
        task_env.plan_task_cfg(spec, "missing", _arm(("jaw",)), SCENE_PLAIN)


def test_variants_resolve_to_the_generic_appliers(task_env):
    spec = load_task(TASK).spec
    assert {variant: spec.entry_point(variant) for variant in spec.all_variants()} == {
        variant: f"sim.workspace.envs.task_env:apply_{variant}" for variant in ("teleop", "eval", "eval_dr")}
    for variant in spec.all_variants():
        assert callable(getattr(task_env, f"apply_{variant}"))
