"""Register generic scene/task factories without importing the simulator runtime."""
import os

import gymnasium as gym

from core.paths import DEFAULT_TASK
from core.scenes import load_scene_spec, selected_scene_name
from core.tasks.loader import load_task, resolve

_ENV_CLASS = "isaaclab.envs:ManagerBasedRLEnv"
BASE_GYM_ID = "S2R-base"


def _register(env_id, env_cfg_entry_point):
    if env_id in gym.registry:
        # Task ids intentionally omit scene/arm. An explicit new composition must
        # replace its factory instead of silently retaining the previous scene.
        existing = gym.registry[env_id]
        if existing.entry_point != _ENV_CLASS:
            raise ValueError(f"Gym id {env_id!r} is owned by another environment")
        existing.kwargs["env_cfg_entry_point"] = env_cfg_entry_point
        return
    gym.register(id=env_id, entry_point=_ENV_CLASS, disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": env_cfg_entry_point})


def scene_gym_id(scene=None):
    return f"S2R-{scene or selected_scene_name()}"


def scene_entry_point(scene=None):
    load_scene_spec(scene)
    return "sim.envs.scene_builder:build_scene_cfg"


def _scene_builder(scene):
    def build_env_cfg():
        return resolve(scene_entry_point(scene))(scene=scene)
    build_env_cfg.__qualname__ = f"build_scene_cfg[{scene}]"
    return build_env_cfg


def _variant_builder(scene, task, variant):
    apply_entry = task.spec.entry_point(variant)

    def build_env_cfg():
        cfg = resolve(scene_entry_point(scene))(scene=scene)
        cfg.task_name = task.name
        resolve(apply_entry)(cfg, task)
        return cfg

    build_env_cfg.__qualname__ = f"build_env_cfg[{scene}:{task.name}:{variant}]"
    return build_env_cfg


def register_envs(scene=None, task=None):
    scene = selected_scene_name() if scene is None else scene
    _register(BASE_GYM_ID, f"{__name__}.base_env_cfg:BaseEnvCfg")
    scene_entry_point(scene)
    _register(scene_gym_id(scene), _scene_builder(scene))
    task_name = task if task is not None else os.environ.get("S2R_TASK", DEFAULT_TASK)
    if not task_name:
        return None
    loaded = load_task(task_name)
    for variant in loaded.spec.all_variants():
        _register(loaded.gym_id(variant), _variant_builder(scene, loaded, variant))
    return loaded


TASK = register_envs()
