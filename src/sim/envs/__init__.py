import os

import gymnasium as gym

from core.scenes import SCENE_NAME, load_scene_spec
from core.tasks.loader import load_task, resolve

_ENV_CLASS = "isaaclab.envs:ManagerBasedRLEnv"
BASE_GYM_ID = "S2R-base"


# Register one gym id unless it already exists (re-importing sim.envs must not raise).
def _register(env_id, env_cfg_entry_point):
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point=_ENV_CLASS,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": env_cfg_entry_point},
    )


# Gym id of a bare scene (arm + room, no task).
def scene_gym_id(scene=None):
    return f"S2R-{scene or SCENE_NAME}"


# The scene's env cfg class as "module:Class", from scene.toml [scene] env_cfg.
def scene_entry_point(scene=None):
    spec = load_scene_spec(scene)
    if "env_cfg" not in spec:
        raise KeyError(f"scenes/{scene or SCENE_NAME}/scene.toml needs [scene] env_cfg = \"module:EnvCfgClass\"")
    return spec["env_cfg"]


# Build the zero-arg function gym calls for one task variant: scene cfg first, then the task adds itself.
def _variant_builder(scene_entry, task, variant):
    apply_entry = task.spec.entry_point(variant)

    def build_env_cfg():
        cfg = resolve(scene_entry)()
        resolve(apply_entry)(cfg, task)
        return cfg

    build_env_cfg.__qualname__ = f"build_env_cfg[{task.name}:{variant}]"
    return build_env_cfg


# Register base, scene and every variant of the selected task; returns the task (None when S2R_TASK is empty).
def register_envs(scene=None, task=None):
    _register(BASE_GYM_ID, f"{__name__}.base_env_cfg:BaseEnvCfg")
    scene_entry = scene_entry_point(scene)
    _register(scene_gym_id(scene), scene_entry)

    task_name = task if task is not None else os.environ.get("S2R_TASK", "pick_place_cube")
    if not task_name:
        return None
    loaded = load_task(task_name)
    for variant in loaded.spec.all_variants():
        _register(loaded.gym_id(variant), _variant_builder(scene_entry, loaded, variant))
    return loaded


TASK = register_envs()
