"""Isaac variants for the task."""
from sim.envs.generic_task_cfg import apply_generate
from sim.envs.generic_task_cfg import apply_teleop as _apply_teleop


def apply_teleop(cfg, task):
    _apply_teleop(cfg, task)
    slots = task.spec.object_sets["cubes"].slot_names()
    for name, rgb in zip(slots, task.spec.palette.values(), strict=True):
        getattr(cfg.scene, name).spawn.visual_material.diffuse_color = rgb
