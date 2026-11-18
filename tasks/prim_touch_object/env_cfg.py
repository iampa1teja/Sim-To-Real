"""Isaac variants for the primitive task."""
from sim.envs.generic_task_cfg import apply_teleop
from sim.envs.generic_task_cfg import apply_generate as _apply_generate


def apply_generate(cfg, task):
    from core.arms.spec import load_arm_spec

    _apply_generate(cfg, task)
    links = load_arm_spec()[0]["links"]
    cfg.scene.contact_touch_fixed = cfg.scene.contact_grasp.replace(
        prim_path="{ENV_REGEX_NS}/Robot/" + links["ee"],
    )
