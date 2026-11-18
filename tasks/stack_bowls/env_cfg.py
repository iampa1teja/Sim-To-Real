"""Isaac variants for the bowl task."""
from sim.envs.generic_task_cfg import apply_teleop
from sim.envs.generic_task_cfg import apply_generate as _apply_generate


def apply_generate(cfg, task):
    _apply_generate(cfg, task)
    # The analytic bowl walls can produce more constraints than the default budget.
    cfg.sim.physics.solver_cfg.njmax = 768
