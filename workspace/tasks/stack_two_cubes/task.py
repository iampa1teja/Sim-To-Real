"""Stack the blue cube (cube_a) on the red cube (cube_b): stages, success, expert and the task-specific rules."""
import numpy as np

from sim.actions import approach, grasped, hold, lifted, pick, place, placed, reached

XY_TOL = 0.004       # footprint may overhang the red cube's top face by this much
CONFIRM_STEPS = 25   # stacked for this many consecutive steps is success

lift = lifted("cube_a", min_lift=0.01, float32_ge=True)
grasp = grasped("cube_a", force=2.0, lift=lift, warmup_steps=30)
stacked = placed("cube_a", "cube_b", "on_top", grasp=grasp, warmup_steps=30,
                 xy_tol=XY_TOL, z_tol=0.004, max_speed=0.05)

stages = {
    "reached": reached("gripper", "cube_a", 0.03),
    "grasped": grasp,
    "lifted": lift,
    "over_target": reached("cube_a", "cube_b", relation="aligned", held=grasp, xy_tol=XY_TOL),
    "stacked": stacked,
    "success": hold(stacked, CONFIRM_STEPS),
}

expert = [
    approach("cube_a", height=.06),
    pick("cube_a", lift=.05),
    place("cube_a", "cube_b", "on_top", transport="direct", lift=.05, retreat=.06, clearance=.0005,
          settle_seconds=1.2, precise=XY_TOL / 4, yaw_precise=.0005, settle_steps=CONFIRM_STEPS + 1,
          feedback="yaw"),
]
expert_options = {"move_seconds": 1.5, "transport_seconds": .5, "waypoint_timeout_seconds": 8.0}


def sample_start(task, rng, sample):
    """Both cubes from the start region, the second redrawn until they are 6 cm apart."""
    first = sample()
    for _ in range(1000):
        second = sample()
        if np.linalg.norm(np.asarray(first.position[:2]) - second.position[:2]) >= .06:
            return {"cube_a": first, "cube_b": second}
    raise ValueError("Task start region cannot fit two separated cubes")
