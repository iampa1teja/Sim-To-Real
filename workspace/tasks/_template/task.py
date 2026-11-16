"""Compose checks and moves; this module must import without Torch or Isaac.

Names and stage order must match task.toml. Put item-specific geometry here;
reuse generic checks/moves instead of copying simulator environment code.
"""
from sim.actions import approach, grasped, hold, lifted, pick, place, placed, reached

lift = lifted("object", min_lift=.01)
grasp = grasped("object", force=2., lift=lift, warmup_steps=30)
placement = placed("object", "target", "on_top", grasp=grasp,
                   xy_tol=.004, z_tol=.004, max_speed=.05)
stages = {
    "reached": reached("gripper", "object", .03),
    "grasped": grasp,
    "lifted": lift,
    "over_target": reached("object", "target", relation="aligned", held=grasp, xy_tol=.004),
    "placed": placement,
    "success": hold(placement, 25),
}
expert = [
    approach("object", height=.06),
    pick("object", lift=.05),
    place("object", "target", "on_top", transport="direct", lift=.05,
          precise=.001, yaw_precise=.0005, settle_steps=26),
]
