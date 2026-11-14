"""pick(obj, lift): descend open onto the object centre, close, then lift along the table normal."""
from sim.actions.action import Move


class Pick(Move):
    def __init__(self, obj, lift=.09, grasp_seconds=.8):
        super().__init__(obj)
        self.lift = lift
        self.grasp_seconds = grasp_seconds

    def plan(self, ctx):
        opened, closed = self.gripper(ctx)
        source = self.position(ctx)
        lift = float(ctx.options.get("lift_height_m", self.lift))
        return (
            self.waypoint(ctx, "descend", source, opened),
            self.waypoint(ctx, "close", source, closed, float(ctx.options.get("grasp_seconds", self.grasp_seconds))),
            self.waypoint(ctx, "lift", source + self.normal(ctx) * lift, closed),
        )


def pick(obj, lift=.09, grasp_seconds=.8):
    return Pick(obj, lift, grasp_seconds)
