"""approach(obj, height): open the gripper above the object along the table normal."""
from sim.actions.action import Move


class Approach(Move):
    def __init__(self, obj, height=.06):
        super().__init__(obj)
        self.height = height

    def plan(self, ctx):
        opened, _ = self.gripper(ctx)
        height = float(ctx.options.get("approach_height_m", self.height))
        return (self.waypoint(ctx, "approach", self.position(ctx) + self.normal(ctx) * height, opened),)


def approach(obj, height=.06):
    return Approach(obj, height)
