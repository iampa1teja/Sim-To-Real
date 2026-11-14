"""A second task plugin: two tracked cubes, with task-owned stacking rules."""
import numpy as np

from core.tasks.base import Pose, StepState, Task, load_task_spec


class StackTwoCubesTask(Task):
    failure_modes = ("never_reached", "missed_grasp", "dropped_before_target", "dropped_outside",
                     "stacked_not_confirmed", "timeout_holding")

    def sample_start(self, rng):
        first = super().sample_start(rng)["cube_a"]
        for _ in range(1000):
            second = super().sample_start(rng)["cube_a"]
            if np.linalg.norm(np.asarray(first.position[:2]) - second.position[:2]) >= .06:
                return {"cube_a": first, "cube_b": second}
        raise ValueError("Task start region cannot fit two separated cubes")

    def signals(self, env):
        from .mdp import stage_signals
        return stage_signals(env)

    def observe(self, state: StepState):
        signals = state.signals
        return {"reached": float(signals.get("reach_distance", np.linalg.norm(
                    np.asarray(state.ee.position) - state.objects["cube_a"].position))) <= .03,
                **{name: bool(signals.get(name, False)) for name in self.stages if name != "reached"}}

    def failure_mode(self, flags, state):
        if self.is_success(flags):
            return None
        if flags.get("stacked"):
            return "stacked_not_confirmed"
        if not flags.get("reached"):
            return "never_reached"
        if not flags.get("grasped"):
            return "missed_grasp"
        if state is not None and state.signals.get("holding", False):
            return "timeout_holding"
        return "dropped_outside" if flags.get("over_target") else "dropped_before_target"


TASK = StackTwoCubesTask(load_task_spec("stack_two_cubes"))
