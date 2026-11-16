"""Task-defined milestones and failure reporting from the furthest latched stage."""
from core.tasks.task import StepState, Task


def failure_mode(stages, latched, holding=False):
    """Report the next stage, even when earlier stages were never observed.

    Holding is a separate timeout flag, independent of which stage failed.
    A successful trace has no failure and is never a holding timeout.
    """
    stages = tuple(stages)
    if not stages:
        raise ValueError("Failure reporting requires at least one stage")
    success = bool(latched.get(stages[-1], False))
    furthest = max((i for i, name in enumerate(stages) if latched.get(name, False)), default=-1)
    return {"failure_mode": None if success else "failed_at_" + stages[furthest + 1],
            "timeout_holding": bool(holding) and not success}


class StageTrace:
    def __init__(self, task: Task):
        self.task = task
        self.stages = dict.fromkeys(task.stages, False)
        self.last_state: StepState | None = None

    def observe(self, state: StepState):
        flags = self.task.observe(state)
        for stage in self.stages:
            self.stages[stage] |= bool(flags.get(stage, False))
        self.last_state = state
        return flags

    @property
    def holding(self):
        return bool(self.last_state and self.last_state.signals.get("holding", False))

    def progress_label(self):
        return self.task.progress_label(self.stages, self.last_state)

    def finish(self, success=None):
        stages = dict(self.stages)
        if success is not None:
            stages[self.task.stages[-1]] = bool(success)
        return {"stages": stages, **failure_mode(self.task.stages, stages, self.holding)}
