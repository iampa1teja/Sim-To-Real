"""Task-defined milestones latched once per live episode."""
from core.tasks.base import StepState, Task


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
        return {"stages": stages, "failure_mode": self.task.failure_mode(stages, self.last_state)}
