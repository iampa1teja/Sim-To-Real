"""The generic task: ``task.toml`` data plus the stages and expert declared in ``task.py``.

``task.py`` exposes plain module-level names:

- ``stages``: dict of stage name -> check, in ``task.toml`` stage order; the last one is success.
- ``expert`` (optional): list of moves, planned in order by the scripted expert.
- optional hooks for genuinely task-specific rules:
  ``start_is_valid(task, start, scene)`` an extra start rule, applied after the generic one;
  ``failure_mode(flags, state)`` the failure name of an unsuccessful episode (default ``"timeout"``);
  ``progress_label(label, flags, state)`` extends the generic progress label;
  ``episode_metadata(metadata)`` extends the generic episode metadata.

Checks and moves come from ``sim.actions``, whose module tops are numpy-only, so loading a task needs
neither torch nor Isaac. Simulator work (signals, the env builder) is imported only when called.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.tasks.base import StepState, Task as BaseTask, TaskSpec
from utils.workspace import load_file_module


def load_task_file(path: Path):
    """Import a task folder's task.py under a unique module name (no sys.path changes)."""
    path = Path(path).resolve()
    return load_file_module(path, "_s2r_task_" + hashlib.sha256(str(path).encode()).hexdigest()[:16])


def _walk(checks):
    seen, ordered = set(), []

    def visit(check):
        if id(check) in seen:
            return
        seen.add(id(check))
        for child in check.children():
            visit(child)
        ordered.append(check)

    for check in checks:
        visit(check)
    return tuple(ordered)


class Task(BaseTask):
    def __init__(self, spec: TaskSpec, module=None):
        if spec.task_py is None:
            raise ValueError(f"Task {spec.name!r} has no task.py")
        super().__init__(spec)
        self.module = load_task_file(spec.task_py) if module is None else module
        stages = getattr(self.module, "stages", None)
        if not isinstance(stages, dict) or tuple(stages) != spec.stages:
            raise ValueError(f"{spec.task_py}: stages must be a dict with the task.toml stages {list(spec.stages)} "
                             f"in order, got {list(stages) if isinstance(stages, dict) else stages!r}")
        self.stage_checks = dict(stages)
        # Every distinct check, children first; the env resets each of them once.
        self.checks = _walk(self.stage_checks.values())
        terms = spec.raw.get("observations", {}).get("subtask_terms", {})
        for stage in (spec.stages[-1], *terms.values()):
            if stage not in self.stage_checks or not self.stage_checks[stage].in_sim:
                raise ValueError(f"{spec.task_py}: stage {stage!r} drives a simulator term and needs a "
                                 "simulator-evaluated check")

    def _hook(self, name):
        return getattr(self.module, name, None)

    def signals(self, env) -> dict[str, Any]:
        from sim.workspace.envs.task_env import task_signals

        return task_signals(env, self)

    def observe(self, state: StepState) -> dict[str, bool]:
        return {name: bool(check.observe(state, name, self.spec.objects)) for name, check in self.stage_checks.items()}

    def start_is_valid(self, start, scene) -> bool:
        if not super().start_is_valid(start, scene):
            return False
        rule = self._hook("start_is_valid")
        return True if rule is None else bool(rule(self, start, scene))

    def failure_mode(self, flags, state):
        if self.is_success(flags):
            return None
        rule = self._hook("failure_mode")
        return "timeout" if rule is None else rule(flags, state)

    def progress_label(self, flags, state):
        label = super().progress_label(flags, state)
        rule = self._hook("progress_label")
        return label if rule is None else rule(label, flags, state)

    def object_colors(self) -> dict[str, dict]:
        """``[task.dr] object_colors``: object name -> {colour name: RGB or "from_setup"}."""
        return dict(self.spec.raw.get("dr", {}).get("object_colors", {}))

    def episode_metadata(self, env, index=0) -> dict[str, Any]:
        metadata = {}
        colors = self.object_colors()
        if colors:
            # The colour-randomized object names the episode colour (its first palette entry by default).
            name, palette = next(iter(colors.items()))
            recorded = getattr(env, "_object_colors", {}).get(name)
            metadata["color"] = recorded[index] if recorded is not None and recorded[index] is not None else next(iter(palette))
        rule = self._hook("episode_metadata")
        return metadata if rule is None else rule(metadata)

    def expert(self):
        moves = getattr(self.module, "expert", None)
        if moves is None:
            raise NotImplementedError(f"Task {self.name!r} has no expert in task.py")
        from sim.actions.action import Expert

        return Expert(moves, self.spec.objects)
