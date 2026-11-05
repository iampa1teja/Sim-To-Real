"""Deterministic smoke coverage and evaluation-only progress deadlines.

These helpers select attempts and stop stalled episodes. They never generate or
modify policy observations or actions, or alter the task's success predicate.
"""
from __future__ import annotations

import math


MILESTONES = ("reached", "grasped", "lifted", "over_box", "placed")


def one_per_zone(schedule):
    """Keep the first scheduled example in each zone, including its original seed."""
    selected, seen = [], set()
    for row in schedule:
        zone = row.get("zone")
        if not isinstance(zone, str) or not zone:
            raise ValueError("Smoke zone selection requires a named zone on every start")
        if zone not in seen:
            selected.append(row)
            seen.add(zone)
    if not selected:
        raise ValueError("Smoke zone selection requires at least one start")
    return selected


class MilestoneDeadline:
    """60s to first reach, then 30s from each genuinely new milestone.

    Durations are configurable for tests. Time is simulated episode time. A
    repeated reach/grasp never buys time, late progress cannot revive an expired
    attempt, and progress exactly at the deadline is accepted. Observing several
    new milestones on one step grants one progress window, not stacked windows.
    """
    def __init__(self, reach_timeout_s=60.0, progress_timeout_s=30.0):
        for value in (reach_timeout_s, progress_timeout_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Smoke deadlines must be finite positive seconds")
        self.reach_timeout_s = float(reach_timeout_s)
        self.progress_timeout_s = float(progress_timeout_s)
        self.reset()

    @property
    def maximum_duration_s(self):
        return self.reach_timeout_s + len(MILESTONES) * self.progress_timeout_s

    def reset(self):
        self.deadline_s = self.reach_timeout_s
        self.highest = -1
        self.elapsed_s = 0.0
        self.times = {}
        self.stop_reason = None

    def observe(self, elapsed_s, stages, *, success=False):
        if not math.isfinite(elapsed_s) or elapsed_s < 0 or elapsed_s < self.elapsed_s - 1e-8:
            raise ValueError("Smoke deadline requires monotonic nonnegative episode time")
        self.elapsed_s = float(elapsed_s)
        if self.stop_reason:
            return True
        # Do not accept progress first observed after the previous deadline.
        if elapsed_s > self.deadline_s + 1e-8:
            self.stop_reason = "reach_timeout" if self.highest < 0 else "progress_timeout"
            return True
        # A pushed/flying cube can set the native "lifted" flag without a
        # grasp. Only an ordered prefix of milestones earns more time.
        highest = -1
        for i, name in enumerate(MILESTONES):
            if not stages.get(name):
                break
            highest = i
        if highest > self.highest:
            for i in range(self.highest + 1, highest + 1):
                if stages.get(MILESTONES[i]):
                    self.times[MILESTONES[i]] = self.elapsed_s
            self.highest = highest
            self.deadline_s = self.elapsed_s + self.progress_timeout_s
        if success:
            return False
        if elapsed_s >= self.deadline_s - 1e-8:
            self.stop_reason = "reach_timeout" if self.highest < 0 else "progress_timeout"
        return self.stop_reason is not None

    def summary(self):
        return dict(reach_timeout_s=self.reach_timeout_s, progress_timeout_s=self.progress_timeout_s,
                    deadline_s=self.deadline_s, milestone_times_s=dict(self.times), stop_reason=self.stop_reason)


def install_milestone_deadline(env, reach_timeout_s, progress_timeout_s):
    """Add truncation to an instrumented single-env benchmark, before auto-reset.

    Isaac normally evaluates time_out before success. Cache the instrumented
    success result per physics step so both terms see the current terminal frame
    and the original success predicate still executes exactly once per step.
    """
    import torch

    if env.num_envs != 1 or not hasattr(env, "_benchmark_trace"):
        raise ValueError("Smoke progress deadlines require one instrumented benchmark environment")
    budget = MilestoneDeadline(reach_timeout_s, progress_timeout_s)
    manager = env.termination_manager
    success_cfg = manager.get_term_cfg("success")
    timeout_cfg = manager.get_term_cfg("time_out")
    original_success, original_timeout = success_cfg.func, timeout_cfg.func
    success_params = dict(success_cfg.params)
    cache = dict(step=None, result=None, expired=False, elapsed=None)

    def observe(raw, **params):
        step = int(raw.common_step_counter)
        if cache["step"] == step:
            return cache["result"]
        elapsed = float(raw.episode_length_buf[0].item()) * float(raw.step_dt)
        if cache["elapsed"] is not None and elapsed < cache["elapsed"]:
            budget.reset()
        result = original_success(raw, **params)
        expired = budget.observe(elapsed, raw._benchmark_trace.stages, success=bool(result[0].item()))
        cache.update(step=step, result=result, expired=expired, elapsed=elapsed)
        return result

    def timeout(raw, **params):
        observe(raw, **success_params)
        result = original_timeout(raw, **params)
        return result | torch.full_like(result, cache["expired"])

    success_cfg.func, timeout_cfg.func = observe, timeout
    manager.set_term_cfg("success", success_cfg)
    manager.set_term_cfg("time_out", timeout_cfg)
    return budget
