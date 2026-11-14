# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task library base: the per-step ``Frame``, stage ``Check``s and expert ``Move``s.

A task's ``task.py`` builds its stages from checks and its scripted expert from moves. Both are
plain objects holding their parameters; nothing here imports torch or Isaac at module top.

Checks are evaluated in two places, as the stage rules are today:
- ``evaluate(frame)``: per-environment torch booleans on the live ``Frame`` (simulator side). Stateful
  checks keep their per-environment state on the instance and update each environment at most once
  per ``frame.step``, so several consumers in one step (termination, observations, the stage tracer)
  agree. ``reset`` clears an environment and lets it update again in the same step.
- ``observe(state, name, objects)``: one environment's flag on a ``StepState`` (core side). By default
  it reads the signal the simulator published under the stage name; pose-only checks recompute it.

Moves plan Cartesian waypoints with numpy only (world metres, XYZW orientations).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import cached_property
import math
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from utils.geometry import Pose

if TYPE_CHECKING:
    from core.tasks.base import ObjectSpec, StepState


# ---------------------------------------------------------------------------
# Per-step frame
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """Live measurements for ``N`` environments, built by ``sim.workspace.envs.frame``.

    Tensors are torch, world metres, XYZW quaternions. ``velocity`` rows are linear then angular.
    ``contact`` maps each contact-filtered object to its jaw contact force norm, summed over bodies.
    ``gripper_point`` is the grasp point used for reach distances (None without a gripper source).
    ``step`` is the env's ``common_step_counter``; ``episode_step`` its ``episode_length_buf``.
    ``owner`` identifies the environment whose state the checks keep.

    ``velocity`` and ``gripper_point`` are read from the simulator only when a check first uses them,
    through the ``velocities`` and ``gripper`` callables, so terms that do not need them touch no
    robot or velocity data.
    """
    position: dict[str, Any]
    orientation: dict[str, Any]
    contact: dict[str, Any]
    step: int
    episode_step: Any
    objects: dict[str, ObjectSpec]
    owner: object = None
    gripper: Callable[[], Any] | None = None
    velocities: Callable[[], dict[str, Any]] | None = None

    @cached_property
    def gripper_point(self):
        return None if self.gripper is None else self.gripper()

    @cached_property
    def velocity(self) -> dict[str, Any]:
        return {} if self.velocities is None else self.velocities()


def object_size(spec) -> tuple[float, float, float]:
    """Cuboid size (or a USD object's outer size) declared in task.toml."""
    size = spec.size if spec.size is not None else spec.params.get("size_outer")
    if size is None:
        raise ValueError(f"Task object {spec.name!r} needs size or size_outer")
    return tuple(float(value) for value in size)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class Check:
    """A stage rule. Subclasses override ``evaluate`` (simulator) and, if pose-only, ``observe``."""

    #: False for checks computed only from the core ``StepState``; they cannot drive simulator terms.
    in_sim = True

    def children(self) -> tuple[Check, ...]:
        return ()

    def reset(self, ids, frame) -> None:
        """Clear per-environment state for ``ids`` after their new object poses were written."""

    def evaluate(self, frame):
        raise NotImplementedError(f"{type(self).__name__} is evaluated on the core StepState only")

    def signals(self, frame, name) -> dict[str, Any]:
        """Values published to core for this stage (``StepState.signals``)."""
        return {name: self.evaluate(frame)}

    def observe(self, state: StepState, name: str, objects) -> bool:
        return bool(state.signals.get(name, False))

    def __and__(self, other):
        return AllOf(self, other)

    def __invert__(self):
        return Not(self)


def _require_sim(check, user):
    if not check.in_sim:
        raise ValueError(f"{user} needs a simulator-evaluated check; {type(check).__name__} is core-only")
    return check


class AllOf(Check):
    def __init__(self, *checks):
        self.checks = tuple(_require_sim(check, "&") for check in checks)

    def children(self):
        return self.checks

    def evaluate(self, frame):
        result = self.checks[0].evaluate(frame)
        for check in self.checks[1:]:
            result = result & check.evaluate(frame)
        return result


class Not(Check):
    def __init__(self, check):
        self.check = _require_sim(check, "~")

    def children(self):
        return (self.check,)

    def evaluate(self, frame):
        return ~self.check.evaluate(frame)


class Tracker(Check):
    """A check with per-environment state, updated once per environment per ``frame.step``.

    State is allocated on first use from the frame, bound to ``frame.owner``; a frame from another
    environment starts fresh state. Each state holds ``step`` (last update, -1 after reset).
    """

    _owner = None
    _state = None

    def initial(self, frame) -> dict[str, Any]:
        raise NotImplementedError

    def clear(self, state, ids, frame) -> None:
        raise NotImplementedError

    def state(self, frame):
        if self._state is None or self._owner is not frame.owner:
            import torch

            state = self.initial(frame)
            first = next(iter(state.values()))
            state["step"] = torch.full((first.shape[0],), -1, dtype=torch.long, device=first.device)
            self._owner, self._state = frame.owner, state
        return self._state

    def due(self, frame, state):
        """Environments not yet updated at this step."""
        return state["step"] != frame.step

    def reset(self, ids, frame):
        state = self.state(frame)
        self.clear(state, ids, frame)
        state["step"][ids] = -1


class Hold(Tracker):
    """True once ``check`` has held for ``steps`` consecutive updates (a bounce does not count)."""

    def __init__(self, check, steps):
        if steps < 1:
            raise ValueError("confirm_steps must be positive.")
        self.check = _require_sim(check, "hold")
        self.steps = int(steps)

    def children(self):
        return (self.check,)

    def initial(self, frame):
        import torch

        n = frame.episode_step.shape[0]
        return {"count": torch.zeros(n, dtype=torch.long, device=frame.episode_step.device)}

    def clear(self, state, ids, frame):
        state["count"][ids] = 0

    def evaluate(self, frame):
        import torch

        value = self.check.evaluate(frame)
        state = self.state(frame)
        update = self.due(frame, state)
        count = torch.where(value, state["count"] + 1, 0)
        state["count"][update] = count[update]
        state["step"][update] = frame.step
        return state["count"] >= self.steps


def hold(check, steps):
    return Hold(check, steps)


# ---------------------------------------------------------------------------
# Expert waypoints (moved unchanged from core/tasks/expert.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpertContext:
    state: StepState
    control_pose: Pose
    table_normal: tuple[float, float, float]
    options: dict = field(default_factory=dict)
    arm_base: Pose | None = None  # actual declared base-link pose in world coordinates
    arm: object | None = None


@dataclass(frozen=True, slots=True)
class Waypoint:
    name: str
    pose: Pose
    gripper_open_fraction: float | None = None
    duration_s: float = 2.0
    tolerance_m: float = .004
    timeout_s: float = 8.0
    gripper_open_gap_m: float | None = None
    object_ref: str | None = None  # world object-centre goal, with fixed pickup TCP offset

    def __post_init__(self):
        if (self.gripper_open_fraction is None) == (self.gripper_open_gap_m is None):
            raise ValueError("Exactly one of gripper_open_fraction and gripper_open_gap_m is required")
        opening = self.gripper_open_fraction if self.gripper_open_gap_m is None else self.gripper_open_gap_m
        values = (*self.pose.position, *self.pose.quat_xyzw, opening,
                  self.duration_s, self.tolerance_m, self.timeout_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Expert waypoints must contain finite poses and timing")
        if not self.name or sum(value * value for value in self.pose.quat_xyzw) < 1e-12:
            raise ValueError("Expert waypoints need a name and nonzero quaternion")
        if self.gripper_open_fraction is not None and not 0 <= self.gripper_open_fraction <= 1:
            raise ValueError("gripper_open_fraction must be within [0, 1]")
        if self.gripper_open_gap_m is not None and self.gripper_open_gap_m < 0:
            raise ValueError("gripper_open_gap_m must be nonnegative")
        if min(self.duration_s, self.tolerance_m, self.timeout_s) <= 0 or self.timeout_s < self.duration_s:
            raise ValueError("Waypoint timing/tolerance must be positive, with timeout >= duration")


def gripper_commands(context, object_width_m):
    """Expert opening options, with measured physical openings for imported arms."""
    options = context.options
    grip = context.arm.gripper if context.arm is not None else None
    measured = grip is not None and grip.open_gap_m is not None
    physical = measured or any(key in options for key in ('gripper_open_gap_m', 'gripper_closed_gap_m'))
    if physical:
        default_open = min(grip.open_gap_m, object_width_m + .02) if measured else object_width_m + .02
        return (dict(gripper_open_gap_m=float(options.get('gripper_open_gap_m', default_open))),
                dict(gripper_open_gap_m=float(options.get('gripper_closed_gap_m', 0.))))
    return (dict(gripper_open_fraction=float(options.get('gripper_open_fraction', .45))),
            dict(gripper_open_fraction=float(options.get('gripper_closed_fraction', 0.))))


def grasp_orientation(context, object_pose, orientation):
    """Nearest cube face normal about the table normal; preserve this after pickup."""
    import numpy as np
    from utils.geometry import rotation_matrix_xyzw, quaternion_multiply

    arm = context.arm
    if arm is None or 'z' not in arm.ik.orientation_axes or arm.gripper is None:
        return orientation
    if arm.gripper.closing_axis is None:
        return orientation
    normal = np.asarray(context.table_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    closing = rotation_matrix_xyzw(orientation) @ arm.gripper.closing_axis
    closing -= np.dot(closing, normal) * normal
    if np.linalg.norm(closing) < 1e-8:
        raise ValueError('Gripper closing axis must lie in the table plane for grasp yaw alignment')
    closing /= np.linalg.norm(closing)
    rotation = rotation_matrix_xyzw(object_pose.quat_xyzw)
    angles = []
    for direction in (rotation[:, 0], -rotation[:, 0], rotation[:, 1], -rotation[:, 1]):
        direction = direction - np.dot(direction, normal) * normal
        if np.linalg.norm(direction) < 1e-8:
            continue
        direction /= np.linalg.norm(direction)
        angles.append(math.atan2(np.dot(normal, np.cross(closing, direction)), np.dot(closing, direction)))
    angle = min(angles, key=abs)
    increment = tuple(normal * math.sin(angle / 2)) + (math.cos(angle / 2),)
    return quaternion_multiply(increment, orientation)


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MoveContext(ExpertContext):
    """An ``ExpertContext`` plus the task's declared objects (sizes, container geometry)."""
    objects: dict = field(default_factory=dict)


class Move:
    """One expert phase for the manipulated object ``obj``; ``plan`` returns its waypoints.

    Shared settings come from the expert options with these defaults: ``move_seconds`` (2.0),
    ``waypoint_timeout_seconds`` (max(8, 3 * move_seconds)), ``position_tolerance_m`` (0.004),
    ``orientation_xyzw`` (the current control orientation, turned to the nearest object face),
    the gripper opening options, and ``object_feedback`` (False) for object-centre goals.
    """

    def __init__(self, obj):
        self.obj = obj

    def validate(self, ctx: MoveContext) -> None:
        """Reject invalid options before any move plans (the expert validates every move first)."""

    def plan(self, ctx: MoveContext) -> tuple[Waypoint, ...]:
        raise NotImplementedError

    @staticmethod
    def normal(ctx):
        normal = np.asarray(ctx.table_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        return normal

    def position(self, ctx, name=None):
        return np.asarray(ctx.state.objects[name or self.obj].position, dtype=float)

    def size(self, ctx, name=None):
        return ctx.objects[name or self.obj].size

    def duration(self, ctx):
        return float(ctx.options.get("move_seconds", 2.0))

    def gripper(self, ctx):
        """(opened, closed) gripper commands for the manipulated object's width."""
        return gripper_commands(ctx, max(self.size(ctx)[:2]))

    def orientation(self, ctx):
        orientation = tuple(ctx.options.get("orientation_xyzw", ctx.control_pose.quat_xyzw))
        return grasp_orientation(ctx, ctx.state.objects[self.obj], orientation)

    def waypoint(self, ctx, name, position, opening, seconds=None, *, object_goal=False):
        options = ctx.options
        duration = self.duration(ctx)
        seconds = duration if seconds is None else seconds
        timeout = float(options.get("waypoint_timeout_seconds", max(8., duration * 3)))
        return Waypoint(name, Pose.from_lists(position, self.orientation(ctx)), duration_s=seconds,
                        tolerance_m=float(options.get("position_tolerance_m", .004)),
                        timeout_s=max(timeout, seconds),
                        object_ref=self.obj if object_goal and options.get("object_feedback", False) else None,
                        **opening)


class Expert:
    """Plans a task's moves in order; the scripted runner calls ``plan(context)``."""

    def __init__(self, moves, objects):
        self.moves = tuple(moves)
        self.objects = objects

    def plan(self, context):
        values = {item.name: getattr(context, item.name) for item in fields(ExpertContext)}
        ctx = MoveContext(**values, objects=self.objects)
        for move in self.moves:
            move.validate(ctx)
        return tuple(point for move in self.moves for point in move.plan(ctx))
