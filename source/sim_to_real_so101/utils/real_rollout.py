"""Closed-loop rollout of a policy on the physical SO-101 follower.

numpy only: the robot (connect-free `get_observation` / `send_action`) and the policy (`get_raw_action`) are
injected, so every safety rule here is tested without hardware, LeRobot or torch.

Units are raw LeRobot units, the same ones the simulator mapping uses: -100..100 for the five body joints and
0..100 for the gripper. Nothing is sent to the arm unless `enable_motion` is set (shadow mode otherwise).
Stopping never releases torque: the arm holds its last goal. Cutting power or `--release-on-exit` lets it go.
"""
from dataclasses import dataclass
import json
import time

import numpy as np

from sim_to_real_so101.utils.joint_calibration import (
    FOLLOWER_JOINTS, follower_mapping_from_calibration, sim_gripper_closed_degrees,
)

BODY_RANGE = (-100.0, 100.0)
GRIPPER_RANGE = (0.0, 100.0)
OUTCOMES = ("success", "failure", "unlabeled")


@dataclass(frozen=True)
class RolloutSettings:
    fps: int = 30
    episode_seconds: float = 30.0
    enable_motion: bool = False
    max_start_error: float = 20.0  # largest allowed gap, in raw units, between the pose and the first target


@dataclass
class EpisodeResult:
    status: str  # completed | stopped | policy_error | robot_error | start_too_far
    steps: int
    seconds: float
    detail: str = ""


def joint_range(name):
    return GRIPPER_RANGE if name.startswith("gripper") else BODY_RANGE


def clamp_action(action, joint_order):
    """Target vector inside each joint's raw range. Raises ValueError for a missing or non-finite value."""
    values = []
    for name in joint_order:
        if name not in action:
            raise ValueError(f"policy action has no value for {name}")
        value = float(action[name])
        if not np.isfinite(value):
            raise ValueError(f"policy action for {name} is not finite ({value})")
        low, high = joint_range(name)
        values.append(min(max(value, low), high))
    return np.asarray(values, dtype=np.float32)


def run_episode(robot, policy, joint_order, cameras, settings, *, should_stop=lambda: False, log=None,
                clock=time.monotonic, sleep=time.sleep):
    """Run one episode at settings.fps for at most episode_seconds; hold the last goal when it ends or fails."""
    period = 1.0 / settings.fps
    max_steps = int(settings.episode_seconds * settings.fps)
    started = clock()
    next_tick = started
    step = 0
    status, detail = "completed", ""
    while step < max_steps:
        if should_stop():
            status = "stopped"
            break
        try:
            observation = robot.get_observation()
            state = np.asarray([observation[name] for name in joint_order], dtype=np.float32)
            images = {camera: observation[camera] for camera in cameras}
        except Exception as error:
            status, detail = "robot_error", f"observation failed: {error!r}"
            break
        try:
            target = clamp_action(policy.get_raw_action(state, images), joint_order)
        except Exception as error:
            status, detail = "policy_error", repr(error)
            break
        if step == 0:
            gap = float(np.max(np.abs(target - state)))
            if gap > settings.max_start_error:
                status = "start_too_far"
                detail = f"first target is {gap:.1f} raw units from the current pose (limit {settings.max_start_error})"
                break
        if settings.enable_motion:
            try:
                robot.send_action(dict(zip(joint_order, target.tolist())))
            except Exception as error:
                status, detail = "robot_error", f"send_action failed: {error!r}"
                break
        if log is not None:
            log({"step": step, "t": round(clock() - started, 4), "state": state.tolist(), "target": target.tolist(),
                 "sent": settings.enable_motion})
        step += 1
        next_tick += period
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)
        else:
            next_tick = clock()  # overran the tick: do not try to catch up with a burst of commands
    return EpisodeResult(status, step, round(clock() - started, 3), detail)


def summarize(episodes):
    """episodes: list of dicts with 'outcome' in OUTCOMES. Only labelled episodes count towards the rate."""
    labelled = [e for e in episodes if e["outcome"] in ("success", "failure")]
    successes = sum(e["outcome"] == "success" for e in labelled)
    return {"episodes": len(episodes), "labelled": len(labelled), "successes": successes,
            "success_rate": (successes / len(labelled)) if labelled else None}


def write_report(path, header, episodes):
    report = {"schema": "so101_real_rollout_v1", **header, "episodes": episodes, "summary": summarize(episodes)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
