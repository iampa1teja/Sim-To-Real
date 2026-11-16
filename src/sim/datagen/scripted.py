"""Generate physical expert rollouts and commit only task-confirmed successes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from utils.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from utils.cli import parse_yaml_args
from sim import add_launch_arguments, apply_selection, close_app, launch
from sim.datagen.rollout import execute_expert, make_report, publish_report, setup_context


def add_generation_launch_arguments(parser):
    add_launch_arguments(parser)
    if not any(action.dest == "headless" for action in parser._actions):
        parser.add_argument("--headless", action="store_true")
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--source", choices=("scripted",), default="scripted")
    for key, default in (("arm", DEFAULT_ARM), ("scene", DEFAULT_SCENE), ("task", DEFAULT_TASK)):
        parser.add_argument("--" + key, default=os.environ.get("S2R_" + key.upper(), default))
    parser.add_argument("--variant", default="teleop")
    parser.add_argument("--unit")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo_id", default="local/scripted")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1, help="Stop after this many accepted episodes; zero runs all attempts")
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1984)
    parser.add_argument("--starts_dir", type=Path)
    parser.add_argument("--start_options", default="{}", help="Per-run object start region: frame, xy_min, xy_max, yaw_deg")
    parser.add_argument("--initial_joints_sim", type=json.loads,
                        help="Optional fixed initial action-joint positions, radians/metres in manifest order")
    parser.add_argument("--start_joint_jitter_rad", type=float, default=0.,
                        help="Seeded initial revolute-arm joint variation; the robot base stays fixed")
    parser.add_argument("--expert_options", default="{}", help="Task-specific expert options as a JSON object")
    parser.add_argument("--ik_integral_gain", type=float, default=1.0, help="Bounded servo-error correction per second; zero disables")
    parser.add_argument("--ik_max_integral_bias_rad", type=float, default=.08)
    parser.add_argument("--ik_max_integral_bias_m", type=float, default=.005)
    parser.add_argument("--ik_damping", type=float, default=.0125,
                        help="Damped least-squares regularization in library IK")
    parser.add_argument("--solver_njmax", type=int,
                        help="Constraint capacity for complex imported collision geometry")
    parser.add_argument("--solver_nconmax", type=int,
                        help="Contact capacity for complex imported collision geometry")
    parser.add_argument("--debug", action="store_true", help="Save rollout diagnostics and waypoint RGB beside the report")
    parser.add_argument("--start_mode", choices=("random", "grid"), default="grid")
    parser.add_argument("--cells_per_axis", type=int, default=5)
    parser.add_argument("--yaw_steps", type=int, default=4)
    parser.add_argument("--max_starts", type=int, default=256)
    parser.add_argument("--disable_fabric", action="store_true")
    add_generation_launch_arguments(parser)
    return parser


def parse_args(argv=None, *, parser=None):
    parser = parser or build_parser()
    import sys
    words=list(sys.argv[1:] if argv is None else argv)
    explicit=lambda key: any(word=='--'+key or word.startswith('--'+key+'=') for word in words)
    for key in ('arm','scene','task'):
        value=os.environ.get('S2R_'+key.upper())
        if value and not explicit(key):
            words += ['--'+key,value]
    args = parse_yaml_args(parser, words, json_fields=("expert_options", "start_options"))
    if min(args.attempts, args.max_steps, args.fps) < 1 or args.episodes < 0 or args.warmup_steps < 0:
        parser.error("attempts/max_steps/fps must be positive; episodes/warmup_steps must be nonnegative")
    if min(args.cells_per_axis, args.yaw_steps, args.max_starts) < 1:
        parser.error("cells_per_axis/yaw_steps/max_starts must be positive")
    args.expert_options = json.loads(args.expert_options) if isinstance(args.expert_options, str) else args.expert_options
    if not isinstance(args.expert_options, dict):
        parser.error("expert_options must be a JSON object")
    if isinstance(args.start_options, str):
        args.start_options = json.loads(args.start_options)
    if not isinstance(args.start_options, dict) or set(args.start_options) - {'frame', 'xy_min', 'xy_max', 'yaw_deg'}:
        parser.error("start_options supports only frame, xy_min, xy_max and yaw_deg")
    for key in ('xy_min', 'xy_max', 'yaw_deg'):
        if key in args.start_options:
            try:
                valid = np.shape(args.start_options[key]) == (2,) and np.isfinite(args.start_options[key]).all()
            except (TypeError, ValueError):
                valid = False
            if not valid:
                parser.error("start_options vectors must contain two finite numbers")
    if not math.isfinite(args.start_joint_jitter_rad) or args.start_joint_jitter_rad < 0:
        parser.error("start_joint_jitter_rad must be nonnegative and finite")
    if args.starts_dir and args.start_options:
        parser.error("Choose starts_dir or start_options")
    correction = (args.ik_integral_gain, args.ik_max_integral_bias_rad, args.ik_max_integral_bias_m)
    if not np.isfinite(correction).all() or correction[0] < 0 or min(correction[1:]) <= 0:
        parser.error("IK correction gain must be nonnegative and its radian/metre caps positive and finite")
    if not math.isfinite(args.ik_damping) or args.ik_damping <= 0:
        parser.error("IK damping must be positive and finite")
    if any(value is not None and value < 1 for value in (args.solver_njmax, args.solver_nconmax)):
        parser.error("Physics solver capacities must be positive")
    if args.debug and args.report is not None and args.report.resolve().is_relative_to(args.destination.resolve()):
        parser.error("debug report must be outside the dataset destination")
    return args


def main(argv=None):
    args = parse_args(argv)
    apply_selection(args)
    app = launch(args, enable_cameras=True)
    context = None
    report = None
    succeeded = False
    try:
        import torch

        context = setup_context(args)
        expert = context.task.expert()
        report = make_report(context, source_provenance={"expert": str(context.task.spec.task_py),
                                                        "options": args.expert_options,
                                                        "solver_capacities": {
                                                            name: getattr(context.raw.cfg.sim.physics.solver_cfg, name)
                                                            for name in ('njmax', 'nconmax')}})
        publish_report(context, report)
        with torch.inference_mode():
            for attempt in range(args.attempts):
                seed = args.seed + attempt
                rollout = context.begin_attempt(seed)
                rollout.attempt = attempt
                try:
                    reason = execute_expert(context, rollout, expert)
                    row = rollout.recorder.finish(attempt=attempt, seed=seed, reason=reason)
                    if hasattr(rollout, "diagnostics"):
                        row["diagnostics"] = rollout.diagnostics
                    if args.debug:
                        row["waypoint_trace"] = rollout.waypoint_trace
                except BaseException:
                    rollout.recorder.abort()
                    raise
                report["episodes"].append(row)
                publish_report(context, report)
                print("[DATAGEN_ATTEMPT] " + json.dumps(row), flush=True)
                if args.episodes and report["accepted_episodes"] >= args.episodes:
                    break
        succeeded = report["accepted_episodes"] > 0 and (not args.episodes or report["accepted_episodes"] >= args.episodes)
        report["attempts_completed"] = len(report["episodes"])
        report["requested_episodes"] = args.episodes
        report["complete"] = (report["accepted_episodes"] >= args.episodes if args.episodes else
                              len(report["episodes"]) == args.attempts)
        publish_report(context, report)
        return 0 if succeeded else 1
    finally:
        try:
            if context is not None:
                try:
                    if report is not None:
                        publish_report(context, report)
                finally:
                    context.close()
        finally:
            close_app(app, success=succeeded)


if __name__ == "__main__":
    raise SystemExit(main())
