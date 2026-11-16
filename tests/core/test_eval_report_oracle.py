"""Frozen pre-merge reports retain JSON bytes, ordering and console output."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from itertools import combinations
import math
from pathlib import Path
from statistics import median
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest

from core.eval import report
from core.eval.eval_set import validate_eval_set
from core.eval.resume import write_report
from core.policy_client.grasp import GraspConfig


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/legacy_eval_report"


class FixedDatetime:
    @staticmethod
    def now(tz):
        return datetime(2026, 11, 16, 12, 0, tzinfo=tz)


def frozen(filename, **bindings):
    """Execute unchanged function bodies with explicit, independent old dependencies."""
    path = FIXTURES / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    module = ModuleType("legacy_" + path.stem)
    module.__dict__.update(math=math, np=np, median=median, datetime=FixedDatetime,
                           timezone=timezone, json=json, sys=sys, **bindings)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def legacy(monkeypatch):
    stats = frozen("stats.py")
    core = frozen("report.py", wilson_interval=stats.wilson_interval)
    sim = frozen("sim_report.py", benchmark_report=core, results_report=core.results_report,
                 write_report=write_report)
    monkeypatch.setattr(report, "datetime", FixedDatetime)
    # Image generation is outside this merge; exercise the dispatch without plotting.
    monkeypatch.setattr(report, "save_heatmap", lambda *args: None)
    core.save_heatmap = lambda *args: None
    return stats, core, sim


def context(tmp_path, benchmark, mode, empty):
    args = NS(checkpoint="fixture", seed=1984, repeats=2, random_fraction=.25,
              action_horizon=16, robot_start="recorded", action_repeat=2,
              arm="test_arm", scene="test_scene", task="test_task", variant="eval",
              seed_per_episode=True, num_episodes=2, policy="example", cube_starts="starts",
              lang_instruction="instruction", lang_instruction_by_color={"red": "red instruction"},
              render_warmup=3, results_json=tmp_path / "report.json", alpha=.5,
              log_timing=True, timing_csv="timing.csv", config_fingerprint="fingerprint")
    rows = [dict(start_id="id-000", split="id", repeat=0, policy_seed=7, zone="left",
                 start_kind="recorded", cube_color="red", success=True, success_step=30,
                 stages={"ready": True, "done": True}, failure_mode=None),
            dict(start_id="id-001", split="ood", repeat=1, policy_seed=8, zone="right",
                 start_kind="random", cube_color="blue", success=False, success_step=None,
                 stages={"ready": True, "done": False}, failure_mode="timeout")]
    if empty:
        rows = []
    cfg = GraspConfig(gripper_latch=mode == "grasp")
    policy = NS(grasp=NS(calibration={"closed": 1., "open": 2.}))
    state = NS(episodes=rows, successes=sum(row["success"] for row in rows),
               video_dir=tmp_path / "videos", video_recorder=None)
    ctx = NS(args_cli=args, env_id="S2R-test-eval", task=NS(stages=("ready", "done")),
             benchmark=NS(data={"eval_set_id": "digest", "legacy_eval_set_id": "legacy"}) if benchmark else None,
             episode_length_s=15., step_dt=.1, replan_interval_s=.2,
             env_cfg=NS(scene=NS(robot=NS(spawn=NS(usd_path="robot.usd")))),
             policy=policy, grasp_config=cfg, grasp_kwargs={"enabled": True} if mode == "grasp" else {},
             smoothing_kwargs={"alpha": .5} if mode == "smoothing" else {},
             smoothing_defaults={"alpha": 1., "log_timing": False, "timing_csv": None},
             policy_options={"temperature": .1}, rename_map={"view": "camera"},
             render_warmup=3, resume=mode == "resume", saved_report={"save_videos": True, "video_encoder": "saved"},
             smoke_test=mode.startswith("smoke"), smoke_selection="one_per_zone",
             smoke_reach=3., smoke_progress=4., smoke_budget=5.)
    if mode == "video":
        state.video_recorder = NS(encoder="current")
        policy.joint_transform, policy.policy_cameras = {"joint": "axis"}, ["view"]
    if mode == "smoke_stdout":
        args.results_json = None
    return ctx, state


@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("mode", ["plain", "video", "resume", "grasp", "smoothing", "smoke_json", "smoke_stdout"])
def test_assembled_and_written_reports_are_byte_identical(legacy, tmp_path, capsys, benchmark, empty, mode):
    _, _, old = legacy
    ctx, state = context(tmp_path, benchmark, mode, empty)
    before = old.build_report(ctx, state)
    after = report.build_report(ctx, state)
    assert json.dumps(after, indent=2, allow_nan=False) == json.dumps(before, indent=2, allow_nan=False)
    old.finish_report(ctx, state)
    expected_stdout = capsys.readouterr().out
    expected_bytes = ctx.args_cli.results_json.read_bytes() if ctx.args_cli.results_json else None
    report.finish_report(ctx, state)
    assert capsys.readouterr().out == expected_stdout
    if expected_bytes is not None:
        assert ctx.args_cli.results_json.read_bytes() == expected_bytes


def test_statistics_and_paired_comparison_json_match_frozen_implementations(legacy):
    stats, _, _ = legacy
    old = frozen("compare.py", mcnemar=stats.mcnemar, validate_eval_set=validate_eval_set, combinations=combinations)
    for total in range(30):
        for successes in range(total + 1):
            assert report.wilson_interval(successes, total) == stats.wilson_interval(successes, total)
    data = json.loads((FIXTURES.parent / "legacy_eval_set_v1.json").read_text())
    runs = []
    for index in range(3):
        runs.append(dict(schema_version=2, complete=True, eval_set=deepcopy(data), eval_set_id=data["eval_set_id"],
                         task="S2R-pick_place_cube-eval", checkpoint=str(index), episode_length_s=15,
                         robot_start="recorded", seed=1984, action_horizon=16,
                         episodes=[dict(start_id=f"id-{n:03d}", split="id", repeat=0,
                                        policy_seed=7+n, success=(n+index) % 3 == 0) for n in range(12)]))
    assert json.dumps(report.compare(runs)) == json.dumps(old.compare(runs))
