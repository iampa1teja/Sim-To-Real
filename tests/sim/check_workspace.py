"""WP14 §16.1 owner check: is the arm's [arm.workspace] reachable by the scripted expert?

Runs on the host once the arm has its measured kinematics (``s2r arm-kinematics`` writes
``arms/<arm>/kinematics.json`` in the container). No Isaac Sim needed.

For each task (default: all 17) it builds ``--n`` success specs with the real start sampler (positions drawn
from the workspace) and runs the scripted expert in the kinematic FakeEnv, printing the success rate and the
target position of every failed start. Target: >= 90 % per task. FakeEnv has no grasp physics, so this checks reach
and timing only.

If failures cluster near the base or the far edge, narrow ``r_m`` / ``bearing_deg`` / the clip in ``arms/<arm>/arm.toml``.

    uv run --group host python tests/sim/check_workspace.py --arm so101 --n 20
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

ALL_TASKS = ("pick_place_cube", "stack_two_cubes", "pick_place_color_cube", "stack_bowls",
             "prim_joint_sweep", "prim_gripper_gap", "prim_reach_point", "prim_test_reach", "prim_calibration_reach",
             "prim_waypoint_path", "prim_touch_object", "prim_nudge", "prim_push_to_region", "prim_lift_test",
             "prim_grasp_lift", "prim_place_at", "prim_regrasp_after_drop")


def expert_runs(arm, tasks, n):
    from core.arms.gripper import Gripper
    from core.arms.kinematics import KinematicChain
    from core.arms.raw_mapping import RawMapping
    from core.generate.errors import SamplingError
    from core.generate.fake_env import FakeEnv
    from core.generate.loop import make_expert_factory, make_spec_fn, run_episode
    from core.generate.plan import attempt_seed
    from core.generate.protocols import Kind
    from core.tasks.loader import load_task
    from v4_support.stubs import ListWriter

    chain, gripper, mapping = KinematicChain.from_arm(arm), Gripper.from_arm(arm), RawMapping.from_setup(arm=arm)
    worst = 1.0
    for name in tasks:
        task = load_task(name)
        spec_fn, factory = make_spec_fn(task, arm, scene="check"), make_expert_factory(task, arm)
        ok = total = 0
        bad = []
        for i in range(4 * n):
            try:
                spec = spec_fn(Kind.SUCCESS, i, attempt_seed(0, name, Kind.SUCCESS, i))
            except SamplingError:
                continue
            if spec.noise is not None:
                continue
            result, _ = run_episode(spec, task, FakeEnv(task, chain, gripper), factory(spec), ListWriter(), mapping)
            total += 1
            if result.status.value == "accepted":
                ok += 1
            else:
                target = spec.objects[spec.roles["target"]].pose.position[:2] if "target" in spec.roles else None
                bad.append({"attempt": i, "reason": getattr(result.reason, "value", None),
                            "furthest": result.furthest_stage,
                            "target_xy": None if target is None else [round(v, 3) for v in target]})
            if total == n:
                break
        rate = ok / max(total, 1)
        worst = min(worst, rate)
        print(json.dumps({"part": "expert", "task": name, "success": ok, "runs": total, "rate": round(rate, 3)}))
        for item in bad:
            print("   ", json.dumps(item))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="so101")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS))
    parser.add_argument("--n", type=int, default=20, help="success specs per task for the expert runs")
    args = parser.parse_args()
    worst = expert_runs(args.arm, [t for t in args.tasks.split(",") if t], args.n)
    passed = worst >= 0.9
    print(json.dumps({"passed": passed, "worst_task_rate": worst}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
