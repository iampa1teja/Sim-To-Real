"""Evaluation metadata, report assembly and output, without simulator imports."""

import json
import sys
from core.eval import report as benchmark_report
from core.eval.report import results_report
from core.eval.resume import write_report


def episode_metadata(context):
    raw = context.env.unwrapped
    index = getattr(raw, "_task_start_index", [None])[0]
    task_metadata = context.task.episode_metadata(raw)
    color = task_metadata.get("color", "unknown")
    instruction = task_metadata.get("instruction", context.args_cli.lang_instruction)
    if context.args_cli.lang_instruction_by_color is not None:
        instruction = context.args_cli.lang_instruction_by_color.get(color)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Colour instructions need a nonempty instruction for {color!r}")
    context.policy.lang_instruction = instruction
    extra = {}
    if context.benchmark:
        start = raw._benchmark_start
        extra = dict(
            start_id=start["id"], split=start["split"], repeat=start["repeat"], policy_seed=start["policy_seed"]
        )
        extra.update({key: start[key] for key in ("x", "y", "yaw", "objects") if key in start})
    return {
        **extra,
        **task_metadata,
        "start_index": index,
        "start_kind": "random" if index == -1 else "recorded" if index is not None else "unknown",
        "zone": getattr(raw, "_task_start_zone", ["unknown"])[0],
        "color": color,
        "instruction": instruction,
    }


def progress_summary(context, state, stage_label=None):
    completed = len(state.episodes)
    rate = f"{state.successes / completed:.1%}" if completed else "N/A"
    if stage_label is None:
        stage_label = context.env.unwrapped._benchmark_trace.progress_label()
    return (f"{stage_label}, " if stage_label else "") + f"success={rate} ({state.successes}/{completed})"


def build_report(context, state):
    report_fn = results_report
    report_extra = {}
    if context.benchmark:
        report_fn = benchmark_report.benchmark_report
        report_extra = dict(
            eval_set=context.benchmark.data, repeats=context.args_cli.repeats, stages=context.task.stages
        )
    report = report_fn(
        state.episodes,
        task=context.env_id,
        checkpoint=context.args_cli.checkpoint,
        seed=context.args_cli.seed,
        **report_extra,
        random_fraction=context.args_cli.random_fraction,
        episode_length_s=context.episode_length_s,
        action_horizon=context.args_cli.action_horizon,
        step_dt=context.step_dt,
    )
    report["robot_start"] = context.args_cli.robot_start
    report["action_repeat"] = context.args_cli.action_repeat
    report["replan_interval_s"] = context.replan_interval_s
    report["robot_usd"] = context.env_cfg.scene.robot.spawn.usd_path
    report["selection"] = dict(
        arm=context.args_cli.arm,
        scene=context.args_cli.scene,
        task=context.args_cli.task,
        variant=context.args_cli.variant,
    )
    if state.video_recorder is not None or (context.resume and context.saved_report.get("save_videos")):
        report["save_videos"] = True
        report["video_dir"] = str(state.video_dir)
        if state.video_recorder is not None:
            report["video_encoder"] = state.video_recorder.encoder
        elif "video_encoder" in context.saved_report:
            report["video_encoder"] = context.saved_report["video_encoder"]
    if getattr(context.policy, "joint_transform", None) is not None:
        report["policy_joint_transform"] = context.policy.joint_transform
    if getattr(context.policy, "policy_cameras", None) is not None:
        report["policy_cameras"] = context.policy.policy_cameras
    if context.grasp_kwargs:
        from dataclasses import asdict

        report["gripper_control"] = asdict(context.grasp_config)
        if context.grasp_config.enabled:
            report["gripper_calibration"] = context.policy.grasp.calibration
        report["gripper_latch"] = context.grasp_config.gripper_latch
        report["seed_per_episode"] = context.args_cli.seed_per_episode
    if context.smoothing_kwargs:
        report["smoothing"] = {
            key: getattr(context.args_cli, key, default) for key, default in context.smoothing_defaults.items()
        }
    report["requested_episodes"] = context.args_cli.num_episodes
    if context.smoke_test:
        report["smoke_test"] = dict(
            selection=context.smoke_selection,
            reach_timeout_s=context.smoke_reach,
            progress_timeout_s=context.smoke_progress,
        )
    report["complete"] = len(state.episodes) == context.args_cli.num_episodes
    report["policy"] = context.args_cli.policy
    fingerprint = getattr(context.args_cli, "config_fingerprint", None)
    if fingerprint is not None:
        report["config_fingerprint"] = fingerprint
    from dataclasses import asdict

    report["resume_settings"] = dict(
        policy=context.args_cli.policy,
        policy_options=context.policy_options,
        cube_starts=context.args_cli.cube_starts,
        rename_map=context.rename_map,
        lang=context.args_cli.lang_instruction,
        lang_by_color=context.args_cli.lang_instruction_by_color,
        render_warmup=context.render_warmup,
        smoothing={
            key: getattr(context.args_cli, key, default)
            for key, default in context.smoothing_defaults.items()
            if key not in ("log_timing", "timing_csv")
        },
        grasp={
            key: value
            for key, value in asdict(context.grasp_config).items()
            if key not in ("log_gripper", "gripper_csv")
        },
        seed_per_episode=context.args_cli.seed_per_episode,
    )
    return report


def finish_report(context, state):
    report = build_report(context, state)
    overall = report["overall"]
    rate = overall["success_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "n/a"
    status = "complete" if report["complete"] else "incomplete"
    print(
        f"[RESULT] {status}: {len(state.episodes)}/{context.args_cli.num_episodes} episodes; success {overall['successes']}/{overall['episodes']} ({rate_text})"
    )
    if overall.get("median_time_to_success_s") is not None:
        print(f"[RESULT] Median time to success: {overall['median_time_to_success_s']:.2f}s sim")
    for group_key, label in (("by_split", "Split"), ("by_zone", "Zone"), ("by_cube_color", "Colour")):
        groups = report.get(group_key, {})
        if groups:
            values = ", ".join((f"{name} {stats['successes']}/{stats['episodes']}" for name, stats in groups.items()))
            print(f"[RESULT] {label}: {values}")
    if context.args_cli.results_json:
        write_report(context.args_cli.results_json, report)
        if context.benchmark:
            benchmark_report.save_heatmap(report, context.args_cli.results_json.with_suffix(".png"))
        print(f"Results JSON: {context.args_cli.results_json}")
    elif context.smoke_test and (context.smoke_selection == "one_per_zone" or context.smoke_budget is not None):
        smoke_result = {
            key: report.get(key)
            for key in (
                "complete",
                "requested_episodes",
                "checkpoint",
                "seed",
                "step_dt",
                "episode_length_s",
                "action_horizon",
                "action_repeat",
                "policy_cameras",
                "policy_joint_transform",
                "resume_settings",
                "smoke_test",
                "overall",
            )
        }
        smoke_result["episodes"] = [
            {
                key: row.get(key)
                for key in (
                    "start_id",
                    "zone",
                    "policy_seed",
                    "success",
                    "elapsed_sim_s",
                    "wall_time_s",
                    "failure_mode",
                    "stages",
                    "smoke_deadline",
                )
            }
            for row in state.episodes
        ]
        print("[SMOKE_RESULT] " + json.dumps(smoke_result, separators=(",", ":"), allow_nan=False), flush=True)
    elif not context.smoke_test:
        print(json.dumps(report, indent=2, allow_nan=False))
    sys.stdout.flush()
