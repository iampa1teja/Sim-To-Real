"""Evaluation episode loop; setup and report formatting live in separate modules."""

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any
import sys
import traceback
import torch
from tqdm import tqdm
from core.benchmark.resume import load_episodes, next_video_attempt, write_report
from sim.eval.report import build_report, episode_metadata, progress_summary, finish_report
from sim.eval.scene import capture_scene, validate_scene_geometry
from sim.eval.stage_tracking import reset_stage_traces


@dataclass
class EvaluationContext:
    args_cli: Any
    simulation_app: Any
    task: Any
    env_id: Any
    env_cfg: Any
    env: Any
    benchmark: Any
    resume: Any
    saved_report: Any
    smoke_test: Any
    smoke_selection: Any
    smoke_reach: Any
    smoke_progress: Any
    smoke_budget: Any
    smoke_deadline: Any
    render_warmup: Any
    episode_length_s: Any
    step_dt: Any
    replan_interval_s: Any
    keyboard_control: Any
    policy: Any
    grasp_config: Any
    grasp_kwargs: Any
    smoothing_defaults: Any
    smoothing_kwargs: Any
    policy_options: Any
    rename_map: Any


@dataclass
class LoopState:
    episodes: list = field(default_factory=list)
    successes: int = 0
    step: int = 0
    pbar: Any = None
    video_recorder: Any = None
    video_attempt: int = 0
    video_dir: Path | None = None


def run_loop(context):
    state = LoopState(video_dir=context.args_cli.video_dir if context.resume else None)
    if context.grasp_config.enabled:
        context.policy.grasp.clock = lambda: state.step * context.step_dt
    if context.resume:
        expected = build_report(context, state)
        identity_keys = (
            "task",
            "checkpoint",
            "seed",
            "random_fraction",
            "episode_length_s",
            "action_horizon",
            "action_repeat",
            "step_dt",
            "robot_start",
            "robot_usd",
            "eval_set_id",
            "repeats",
            "requested_episodes",
            "policy_joint_transform",
            "policy_cameras",
            "resume_settings",
            "config_fingerprint",
        )
        context.saved_report = load_episodes(
            context.args_cli.results_json,
            {key: expected[key] for key in identity_keys if key in expected},
            context.benchmark.schedule,
        )
        state.episodes = context.saved_report["episodes"].copy()
        state.successes = sum((row["success"] for row in state.episodes))
        print(
            f"[RESUME] Retained {len(state.episodes)}/{context.args_cli.num_episodes} completed episodes; continuing at the next unfinished attempt.",
            flush=True,
        )
    try:
        if len(state.episodes) == context.args_cli.num_episodes:
            print("[RESUME] Evaluation is already complete; no attempts to run.", flush=True)
            return
        if getattr(context.args_cli, "save_videos", False):
            from sim.eval.video import EpisodeVideoRecorder

            state.video_dir = getattr(context.args_cli, "video_dir", None)
            if state.video_dir is None:
                if context.args_cli.results_json:
                    state.video_dir = context.args_cli.results_json.with_name(
                        context.args_cli.results_json.stem + "_videos"
                    )
                else:
                    from datetime import datetime, timezone

                    state.video_dir = Path("outputs/eval_videos") / datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%S%f"
                    )
            state.video_dir = state.video_dir.expanduser().resolve()
            state.video_recorder = EpisodeVideoRecorder(
                state.video_dir,
                fps=1 / context.step_dt,
                rename_map=context.rename_map,
                encoder=getattr(context.args_cli, "video_encoder", "libx264"),
            )
            if context.resume:
                state.video_attempt = next_video_attempt(state.video_dir)
            print(
                f"[INFO]: Saving attempt videos to {state.video_dir} (background {state.video_recorder.encoder} encoding)",
                flush=True,
            )
        if context.args_cli.results_json:
            write_report(context.args_cli.results_json, build_report(context, state))
        obs, _ = context.env.reset()
        metadata = episode_metadata(context)
        if context.benchmark:
            live = capture_scene(context.env.unwrapped, context.task)
            validate_scene_geometry(live, context.benchmark.data["scene"], context.task)
            context.policy.reset(seed=metadata["policy_seed"])
        else:
            context.policy.reset()
        actions = torch.zeros(context.env.action_space.shape, device=context.env.unwrapped.device)

        def settling_pose(observation):
            return observation["policy"]["joint_pos_obs"][0].clone()

        settling_action = settling_pose(obs)
        state.step = 0
        episode_wall_start = monotonic()
        render_every = context.env.unwrapped.cfg.sim.render_interval if context.render_warmup else None
        while context.simulation_app.is_running() and len(state.episodes) < context.args_cli.num_episodes:
            with torch.inference_mode():
                if state.step == 0:
                    if state.video_recorder is not None:
                        state.video_recorder.start_episode(state.video_attempt)
                        state.video_attempt += 1
                    state.pbar = tqdm(
                        total=context.env.unwrapped.max_episode_length,
                        desc=f"Rollout (ep {len(state.episodes) + 1}/{context.args_cli.num_episodes})",
                        unit="step",
                    )
                    progress_text = progress_summary(context, state)
                    state.pbar.set_postfix_str(progress_text)
                if state.step < 10:
                    actions[:] = settling_action
                else:
                    joint_positions = obs["policy"]["joint_pos_obs"][0].clone()
                    actions[:] = context.policy.get_action(joint_positions, obs["visual"], log=context.args_cli.rerun)
                if context.render_warmup:
                    steps_to_query = 9 - state.step if state.step < 10 else context.policy.steps_until_query
                    context.env.unwrapped.cfg.sim.render_interval = (
                        render_every if steps_to_query < context.render_warmup else 10**12
                    )
                if state.video_recorder is not None:
                    state.video_recorder.write(obs["visual"])
                obs, _, terminated, truncated, _ = context.env.step(actions)
                state.step += 1
                if context.grasp_config.enabled and context.env.unwrapped._benchmark_trace.stages.get("grasped", False):
                    context.policy.grasp.mark_grasp(state.step * context.step_dt)
                state.pbar.update(1)
                is_terminated = bool(terminated.any().item())
                is_truncated = bool(truncated.any().item())
                if is_terminated or is_truncated:
                    terminal_stage = context.env.unwrapped._benchmark_trace.progress_label()
                    success = bool(context.env.unwrapped._benchmark_trace.stages[context.task.stages[-1]])
                    row = {
                        **metadata,
                        "episode": len(state.episodes),
                        "steps": state.step,
                        "success": success,
                        "success_step": state.step if success else None,
                        "inference_calls": getattr(context.policy, "inference_calls", None),
                        "wall_time_s": monotonic() - episode_wall_start,
                    }
                    row["elapsed_sim_s"] = state.step * context.step_dt
                    row["time_to_success_s"] = row["elapsed_sim_s"] if row["success"] else None
                    row["wall_time_to_success_s"] = row["wall_time_s"] if row["success"] else None
                    if state.video_recorder is not None:
                        row["videos"] = state.video_recorder.finish_episode()
                        row["video_encoder"] = state.video_recorder.encoder
                    row.update(context.env.unwrapped._benchmark_trace.finish(row["success"]))
                    reset_stage_traces(context.env.unwrapped)
                    if context.grasp_config.enabled:
                        row.update(
                            context.policy.grasp.finish_episode(reason="success" if row["success"] else "timeout")
                        )
                    if context.smoke_deadline is not None:
                        row["smoke_deadline"] = context.smoke_deadline.summary()
                    state.episodes.append(row)
                    state.successes += int(row["success"])
                    if context.args_cli.results_json:
                        write_report(context.args_cli.results_json, build_report(context, state))
                    state.pbar.set_postfix_str(progress_summary(context, state, terminal_stage))
                    state.pbar.close()
                    state.pbar = None
                    start = "random" if row["start_index"] == -1 else row["start_index"]
                    timing_label = "time_to_success" if row["success"] else "timeout_elapsed"
                    final_status = f" {terminal_stage}" if terminal_stage else ""
                    if row.get("failure_mode"):
                        final_status += f" failure={row['failure_mode']}"
                    if row.get("smoke_deadline", {}).get("stop_reason"):
                        final_status += " stop=" + row["smoke_deadline"]["stop_reason"]
                    print(
                        f"[EPISODE {len(state.episodes)}] start={start} zone={row['zone']} color={row['color']} {('success' if row['success'] else 'fail')} steps={state.step} {timing_label}={row['elapsed_sim_s']:.2f}s sim ({row['wall_time_s']:.2f}s wall){final_status}",
                        flush=True,
                    )
                    if len(state.episodes) == context.args_cli.num_episodes:
                        break
                    metadata = episode_metadata(context)
                    if context.smoke_deadline is not None:
                        context.smoke_deadline.reset()
                    if context.benchmark:
                        context.policy.reset(seed=metadata["policy_seed"])
                    else:
                        context.policy.reset()
                    settling_action = settling_pose(obs)
                    state.step = 0
                    episode_wall_start = monotonic()
                elif context.keyboard_control.reset_world:
                    if context.benchmark:
                        raise RuntimeError("Manual reset interrupted fixed benchmark; saving incomplete report")
                    context.keyboard_control.reset_world = False
                    state.pbar.close()
                    state.pbar = None
                    print(f"[MANUAL RESET] Episode interrupted at step {state.step}; excluded from results")
                    if state.video_recorder is not None:
                        state.video_recorder.finish_episode()
                    obs, _ = context.env.reset()
                    context.policy.reset()
                    metadata = episode_metadata(context)
                    settling_action = settling_pose(obs)
                    state.step = 0
                    episode_wall_start = monotonic()
                else:
                    latest_progress = progress_summary(context, state)
                    if latest_progress != progress_text:
                        state.pbar.set_postfix_str(latest_progress, refresh=False)
                        progress_text = latest_progress
        if len(state.episodes) != context.args_cli.num_episodes:
            raise RuntimeError(
                f"Evaluation stopped after {len(state.episodes)}/{context.args_cli.num_episodes} episodes"
            )
    except Exception as error:
        traceback.print_exc()
        sys.stderr.flush()
        error._s2r_traceback_printed = True
        raise
    finally:
        if state.pbar is not None:
            state.pbar.close()
        try:
            finish_report(context, state)
        finally:
            try:
                if state.video_recorder is not None:
                    state.video_recorder.close()
                if context.smoothing_kwargs.get("log_timing") or context.grasp_config.enabled:
                    context.policy.close_timing()
            finally:
                if context.args_cli.rerun:
                    import rerun as rr

                    recording = rr.get_global_data_recording()
                    if recording is not None:
                        recording.flush(timeout_sec=5)
                    rr.disconnect()
