"""Run task generation in Isaac, or report capacity without importing a simulator."""

import argparse as _argparse
from contextlib import contextmanager as _contextmanager
import json as _json
import os as _os
import sys as _sys


def build_parser():
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--task")
    parser.add_argument("--shard")
    parser.add_argument("--dst_root")
    parser.add_argument("--master_seed", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit_attempts", type=int, default=0)
    parser.add_argument("--disable_fabric", action="store_true")
    return parser


@_contextmanager
def _selection(cfg, task):
    values = dict(S2R_ARM=cfg.arm, S2R_SCENE=cfg.scene, S2R_TASK=task)
    previous = {name: _os.environ.get(name) for name in values}
    _os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                _os.environ.pop(name, None)
            else:
                _os.environ[name] = value


def _print_plan(cfg, tasks):
    from core.generate.loop import make_spec_fn
    from core.generate.plan import KINDS, check_capacity, shard_quota
    from core.tasks.loader import load_task
    for name in tasks:
        task = load_task(name)
        spec_fn = make_spec_fn(task, cfg.arm, scene=cfg.scene, noise=cfg.noise,
                               weights_override=cfg.perturbation_weights, fps=cfg.fps)
        report = check_capacity(name, cfg.master_seed, spec_fn, cfg.quotas[name],
                                 max_attempt_factor=cfg.max_attempt_factor,
                                 max_consecutive_skips=cfg.max_consecutive_skips,
                                 unique_across_kinds=cfg.scenario_unique_across_kinds, near_duplicates=True)
        quotas = {kind.value: getattr(cfg.quotas[name], kind.value) for kind in KINDS}
        print(_json.dumps(dict(task=name, shard=list(cfg.shard), quotas=quotas,
                                shard_quotas={k: shard_quota(v, *cfg.shard) for k, v in quotas.items()},
                                non_skipped={k.value: report.non_skipped[k] for k in KINDS}, horizon=report.horizon), sort_keys=True))


class _AttemptLimit(Exception):
    pass


class _LimitedWriter:
    """Keep the attempt boundary outside WP7, before any new episode starts."""
    def __init__(self, writer, maximum):
        self._writer, self._maximum, self._executed = writer, maximum, 0

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def expert_factory(self, factory):
        def make(spec):
            if self._maximum > 0 and self._executed >= self._maximum:
                raise _AttemptLimit()
            self._executed += 1
            return factory(spec)
        return make


def _current_summary(root):
    from core.generate.loop import GenerationSummary
    from core.generate.manifest import Manifest
    from core.generate.protocols import AttemptStatus, Kind
    accepted, rejected, skipped = dict.fromkeys(Kind, 0), {}, 0
    lines = Manifest(root / "meta" / "generation_manifest.jsonl").lines()
    for line in lines:
        if line.status is AttemptStatus.ACCEPTED:
            accepted[line.kind] += 1
        elif line.status is AttemptStatus.REJECTED:
            rejected[line.reason.value] = rejected.get(line.reason.value, 0)+1
        else:
            skipped += 1
    return GenerationSummary(accepted, rejected, skipped, len(lines))


def _make_env(args, cfg, task):
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from core.scenes import load_scene_spec
    import sim.envs
    sim.envs.register_envs(scene=cfg.scene, task=task.name)
    env_id = task.gym_id("generate")
    env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    impratio = load_scene_spec(cfg.scene).get("generation_mjwarp_impratio")
    if impratio is not None:
        env_cfg.sim.physics.solver_cfg.impratio = impratio
    env_cfg.seed = cfg.master_seed
    env_cfg.decimation = round(1/(cfg.fps*env_cfg.sim.dt))
    env_cfg.sim.render_interval = env_cfg.decimation
    if env_cfg.decimation < 1 or abs(env_cfg.decimation*env_cfg.sim.dt-1/cfg.fps) > 1e-8:
        raise ValueError("Requested fps is not a multiple of the physics timestep")
    env_cfg.terminations = None
    env_cfg.episode_length_s = (task.spec.raw.get("expert", {}).get("max_steps", 600)+1)/cfg.fps
    if not cfg.cameras:
        env_cfg.observations.visual = None
        for name in dir(env_cfg.scene):
            if name.startswith("camera_"):
                setattr(env_cfg.scene, name, None)
    return gym.make(env_id, cfg=env_cfg)


def _run(args, cfg, tasks, parser, argv):
    from isaaclab.app import AppLauncher
    from sim.generate.dr_replay import discover_sim_cameras, rendering_kit_args, git_commit
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.enable_cameras = cfg.cameras
    args.kit_args = rendering_kit_args(args.kit_args)
    app = AppLauncher(args).app
    try:
        import torch
        from core.arms.gripper import Gripper
        from core.arms.kinematics import KinematicChain
        from core.arms.raw_mapping import RawMapping
        from core.arms.spec import load_arm_spec
        from core.datasets.task_meta import LABEL_FEATURES
        from core.generate.errors import GenerationError
        from core.generate.loop import run_generation, make_expert_factory, make_spec_fn
        from core.tasks.loader import load_task
        from sim.generate.isaac_gen_env import IsaacGenEnv, RecorderEpisodeWriter
        from sim.recording.lerobot_recorder import LeRobotRecorder
        executed_total = 0
        failed = False
        for name in tasks:
            if args.limit_attempts > 0 and executed_total >= args.limit_attempts:
                break
            env = writer = None
            raw_env = None
            try:
                task = load_task(name)
                raw_env = _make_env(args, cfg, task)
                cameras = discover_sim_cameras(raw_env) if cfg.cameras else {}
                mapping = RawMapping.from_setup(arm=cfg.arm)
                env = IsaacGenEnv(raw_env, task, arm=cfg.arm, mapping=mapping,
                                  chain=KinematicChain.from_arm(cfg.arm), gripper=Gripper.from_arm(cfg.arm),
                                  cameras=cameras, palette_rgb=task.spec.palette)
                arm_spec = load_arm_spec(cfg.arm)[0]
                feature_names = [f"{joint}.pos" for joint in arm_spec["lerobot"]["names"]]
                features = {"observation.state": dict(dtype="float32", shape=(len(feature_names),), names=feature_names)}
                for camera, size in cameras.items():
                    features[f"observation.images.{camera}"] = dict(dtype="video", shape=(size["height"], size["width"], 3),
                                                                    names=["height", "width", "channels"])
                features["action"] = dict(dtype="float32", shape=(len(feature_names),), names=feature_names)
                recorder = LeRobotRecorder(task_name=task.instruction, repo_id=cfg.dataset_repo_id(name),
                                             dataset_root=cfg.dataset_root(name), fps=cfg.fps, device=raw_env.unwrapped.device,
                                             cameras=cameras, features=features, extra_features=LABEL_FEATURES,
                                             robot_type=arm_spec["lerobot"]["follower_type"])
                writer = RecorderEpisodeWriter(recorder)
                recorder.init_dataset()
                maximum = args.limit_attempts-executed_total if args.limit_attempts > 0 else 0
                limited = _LimitedWriter(writer, maximum)
                try:
                    with torch.inference_mode():
                        summary = run_generation(cfg, name, env, limited,
                                                   expert_factory=limited.expert_factory(make_expert_factory(task, cfg.arm)),
                                                   mapping=mapping,
                                                   spec_fn=make_spec_fn(task, cfg.arm, scene=cfg.scene, noise=cfg.noise,
                                                                         weights_override=cfg.perturbation_weights, fps=cfg.fps),
                                                   git_commit=git_commit())
                except _AttemptLimit:
                    summary = _current_summary(writer.root)
                except GenerationError as exc:
                    print(f"{name}: {type(exc).__name__}: {exc}", file=_sys.stderr)
                    summary = _current_summary(writer.root)
                    failed = True
                executed_total += limited._executed
                print(f"{name}: {summary}")
            finally:
                try:
                    if writer is not None:
                        writer.finalize()
                finally:
                    if env is not None:
                        env.close()
                    elif raw_env is not None:
                        raw_env.close()
        return int(failed)
    finally:
        app.close()


def main(argv=None):
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    from core.generate.config import load_generate_config
    from core.generate.errors import ConfigError, GenerationError
    try:
        overrides = {key: getattr(args, key) for key in ("shard", "dst_root", "master_seed") if getattr(args, key) is not None}
        cfg = load_generate_config(args.config_path, overrides)
        tasks = (args.task,) if args.task else cfg.tasks
        if any(name not in cfg.tasks for name in tasks):
            raise ConfigError(f"Task is not configured: {args.task}")
        if args.limit_attempts < 0:
            raise ConfigError("limit_attempts must be nonnegative")
        with _selection(cfg, tasks[0]):
            if args.dry_run:
                _print_plan(cfg, tasks)
                return 0
            return _run(args, cfg, tasks, parser, argv)
    except GenerationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
