"""Re-render recorded pick-place episodes with blue/red cubes and domain randomization.

Kinematic replay: for every frame the recorded joint state and cube pose are written into the
task scene and the cameras are rendered (no physics step). Actions and observation.state are
copied unchanged; only the camera images and the task text change. Writes a NEW LeRobot dataset
plus task_meta/ object sidecars and meta/dr_replay_manifest.json.

  # plan only (no Isaac Sim; needs LeRobot)
  s2r datagen --source dr_replay --src_root <src> --src_repo_id local/pick_place_v1 \
      --dst_root <dst> --dst_repo_id local/pick_place_v2_dr --exclude_episodes 0,6 --dry_run
  # fidelity check: blue, no DR, 2 episodes -> frames must match the originals
  s2r datagen --source dr_replay ... --colors blue --dr "" --limit 2 --headless
"""
# - --arm/--scene/--task/--variant select the env. S2R_* are exported before importing any core/sim module
#   that reads them at import.
# - DR ranges come from task.toml [task.dr] (the same values apply_eval_dr uses).
# - Joint units go through JointMapping; the recorder is sim.recording.recorder.LeRobotRecorder.

import argparse
import json
import os
from pathlib import Path

from utils.paths import DEFAULT_ARM, DEFAULT_SCENE, DEFAULT_TASK
from utils.provenance import git_commit
from sim import apply_selection, add_launch_arguments, launch, close_app

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_DR_TERMS = ("light", "robot_color")  # camera views stay fixed to the calibrated real setup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay recorded episodes with visual domain randomization.")
    parser.add_argument("--config_path", type=Path, help="flat datagen YAML; flags override defaults")
    parser.add_argument("--source", choices=["dr_replay"], default="dr_replay")
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--src_repo_id", required=True)
    parser.add_argument("--dst_root", required=True)
    parser.add_argument("--dst_repo_id", required=True)
    parser.add_argument("--exclude_episodes", default="", help="Source episodes to skip, e.g. 0,6")
    parser.add_argument("--colors", default="blue,red")
    parser.add_argument("--dr", default="light,robot_color",
                        help='Comma list of DR terms (light, robot_color), or "" for none (fidelity check). '
                             "Cameras are never randomized: they are calibrated to the real setup.")
    parser.add_argument("--include_originals", action="store_true",
                        help="Also re-render every kept episode blue with no DR, as the first episodes")
    parser.add_argument("--lang_blue", default="Pick up the blue cube and place it in the white box")
    parser.add_argument("--lang_red", default="Pick up the red cube and place it in the white box")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit (no Isaac Sim)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Replay only the first N planned episodes (0 = all); for quick checks")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--unit", help="Calibration unit; required for legacy per-unit datasets without saved mapping")
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
    parser.add_argument("--variant", default="teleop", help="Task variant without terminations")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disable_fabric", action="store_true")
    return parser


def parse_args(argv=None):
    from utils.cli import parse_yaml_args

    parser = build_parser()
    add_launch_arguments(parser)
    return parse_yaml_args(parser, argv)


def parse_episode_list(text) -> set[int]:
    if not text.strip():
        return set()
    parts = [part.strip() for part in text.split(",")]
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid episode list: {text!r}")
    return {int(part) for part in parts}


def parse_dr_terms(text) -> list[str]:
    from sim.datagen.dr_variants import DR_TERMS, parse_csv

    if text.strip().lower() in ("", "none"):
        return []
    camera = {t.strip() for t in text.split(",")} & (set(DR_TERMS) - set(REPLAY_DR_TERMS))
    if camera:
        raise ValueError(f"{sorted(camera)}: camera views are fixed to the calibrated real setup "
                         "and are not randomized")
    return parse_csv(text, REPLAY_DR_TERMS)


def check_source_fps(src_root, fps) -> None:
    info = json.loads((Path(src_root) / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info["fps"]) != fps:
        raise ValueError(f"Source dataset is {info['fps']} FPS; --fps is {fps}")


def read_dr_ranges(task):
    """DR ranges exactly as the task's eval_dr variant uses them (needs a launched app for ROBOT_COLORS)."""
    from sim.workspace.envs.mdp.appearance import ROBOT_COLORS
    from sim.datagen.dr_variants import DRRanges

    dr = task.spec.raw.get("dr", {})
    return DRRanges(
        exposure=tuple(dr.get("room_light_exposure_range", (-3.0, 1.0))),
        robot_colors=dict(ROBOT_COLORS),
        # Cameras are fixed (not randomized in the DR eval either), so no camera ranges.
        camera_pos={},
        camera_rot={},
        focal_length=None,
    )


def make_env(args):
    """One-env teleop scene with automatic object placement disabled (the replay places it every frame)."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import sim.workspace.envs

    env_id = sim.workspace.envs.TASK.gym_id(args.variant)
    env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    if env_cfg.terminations is not None and any(
        value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
    ):
        raise ValueError(f"{env_id} has terminations; replay needs a variant without them (e.g. teleop).")
    env_cfg.events.reset_from_starts = None
    env = gym.make(env_id, cfg=env_cfg)
    env.reset()
    return env


def discover_sim_cameras(env) -> dict:
    """Recorded scene keys camera_* -> {height, width}, as in teleop.agent."""
    scene = env.unwrapped.scene
    return {
        key.removeprefix("camera_"): {"height": getattr(scene.cfg, key).height, "width": getattr(scene.cfg, key).width}
        for key in scene.keys() if key.startswith("camera_")
    }


def make_mapping(env, *, arm=None, mapping=None, source_root=None, unit=None):
    """Use recorded calibration or an explicit unit; no scene calibration or hardware reads."""
    from utils.arms.mapping import JointMapping
    from utils.arms.arms import load_arm_spec
    from utils.datasets.sources import replay_joint_mapping

    arm = arm or load_arm_spec()[0]
    if mapping is None:
        mapping, _ = replay_joint_mapping(source_root, arm, unit)
    return JointMapping.from_arm(arm, mapping, device=env.unwrapped.device)


def make_writer(args, sim_cameras, device, *, arm=None):
    from sim.recording.recorder import LeRobotRecorder

    writer = LeRobotRecorder(
        task_name=args.lang_blue,  # default only; every frame's task is set per episode
        repo_id=args.dst_repo_id, dataset_root=args.dst_root, fps=args.fps, device=device, cameras=sim_cameras,
        arm=arm,
    )
    writer.init_dataset()
    return writer


def replay_episode(replayer, writer, src, planned, fps) -> None:
    """Render one planned episode from its source episode and save it with its object sidecar."""
    from sim.datagen.dr_variants import cube_colors
    from utils.datasets.task_meta import save_task_trajectory

    replayer.apply(cube_colors()[planned.color], planned.color, planned.draw)
    for i in range(src.length):
        replayer.pose_frame(src.states[i], {name: poses[i] for name, poses in src.objects_base.items()})
        images = replayer.render_images()
        frame = writer.make_sim_frame(src.actions[i], src.states[i], images)
        frame["task"] = planned.task_text
        writer.add_frame(frame)
    index = writer.save_episode()
    if index != planned.dst_index:
        raise RuntimeError(f"Saved episode {index}, planned {planned.dst_index}")
    save_task_trajectory(writer.dataset_root, index, src.objects_base, src.object_sizes, fps,
                         task=replayer.task.name, arm=replayer.arm.name, reference_frame=src.reference_frame)


def print_plan(plan) -> None:
    print(f"{'dst':>4} {'src':>4} {'color':<5} {'dr_seed':>20}  task")
    for e in plan:
        print(f"{e.dst_index:>4} {e.src_index:>4} {e.color:<5} {e.dr_seed:>20}  {e.task_text}")
    generated = [e for e in plan if e.draw is not None]
    print(f"{len(plan)} episodes: {len(plan) - len(generated)} originals, "
          f"{sum(e.color == 'blue' for e in generated)} blue, {sum(e.color == 'red' for e in generated)} red")


def _json_args(args) -> dict:
    return {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}


def main(argv=None):
    args = parse_args(argv)
    if not args.task:
        raise SystemExit("dr_replay needs a task.")
    apply_selection(args)

    import torch

    from sim.datagen.dr_variants import cube_colors, build_manifest, parse_csv, plan_episodes, write_json_atomic
    from utils.datasets.sources import load_source, replay_joint_mapping
    from utils.arms.arms import load_arm_spec
    from core.tasks.loader import load_task

    arm, _ = load_arm_spec(args.arm)
    task = load_task(args.task)

    colors = parse_csv(args.colors, cube_colors())
    terms = parse_dr_terms(args.dr)
    exclude = parse_episode_list(args.exclude_episodes)
    check_source_fps(args.src_root, args.fps)
    if Path(args.dst_root).exists():
        raise FileExistsError(f"Destination already exists: {args.dst_root}")

    if args.dry_run:
        source = load_source(args.src_root, args.src_repo_id, exclude, arm=arm, task=task)
        # Draws depend only on each episode's seed + the config ranges, so they are reproduced
        # exactly in the real run; the ranges themselves need Isaac Sim, so they are not shown here.
        print_plan(plan_episodes([s.index for s in source], colors, [], None, args.seed,
                                 args.include_originals, args.lang_blue, args.lang_red))
        print(f"DR terms for the real run: {terms or 'none'}")
        return

    joint_mapping, calibration = replay_joint_mapping(args.src_root, arm, args.unit)
    print(f"[INFO]: Replay calibration: {calibration}")

    app = launch(args, enable_cameras=True)

    env = writer = None
    try:
        import sim.workspace.envs
        from sim.recording.replay_scene import SceneReplayer

        source = load_source(args.src_root, args.src_repo_id, exclude, arm=arm, task=task)
        ranges = read_dr_ranges(sim.workspace.envs.TASK)
        plan = plan_episodes([s.index for s in source], colors, terms, ranges, args.seed,
                             args.include_originals, args.lang_blue, args.lang_red)
        if args.limit > 0:
            plan = plan[:args.limit]
        print_plan(plan)

        env = make_env(args)
        sim_cameras = discover_sim_cameras(env)
        mapping = make_mapping(env, arm=arm, mapping=joint_mapping)
        replayer = SceneReplayer(env.unwrapped, mapping, ranges, sim_cameras, task=args.task, arm=arm)
        writer = make_writer(args, sim_cameras, env.unwrapped.device, arm=arm)

        by_index = {s.index: s for s in source}
        with torch.inference_mode():
            for n, planned in enumerate(plan, 1):
                print(f"[{n}/{len(plan)}] src {planned.src_index} -> dst {planned.dst_index} ({planned.color})")
                replay_episode(replayer, writer, by_index[planned.src_index], planned, args.fps)

        manifest = build_manifest(plan, _json_args(args), git_commit(REPO_ROOT))
        manifest.update(arm=arm.name, joint_names=list(arm.joint_names), joint_mapping=joint_mapping,
                        joint_units=[joint.unit for joint in arm.action_joints],
                        joint_roles=[joint.role for joint in arm.action_joints], use_degrees=arm.lerobot.use_degrees,
                        calibration=calibration)
        write_json_atomic(Path(args.dst_root) / "meta" / "dr_replay_manifest.json", manifest)
        print(f"[INFO]: Wrote {len(plan)} episodes to {args.dst_root}")
    finally:
        try:
            if writer is not None:
                writer.finalize()
            if env is not None:
                env.close()
        finally:
            close_app(app)


if __name__ == "__main__":
    main()
