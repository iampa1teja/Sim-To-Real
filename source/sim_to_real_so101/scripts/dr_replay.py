"""Re-render recorded pick-place episodes with blue/red cubes and domain randomization.

Kinematic replay: for every frame the recorded joint state and cube pose are written into the
Pick-Place scene and the cameras are rendered (no physics step). Actions and observation.state are
copied unchanged; only the camera images and the task text change. Writes a NEW LeRobot dataset
plus pick_place_meta/ cube sidecars and meta/dr_replay_manifest.json.

  # plan only (no Isaac Sim; needs LeRobot)
  lerobot_dr_replay --src_root <src> --src_repo_id local/pick_place_v1 \
      --dst_root <dst> --dst_repo_id local/pick_place_v2_dr --exclude_episodes 0,6 --dry_run
  # fidelity check: blue, no DR, 2 episodes -> frames must match the originals
  lerobot_dr_replay ... --colors blue --dr "" --limit 2 --headless
"""

import argparse
import json
from pathlib import Path
import subprocess

import torch

from sim_to_real_so101.utils.dr_variants import (
    COLORS,
    DR_TERMS,
    DRRanges,
    build_manifest,
    parse_csv,
    plan_episodes,
    write_json_atomic,
)
from sim_to_real_so101.utils.replay_source import load_source

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay SO-101 episodes with visual domain randomization.")
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
    parser.add_argument("--task", default="Lerobot-So101-Teleop-Pick-Place")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disable_fabric", action="store_true")
    return parser


def parse_episode_list(text) -> set[int]:
    if not text.strip():
        return set()
    parts = [part.strip() for part in text.split(",")]
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid episode list: {text!r}")
    return {int(part) for part in parts}


REPLAY_DR_TERMS = ("light", "robot_color")  # camera views stay fixed to the calibrated real setup


def parse_dr_terms(text) -> list[str]:
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_dr_ranges(cfg=None) -> DRRanges:
    """DR ranges exactly as the pick-place DR evaluation uses them (needs a launched app).

    The eval_* events are created in __post_init__, so read them from an INSTANCE of the config.
    """
    from sim_to_real_so101.mdp import ROBOT_COLORS
    if cfg is None:
        from sim_to_real_so101.tasks.pick_place_env_cfg import PickPlaceEvalDREnvCfg
        cfg = PickPlaceEvalDREnvCfg()
    events = cfg.events
    return DRRanges(
        exposure=tuple(events.eval_room_light.params["exposure_range"]),
        robot_colors={name: ROBOT_COLORS[name] for name in events.eval_robot_color.params["color_names"]},
        # Cameras are fixed (not randomized in the DR eval either), so no camera ranges.
        camera_pos={},
        camera_rot={},
        focal_length=None,
    )


def make_env(args):
    """One-env Pick-Place teleop scene with automatic cube spawning disabled."""
    from importlib import import_module
    import gymnasium as gym

    import_module("isaaclab_tasks")
    import_module("sim_to_real_so101.tasks")
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    if env_cfg.terminations is not None and any(
        value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
    ):
        raise ValueError(f"{args.task} has terminations; replay needs a teleop task without them.")
    env_cfg.events.spawn_cube = None  # the replay places the cube every frame
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()
    return env


def discover_sim_cameras(env) -> dict:
    """Same camera discovery as pick_place_agent: scene keys camera_* -> {height, width}."""
    scene = env.unwrapped.scene
    return {
        key.removeprefix("camera_"): {"height": getattr(scene.cfg, key).height, "width": getattr(scene.cfg, key).width}
        for key in scene.keys() if key.startswith("camera_")
    }


def make_interface(env, sim_cameras, fps):
    """Joint/image conversion only; no robot hardware is opened."""
    from sim_to_real_so101.utils.lerobot_interface import LeRobotSO101Interface
    return LeRobotSO101Interface(
        device=env.unwrapped.device, port="", id="dr_replay", cameras=sim_cameras, fps=fps,
        kind="follower", joint_mapping=getattr(env.unwrapped.cfg, "sim_joint_mapping", None),
    )


def make_writer(args, sim_cameras, device):
    from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder
    writer = LeRobotRecorder(
        task_name=args.lang_blue,  # default only; every frame's task is set per episode
        repo_id=args.dst_repo_id, dataset_root=args.dst_root, fps=args.fps, device=device,
        cameras=sim_cameras, robot_type="so101_follower",
    )
    writer.init_dataset()
    return writer


def replay_episode(replayer, writer, src, planned, fps) -> None:
    """Render one planned episode from its source episode and save it with its cube sidecar."""
    from sim_to_real_so101.utils.episode_metadata import save_cube_trajectory

    replayer.apply(COLORS[planned.color], planned.color, planned.draw)
    for i in range(src.length):
        replayer.pose_frame(src.states[i], src.poses_base[i])
        images = replayer.render_images()
        frame = writer.make_sim_frame(src.actions[i], src.states[i], images)
        frame["task"] = planned.task_text
        writer.add_frame(frame)
    index = writer.save_episode()
    if index != planned.dst_index:
        raise RuntimeError(f"Saved episode {index}, planned {planned.dst_index}")
    save_cube_trajectory(writer.dataset_root, index, src.poses_base.tolist(), src.cube_size, fps)


def print_plan(plan) -> None:
    print(f"{'dst':>4} {'src':>4} {'color':<5} {'dr_seed':>20}  task")
    for e in plan:
        print(f"{e.dst_index:>4} {e.src_index:>4} {e.color:<5} {e.dr_seed:>20}  {e.task_text}")
    generated = [e for e in plan if e.draw is not None]
    print(f"{len(plan)} episodes: {len(plan) - len(generated)} originals, "
          f"{sum(e.color == 'blue' for e in generated)} blue, {sum(e.color == 'red' for e in generated)} red")


def _json_args(args) -> dict:
    return {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}


@torch.inference_mode()
def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()
    colors = parse_csv(args.colors, COLORS)
    terms = parse_dr_terms(args.dr)
    exclude = parse_episode_list(args.exclude_episodes)
    check_source_fps(args.src_root, args.fps)
    if Path(args.dst_root).exists():
        raise FileExistsError(f"Destination already exists: {args.dst_root}")

    if args.dry_run:
        source = load_source(args.src_root, args.src_repo_id, exclude)
        # Draws depend only on each episode's seed + the config ranges, so they are reproduced
        # exactly in the real run; the ranges themselves need Isaac Sim, so they are not shown here.
        print_plan(plan_episodes([s.index for s in source], colors, [], None, args.seed,
                                 args.include_originals, args.lang_blue, args.lang_red))
        print(f"DR terms for the real run: {terms or 'none'}")
        return

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app

    env = writer = None
    try:
        source = load_source(args.src_root, args.src_repo_id, exclude)
        ranges = read_dr_ranges()
        plan = plan_episodes([s.index for s in source], colors, terms, ranges, args.seed,
                             args.include_originals, args.lang_blue, args.lang_red)
        if args.limit > 0:
            plan = plan[:args.limit]
        print_plan(plan)

        env = make_env(args)
        sim_cameras = discover_sim_cameras(env)
        interface = make_interface(env, sim_cameras, args.fps)
        from sim_to_real_so101.utils.replay_scene import SceneReplayer
        replayer = SceneReplayer(env.unwrapped, interface, ranges)
        writer = make_writer(args, sim_cameras, env.unwrapped.device)

        by_index = {s.index: s for s in source}
        for n, planned in enumerate(plan, 1):
            print(f"[{n}/{len(plan)}] src {planned.src_index} -> dst {planned.dst_index} ({planned.color})")
            replay_episode(replayer, writer, by_index[planned.src_index], planned, args.fps)

        write_json_atomic(Path(args.dst_root) / "meta" / "dr_replay_manifest.json",
                          build_manifest(plan, _json_args(args), git_commit()))
        print(f"[INFO]: Wrote {len(plan)} episodes to {args.dst_root}")
    finally:
        if writer is not None:
            writer.finalize()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
