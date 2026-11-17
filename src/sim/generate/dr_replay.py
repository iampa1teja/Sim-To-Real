# sim/generate/dr_replay.py  (ported from v1 scripts/dr_replay.py)
"""Re-render recorded pick-place episodes with blue/red cubes and domain randomization.

Kinematic replay: for every frame the recorded joint state and cube pose are written into the
task scene and the cameras are rendered (no physics step). Actions and observation.state are
copied unchanged; only the camera images and the task text change. Writes a NEW LeRobot dataset
plus pick_place_meta/ cube sidecars and meta/dr_replay_manifest.json.

  # plan only (no Isaac Sim; needs LeRobot)
  dr_replay --src_root <src> --src_repo_id local/pick_place_v1 \
      --dst_root <dst> --dst_repo_id local/pick_place_v2_dr --exclude_episodes 0,6 --dry_run
  # fidelity check: blue, no DR, 2 episodes -> frames must match the originals
  dr_replay ... --colors blue --dr "" --limit 2 --headless
"""
# Changes from v1:
# - --arm/--scene/--task/--variant select the env (S2R_* exported before core/sim imports); v1's --task gym id is gone.
# - DR ranges come from task.toml [task.dr] (the same values apply_eval_dr uses) instead of PickPlaceEvalDREnvCfg.
# - Joint units go through JointMapping; the recorder is sim.recording.lerobot_recorder.LeRobotRecorder.

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_DR_TERMS = ("light", "robot_color")  # camera views stay fixed to the calibrated real setup


def rendering_kit_args(existing: str = "") -> str:
    """Enable the installed material context-menu dependency before Kit starts.

    The material library initializes its menus asynchronously during startup.
    Enabling this extension after app creation can miss that initialization.
    """
    extension = "omni.kit.context_menu"
    words = existing.split()
    for index, word in enumerate(words):
        if word == f"--enable={extension}" or (
            word == "--enable" and words[index + 1:index + 2] == [extension]
        ):
            return existing
    return f"{existing} --enable {extension}".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay recorded episodes with visual domain randomization.")
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
    parser.add_argument("--copies", type=int, default=2, help="Number of appearance copies per v2 episode")
    parser.add_argument("--no_recolor", action="store_true", help="Keep source object colors in v2 copies")
    parser.add_argument("--camera_jitter", metavar="POS_M,ROT_DEG", help="Jitter camera poses in v2 copies")
    parser.add_argument("--lang_blue", default="Pick up the blue cube and place it in the white box")
    parser.add_argument("--lang_red", default="Pick up the red cube and place it in the white box")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Print the plan and exit (no Isaac Sim)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Replay only the first N planned episodes (0 = all); for quick checks")
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", "so101"))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "rpd_meeting_room"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", "pick_place_cube"))
    parser.add_argument("--variant", default="teleop", help="Task variant without terminations")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--disable_fabric", action="store_true")
    return parser


def apply_selection(args) -> None:
    """Export S2R_ARM / S2R_SCENE / S2R_TASK before anything imports core or sim."""
    from sim.teleop.config import apply_selection as apply

    if not args.task:
        raise SystemExit("dr_replay needs a task.")
    apply(args)


def parse_episode_list(text) -> set[int]:
    if not text.strip():
        return set()
    parts = [part.strip() for part in text.split(",")]
    if any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid episode list: {text!r}")
    return {int(part) for part in parts}


def parse_dr_terms(text) -> list[str]:
    from sim.generate.dr_variants import DR_TERMS, parse_csv

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


def read_dr_ranges(task):
    """DR ranges exactly as the task's eval_dr variant uses them (needs a launched app for ROBOT_COLORS)."""
    from sim.envs.mdp.resets import ROBOT_COLORS
    from sim.generate.dr_variants import DRRanges

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

    import sim.envs
    from sim.envs.mdp.resets import park_object, reset_object_pose

    env_id = sim.envs.TASK.gym_id(args.variant)
    env_cfg = parse_env_cfg(env_id, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    if env_cfg.terminations is not None and any(
        value is not None for key, value in vars(env_cfg.terminations).items() if not key.startswith("_")
    ):
        raise ValueError(f"{env_id} has terminations; replay needs a variant without them (e.g. teleop).")
    if env_cfg.events is not None:
        for name, term in vars(env_cfg.events).items():
            fn = getattr(term, "func", None)
            fn_name = getattr(fn, "__name__", str(fn).rsplit(":", 1)[-1])
            if (fn in (park_object, reset_object_pose) or name.startswith("spawn")
                    or fn_name.startswith("spawn") or fn_name == "reset_cube_from_recorded_starts"):
                setattr(env_cfg.events, name, None)
    env = gym.make(env_id, cfg=env_cfg)
    env.reset()
    return env


def discover_sim_cameras(env) -> dict:
    """Same camera discovery as lerobot_agent: scene keys camera_* -> {height, width}."""
    scene = env.unwrapped.scene
    return {
        key.removeprefix("camera_"): {"height": getattr(scene.cfg, key).height, "width": getattr(scene.cfg, key).width}
        for key in scene.keys() if key.startswith("camera_")
    }


def make_mapping(env):
    """Raw dataset units <-> sim radians with the env's joint mapping; no robot hardware is opened."""
    from core.arms.mapping import JointMapping

    mapping = getattr(env.unwrapped.cfg, "sim_joint_mapping", None)
    return JointMapping.from_setup(SimpleNamespace(mapping=mapping), device=env.unwrapped.device)


def make_writer(args, sim_cameras, device):
    from sim.recording.lerobot_recorder import LeRobotRecorder

    writer = LeRobotRecorder(
        task_name=args.lang_blue,  # default only; every frame's task is set per episode
        repo_id=args.dst_repo_id, dataset_root=args.dst_root, fps=args.fps, device=device, cameras=sim_cameras,
    )
    writer.init_dataset()
    return writer


def replay_episode(replayer, writer, src, planned, fps) -> None:
    """Render one planned episode from its source episode and save it with its cube sidecar."""
    from sim.generate.dr_variants import COLORS
    from sim.recording.episode_metadata import save_cube_trajectory

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


def _json_args(args, *, legacy=False) -> dict:
    return {k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, type(None)))
            and (not legacy or k not in {"copies", "no_recolor", "camera_jitter"})}


def _main_v2(parser, args, argv):
    from core.datasets.task_meta import list_sidecars, read_sidecar
    from core.generate.config import DRConfig
    from core.tasks.base import load_task_spec
    from sim.generate.dr_variants import exposure_range, plan_episodes_v2, print_plan_v2, robot_color_names

    exclude = parse_episode_list(args.exclude_episodes)
    indices = list_sidecars(args.src_root)
    unknown = exclude - set(indices)
    if unknown:
        raise ValueError(f"Unknown excluded episode(s): {sorted(unknown)}")
    source_meta = [read_sidecar(args.src_root, index) for index in indices]
    tasks = {sidecar["task"] for sidecar in source_meta}
    if len(tasks) != 1 or None in tasks:
        raise ValueError("V2 replay requires a source dataset with one task")
    task_spec = load_task_spec(tasks.pop())
    camera_jitter = None
    if args.camera_jitter is not None:
        try:
            position, rotation = map(float, args.camera_jitter.split(","))
        except ValueError as exc:
            raise ValueError("--camera_jitter expects POS_M,ROT_DEG") from exc
        camera_jitter = {"position_m": position, "rotation_deg": rotation}
    config = DRConfig(copies=args.copies, terms=tuple(parse_dr_terms(args.dr)),
                      recolor=not args.no_recolor, seed=args.seed, camera_jitter=camera_jitter)
    if Path(args.dst_root).exists():
        raise FileExistsError(f"Destination already exists: {args.dst_root}")
    check_source_fps(args.src_root, args.fps)
    kept = [sidecar for sidecar in source_meta if sidecar["episode_index"] not in exclude]

    def make_plan(cameras):
        plan = plan_episodes_v2(kept, task_spec=task_spec, cfg=config, cameras=cameras,
                                exposure_range=exposure_range(task_spec), robot_colors=robot_color_names(),
                                include_originals=args.include_originals)
        if args.limit > 0:
            plan = plan[:args.limit]
        print_plan_v2(plan)
        return plan

    if args.dry_run:
        info = json.loads((Path(args.src_root) / "meta" / "info.json").read_text())
        prefix = "observation.images."
        make_plan([key.removeprefix(prefix) for key in info.get("features", {}) if key.startswith(prefix)])
        return 0

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    arms = {sidecar["arm"] for sidecar in source_meta}
    scenes = {sidecar["scene"] for sidecar in source_meta}
    if len(arms) != 1 or None in arms or len(scenes) != 1 or None in scenes:
        raise ValueError("V2 replay requires one source arm and scene")
    args.task, args.arm, args.scene = task_spec.name, arms.pop(), scenes.pop()
    if any(obj.params.get("size_variants") for obj in task_spec.objects.values()):
        args.variant = "generate"
    apply_selection(args)
    args.enable_cameras = True
    args.kit_args = rendering_kit_args(args.kit_args)
    app = AppLauncher(args).app

    env = writer = None
    try:
        import torch
        from core.datasets.frames import read_frames
        from core.datasets.task_meta import LABEL_FEATURES, write_sidecar
        from core.generate.labels import label_columns
        from core.generate.protocols import FrameLabels
        from sim.envs.mdp.objects import generation_assets, park_all, select_generation_assets
        from sim.envs.mdp.resets import ROBOT_COLORS
        from sim.generate.dr_variants import DRRanges, build_manifest_v2, sidecar_v2, write_json_atomic
        from sim.recording.lerobot_recorder import LeRobotRecorder
        from sim.recording.replay_scene import SceneReplayer
        from sim.recording.replay_source import load_source

        sources = load_source(args.src_root, args.src_repo_id, exclude,
                              task=args.task, arm=args.arm, scene=args.scene)
        by_index = {src.index: src for src in sources}
        frames = read_frames(args.src_root)
        env = make_env(args)
        cameras = discover_sim_cameras(env)
        plan = make_plan(cameras)
        physical_assets = tuple(asset for variants in generation_assets(task_spec).values() for asset in variants)
        ranges = DRRanges(exposure_range(task_spec), ROBOT_COLORS, {}, {}, None)
        replayer = SceneReplayer(env.unwrapped, make_mapping(env), ranges, cameras,
                                 task=args.task, palette=task_spec.palette)
        writer = LeRobotRecorder(task_name=task_spec.instruction, repo_id=args.dst_repo_id,
                                 dataset_root=args.dst_root, fps=args.fps, device=env.unwrapped.device,
                                 cameras=cameras, extra_features=LABEL_FEATURES)
        writer.init_dataset()
        with torch.inference_mode():
            for planned in plan:
                src = by_index[planned.src_index]
                payload = sidecar_v2(planned, src.meta, source_repo_id=args.src_repo_id)
                data = frames[src.index]
                sizes = {name: obj["size_m"] for name, obj in payload["objects"].items()}
                assets = select_generation_assets(task_spec, sizes)
                env.unwrapped._generation_object_assets = assets
                env.reset()
                park_all(env.unwrapped, physical_assets)
                replayer.apply_v2(payload, assets)
                with replayer.camera_jitter(payload["appearance"].get("camera_offsets", {})):
                    for t in range(src.length):
                        replayer.pose_frame(src.states[t], {assets[name]: poses[t] for name, poses in src.objects.items()})
                        labels = FrameLabels(int(data["label.stage"][t]), bool(data["label.imitate"][t]),
                                             data["label.ee_pose"][t])
                        frame = writer.make_sim_frame(src.actions[t], src.states[t], replayer.render_images(),
                                                      extra=label_columns(labels))
                        frame["task"] = payload["instruction"]
                        writer.add_frame(frame)
                index = writer.save_episode()
                if index != planned.dst_index:
                    raise RuntimeError(f"Saved episode {index}, planned {planned.dst_index}")
                write_sidecar(writer.dataset_root, payload)
        write_json_atomic(Path(args.dst_root) / "meta" / "dr_replay_manifest.json",
                          build_manifest_v2(plan, _json_args(args), git_commit()))
        print(f"[INFO]: Wrote {len(plan)} episodes to {args.dst_root}")
        return 0
    finally:
        try:
            if writer is not None:
                writer.finalize()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


def main(argv=None):
    parser = build_parser()
    args, _ = parser.parse_known_args(argv)
    if (Path(args.src_root) / "task_meta").is_dir():
        return _main_v2(parser, args, argv)
    if args.camera_jitter is not None:
        raise ValueError("Camera jitter requires a v2 dataset")
    apply_selection(args)

    import torch

    from sim.generate.dr_variants import COLORS, build_manifest, parse_csv, plan_episodes, write_json_atomic
    from sim.recording.replay_source import load_source

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
    args = parser.parse_args(argv)
    args.enable_cameras = True
    args.kit_args = rendering_kit_args(args.kit_args)
    app = AppLauncher(args).app

    env = writer = None
    try:
        import sim.envs
        from sim.recording.replay_scene import SceneReplayer

        source = load_source(args.src_root, args.src_repo_id, exclude)
        ranges = read_dr_ranges(sim.envs.TASK)
        plan = plan_episodes([s.index for s in source], colors, terms, ranges, args.seed,
                             args.include_originals, args.lang_blue, args.lang_red)
        if args.limit > 0:
            plan = plan[:args.limit]
        print_plan(plan)

        env = make_env(args)
        sim_cameras = discover_sim_cameras(env)
        replayer = SceneReplayer(env.unwrapped, make_mapping(env), ranges, sim_cameras, task=args.task)
        writer = make_writer(args, sim_cameras, env.unwrapped.device)

        by_index = {s.index: s for s in source}
        with torch.inference_mode():
            for n, planned in enumerate(plan, 1):
                print(f"[{n}/{len(plan)}] src {planned.src_index} -> dst {planned.dst_index} ({planned.color})")
                replay_episode(replayer, writer, by_index[planned.src_index], planned, args.fps)

        write_json_atomic(Path(args.dst_root) / "meta" / "dr_replay_manifest.json",
                          build_manifest(plan, _json_args(args, legacy=True), git_commit()))
        print(f"[INFO]: Wrote {len(plan)} episodes to {args.dst_root}")
    finally:
        if writer is not None:
            writer.finalize()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
