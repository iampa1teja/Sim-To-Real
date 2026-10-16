"""Show an evaluation's first reset pose in Isaac Sim without loading a policy.

Run inside teleop with Isaac's Python wrapper. The YAML selects the robot USD
and benchmark starts, just as docker/eval.sh does. Close Isaac to end preview.
"""
import argparse
import json
from pathlib import Path
import random
import runpy
import time
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=Path("pick_place_molmo_base.yaml"))
parser.add_argument("--seed", type=int, default=1984)
parser.add_argument("--out_dir", type=Path, default=ROOT / "outputs/preview_molmo")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
config_path = args.config
if not config_path.is_file() and config_path.parent == Path("."):
    config_path = ROOT / "configs/eval" / config_path
config_tools = runpy.run_path(str(ROOT / "docker/eval_config.py"))
settings = config_tools["build"](config_tools["load"](config_path))
launcher = AppLauncher(args)


def main():
    import gymnasium as gym
    import numpy as np
    from PIL import Image
    import torch
    import isaaclab_tasks  # noqa: F401
    import sim_to_real_so101.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = parse_env_cfg(settings["task"], device=args.device, num_envs=1)
    cfg.seed = args.seed
    if settings["robot_usd"]:
        asset = Path(settings["robot_usd"])
        if not asset.is_file():
            raise ValueError(f"Robot USD does not exist: {asset}")
        cfg.scene.robot.spawn.usd_path = str(asset)
    params = cfg.events.spawn_cube.params
    params.update(starts_dir=settings["dataset"] + "/pick_place_meta",
                  mode=settings["start_mode"], random_fraction=settings["random_fraction"],
                  reset_robot=settings["robot_start"] == "recorded", seed=args.seed)
    if settings["eval_set"]:
        params.update(eval_set=settings["eval_set"], splits=settings["splits"] or None,
                      repeats=settings["repeats"])
    # Preview one reset only, with no automatic episode changes.
    cfg.terminations = None
    env = gym.make(settings["task"], cfg=cfg)
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            if settings["robot_start"] == "recorded":
                hold = obs["policy"]["joint_pos_obs"].clone()
            else:
                # Match the evaluator's ten settling steps for default starts.
                hold = torch.tensor([[-.2736, -.6109, -.0745, 1.5148, -1.6034, -.1465]],
                                    device=env.unwrapped.device)
            for _ in range(10):
                env.step(hold)
            for _ in range(20):
                env.unwrapped.sim.render()
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for name in ("camera_realsense_rgb", "camera_wrist_cam"):
                camera = env.unwrapped.scene[name]
                camera.update(env.unwrapped.step_dt, force_recompute=True)
                pixels = camera.data.output["rgb"][0].detach().cpu().numpy()
                Image.fromarray(pixels[..., :3]).save(args.out_dir / f"{name}.png")
            robot = env.unwrapped.scene["robot"]
            snapshot = {
                "config": str(config_path),
                "robot_usd": cfg.scene.robot.spawn.usd_path,
                "robot_start": settings["robot_start"],
                "target_joint_positions_rad": hold[0].cpu().tolist(),
                "actual_joint_positions_rad": robot.data.joint_pos[0].cpu().tolist(),
                "benchmark_start": getattr(env.unwrapped, "_benchmark_start", None),
            }
            (args.out_dir / "pose.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            env.unwrapped.sim.pause()
            print(f"PREVIEW_READY: {cfg.scene.robot.spawn.usd_path}", flush=True)
            print(f"Initial pose is paused. Images and pose: {args.out_dir}", flush=True)
            while launcher.app.is_running():
                env.unwrapped.sim.render()
                time.sleep(1 / 30)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
