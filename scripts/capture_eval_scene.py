"""Capture live geometry after one reset; no policy, rollout or geometry edits.

Run with Isaac's Python wrapper inside teleop, --headless --out <snapshot.json>.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
try:
    import gymnasium as gym
    import isaaclab_tasks
    import sim_to_real_so101.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from sim_to_real_so101.utils.pick_place_benchmark_runtime import capture_scene
    cfg = parse_env_cfg('Lerobot-So101-Teleop-Pick-Place', device=args.device, num_envs=1)
    env = gym.make('Lerobot-So101-Teleop-Pick-Place', cfg=cfg)
    try:
        env.reset()
        data = capture_scene(env.unwrapped)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        print(f'Live scene snapshot: {args.out}', flush=True)
    finally:
        env.close()
finally:
    launcher.app.close()
