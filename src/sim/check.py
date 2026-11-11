"""Headless composition check used by s2r check after static plugin validation."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from sim.app import launch


def check(args):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import sim.envs as registry
    from sim.envs.mdp._common import as_torch

    task = registry.TASK
    env_id = task.gym_id(args.variant or task.spec.default_variant) if task else registry.scene_gym_id(args.scene)
    cfg = parse_env_cfg(env_id, device=args.device, num_envs=1)
    env = gym.make(env_id, cfg=cfg)
    try:
        raw = env.unwrapped
        arm = cfg.arm_spec
        robot = raw.scene["robot"]
        ids, names = robot.find_joints(list(arm.usd_joint_names), preserve_order=True)
        if names != list(arm.usd_joint_names):
            raise ValueError(f"Articulation joints differ from manifest: {names}")
        bodies = {arm.links.base, arm.links.ee}
        bodies.update(camera.parent_link for camera in arm.cameras.values())
        if arm.gripper:
            bodies.update(arm.gripper.contact_bodies)
            bodies.update(arm.gripper.friction_bodies)
        for name in sorted(bodies):
            found, resolved = robot.find_bodies(name)
            if len(found) != 1 or resolved != [name]:
                raise ValueError(f"Arm body {name!r} does not resolve uniquely: {resolved}")
        obs, _ = env.reset()
        action_ids = robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)[0]
        target = as_torch(robot.data.joint_pos)[:, action_ids].clone()
        for _ in range(args.steps):
            obs, *_ = env.step(target)
        if not torch.isfinite(as_torch(robot.data.joint_pos)).all():
            raise ValueError("Nonfinite joint state after stepping")
        cameras = dict(cfg.scene_spec.cameras, **arm.cameras)
        rendered = {}
        for name, camera in cameras.items():
            pixels = obs["visual"][f"rgb_{name}"]
            if tuple(pixels.shape) != (1, camera.height, camera.width, 3):
                raise ValueError(f"Camera {name!r} returned shape {tuple(pixels.shape)}")
            if not torch.isfinite(pixels).all() or pixels.float().std().item() == 0:
                raise ValueError(f"Camera {name!r} returned invalid or constant pixels")
            rendered[name] = list(pixels.shape)
        for name in task.spec.objects if task else ():
            asset = raw.scene[name]
            if not torch.isfinite(as_torch(asset.data.root_pos_w)).all():
                raise ValueError(f"Task object {name!r} has a nonfinite position")
        if args.scene_json is not None:
            if task is None:
                raise ValueError("--scene-json needs a task with declared objects")
            from sim.eval.scene import capture_scene

            # Capture the instantiated geometry before closing its stage, and
            # publish only after the composition has passed every check above.
            encoded = json.dumps(capture_scene(raw, task), indent=2, allow_nan=False)
            args.scene_json.parent.mkdir(parents=True, exist_ok=True)
            args.scene_json.write_text(encoded + "\n")
        return dict(passed=True, arm=arm.name, scene=args.scene, task=task.name if task else None,
                    variant=args.variant, joints=names, bodies=sorted(bodies), cameras=rendered,
                    objects=list(task.spec.objects) if task else [], steps=args.steps)
    finally:
        env.close()


def main(argv=None):
    from core.paths import DEFAULT_ARM, DEFAULT_TASK

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM", DEFAULT_ARM))
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", "blank_table"))
    parser.add_argument("--task", default=os.environ.get("S2R_TASK", DEFAULT_TASK))
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scene-json", type=Path, help="save live named-object geometry for s2r eval-set")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("steps must be positive")
    app = launch(selection=args, headless=True, enable_cameras=True, device=args.device)
    exit_code = 0
    try:
        result = check(args)
        encoded = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")
        print(encoded, flush=True)
    except Exception:
        exit_code = 1
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
