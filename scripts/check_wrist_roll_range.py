"""Drive simulated MyRoom wrist roll past the USD bounds; never open hardware."""
import sys
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
    env = None
    exit_code = 1
    try:
        import gymnasium as gym
        import torch
        from importlib import import_module
        from isaaclab_tasks.utils import parse_env_cfg

        import_module("isaaclab_tasks")
        import_module("sim_to_real_so101.tasks")
        task = "Lerobot-So101-Teleop-MyRoom"
        cfg = parse_env_cfg(task, device="cuda:0", num_envs=1, use_fabric=True)
        env = gym.make(task, cfg=cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        roll = robot.joint_names.index("Wrist_Roll")
        limits = torch.rad2deg(robot.root_physx_view.get_dof_limits()[0, roll])
        print(f"PhysX wrist_roll limits: {limits.tolist()} degrees", flush=True)
        expected = torch.tensor([-180., 179.912109375], device=limits.device)
        torch.testing.assert_close(limits, expected, atol=0.001, rtol=0)
        dt = env.sim.get_physics_dt()
        for target in (-179., 179.):
            # Start each drive from zero; only the position target sets the endpoint.
            q = torch.zeros_like(robot.data.joint_pos)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            q[:, roll] = target * torch.pi / 180
            robot.set_joint_position_target(q)
            for _ in range(round(5 / dt)):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt)
            actual = torch.rad2deg(robot.root_physx_view.get_dof_positions()[0, roll]).item()
            print(f"target={target:+.3f} deg, PhysX readback={actual:+.3f} deg", flush=True)
            if abs(actual - target) > 0.5:
                raise RuntimeError(f"Wrist roll did not reach {target:+g} degrees: {actual:+.3f}")
        print("PASS: both targets reached without a clamp at 160 degrees", flush=True)
        exit_code = 0
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app._app.post_quit(exit_code)
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
