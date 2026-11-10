"""Live stacking measurements and confirmed success, owned by the task."""
import torch
import isaaclab.utils.math as math_utils
from isaaclab.sim import get_current_stage
from sim.envs.mdp import as_torch, env_indices
from sim.envs.mdp._common import _ee_body


def reset_target(env, env_ids):
    """Restore the support cube for a fresh teleoperation demonstration."""
    ids = env_indices(env, env_ids)
    target = env.scene["cube_b"]
    pose = as_torch(target.data.default_root_state)[ids, :7].clone()
    pose[:, :3] += as_torch(env.scene.env_origins)[ids]
    target.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    target.write_root_velocity_to_sim_index(root_velocity=pose.new_zeros((len(ids), 6)), env_ids=ids)


def reset_tracking(env, env_ids):
    ids = env_indices(env, env_ids)
    if not hasattr(env, "_stack_held"):
        env._stack_held = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._stack_ever_grasped = torch.zeros_like(env._stack_held)
        env._stack_rest_z = as_torch(env.scene["cube_a"].data.root_pos_w)[:, 2].clone()
        env._stack_confirm = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._stack_step = torch.full_like(env._stack_confirm, -1)
    env._stack_held[ids] = False
    env._stack_ever_grasped[ids] = False
    env._stack_rest_z[ids] = as_torch(env.scene["cube_a"].data.root_pos_w)[ids, 2]
    env._stack_confirm[ids] = 0
    env._stack_step[ids] = -1
    env.scene["contact_grasp"].reset(ids)


def stage_signals(env):
    if not hasattr(env, "_stack_held"):
        reset_tracking(env, None)
    if bool(torch.all(env._stack_step == env.common_step_counter)):
        return env._stack_signals
    from . import TASK
    settings = TASK.spec.raw["success"]
    first, target = env.scene["cube_a"], env.scene["cube_b"]
    p = as_torch(first.data.root_pos_w)
    q = as_torch(first.data.root_quat_w)
    target_p = as_torch(target.data.root_pos_w)
    target_q = as_torch(target.data.root_quat_w)
    update = env._stack_step != env.common_step_counter
    env._stack_rest_z[update] = torch.minimum(env._stack_rest_z[update], p[update, 2])
    lifted = p[:, 2] - env._stack_rest_z >= settings["min_lift"]
    force = as_torch(env.scene["contact_grasp"].data.force_matrix_w)
    contact = torch.linalg.vector_norm(force, dim=-1).sum(dim=1)[:, 0]
    held = ((env._stack_ever_grasped & (contact > 0)) |
            ((contact > settings["force_threshold"]) & lifted))
    env._stack_held[update] = held[update]
    env._stack_ever_grasped[update] |= held[update]
    inverse = math_utils.quat_inv(target_q)
    relative = math_utils.quat_apply(inverse, p - target_p)
    axes = math_utils.matrix_from_quat(math_utils.quat_mul(inverse, q))
    half = p.new_tensor(first.cfg.spawn.size) / 2
    extent = torch.matmul(axes.abs(), half)
    target_half = p.new_tensor(target.cfg.spawn.size) / 2
    aligned = torch.all(relative[:, :2].abs() + extent[:, :2] <= target_half[:2] + settings["xy_tolerance"], dim=-1)
    on_top = (relative[:, 2] - target_half[2] - extent[:, 2]).abs() <= settings["z_tolerance"]
    stable = torch.linalg.vector_norm(as_torch(first.data.root_vel_w)[:, :3], dim=-1) <= settings["max_speed"]
    warmed = env.episode_length_buf >= settings["warmup_steps"]
    stacked = env._stack_ever_grasped & aligned & on_top & stable & ~held & warmed
    next_count = torch.where(stacked, env._stack_confirm + 1, 0)
    env._stack_confirm[update] = next_count[update]
    robot = env.scene["robot"]
    ee_index, ee_path = _ee_body(env, robot)
    ee_position = as_torch(robot.data.body_link_pos_w)[:, ee_index]
    ee_quaternion = as_torch(robot.data.body_link_quat_w)[:, ee_index]
    gripper = env.cfg.arm_spec.gripper
    if gripper is not None and gripper.tip_mesh:
        from sim.scene.geometry import gripper_tip_local_offset
        if not hasattr(env, "_stack_grasp_offset"):
            env._stack_grasp_offset = ee_position.new_tensor(gripper_tip_local_offset(
                get_current_stage(), float(first.cfg.spawn.size[0]) / 2, 0.0, ee_path, gripper.tip_mesh))
        ee_position = ee_position + math_utils.quat_apply(ee_quaternion, env._stack_grasp_offset.expand_as(ee_position))
    env._stack_signals = {"holding": held, "grasped": held & warmed, "lifted": lifted,
                          "over_target": held & aligned, "stacked": stacked,
                          "success": env._stack_confirm >= settings["confirm_steps"],
                          "reach_distance": torch.linalg.vector_norm(ee_position - p, dim=-1)}
    env._stack_step[update] = env.common_step_counter
    return env._stack_signals


def success(env):
    return stage_signals(env)["success"]


def signal(env, name):
    return stage_signals(env)[name].float().unsqueeze(-1)
