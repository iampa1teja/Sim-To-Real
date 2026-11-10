"""Simulator placement events."""
import numpy as np
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import get_current_stage
from ._common import _arm_spec, _ee_body, as_torch, env_indices


def reset_object_pose(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    side_clearance_m: float = 0.003,
    half_extents: tuple[float, float, float] | None = None,
) -> None:
    """Place an object flush beside the gripper's fixed jaw, then zero its velocity.

    Measures the fixed-jaw mesh (same method as v1's cube_spawner) and puts the
    object's centre just beside the jaw tip with ``side_clearance_m`` clearance, oriented
    like the live end-effector. Optional ``pose_range`` jitter (x/y/z metres, roll/pitch/yaw
    radians) is layered on top. v1 read the end effector from the ``ee_frame``
    FrameTransformer; this reads the arm.toml ee link's body pose instead.

    ``half_extents`` (metres, object frame) sizes any object; when None it comes from a
    cuboid spawn's ``size`` (USD objects pass it from task.toml ``size``).
    """
    from sim.scene.geometry import gripper_tip_local_offset, table_plane
    from core.scenes import load_scene_spec

    env_ids = env_indices(env, env_ids)
    if len(env_ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    robot = env.scene["robot"]

    # Joint reset events write DOFs before body transforms refresh; without a forward
    # update the ee pose is stale (the USD bind pose on first reset), placing the cube in mid-air.
    env.sim.forward()
    ee_id, ee_path = _ee_body(env, robot)
    eef_pos_w = as_torch(robot.data.body_link_pos_w)[:, ee_id]
    eef_quat_w = as_torch(robot.data.body_link_quat_w)[:, ee_id]

    if half_extents is None:
        size = getattr(asset.cfg.spawn, "size", None)
        if size is None or len(size) != 3:
            raise ValueError(f"{asset_cfg.name}: jaw-relative placement needs half_extents or a cuboid spawn size.")
        half_extents = tuple(float(v) / 2.0 for v in size)
    # The widest half-extent keeps any face clear of the jaw.
    half_size = max(float(v) for v in half_extents)
    stage = get_current_stage()
    gripper = _arm_spec(env).gripper
    if gripper is None or not gripper.tip_mesh:
        raise ValueError("Jaw-relative placement needs [arm.gripper] tip_mesh")
    local_offset = eef_pos_w.new_tensor(
        gripper_tip_local_offset(stage, half_size, side_clearance_m, ee_path, gripper.tip_mesh)
    ).unsqueeze(0).repeat(len(env_ids), 1)

    world_offset = math_utils.quat_apply(eef_quat_w[env_ids], local_offset)
    cube_pos_w = eef_pos_w[env_ids] + world_offset
    cube_quat_w = eef_quat_w[env_ids].clone()

    if pose_range:
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
        cube_pos_w = cube_pos_w + rand_samples[:, 0:3]
        jitter_quat = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        cube_quat_w = math_utils.quat_mul(cube_quat_w, jitter_quat)

    # Fit each environment's tabletop in world metres; raising Z alone keeps the cube beside the gripper.
    scene = getattr(env.cfg, "scene_spec", None) or load_scene_spec()
    table = scene.anchors.get("table")
    if table is None:
        raise ValueError("Jaw-relative placement needs a scene table anchor")
    planes = [table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]), table)
              for index in env_ids.tolist()]
    plane = cube_pos_w.new_tensor(np.array(planes))
    normal = torch.cat((-plane[:, :2], torch.ones_like(plane[:, 2:])), dim=-1)
    rotation = math_utils.matrix_from_quat(cube_quat_w)
    extents = cube_pos_w.new_tensor(half_extents)
    support = (torch.einsum("bi,bij->bj", normal, rotation).abs() * extents).sum(dim=-1)
    table_height = (plane[:, :2] * cube_pos_w[:, :2]).sum(dim=-1) + plane[:, 2]
    cube_pos_w[:, 2] = torch.maximum(cube_pos_w[:, 2], table_height + support + 0.0002)

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([cube_pos_w, cube_quat_w], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=cube_pos_w.new_zeros((len(env_ids), 6)), env_ids=env_ids)


def park_object(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    park_offset: tuple[float, float, float] = (0.0, 0.0, -2.0),
) -> None:
    """Move an object out of the scene (env origin + ``park_offset``) with zero velocity.

    Newton builds its model once, so objects cannot be created at runtime. Spawnable task
    objects are declared up front and parked here on reset; ``sim.datagen.spawner``
    keeps them parked every tick and teleports one to the jaw when asked.
    """
    env_ids = env_indices(env, env_ids)
    if len(env_ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    origins = as_torch(env.scene.env_origins)[env_ids]
    position = origins + origins.new_tensor(park_offset)
    quat = origins.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(len(env_ids), 4)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([position, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=position.new_zeros((len(env_ids), 6)), env_ids=env_ids)
