# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import torch
import glob
import os
import yaml

import numpy as np
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import get_current_stage
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer
from isaacsim.core.prims import XFormPrim
from sim_to_real_so101.utils.gripper_geometry import gripper_tip_local_offset, table_plane



ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "beige": (0.960784, 0.960784, 0.862745),  
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def randomize_robot_color(env, 
    env_ids: torch.Tensor | None,
    color_names: list[str] = list(ROBOT_COLORS.keys()),
):
    """Randomly set robot color from predefined palette on each reset."""
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    selected_color = ROBOT_COLORS[color_names[idx]]
    
    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_prim_path = robot.cfg.prim_path + "/Looks/material_a_3d_printed/Shader"
        material_prim = sim_utils.find_matching_prims(material_prim_path)[0]
        material_prim.GetAttribute("inputs:diffuse_color_constant").Set(selected_color)

def randomize_mat_rotation(
    env,
    env_ids: torch.Tensor | None,
    yaw_range: tuple[float, float] = (-0.3, 0.3),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize mat yaw rotation on reset.
    
    Args:
        yaw_range: Range of yaw rotation in radians (default ±0.3 rad ≈ ±17°).
    """
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]
    
    # Sample random yaw for each environment
    yaw = math_utils.sample_uniform(
        yaw_range[0], yaw_range[1], (len(env_ids),), device=env.unwrapped.device
    )
    # Keep roll and pitch at 0, add 90° base yaw (mat's default orientation)
    roll = torch.zeros_like(yaw)
    pitch = torch.zeros_like(yaw)
    base_yaw = torch.full_like(yaw, math.pi / 2)  # 90° in radians
    
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, base_yaw + yaw)
    
    asset_xform = XFormPrim(prim_paths_expr=asset_prim_path)
    
    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations)


def randomize_camera_focal_length(
    env,
    env_ids: torch.Tensor | None,
    focal_length_range: tuple[float, float] = (12.0, 15.0),
    asset_cfg: SceneEntityCfg = None,
):
    """Randomize camera focal length on reset (affects field of view).
    
    Args:
        focal_length_range: Range of focal lengths in mm (default 12-15mm).
            Lower = wider FOV, higher = narrower FOV.
    """
    camera = env.scene[asset_cfg.name]
    camera_prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    
    focal_length = math_utils.sample_uniform(
        focal_length_range[0], focal_length_range[1], (1,), device="cpu"
    ).item()
    
    camera_prims = sim_utils.find_matching_prims(camera_prim_path)
    
    with Sdf.ChangeBlock():
        for prim in camera_prims:
            if prim.IsValid():
                focal_attr = prim.GetAttribute("focalLength")
                if focal_attr.IsValid():
                    focal_attr.Set(focal_length)


# Cache for storing default poses read from USD
_default_poses_cache = {}


def randomize_camera_pose(
    env,
    env_ids: torch.Tensor | None,
    prim_path_pattern: str,
    pos_range: dict[str, tuple[float, float]] = None,
    rot_range: dict[str, tuple[float, float]] = None,
):
    """Randomize camera mount position and orientation relative to its USD default.
    
    Args:
        prim_path_pattern: Prim path pattern for the camera mount xform.
        pos_range: Dict with x, y, z keys and (min, max) tuple values in meters.
        rot_range: Dict with roll, pitch, yaw keys and (min, max) tuple values in radians.
    """
    if pos_range is None:
        pos_range = {}
    if rot_range is None:
        rot_range = {}
    
    prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_path)
    
    if not prims:
        return
    
    # Read and cache default pose from first prim on first call
    if prim_path_pattern not in _default_poses_cache:
        prim = prims[0]
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        
        default_pos = (0.0, 0.0, 0.0)
        default_quat = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
        
        if translate_attr.IsValid():
            val = translate_attr.Get()
            if val is not None:
                default_pos = (val[0], val[1], val[2])
        
        if orient_attr.IsValid():
            val = orient_attr.Get()
            if val is not None:
                default_quat = (val.GetReal(), val.GetImaginary()[0], val.GetImaginary()[1], val.GetImaginary()[2])
        
        _default_poses_cache[prim_path_pattern] = {"pos": default_pos, "quat": default_quat}
    
    base_pos = _default_poses_cache[prim_path_pattern]["pos"]
    base_quat = _default_poses_cache[prim_path_pattern]["quat"]
    
    # Sample random offsets
    x = base_pos[0] + math_utils.sample_uniform(*pos_range.get("x", (0, 0)), (1,), device="cpu").item()
    y = base_pos[1] + math_utils.sample_uniform(*pos_range.get("y", (0, 0)), (1,), device="cpu").item()
    z = base_pos[2] + math_utils.sample_uniform(*pos_range.get("z", (0, 0)), (1,), device="cpu").item()
    
    # Sample rotation offsets and combine with base quaternion
    roll = math_utils.sample_uniform(*rot_range.get("roll", (0, 0)), (1,), device="cpu").item()
    pitch = math_utils.sample_uniform(*rot_range.get("pitch", (0, 0)), (1,), device="cpu").item()
    yaw = math_utils.sample_uniform(*rot_range.get("yaw", (0, 0)), (1,), device="cpu").item()
    
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw])
    )[0]
    
    # Combine base quaternion with delta
    base_quat_tensor = torch.tensor([base_quat])
    final_quat = math_utils.quat_mul(base_quat_tensor, delta_quat.unsqueeze(0))[0]
    
    with Sdf.ChangeBlock():
        for prim in prims:
            if prim.IsValid():
                # Set translation
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(x, y, z))
                # Set orientation
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid():
                    orient_attr.Set(Gf.Quatd(
                        final_quat[0].item(), final_quat[1].item(), final_quat[2].item(), final_quat[3].item()
                    ))


def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)


def randomize_sky_light(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    temperature_range: tuple[float, float],
    textures_root: str,
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    temperature = math_utils.sample_uniform(*temperature_range, (1,), device="cpu").item()

    # list all .exr files in the textures_root
    textures = glob.glob(os.path.join(textures_root, "*.exr"))
    # get a random texture (using torch)

    if not textures:
        print("[WARNING] No textures found in the textures_root")
        return

    texture = torch.randint(0, len(textures), (1,), device="cpu").item()
    texture = textures[texture]

    yaw_mapping_path = os.path.join(textures_root, "yaw_mapping.yaml")
    with open(yaw_mapping_path, 'r') as f:
        yaw_mapping = yaml.safe_load(f)

    yaw_mapping = yaw_mapping.get(os.path.basename(texture), None)

    range_list = [
        (0.0, 0.0),
        (0.0, 0.0),
        (
            yaw_mapping[0] if yaw_mapping else 0.0,
            yaw_mapping[1] if yaw_mapping else 0.0,
        ),
    ]
    ranges = torch.tensor(range_list, device=env.unwrapped.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.unwrapped.device
    )
    orientations = math_utils.quat_from_euler_xyz(
        rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2]
    )

    asset_xform = XFormPrim(prim_paths_expr=asset_prim_path)

    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations)
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)
            prim.GetAttribute("inputs:colorTemperature").Set(temperature)
            texture = Sdf.AssetPath(texture)
            prim.GetAttribute("inputs:texture:file").Set(texture)


def random_asset_pose(
        env, 
        env_ids, 
        asset, 
        pose_range, 
        pos_offset
):

    root_states = asset.data.default_root_state[env_ids].clone()
    pos_offset_list = [pos_offset.get(key, 0.0) for key in ["x", "y", "z"]]
    pos_offset = torch.tensor(pos_offset_list, device=asset.device)
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    positions = (
        root_states[:, 0:3]
        + env.scene.env_origins[env_ids]
        + rand_samples[:, 0:3]
        + pos_offset
    )
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    asset.write_root_pose_to_sim(
        torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return positions, orientations


def reset_vials_rack(
        env,
        env_ids: torch.Tensor,
        vials: list[str],
        rack: str,
        rack_pose_range: dict[str, tuple[float, float]],
        pose_range: dict[str, tuple[float, float]],
        fixed_vial_z: float,
        rack_placement_prob: float = 0.33,
):

    vial_objects: list[RigidObject | Articulation] = [
        env.scene[asset_name] for asset_name in vials
    ]

    rack = env.scene[rack]
    slots_xform_view = XFormPrim(prim_paths_expr=f"{rack.cfg.prim_path}/Body1/Mesh/top_*")
    total_slots = len(slots_xform_view.prims)

    # randomize rack pose
    new_rack_positions, new_rack_orientations = random_asset_pose(env, env_ids, rack, rack_pose_range, {})
    # Clear velocities immediately after positioning
    zero_velocity = torch.zeros((len(env_ids), 6), device=rack.device)
    rack.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    placed_on_rack_indices = []
    if torch.rand(1, device=env.unwrapped.device).item() < rack_placement_prob:
        # Pick a random vial to place on the rack
        vial_idx = torch.randint(0, len(vial_objects), (1,), device=env.unwrapped.device).item()
        placed_on_rack_indices.append(vial_idx)
    
    slot_positions_local, slot_orientations_local = slots_xform_view.get_local_poses()
    # Place selected vials on rack
    for vial_idx in placed_on_rack_indices:
        vial = vial_objects[vial_idx]
        
        # Select a random slot for this vial
        slot_idx = torch.randint(0, total_slots, (1,), device=env.unwrapped.device).item()
        
        # Thank you Sonnet lol
        # IMPORTANT: Cannot use get_world_poses() here because write_root_pose_to_sim() doesn't
        # update USD until sim.step(). Instead, manually compute slot positions from local transforms.
        # Transform slot positions from rack local frame to world frame using the NEW rack pose
        # For each environment, we need to transform the slot position
        slot_position_local = slot_positions_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        slot_orientation_local = slot_orientations_local[slot_idx].unsqueeze(0).repeat(len(env_ids), 1)
        
        # Combine transforms: world_pose = rack_pose ⊕ slot_local_pose
        slot_position, slot_orientation = math_utils.combine_frame_transforms(
            new_rack_positions, new_rack_orientations, slot_position_local, slot_orientation_local
        )
        # slot_position and slot_orientation are already per-env tensors (shape: len(env_ids), 3/4)
        slot_pose = torch.cat([slot_position, slot_orientation], dim=-1)
        vial.write_root_pose_to_sim(slot_pose, env_ids=env_ids)

        zero_velocity = torch.zeros((len(env_ids), 6), device=vial.device)
        vial.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    pose_range_z_fixed = {**pose_range, "z": (0.0, 0.0)}
    for i, v in enumerate(vial_objects):
        if i not in placed_on_rack_indices:
            default_z = v.data.default_root_state[env_ids[0], 2].item()
            pos_offset = {"z": fixed_vial_z - default_z}
            _, _ = random_asset_pose(env, env_ids, v, pose_range_z_fixed, pos_offset)
            zero_velocity = torch.zeros((len(env_ids), 6), device=v.device)
            v.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def reset_object_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    side_clearance_m: float = 0.003,
) -> None:
    """Place an object flush beside the gripper's fixed jaw, then zero its velocity.

    Measures the actual fixed-jaw mesh (same method as
    cube_spawner.spawn_blue_cube) to place the object's centre just beside the
    jaw tip with ``side_clearance_m`` clearance, oriented to match the live
    end-effector pose — tight/adjacent placement, not an arbitrary offset. If
    ``pose_range`` is non-empty, additional uniform jitter (x/y/z metres,
    roll/pitch/yaw radians) is layered on top, using the same sampling
    pattern as ``random_asset_pose``.

    Args:
        asset_cfg:         SceneEntityCfg naming the target rigid object (e.g. "blue_cube").
        pose_range:         Optional per-axis (min, max) jitter on top of the
                            jaw-relative placement; missing keys default to no
                            jitter. Pass ``{}`` to disable jitter entirely.
        eef_frame_cfg:      SceneEntityCfg for the FrameTransformer sensor that
                            tracks the gripper tip (default: "ee_frame").
        side_clearance_m:   Extra clearance in metres beyond the object's own
                            half-size when measuring the jaw tip (default
                            0.003 m, matching cube_spawner.spawn_blue_cube).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    asset = env.scene[asset_cfg.name]
    ee_frame: FrameTransformer = env.scene[eef_frame_cfg.name]

    # Joint reset events write DOFs before PhysX refreshes link transforms.
    # Querying the sensor here without a forward update uses the previous
    # pose (the USD bind pose on first reset), placing the cube in mid-air.
    env.sim.forward()
    ee_frame.reset(env_ids)
    eef_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    eef_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    size = getattr(asset.cfg.spawn, "size", None)
    if size is None or len(size) != 3 or len(set(size)) != 1:
        raise ValueError("Jaw-relative placement requires a cuboid with an explicit cubic size.")
    half_size = float(size[0]) / 2.0
    stage = get_current_stage()
    target_path = ee_frame.cfg.target_frames[0].prim_path
    gripper_prims = sim_utils.find_matching_prims(target_path)
    if not gripper_prims:
        raise RuntimeError(f"Gripper frame not found: {target_path}")
    local_offset = eef_pos_w.new_tensor(gripper_tip_local_offset(
        stage, half_size, side_clearance_m, str(gripper_prims[0].GetPath())
    )).unsqueeze(0).repeat(len(env_ids), 1)

    world_offset = math_utils.quat_apply(eef_quat_w[env_ids], local_offset)
    cube_pos_w = eef_pos_w[env_ids] + world_offset

    # Match the gripper's live orientation, same as spawn_blue_cube does.
    cube_quat_w = eef_quat_w[env_ids].clone()

    if pose_range:
        range_list = [
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
        )
        cube_pos_w = cube_pos_w + rand_samples[:, 0:3]
        jitter_quat = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        cube_quat_w = math_utils.quat_mul(cube_quat_w, jitter_quat)

    # Fit each environment's actual tabletop in world metres. Raising Z alone
    # preserves the requested position directly beside the gripper.
    planes = [table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]))
              for index in env_ids.tolist()]
    plane = cube_pos_w.new_tensor(np.array(planes))
    normal = torch.cat((-plane[:, :2], torch.ones_like(plane[:, 2:])), dim=-1)
    rotation = math_utils.matrix_from_quat(cube_quat_w)
    support = half_size * torch.einsum("bi,bij->bj", normal, rotation).abs().sum(dim=-1)
    table_height = (plane[:, :2] * cube_pos_w[:, :2]).sum(dim=-1) + plane[:, 2]
    cube_pos_w[:, 2] = torch.maximum(cube_pos_w[:, 2], table_height + support + 0.0002)

    pose = torch.cat([cube_pos_w, cube_quat_w], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)

    zero_velocity = cube_pos_w.new_zeros((len(env_ids), 6))
    asset.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    if asset_cfg.name == "blue_cube":
        from .terms import reset_pick_place_state
        reset_pick_place_state(env, env_ids, asset_cfg.name)
        if "contact_grasp" in env.scene.keys():
            env.scene["contact_grasp"].reset(env_ids)


def reset_cube_from_recorded_starts(
    env, env_ids, asset_cfg, starts_dir, mode="cycle", random_fraction=0.0, seed=None,
):
    """Evaluation-only placement from base-link frame trajectory starts."""
    from sim_to_real_so101.utils.pick_place_eval import (
        RecordedStarts, resting_pose, rotation_matrix, footprint, footprints_overlap, upright_orientation,
    )
    from sim_to_real_so101.tasks.pick_place_env_cfg import WHITE_BOX_SIZE
    from .terms import reset_pick_place_state

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not len(env_ids):
        return
    starts_dir = starts_dir or os.environ.get("PICK_PLACE_EVAL_STARTS")
    seed = seed if seed is not None else env.cfg.seed
    key = (starts_dir, mode, random_fraction, seed)
    if getattr(env, "_pick_place_starts_key", None) != key:
        env._pick_place_starts = RecordedStarts(starts_dir, mode, random_fraction, seed)
        env._pick_place_starts_key = key
    sampler = env._pick_place_starts
    asset, robot, box = env.scene[asset_cfg.name], env.scene["robot"], env.scene["white_box"]
    env.sim.forward()
    base_ids, _ = robot.find_bodies("base")
    if len(base_ids) != 1:
        raise ValueError('Recorded starts require exactly one robot body named base')
    base_id = base_ids[0]
    size = asset.cfg.spawn.size
    stage = get_current_stage()
    if not hasattr(env, "_pick_place_start_index"):
        env._pick_place_start_index = [-1] * env.num_envs
        env._pick_place_start_zone = [None] * env.num_envs
        env._pick_place_cube_color = ["blue"] * env.num_envs
    positions, orientations = [], []
    for index in env_ids.tolist():
        base_p = robot.data.body_pos_w[index, base_id].detach().cpu().numpy()
        base_q = robot.data.body_quat_w[index, base_id].detach().cpu().numpy()
        plane = table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]))
        box_p = box.data.root_pos_w[index].detach().cpu().numpy()
        box_q = box.data.root_quat_w[index].detach().cpu().numpy()
        box_polygon = footprint(box_p, rotation_matrix(box_q), WHITE_BOX_SIZE, bottom_origin=True)

        def accepts_random(p, q):
            q = upright_orientation(base_q, plane, q)
            world, rotation = resting_pose(p, q, base_p, base_q, plane, size)
            return not footprints_overlap(footprint(world, rotation, size), box_polygon)

        start, p, q, zone = sampler.sample(accepts_random)
        if start == -1:
            q = upright_orientation(base_q, plane, q)
        world, rotation = resting_pose(p, q, base_p, base_q, plane, size)
        if start != -1 and footprints_overlap(footprint(world, rotation, size), box_polygon):
            raise ValueError(f"Recorded start {start} overlaps the live white box; verify the dataset and box pose")
        positions.append(world)
        orientations.append(math_utils.quat_mul(
            torch.as_tensor(base_q, device=env.device),
            torch.as_tensor(q, device=env.device, dtype=robot.data.body_quat_w.dtype),
        ))
        env._pick_place_start_index[index] = start
        env._pick_place_start_zone[index] = zone
    pose = torch.cat((torch.as_tensor(np.array(positions), device=env.device, dtype=asset.data.root_pos_w.dtype),
                      torch.stack(orientations)), dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)
    reset_pick_place_state(env, env_ids, asset_cfg.name)
    env.scene["contact_grasp"].reset(env_ids)


def randomize_cube_color(env, env_ids, colors):
    """Set the bound live cube surface shader; remember the rendered colour."""
    from pxr import Usd, UsdShade

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset = env.scene["blue_cube"]
    stage = get_current_stage()
    names = list(colors)
    if not names:
        raise ValueError('Cube colour palette must not be empty')
    if not hasattr(env, "_pick_place_cube_color"):
        env._pick_place_cube_color = ["blue"] * env.num_envs
    for index in env_ids.tolist():
        name = names[torch.randint(len(names), (1,)).item()]
        path = asset.root_physx_view.prim_paths[index]
        root = stage.GetPrimAtPath(path)
        shaders = {}
        for prim in Usd.PrimRange(root):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                shader, _, _ = material.ComputeSurfaceSource()
                if shader:
                    shaders[str(shader.GetPath())] = shader
        if not shaders:
            raise RuntimeError(f'No bound surface shader found for cube at {path}')
        for shader in shaders.values():
            diffuse = shader.GetInput('diffuseColor')
            if not diffuse:
                raise RuntimeError(f'Cube shader {shader.GetPath()} has no diffuseColor input')
            diffuse.Set(Gf.Vec3f(*colors[name]))
        env._pick_place_cube_color[index] = name


def randomize_pick_place_room_light(env, env_ids, exposure_range):
    """Reuse light exposure DR on MyRoom's startup-created measured tube light."""
    from types import SimpleNamespace

    stage = get_current_stage()
    for index in env_ids.tolist():
        path = env.scene["room"].prim_paths[index] + "/Room/Lighting/Tubelight"
        if not stage.GetPrimAtPath(path).IsValid():
            # Custom ROOM_USD_PATH may not have the measured room's tube light.
            continue
        view = SimpleNamespace(scene={"light": SimpleNamespace(prim_paths=[path])})
        randomize_light_exposure(view, env_ids, exposure_range, SceneEntityCfg("light"))


def check_pick_place_eval_event_order(env, env_ids):
    """Check the actual manager's prepared reset order at environment creation."""
    from sim_to_real_so101.utils.pick_place_eval import RecordedStarts

    params = env.event_manager.get_term_cfg("spawn_cube").params
    directory = params["starts_dir"] or os.environ.get("PICK_PLACE_EVAL_STARTS")
    seed = params["seed"] if params["seed"] is not None else env.cfg.seed
    params["starts_dir"], params["seed"] = directory, seed
    env._pick_place_starts = RecordedStarts(directory, params["mode"], params["random_fraction"], seed)
    env._pick_place_starts_key = (directory, params["mode"], params["random_fraction"], seed)
    names = env.event_manager.active_terms['reset']
    if not names.index('reset_robot_position') < names.index('spawn_cube') < names.index('reset_tracking'):
        raise RuntimeError(f'Invalid pick-place reset order: {names}')
    if 'eval_robot_color' in names and not names.index('reset_set_robot_visual_material') < names.index('eval_robot_color'):
        raise RuntimeError(f'Robot colour DR would be overwritten: {names}')
    print(f'[Pick-place eval] Reset event order: {names}')


def sync_pick_place_camera_intrinsics(env, env_ids):
    """Make reused focal-length DR affect MyRoom's OpenCV RTX lens model too."""
    for name in ("camera_realsense_rgb", "camera_wrist_cam", "realsense_depth"):
        camera = env.scene[name]
        calibration = camera.cfg.spawn.intrinsics
        if calibration is None:
            continue
        for prim in sim_utils.find_matching_prims(camera.cfg.prim_path):
            scale = prim.GetAttribute('focalLength').Get() / camera.cfg.spawn.focal_length
            for axis in ('fx', 'fy'):
                prim.GetAttribute('omni:lensdistortion:opencvPinhole:' + axis).Set(calibration[axis] * scale)
