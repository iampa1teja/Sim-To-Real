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
"""Reset and domain-randomization events (Isaac Lab 3.0, Newton).

Quaternions: Isaac Lab 3.0 tensors are XYZW. Recorded starts, eval sets and the
geometry helpers in ``tasks/pick_place_cube/eval.py`` keep v1's WXYZ on disk, so this
module converts at that boundary with ``_wxyz`` / ``_xyzw``.

The vial-task randomizers (mat rotation, camera pose/focal/light-exposure DR) are
not ported; the camera setters stay because replay_scene uses them.
"""

import os
import re

import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import get_current_stage

from ._common import as_torch

def env_indices(env, env_ids):
    """Any env_ids an event can receive (None, slice, list, tensor) as a 1-D index tensor.

    Isaac Lab's EventManager passes startup, reset and interval terms different shapes
    (and different versions differ), so every event that indexes by environment
    normalizes through this instead of assuming a tensor.
    """
    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "beige": (0.960784, 0.960784, 0.862745),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def _wxyz(q):
    """XYZW (Isaac Lab 3.0) -> WXYZ (v1 on-disk / numpy geometry helpers)."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def _xyzw(q):
    """WXYZ -> XYZW."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def _env_prim_path(env, asset, index):
    """Concrete prim path of ``asset`` in environment ``index``."""
    path = asset.cfg.prim_path
    env_paths = getattr(env.scene, "env_prim_paths", None)
    # Asset configs may retain the namespace token or contain either version's
    # expanded regex. USD lookups need the actual scene namespace in all cases.
    namespace_pattern = r"(?:^|/)env_(?:\.\*|\[\^/\]\+)(?=/|$)"
    if env_paths is not None:
        env_path = env_paths[index]
        path = path.replace("{ENV_REGEX_NS}", env_path)
        match = re.search(namespace_pattern, path)
        return env_path + path[match.end():] if match else path
    return re.sub(
        namespace_pattern,
        lambda match: ("/" if match.group().startswith("/") else "") + f"env_{index}",
        path,
    )


def _ee_body(env, robot):
    """(ee body index, ee prim path in env 0) for the arm's end-effector link from arm.toml."""
    from core.arms.spec import load_arm_spec

    ee_link = load_arm_spec()[0].links.ee
    ids, _ = robot.find_bodies(ee_link)
    if len(ids) != 1:
        raise ValueError(f"Expected exactly one robot body named {ee_link!r}")
    return ids[0], _env_prim_path(env, robot, 0) + "/" + ee_link


# ---------------------------------------------------------------------------
# Robot appearance
# ---------------------------------------------------------------------------


def randomize_robot_color(env, env_ids, color_names: list[str] = list(ROBOT_COLORS.keys())):
    """Randomly set robot color from predefined palette on each reset."""
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    selected_name = color_names[idx]
    set_robot_color(env, ROBOT_COLORS[selected_name])
    return selected_name


def set_robot_color(env, rgb: tuple[float, float, float]):
    """Set color of the arm to a desired RGB value (USD material edit, independent of physics backend)."""
    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        from core.arms.spec import load_arm_spec

        material_path = load_arm_spec()[0].appearance.get("body_material")
        if not material_path:
            return
        material_prim_path = robot.cfg.prim_path + "/" + material_path.lstrip("/")
        for material_prim in sim_utils.find_matching_prims(material_prim_path):
            shader = UsdShade.Shader(material_prim)
            name = "diffuseColor" if shader.GetIdAttr().Get() == "UsdPreviewSurface" else "diffuse_color_constant"
            shader.GetInput(name).Set(Gf.Vec3f(*rgb))


# ---------------------------------------------------------------------------
# Cube placement
# ---------------------------------------------------------------------------


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
    from sim.scene.gripper_geometry import gripper_tip_local_offset, table_plane

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
    local_offset = eef_pos_w.new_tensor(
        gripper_tip_local_offset(stage, half_size, side_clearance_m, ee_path)
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
    planes = [table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index])) for index in env_ids.tolist()]
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
    objects are declared up front and parked here on reset; ``sim.generate.object_spawner``
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


# ---------------------------------------------------------------------------
# Camera setters (used by replay_scene)
# ---------------------------------------------------------------------------

# Cache for storing default poses read from USD
_default_poses_cache = {}


def sync_pick_place_camera_intrinsics(env, env_ids):
    """Make focal-length changes affect the OpenCV RTX lens model too."""
    for name in ("camera_realsense_rgb", "camera_wrist_cam", "realsense_depth"):
        if name not in env.scene.sensors:  # None cfgs (e.g. realsense_depth in eval) are never spawned
            continue
        camera = env.scene[name]
        calibration = camera.cfg.spawn.intrinsics
        if calibration is None:
            continue
        for prim in sim_utils.find_matching_prims(camera.cfg.prim_path):
            scale = prim.GetAttribute("focalLength").Get() / camera.cfg.spawn.focal_length
            for axis in ("fx", "fy"):
                prim.GetAttribute("omni:lensdistortion:opencvPinhole:" + axis).Set(calibration[axis] * scale)


def get_camera_base_pose(env, prim_path_pattern):
    """Read and cache the original camera pose (quat XYZW) for an offset setter."""
    if prim_path_pattern not in _default_poses_cache:
        prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        prims = sim_utils.find_matching_prims(prim_path)
        if not prims:
            return None
        prim = prims[0]
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        default_pos = (0.0, 0.0, 0.0)
        default_quat = (0.0, 0.0, 0.0, 1.0)
        if translate_attr.IsValid() and translate_attr.Get() is not None:
            value = translate_attr.Get()
            default_pos = (value[0], value[1], value[2])
        if orient_attr.IsValid() and orient_attr.Get() is not None:
            value = orient_attr.Get()
            default_quat = (*value.GetImaginary(), value.GetReal())
        _default_poses_cache[prim_path_pattern] = {"pos": default_pos, "quat": default_quat}
    return _default_poses_cache[prim_path_pattern]


def set_camera_pose_offset(env, prim_path_pattern, pos_offset: dict, rot_offset: dict):
    """Apply a position and Euler-angle offset to the camera's base pose."""
    prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_path)
    base_pose = get_camera_base_pose(env, prim_path_pattern)
    if not prims or base_pose is None:
        return
    base_pos = base_pose["pos"]
    position = tuple(base_pos[index] + pos_offset.get(axis, 0.0) for index, axis in enumerate(("x", "y", "z")))
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.tensor([rot_offset.get("roll", 0.0)]),
        torch.tensor([rot_offset.get("pitch", 0.0)]),
        torch.tensor([rot_offset.get("yaw", 0.0)]),
    )[0]
    x, y, z, w = math_utils.quat_mul(torch.tensor([base_pose["quat"]]), delta_quat.unsqueeze(0))[0].tolist()
    with Sdf.ChangeBlock():
        for prim in prims:
            if prim.IsValid():
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(*position))
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid():
                    # Gf takes the real part first. Match the attribute's precision (Quatf or Quatd).
                    quat_type = Gf.Quatf if isinstance(orient_attr.Get(), Gf.Quatf) else Gf.Quatd
                    orient_attr.Set(quat_type(w, x, y, z))


def set_camera_focal_length(env, asset_cfg: SceneEntityCfg, focal_length):
    """Set one focal length on every matching camera prim."""
    camera = env.scene[asset_cfg.name]
    camera_prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    with Sdf.ChangeBlock():
        for prim in sim_utils.find_matching_prims(camera_prim_path):
            if prim.IsValid():
                focal_attr = prim.GetAttribute("focalLength")
                if focal_attr.IsValid():
                    focal_attr.Set(focal_length)


def apply_runtime_joint_limits(env, env_ids):
    """Apply per-joint arm limits: revolute degrees become radians; prismatic limits stay metres."""
    from core.arms.spec import load_arm_spec

    arm, _ = load_arm_spec()
    robot = env.scene["robot"]
    missing = [joint.usd for joint in arm.joints if joint.usd not in robot.joint_names]
    if missing:
        raise ValueError(f"arm.toml joint names {missing} not in the USD articulation {robot.joint_names}")
    joint_ids = [robot.joint_names.index(joint.usd) for joint in arm.joints]
    values = torch.tensor([joint.limits for joint in arm.joints], device=env.device, dtype=torch.float32)
    revolute = torch.tensor([joint.type == "revolute" for joint in arm.joints], device=env.device)
    values[revolute] = torch.deg2rad(values[revolute])
    env_ids = env_indices(env, env_ids)
    limits = values.unsqueeze(0).repeat(len(env_ids), 1, 1)
    robot.write_joint_position_limit_to_sim_index(limits=limits, joint_ids=joint_ids, env_ids=env_ids)
    print(f"[INFO]: Runtime joint limits applied from arm {arm.name!r}: {arm.runtime_joint_limits}")
