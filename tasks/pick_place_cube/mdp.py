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

import numpy as np
import torch
from pxr import Gf, Usd, UsdShade

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import get_current_stage

from sim.envs.mdp._common import _env_prim_path, _wxyz, _xyzw, as_torch, env_indices


# ---------------------------------------------------------------------------
# Cube colour DR
# ---------------------------------------------------------------------------


def randomize_cube_color(env, env_ids, colors):
    """Pick a random palette colour per env, apply it, and return the chosen names."""
    env_ids = env_indices(env, env_ids)
    names = list(colors)
    if not names:
        raise ValueError("Cube colour palette must not be empty")
    if not hasattr(env, "_pick_place_cube_color"):
        env._pick_place_cube_color = ["blue"] * env.num_envs
    selected_names = []
    for index in env_ids.tolist():
        name = names[torch.randint(len(names), (1,)).item()]
        set_cube_color(env, torch.tensor([index], device=env.device), colors[name], name)
        selected_names.append(name)
    return selected_names


def set_cube_color(env, env_ids, rgb: tuple[float, float, float], name):
    """Set the cube's bound surface shader colour in each listed environment."""
    env_ids = env_indices(env, env_ids)
    asset = env.scene["blue_cube"]
    stage = get_current_stage()
    if not hasattr(env, "_pick_place_cube_color"):
        env._pick_place_cube_color = ["blue"] * env.num_envs

    for index in env_ids.tolist():
        path = _env_prim_path(env, asset, index)  # v1: asset.root_physx_view.prim_paths[index]
        # Newton's scene topology and material bindings are fixed after startup.
        # Cache the resolved inputs on the env, not globally across USD stages.
        cache_stage, inputs = getattr(env, "_cube_color_inputs", (None, {}))
        if cache_stage != stage:
            inputs = {}
            env._cube_color_inputs = (stage, inputs)
        if path not in inputs:
            root = stage.GetPrimAtPath(path)
            shaders = {}
            for prim in Usd.PrimRange(root):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if material:
                    shader, _, _ = material.ComputeSurfaceSource()
                    if shader:
                        shaders[str(shader.GetPath())] = shader
            if not shaders:
                raise RuntimeError(f"No bound surface shader found for cube at {path}")
            resolved = []
            for shader in shaders.values():
                diffuse = shader.GetInput("diffuseColor")
                if not diffuse:
                    raise RuntimeError(f"Cube shader {shader.GetPath()} has no diffuseColor input")
                resolved.append(diffuse)
            inputs[path] = tuple(resolved)
        for diffuse in inputs[path]:
            diffuse.Set(Gf.Vec3f(*rgb))
        env._pick_place_cube_color[index] = name


# ---------------------------------------------------------------------------
# Grasp / place state, observations and termination (from terms.py)
# ---------------------------------------------------------------------------


def _ensure_pick_place_state(env, object_name):
    """Allocate episode trackers without changing already initialized environments."""
    if not hasattr(env, "_pick_place_holding"):
        env._pick_place_holding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._pick_place_ever_grasped = torch.zeros_like(env._pick_place_holding)
        env._pick_place_rest_z = as_torch(env.scene[object_name].data.root_pos_w)[:, 2].clone()
        env._pick_place_placed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._pick_place_grasp_step = torch.full_like(env._pick_place_placed_steps, -1)
        env._pick_place_placed_step = torch.full_like(env._pick_place_placed_steps, -1)


def reset_pick_place_state(env, env_ids, object_name="blue_cube"):
    """Clear only the reset environments, after writing their new object poses."""
    _ensure_pick_place_state(env, object_name)
    ids = slice(None) if env_ids is None else env_ids
    env._pick_place_holding[ids] = False
    env._pick_place_ever_grasped[ids] = False
    env._pick_place_rest_z[ids] = as_torch(env.scene[object_name].data.root_pos_w)[ids, 2]
    env._pick_place_placed_steps[ids] = 0
    env._pick_place_grasp_step[ids] = -1
    env._pick_place_placed_step[ids] = -1
    if "contact_grasp" in env.scene.keys():
        env.scene["contact_grasp"].reset(env_ids)


def _grasp_state(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    object_name: str,
    min_lift: float,
    force_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update and return ``(holding, ever_grasped)`` for ``object_name``.

    Shared by the grasp observation and the placement checks. Calling it more
    than once in the same step gives the same result, so the observation and
    termination managers can both use it.

    Lift is measured from the object's lowest height since reset, not an
    absolute world Z: the room's table is ~0.76 m up, so a world-Z threshold
    would read "lifted" while the cube is still on the table.
    """
    _ensure_pick_place_state(env, object_name)
    obj: RigidObject = env.scene[object_name]
    z = as_torch(obj.data.root_pos_w)[:, 2]
    update = env._pick_place_grasp_step != env.common_step_counter
    env._pick_place_rest_z[update] = torch.minimum(env._pick_place_rest_z[update], z[update])

    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    # force_matrix_w is (num_envs, num_bodies, num_filters) of vec3 -> torch (..., 3); the sole filter is the object.
    force_matrix = as_torch(contact_sensor.data.force_matrix_w)
    contact_force = torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=1)[:, 0]
    has_contact = contact_force > 0.0
    grasp_contact = contact_force > force_threshold
    is_lifted = (z - env._pick_place_rest_z) > min_lift

    # Acquisition needs a firm lift. A weaker remaining contact is not release;
    # recontact after an earlier grasp also prevents placement while touching.
    holding = (env._pick_place_ever_grasped & has_contact) | (grasp_contact & is_lifted)
    env._pick_place_holding[update] = holding[update]
    env._pick_place_ever_grasped[update] |= holding[update]
    env._pick_place_grasp_step[update] = env.common_step_counter
    return env._pick_place_holding, env._pick_place_ever_grasped


def object_grasped(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    object_name: str,
    min_lift: float = 0.01,
    warmup_steps: int = 30,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Grasp success: the jaw is in contact with the object and has lifted it.

    Args:
        contact_sensor_cfg: Jaw contact sensor whose only filter is ``object_name``.
        object_name: Scene name of the object (e.g. "blue_cube").
        min_lift: Metres above its resting height the object must rise to count as lifted.
        warmup_steps: Steps after reset during which nothing is reported.
        force_threshold: Minimum jaw-object contact force (N).

    Returns:
        Float tensor of shape (num_envs, 1): 1.0 while held, else 0.0.
    """
    holding, _ = _grasp_state(env, contact_sensor_cfg, object_name, min_lift, force_threshold)
    in_warmup = env.episode_length_buf < warmup_steps
    return (holding & ~in_warmup).float().unsqueeze(-1)


def _object_inside_container(
    obj: RigidObject,
    container: RigidObject,
    container_size: tuple[float, float, float],
    container_floor_z: float,
    container_inner_half_size: tuple[float, float],
) -> torch.Tensor:
    """Check an oriented cuboid against the measured inner walls, floor and rim.

    The box origin is the centre of its outer bottom face. Its live (XYZW)
    orientation transforms both the object centre and object axes into box space.
    """
    box_pos = as_torch(container.data.root_pos_w)
    box_quat = as_torch(container.data.root_quat_w)
    obj_pos = as_torch(obj.data.root_pos_w)
    obj_quat = as_torch(obj.data.root_quat_w)
    inverse = math_utils.quat_inv(box_quat)
    rel = math_utils.quat_apply(inverse, obj_pos - box_pos)
    rotation = math_utils.matrix_from_quat(math_utils.quat_mul(inverse, obj_quat))
    half_size = rel.new_tensor(obj.cfg.spawn.size) / 2
    extent = torch.matmul(rotation.abs(), half_size)
    inner = rel.new_tensor(container_inner_half_size)
    within_xy = torch.all(rel[:, :2].abs() + extent[:, :2] <= inner, dim=-1)
    # Use the centre at the floor so small contact-solver penetrations do not
    # reject resting cubes. The entire cube must still clear the inner walls
    # and fit below the rim (with only floating-point tolerance at the rim).
    tolerance = 8 * torch.finfo(rel.dtype).eps
    within_z = (rel[:, 2] >= container_floor_z) & (rel[:, 2] + extent[:, 2] <= container_size[2] + tolerance)
    return within_xy & within_z


def object_placed_in_container(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    object_name: str,
    container_name: str,
    container_size: tuple[float, float, float],
    container_floor_z: float,
    container_inner_half_size: tuple[float, float],
    min_lift: float = 0.01,
    warmup_steps: int = 30,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Place success: the object was grasped earlier this episode, is now inside
    the container, and has been released.

    Not latched; it reports the current state each step.
    ``object_placed_in_container_termination`` requires it to hold for several steps.

    Args:
        container_size: (length, width, height) of the container in metres, along its local X, Y, Z.
        container_floor_z: Height of the inside floor above the container's origin, in metres.
        container_inner_half_size: Measured inner X and Y half widths, in metres.

    Returns:
        Float tensor of shape (num_envs, 1) indicating placement status.
    """
    holding, ever_grasped = _grasp_state(env, contact_sensor_cfg, object_name, min_lift, force_threshold)
    inside = _object_inside_container(
        env.scene[object_name], env.scene[container_name], container_size, container_floor_z, container_inner_half_size
    )
    in_warmup = env.episode_length_buf < warmup_steps
    return (ever_grasped & inside & ~holding & ~in_warmup).float().unsqueeze(-1)


def object_placed_in_container_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    object_name: str,
    container_name: str,
    container_size: tuple[float, float, float],
    container_floor_z: float,
    container_inner_half_size: tuple[float, float],
    min_lift: float = 0.01,
    warmup_steps: int = 30,
    force_threshold: float = 2.0,
    confirm_steps: int = 25,
) -> torch.Tensor:
    """Task success: grasp then place, held for ``confirm_steps`` consecutive steps.

    The hold requirement stops a cube bouncing through the box from counting.

    Returns:
        Boolean tensor of shape (num_envs,) for termination.
    """
    placed = object_placed_in_container(
        env,
        contact_sensor_cfg,
        object_name,
        container_name,
        container_size,
        container_floor_z,
        container_inner_half_size,
        min_lift,
        warmup_steps,
        force_threshold,
    ).squeeze(-1).bool()

    if confirm_steps < 1:
        raise ValueError("confirm_steps must be positive.")
    update = env._pick_place_placed_step != env.common_step_counter
    steps = torch.where(placed, env._pick_place_placed_steps + 1, 0)
    env._pick_place_placed_steps[update] = steps[update]
    env._pick_place_placed_step[update] = env.common_step_counter
    return env._pick_place_placed_steps >= confirm_steps


def stage_signals(env):
    """Task-owned live measurements used by the generic episode tracer."""
    from sim.scene.geometry import gripper_tip_local_offset
    from sim.envs.mdp._common import _ee_body

    params = env.termination_manager.get_term_cfg("success").params
    observation = {key: value for key, value in params.items() if key != "confirm_steps"}
    grasp = {key: value for key, value in observation.items() if not key.startswith("container_")}
    grasped = object_grasped(env, **grasp).squeeze(-1).bool()
    placed = object_placed_in_container(env, **observation).squeeze(-1).bool()
    obj = env.scene[params["object_name"]]
    position = as_torch(obj.data.root_pos_w)
    robot = env.scene["robot"]
    ee_index, ee_path = _ee_body(env, robot)
    ee_position = as_torch(robot.data.body_link_pos_w)[:, ee_index]
    ee_quaternion = as_torch(robot.data.body_link_quat_w)[:, ee_index]
    if not hasattr(env, "_pick_place_grasp_offset"):
        half = float(obj.cfg.spawn.size[0]) / 2
        arm = env.cfg.arm_spec
        env._pick_place_grasp_offset = ee_position.new_tensor(gripper_tip_local_offset(
            get_current_stage(), half, 0.0, ee_path, arm.gripper.tip_mesh,
        ))
    point = ee_position + math_utils.quat_apply(ee_quaternion, env._pick_place_grasp_offset.expand_as(ee_position))
    return {"grasped": grasped, "placed": placed, "holding": env._pick_place_holding,
            "lifted": (position[:, 2] - env._pick_place_rest_z).double() >= params["min_lift"],
            "reach_distance": torch.linalg.vector_norm(point - position, dim=-1)}
