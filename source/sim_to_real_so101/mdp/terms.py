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
import torch



import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg



def _ensure_pick_place_state(env, object_name):
    """Allocate episode trackers without changing already initialized environments."""
    if not hasattr(env, "_pick_place_holding"):
        env._pick_place_holding = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._pick_place_ever_grasped = torch.zeros_like(env._pick_place_holding)
        env._pick_place_rest_z = env.scene[object_name].data.root_pos_w[:, 2].clone()
        env._pick_place_placed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._pick_place_grasp_step = torch.full_like(env._pick_place_placed_steps, -1)
        env._pick_place_placed_step = torch.full_like(env._pick_place_placed_steps, -1)


def reset_pick_place_state(env, env_ids, object_name="blue_cube"):
    """Clear only the reset environments, after writing their new object poses."""
    _ensure_pick_place_state(env, object_name)
    ids = slice(None) if env_ids is None else env_ids
    env._pick_place_holding[ids] = False
    env._pick_place_ever_grasped[ids] = False
    env._pick_place_rest_z[ids] = env.scene[object_name].data.root_pos_w[ids, 2]
    env._pick_place_placed_steps[ids] = 0
    env._pick_place_grasp_step[ids] = -1
    env._pick_place_placed_step[ids] = -1


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
    absolute world Z: the MyRoom table is ~0.76 m up, so a world-Z threshold
    would read "lifted" while the cube is still on the table.
    """
    _ensure_pick_place_state(env, object_name)
    obj: RigidObject = env.scene[object_name]
    z = obj.data.root_pos_w[:, 2]
    update = env._pick_place_grasp_step != env.common_step_counter
    env._pick_place_rest_z[update] = torch.minimum(env._pick_place_rest_z[update], z[update])

    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    # force_matrix_w is (num_envs, num_bodies, num_filters, 3); the sole filter is the object.
    contact_force = torch.linalg.vector_norm(contact_sensor.data.force_matrix_w, dim=-1).sum(dim=1)[:, 0]
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

    The box origin is the centre of its outer bottom face. Its live wxyz
    orientation transforms both the object centre and object axes into box space.
    """
    inverse = math_utils.quat_inv(container.data.root_quat_w)
    rel = math_utils.quat_apply(inverse, obj.data.root_pos_w - container.data.root_pos_w)
    rotation = math_utils.matrix_from_quat(math_utils.quat_mul(inverse, obj.data.root_quat_w))
    half_size = rel.new_tensor(obj.cfg.spawn.size) / 2
    extent = torch.matmul(rotation.abs(), half_size)
    inner = rel.new_tensor(container_inner_half_size)
    within_xy = torch.all(rel[:, :2].abs() + extent[:, :2] <= inner, dim=-1)
    # Use the centre at the floor so small contact-solver penetrations do not
    # reject resting cubes. The entire cube must still clear the inner walls
    # and fit below the rim (with only floating-point tolerance at the rim).
    tolerance = 8 * torch.finfo(rel.dtype).eps
    within_z = ((rel[:, 2] >= container_floor_z)
                & (rel[:, 2] + extent[:, 2] <= container_size[2] + tolerance))
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
