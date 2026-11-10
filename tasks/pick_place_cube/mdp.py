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

import os

import numpy as np
import torch
from pxr import Gf, Usd, UsdShade

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim import get_current_stage

from sim.envs.mdp.resets import _env_prim_path, _wxyz, _xyzw, as_torch, env_indices


# ---------------------------------------------------------------------------
# Recorded-start resets and eval checks (from resets.py)
# ---------------------------------------------------------------------------


def _robot_states_root(starts_dir, reset_robot):
    """Recorded robot starts live in the dataset that owns <dataset>/pick_place_meta."""
    from pathlib import Path

    return str(Path(starts_dir).parent) if reset_robot and starts_dir else None


def reset_cube_from_recorded_starts(
    env, env_ids, asset_cfg, starts_dir, mode="cycle", random_fraction=0.0, seed=None, reset_robot=True,
    eval_set=None, splits=None, repeats=1, start_offset=0, selection="all",
):
    """Evaluation-only placement from base-link frame trajectory starts.

    With reset_robot, the arm is also put at the recorded first-frame state of the same
    episode (the mean recorded state for random starts), so the policy starts from the
    same initial conditions it was trained on.
    """
    from .eval import (
        RecordedStarts, dataset_state_to_radians, resting_pose, rotation_matrix,
        footprint, footprints_overlap, upright_orientation,
    )
    from . import TASK
    from core.arms.spec import load_arm_spec
    from sim.scene.gripper_geometry import table_plane

    arm = load_arm_spec()[0]

    env_ids = env_indices(env, env_ids)
    if not len(env_ids):
        return
    starts_dir = starts_dir or os.environ.get("PICK_PLACE_EVAL_STARTS")
    seed = seed if seed is not None else env.cfg.seed
    key = (starts_dir, mode, random_fraction, seed, bool(reset_robot))
    if eval_set:
        key += (eval_set, splits, repeats, start_offset, selection)
    if getattr(env, "_pick_place_starts_key", None) != key:
        if eval_set:
            from .benchmark import EvalSetStarts

            env._pick_place_starts = EvalSetStarts(eval_set, splits, repeats, seed, start_offset=start_offset,
                                                   selection=selection, arm=arm)
        else:
            env._pick_place_starts = RecordedStarts(starts_dir, mode, random_fraction, seed,
                                                    robot_states_root=_robot_states_root(starts_dir, reset_robot), arm=arm)
        env._pick_place_starts_key = key
    sampler = env._pick_place_starts
    asset, robot, box = env.scene[asset_cfg.name], env.scene["robot"], env.scene["white_box"]
    env.sim.forward()
    base_ids, _ = robot.find_bodies(arm.links.base)
    if len(base_ids) != 1:
        raise ValueError(f"Recorded starts require exactly one robot base body named {arm.links.base!r}")
    base_id = base_ids[0]
    size = asset.cfg.spawn.size
    stage = get_current_stage()
    body_pos = as_torch(robot.data.body_link_pos_w)
    body_quat = as_torch(robot.data.body_link_quat_w)
    box_pos = as_torch(box.data.root_pos_w)
    box_quat = as_torch(box.data.root_quat_w)
    if not hasattr(env, "_pick_place_start_index"):
        env._pick_place_start_index = [-1] * env.num_envs
        env._pick_place_start_zone = [None] * env.num_envs
        env._pick_place_cube_color = ["blue"] * env.num_envs
    positions, orientations, joint_starts = [], [], []
    for index in env_ids.tolist():
        base_p = body_pos[index, base_id].detach().cpu().numpy()
        # Geometry helpers and recorded starts are WXYZ; convert the live XYZW poses.
        base_q = _wxyz(body_quat[index, base_id].detach().cpu().numpy())
        plane = table_plane(stage, stage.GetPrimAtPath(env.scene.env_prim_paths[index]))
        box_p = box_pos[index].detach().cpu().numpy()
        box_q = _wxyz(box_quat[index].detach().cpu().numpy())
        box_polygon = footprint(box_p, rotation_matrix(box_q), tuple(TASK.spec.objects["white_box"].params["size_outer"]), bottom_origin=True)

        def accepts_random(p, q):
            q = upright_orientation(base_q, plane, q)
            world, rotation = resting_pose(p, q, base_p, base_q, plane, size)
            return not footprints_overlap(footprint(world, rotation, size), box_polygon)

        start, p, q, zone = sampler.sample(accepts_random)
        if start == -1 or (eval_set and sampler.current["split"] != "train"):
            q = upright_orientation(base_q, plane, q)
        world, rotation = resting_pose(p, q, base_p, base_q, plane, size)
        if start != -1 and footprints_overlap(footprint(world, rotation, size), box_polygon):
            raise ValueError(f"Recorded start {start} overlaps the live white box; verify the dataset and box pose")
        positions.append(world)
        # World orientation = base (XYZW) * base-relative start (WXYZ on disk -> XYZW).
        orientations.append(math_utils.quat_mul(
            body_quat[index, base_id],
            torch.as_tensor(_xyzw(q), device=env.device, dtype=body_quat.dtype),
        ))
        env._pick_place_start_index[index] = start
        env._pick_place_start_zone[index] = zone
        if eval_set:
            env._benchmark_start = dict(sampler.current)
        if reset_robot:
            start_mapping = getattr(env.cfg, "sim_joint_mapping", None)
            if eval_set:
                # Frozen benchmark arm units belong to their saved mapping.
                # Preserve the physical start when the active command mapping changes.
                start_mapping = sampler.data.get("scene", {}).get("joint_mapping", start_mapping)
            joint_starts.append(dataset_state_to_radians(sampler.robot_state(start), start_mapping, arm=arm))
    if reset_robot:
        joint_ids, _ = robot.find_joints(list(arm.usd_joint_names), preserve_order=True)
        joint_dtype = as_torch(robot.data.joint_pos).dtype
        joint_pos = torch.as_tensor(np.array(joint_starts), device=env.device, dtype=joint_dtype)
        robot.write_joint_state_to_sim_index(
            position=joint_pos, velocity=torch.zeros_like(joint_pos), joint_ids=joint_ids, env_ids=env_ids
        )
        # Hold the pose until the first policy action (otherwise the old targets pull the arm away).
        robot.set_joint_position_target_index(target=joint_pos, joint_ids=joint_ids, env_ids=env_ids)
    pose = torch.cat((torch.as_tensor(np.array(positions), device=env.device, dtype=box_pos.dtype),
                      torch.stack(orientations)), dim=-1)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids
    )
    reset_pick_place_state(env, env_ids, asset_cfg.name)
    env.scene["contact_grasp"].reset(env_ids)


def check_pick_place_eval_event_order(env, env_ids):
    """Check the actual manager's prepared reset order at environment creation."""
    from .eval import RecordedStarts

    params = env.event_manager.get_term_cfg("spawn_cube").params
    directory = params["starts_dir"] or os.environ.get("PICK_PLACE_EVAL_STARTS")
    seed = params["seed"] if params["seed"] is not None else env.cfg.seed
    reset_robot = bool(params.get("reset_robot", True))
    params["starts_dir"], params["seed"] = directory, seed
    if not params.get("eval_set"):
        env._pick_place_starts = RecordedStarts(directory, params["mode"], params["random_fraction"], seed,
                                                robot_states_root=_robot_states_root(directory, reset_robot))
        env._pick_place_starts_key = (directory, params["mode"], params["random_fraction"], seed, reset_robot)
    names = env.event_manager.active_terms["reset"]
    if not names.index("reset_robot_position") < names.index("spawn_cube") < names.index("reset_tracking"):
        raise RuntimeError(f"Invalid pick-place reset order: {names}")
    if "eval_robot_color" in names and not names.index("reset_set_robot_visual_material") < names.index(
        "eval_robot_color"
    ):
        raise RuntimeError(f"Robot colour DR would be overwritten: {names}")
    print(f"[Pick-place eval] Reset event order: {names}")


# ---------------------------------------------------------------------------
# Cube colour and room light DR
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


def randomize_pick_place_room_light(env, env_ids, exposure_range):
    """Randomize the exposure of the room's measured tube light."""
    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    set_pick_place_room_light(env, env_ids, exposure)
    return float(exposure)


def set_light_exposure_at(stage, prim_path, exposure):
    """Set ``inputs:exposure`` on one light prim if it exists."""
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        prim.GetAttribute("inputs:exposure").Set(exposure)


def set_pick_place_room_light(env, env_ids, exposure):
    """Set exposure on each measured room tube light that exists."""
    env_ids = env_indices(env, env_ids)
    stage = get_current_stage()
    room = env.scene["room"]
    for index in env_ids.tolist():
        set_light_exposure_at(stage, _env_prim_path(env, room, index) + "/Room/Lighting/Tubelight", exposure)


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
