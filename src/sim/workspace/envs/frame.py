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

"""Build the per-step ``Frame`` that task checks read; the only place they touch live Isaac data.

Built afresh on every call: an environment reset inside a step writes new object poses, and the
reset environments must be evaluated again on them (state updates stay once per step per env).
The module top imports nothing from Isaac or this package, so it can be loaded on its own.
"""
from sim.actions.action import Frame, object_size

CONTACT_SENSOR = "contact_grasp"


def contact_objects(spec):
    """Objects filtered by the jaw contact sensor, in filter order: the task's start object."""
    return (spec.start.get("object", spec.tracked_objects[0]),)


def _tensor(value):
    import torch

    return value if isinstance(value, torch.Tensor) else value.torch


def _grasp_offset(env, spec, ee_position, ee_path):
    """Grasp point in the end-effector frame, cached on the env (None: the end-effector origin)."""
    if not hasattr(env, "_task_grasp_offset"):
        arm = env.cfg.arm_spec
        gripper = arm.gripper
        if gripper is None:
            offset = None
        elif gripper.kind == "parallel" or not gripper.tip_mesh:
            offset = arm.ik.position_offset_m
        else:
            from isaaclab.sim import get_current_stage
            from sim.workspace.scene.geometry import gripper_tip_local_offset

            half = object_size(spec.objects[contact_objects(spec)[0]])[0] / 2
            offset = gripper_tip_local_offset(get_current_stage(), half, 0.0, ee_path, gripper.tip_mesh)
        env._task_grasp_offset = None if offset is None else ee_position.new_tensor(offset)
    return env._task_grasp_offset


def gripper_point(env, spec):
    import isaaclab.utils.math as math_utils
    from sim.workspace.envs.mdp._common import _ee_body

    robot = env.scene["robot"]
    ee_index, ee_path = _ee_body(env, robot)
    ee_position = _tensor(robot.data.body_link_pos_w)[:, ee_index]
    ee_quaternion = _tensor(robot.data.body_link_quat_w)[:, ee_index]
    offset = _grasp_offset(env, spec, ee_position, ee_path)
    if offset is None:
        return ee_position
    return ee_position + math_utils.quat_apply(ee_quaternion, offset.expand_as(ee_position))


def build_frame(env, task, *, gripper=True):
    """Frame for ``task`` in ``env``; ``gripper=False`` leaves out the grasp point (reset events)."""
    import torch

    spec = task.spec
    data = {name: env.scene[name].data for name in spec.objects}
    contact = {}
    if CONTACT_SENSOR in env.scene.keys():
        # force_matrix_w is (num_envs, num_bodies, num_filters) of vec3 -> torch (..., 3).
        force = torch.linalg.vector_norm(_tensor(env.scene[CONTACT_SENSOR].data.force_matrix_w), dim=-1).sum(dim=1)
        contact = {name: force[:, index] for index, name in enumerate(contact_objects(spec))}
    return Frame(
        position={name: _tensor(value.root_pos_w) for name, value in data.items()},
        orientation={name: _tensor(value.root_quat_w) for name, value in data.items()},
        contact=contact,
        step=env.common_step_counter,
        episode_step=env.episode_length_buf,
        objects=spec.objects,
        owner=env,
        # Read only when a check needs them (reach distance, on-top stability).
        gripper=(lambda: gripper_point(env, spec)) if gripper else None,
        velocities=lambda: {name: _tensor(value.root_vel_w) for name, value in data.items()},
    )
