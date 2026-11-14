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

"""placed(obj, target, relation): the grasped object was released inside or on top of the target.

The object must have been grasped earlier this episode, be in the target relation now, no longer be
held, and the episode must be past its warmup window. It reports the current state each step; wrap
it in ``hold(...)`` to require it for several consecutive steps.

Relations (target frame from its live XYZW pose):
- ``inside``: the oriented object box clears the measured inner walls (``inner_half_size``), its
  centre is at or above ``floor_z`` and it fits below the rim (``size_outer`` height). The target
  origin is the centre of its outer bottom face.
- ``on_top``: the object footprint lies within the target's top face plus ``xy_tol``, its bottom is
  within ``z_tol`` of the target top, and it moves slower than ``max_speed`` (m/s).
"""
from sim.actions.action import Check, object_size


class Placed(Check):
    def __init__(self, obj, target, relation="inside", *, grasp, warmup_steps=30,
                 xy_tol=None, z_tol=None, max_speed=None):
        if relation not in ("inside", "on_top"):
            raise ValueError(f"placed relation must be 'inside' or 'on_top', got {relation!r}")
        if relation == "on_top" and None in (xy_tol, z_tol, max_speed):
            raise ValueError("placed(relation='on_top') needs xy_tol, z_tol and max_speed")
        self.obj, self.target, self.relation = obj, target, relation
        self.grasp = grasp
        self.warmup_steps = warmup_steps
        self.xy_tol, self.z_tol, self.max_speed = xy_tol, z_tol, max_speed

    def children(self):
        return (self.grasp,)

    def _relative(self, frame):
        import isaaclab.utils.math as math_utils
        import torch

        target_pos, target_quat = frame.position[self.target], frame.orientation[self.target]
        inverse = math_utils.quat_inv(target_quat)
        rel = math_utils.quat_apply(inverse, frame.position[self.obj] - target_pos)
        rotation = math_utils.matrix_from_quat(math_utils.quat_mul(inverse, frame.orientation[self.obj]))
        half_size = rel.new_tensor(object_size(frame.objects[self.obj])) / 2
        return rel, torch.matmul(rotation.abs(), half_size)

    def _inside(self, frame):
        import torch

        params = frame.objects[self.target].params
        size, floor_z = params["size_outer"], params["floor_z"]
        rel, extent = self._relative(frame)
        inner = rel.new_tensor(tuple(params["inner_half_size"]))
        within_xy = torch.all(rel[:, :2].abs() + extent[:, :2] <= inner, dim=-1)
        # Use the centre at the floor so small contact-solver penetrations do not
        # reject resting cubes. The entire cube must still clear the inner walls
        # and fit below the rim (with only floating-point tolerance at the rim).
        tolerance = 8 * torch.finfo(rel.dtype).eps
        within_z = (rel[:, 2] >= floor_z) & (rel[:, 2] + extent[:, 2] <= size[2] + tolerance)
        return within_xy & within_z

    def _on_top(self, frame):
        import torch

        rel, extent = self._relative(frame)
        target_half = rel.new_tensor(object_size(frame.objects[self.target])) / 2
        aligned = torch.all(rel[:, :2].abs() + extent[:, :2] <= target_half[:2] + self.xy_tol, dim=-1)
        on_top = (rel[:, 2] - target_half[2] - extent[:, 2]).abs() <= self.z_tol
        stable = torch.linalg.vector_norm(frame.velocity[self.obj][:, :3], dim=-1) <= self.max_speed
        return aligned & on_top & stable

    def evaluate(self, frame):
        holding, ever = self.grasp.update(frame)
        relation = self._inside(frame) if self.relation == "inside" else self._on_top(frame)
        return ever & relation & ~holding & ~(frame.episode_step < self.warmup_steps)


def placed(obj, target, relation="inside", *, grasp, warmup_steps=30, xy_tol=None, z_tol=None, max_speed=None):
    return Placed(obj, target, relation, grasp=grasp, warmup_steps=warmup_steps,
                  xy_tol=xy_tol, z_tol=z_tol, max_speed=max_speed)
