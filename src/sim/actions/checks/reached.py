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

"""reached(a, b, dist): ``a`` is near ``b``.

- ``reached("gripper", obj, dist)``: the gripper grasp point is within ``dist`` metres of the object
  centre. The simulator publishes the distance as the ``reach_distance`` signal; core compares it
  (falling back to the end-effector position when a state has no distance).
- ``reached(obj, target, relation="over", held=grasp)``: while ``grasp`` reports holding, the object
  centre lies inside the convex hull of the target's footprint. Today this is computed from the
  core ``StepState`` poses only, so it publishes no signal and has no simulator evaluation.
"""
import numpy as np

from sim.actions.action import Check, object_size


class Reached(Check):
    def __init__(self, a, b, dist=None, *, relation="near", held=None, signal="reach_distance"):
        if (a == "gripper") != (relation == "near"):
            raise ValueError("reached supports ('gripper', obj, dist) and (obj, target, relation='over')")
        if relation == "near" and dist is None:
            raise ValueError("reached('gripper', obj) needs dist")
        self.a, self.b, self.dist = a, b, dist
        self.relation, self.held, self.signal = relation, held, signal
        self.in_sim = relation == "near"

    def children(self):
        return () if self.held is None else (self.held,)

    # Simulator side ------------------------------------------------------------------------------

    def distance(self, frame):
        import torch

        if frame.gripper_point is None:
            raise ValueError("reached('gripper', ...) needs a frame with the gripper point")
        return torch.linalg.vector_norm(frame.gripper_point - frame.position[self.b], dim=-1)

    def evaluate(self, frame):
        if self.relation != "near":
            return super().evaluate(frame)
        return self.distance(frame) <= self.dist

    def signals(self, frame, name):
        return {self.signal: self.distance(frame)} if self.relation == "near" else {}

    # Core side -----------------------------------------------------------------------------------

    def observe(self, state, name, objects):
        if self.relation == "near":
            target = np.asarray(state.objects[self.b].position)
            fallback = np.linalg.norm(np.asarray(state.ee.position) - target)
            return float(state.signals.get(self.signal, fallback)) <= self.dist
        if self.held is not None and not bool(state.signals.get("holding", False)):
            return False
        return self._over(np.asarray(state.objects[self.a].position), state.objects[self.b], objects[self.b])

    @staticmethod
    def _over(point, pose, spec):
        from utils.geometry import footprint, hull, inside_polygon, rotation_matrix_wxyz

        qx, qy, qz, qw = pose.quat_xyzw
        polygon = footprint(pose.position, rotation_matrix_wxyz((qw, qx, qy, qz)), object_size(spec),
                            bottom_origin=bool(spec.params.get("bottom_origin", False)))
        return inside_polygon(point[:2], hull(polygon))


def reached(a, b, dist=None, *, relation="near", held=None, signal="reach_distance"):
    return Reached(a, b, dist, relation=relation, held=held, signal=signal)
