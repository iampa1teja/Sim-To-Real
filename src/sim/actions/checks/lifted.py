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

"""lifted(obj, min_lift): the object has risen above its lowest height since reset.

Lift is measured from the object's lowest height since reset, not an absolute world Z: the room's
table is ~0.76 m up, so a world-Z threshold would read "lifted" while the cube is still on the table.

Two comparisons are used today and both are kept: the stage itself compares in float64 with ``>=``
(``evaluate``), while the grasp hysteresis in ``grasped`` uses the strict float comparison ``>``
(``raised``).
"""
from sim.actions.action import Tracker


class Lifted(Tracker):
    def __init__(self, obj, min_lift=0.01):
        self.obj = obj
        self.min_lift = min_lift

    def initial(self, frame):
        return {"rest_z": frame.position[self.obj][:, 2].clone()}

    def clear(self, state, ids, frame):
        state["rest_z"][ids] = frame.position[self.obj][ids, 2]

    def height(self, frame):
        """Height above the lowest point since reset, after this step's update."""
        import torch

        state = self.state(frame)
        z = frame.position[self.obj][:, 2]
        update = self.due(frame, state)
        state["rest_z"][update] = torch.minimum(state["rest_z"][update], z[update])
        state["step"][update] = frame.step
        return z - state["rest_z"]

    def raised(self, frame):
        return self.height(frame) > self.min_lift

    def evaluate(self, frame):
        return self.height(frame).double() >= self.min_lift


def lifted(obj, min_lift=0.01):
    return Lifted(obj, min_lift)
