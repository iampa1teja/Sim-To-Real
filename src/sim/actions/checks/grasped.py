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

"""grasped(obj, force, lift): the jaw holds the object, with hysteresis.

Acquisition needs a firm lift: jaw contact above ``force`` newtons while ``lift`` reports the object
raised. Once grasped this episode, any remaining contact keeps it held; a weaker remaining contact is
not a release, and re-contact after an earlier grasp counts as holding again.

The stage is "holding and past the warmup window". ``holding`` and ``ever`` are the raw states that
``placed`` and the published ``holding`` signal use.
"""
from sim.actions.action import Tracker


class Grasped(Tracker):
    def __init__(self, obj, force=2.0, *, lift, warmup_steps=30):
        self.obj = obj
        self.force = force
        self.lift = lift
        self.warmup_steps = warmup_steps

    def children(self):
        return (self.lift,)

    def initial(self, frame):
        import torch

        holding = torch.zeros(frame.position[self.obj].shape[0], dtype=torch.bool,
                              device=frame.position[self.obj].device)
        return {"holding": holding, "ever": torch.zeros_like(holding)}

    def clear(self, state, ids, frame):
        state["holding"][ids] = False
        state["ever"][ids] = False

    def update(self, frame):
        """Return ``(holding, ever_grasped)`` after this step's update."""
        state = self.state(frame)
        update = self.due(frame, state)
        raised = self.lift.raised(frame)
        contact = frame.contact[self.obj]
        holding = (state["ever"] & (contact > 0.0)) | ((contact > self.force) & raised)
        state["holding"][update] = holding[update]
        state["ever"][update] |= holding[update]
        state["step"][update] = frame.step
        return state["holding"], state["ever"]

    def evaluate(self, frame):
        holding, _ = self.update(frame)
        return holding & ~(frame.episode_step < self.warmup_steps)

    def signals(self, frame, name):
        stage = self.evaluate(frame)
        holding, _ = self.update(frame)
        return {name: stage, "holding": holding.clone()}


def grasped(obj, force=2.0, *, lift, warmup_steps=30):
    return Grasped(obj, force, lift=lift, warmup_steps=warmup_steps)


class Carried(Tracker):
    """Track distinct objects lifted under firm contact, without holding hysteresis.

    ``reset_all`` preserves batch-cached legacy tracking: resetting some rows invalidates
    the measurement step for every row, while clearing history only for the reset rows.
    """
    def __init__(self, objects, *, min_lift, force, reset_all=False):
        self.objects, self.min_lift, self.force = tuple(objects), min_lift, force
        self.reset_all = reset_all

    def positions(self, frame):
        import torch
        return torch.stack([frame.position[name] for name in self.objects], 1)

    def initial(self, frame):
        import torch
        positions = self.positions(frame)
        ever = torch.zeros(positions.shape[:2], dtype=torch.bool, device=positions.device)
        return {"rest_z": positions[:, :, 2].clone(), "grasped": ever, "lifted": ever.clone()}

    def clear(self, state, ids, frame):
        state["rest_z"][ids] = self.positions(frame)[ids, :, 2]
        state["grasped"][ids] = False
        state["lifted"][ids] = False

    def update(self, frame):
        import torch
        state = self.state(frame)
        update = self.due(frame, state)
        z = self.positions(frame)[:, :, 2]
        state["rest_z"][update] = torch.minimum(state["rest_z"][update], z[update])
        lifted = z - state["rest_z"] >= self.min_lift
        contact = torch.stack([frame.contact[name] for name in self.objects], 1)
        held = (contact > self.force) & lifted
        state["grasped"][update] |= held[update]
        state["lifted"][update] |= (lifted & state["grasped"])[update]
        state["step"][update] = frame.step
        return held, state["grasped"], state["lifted"]
