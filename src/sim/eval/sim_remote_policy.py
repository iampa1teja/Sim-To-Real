"""Drive the simulated arm from a served policy (v1 lerobot_interface.SimRemotePolicy, split in two).

core.policy_client.remote_policy.RemotePolicy does the robot-agnostic work in raw LeRobot units; this
wrapper converts the sim side: joint radians <-> raw units through the arm's JointMapping, and Isaac
``rgb_<camera>`` observations to policy camera names.
"""
from __future__ import annotations

import numpy as np
import torch

from core.policy_client.remote_policy import RemotePolicy


class SimRemotePolicy(RemotePolicy):
    def __init__(self, mapping, cameras, rename_map=None, **kwargs):
        """mapping: core.arms.mapping.JointMapping; cameras: sim camera names; rename_map: sim -> policy name."""
        self.mapping = mapping
        self.cameras = list(cameras)
        self.rename_map = dict(rename_map or {})
        super().__init__([f"{name}.pos" for name in mapping.names], mapping=mapping, **kwargs)

    def _sim_inputs(self, joint_positions, visual_obs) -> dict:
        state = self.mapping.to_raw(joint_positions).detach().cpu().numpy().astype(np.float32)
        images = {self.rename_map.get(camera, camera): visual_obs[f"rgb_{camera}"][0].cpu().numpy()
                  for camera in self.cameras}
        return {"state": state, "images": images, "instruction": self.lang_instruction}

    def get_action(self, joint_positions: torch.Tensor, visual_obs: dict, log: bool = False) -> torch.Tensor:
        """Next sim action in radians for env.step(); joint_positions are the current sim radians (J,)."""
        measured = None
        if self.grasp is not None:
            measured = float(self.mapping.to_raw(joint_positions)[self.grasp.gripper_index].item())
        action = self.next_raw_action(lambda: self._sim_inputs(joint_positions, visual_obs), measured)
        raw = torch.tensor([action[key] for key in self.joint_keys], dtype=self.mapping.mins.dtype,
                           device=self.mapping.mins.device)
        if log:
            from sim.teleop.rerun_logging import log_rerun_data

            inputs = self._sim_inputs(joint_positions, visual_obs)
            observation = dict(inputs["images"])
            observation.update({key: float(value) for key, value in zip(self.joint_keys, inputs["state"])})
            log_rerun_data(observation=observation, action=action)
        return self.mapping.to_sim(raw)
