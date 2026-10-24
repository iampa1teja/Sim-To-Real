"""Workshop training correction, registered so saved LeRobot processors reload."""
from dataclasses import dataclass

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ProcessorStepRegistry
from lerobot.policies.molmoact2 import processor_molmoact2 as upstream


@ProcessorStepRegistry.register(name="workshop_molmo_joint_frame")
@dataclass
class WorkshopMolmoJointFrame(upstream.MolmoAct2StateFrameTransformStep):
    """Apply the same robot-to-model frame to states and supervised actions."""

    def __call__(self, transition):
        transition = super().__call__(transition)
        action = transition.get(TransitionKey.ACTION)
        if action is not None and self.joint_signs is not None:
            transition = transition.copy()
            action = torch.as_tensor(action, dtype=torch.float32).clone()
            n = len(self.joint_signs)
            signs = torch.tensor(self.joint_signs, device=action.device)
            offsets = torch.tensor(self.joint_offsets, device=action.device)
            action[..., :n] = action[..., :n] * signs + offsets
            transition[TransitionKey.ACTION] = action
        return transition


def install_training_correction():
    """Scope the upstream factory correction to this training entrypoint."""
    original = upstream.make_molmoact2_pre_post_processors

    def corrected(config, *args, **kwargs):
        if not config.norm_tag or not config.norm_stats_path:
            raise ValueError("Workshop training requires the prepared model-frame quantiles")
        pre, post = original(config, *args, **kwargs)
        for index, step in enumerate(pre.steps):
            if type(step) is upstream.MolmoAct2StateFrameTransformStep:
                pre.steps[index] = WorkshopMolmoJointFrame(
                    joint_signs=config.joint_signs, joint_offsets=config.joint_offsets)
        return pre, post

    upstream.make_molmoact2_pre_post_processors = corrected
