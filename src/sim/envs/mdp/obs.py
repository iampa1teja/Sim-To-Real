import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def as_torch(x):
    """Return a torch view of 3.0 data (a ProxyArray with a zero-copy ``.torch``); plain tensors pass through."""
    return x if isinstance(x, torch.Tensor) else x.torch


def ee_frame_state(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector pose in the base-link frame: pos (3) + quat xyzw (4).

    ``robot_cfg.body_names`` must be ``[base_link, ee_link]`` with ``preserve_order=True``.
    Replaces v1's FrameTransformer sensor; computed from body poses so it works on Newton.
    """
    robot = env.scene[robot_cfg.name]
    base_id, ee_id = robot_cfg.body_ids
    pos = as_torch(robot.data.body_link_pos_w)
    quat = as_torch(robot.data.body_link_quat_w)
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(
        pos[:, base_id], quat[:, base_id], pos[:, ee_id], quat[:, ee_id]
    )
    return torch.cat([rel_pos, rel_quat], dim=-1)


def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """Unprocessed camera output for one data type (depth, segmentation)."""
    sensor = env.scene[sensor_cfg.name]
    return as_torch(sensor.data.output[data_type]).clone()


def image_rgb_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """RGB without normalisation (v1's ``image(normalize=False)``), with the camera's fitted colour response.

    3.0 replaced ``mdp.image`` with the ``image_rgb`` class; this keeps v1's output and
    applies ``CalibratedTiledCamera.apply_response`` when the sensor has one.
    """
    sensor = env.scene[sensor_cfg.name]
    images = as_torch(sensor.data.output[data_type])[..., :3].clone()
    apply_response = getattr(sensor, "apply_response", None)
    return apply_response(images) if apply_response is not None else images
