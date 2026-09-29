"""Pure torch frame-transform helpers used by replay code.

Quaternions use wxyz order and support normal torch broadcasting.
"""

import torch


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_w, first_x, first_y, first_z = first.unbind(dim=-1)
    second_w, second_x, second_y, second_z = second.unbind(dim=-1)
    return torch.stack(
        (
            first_w * second_w - first_x * second_x - first_y * second_y - first_z * second_z,
            first_w * second_x + first_x * second_w + first_y * second_z - first_z * second_y,
            first_w * second_y - first_x * second_z + first_y * second_w + first_z * second_x,
            first_w * second_z + first_x * second_y - first_y * second_x + first_z * second_w,
        ),
        dim=-1,
    )


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return quat * quat.new_tensor((1.0, -1.0, -1.0, -1.0))


def _rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(vector.shape[:-1] + (1,), dtype=vector.dtype, device=vector.device)
    vector_quat = torch.cat((zeros, vector), dim=-1)
    return _quat_mul(_quat_mul(quat, vector_quat), _quat_conjugate(quat))[..., 1:]


def base_to_world(base_pos, base_quat, pos_b, quat_b):
    """Transform a pose from a base frame into the world frame."""
    return base_pos + _rotate(base_quat, pos_b), _quat_mul(base_quat, quat_b)


def world_to_base(base_pos, base_quat, pos_w, quat_w):
    """Transform a world-frame pose into the base frame."""
    inverse = _quat_conjugate(base_quat)
    return _rotate(inverse, pos_w - base_pos), _quat_mul(inverse, quat_w)
