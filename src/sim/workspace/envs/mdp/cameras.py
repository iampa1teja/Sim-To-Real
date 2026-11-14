"""Simulator cameras events."""
import torch
from pxr import Gf, Sdf
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

_default_poses_cache = {}


def get_camera_base_pose(env, prim_path_pattern):
    """Read and cache the original camera pose (quat XYZW) for an offset setter."""
    if prim_path_pattern not in _default_poses_cache:
        prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        prims = sim_utils.find_matching_prims(prim_path)
        if not prims:
            return None
        prim = prims[0]
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        default_pos = (0.0, 0.0, 0.0)
        default_quat = (0.0, 0.0, 0.0, 1.0)
        if translate_attr.IsValid() and translate_attr.Get() is not None:
            value = translate_attr.Get()
            default_pos = (value[0], value[1], value[2])
        if orient_attr.IsValid() and orient_attr.Get() is not None:
            value = orient_attr.Get()
            default_quat = (*value.GetImaginary(), value.GetReal())
        _default_poses_cache[prim_path_pattern] = {"pos": default_pos, "quat": default_quat}
    return _default_poses_cache[prim_path_pattern]


def set_camera_pose_offset(env, prim_path_pattern, pos_offset: dict, rot_offset: dict):
    """Apply a position and Euler-angle offset to the camera's base pose."""
    prim_path = prim_path_pattern.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    prims = sim_utils.find_matching_prims(prim_path)
    base_pose = get_camera_base_pose(env, prim_path_pattern)
    if not prims or base_pose is None:
        return
    base_pos = base_pose["pos"]
    position = tuple(base_pos[index] + pos_offset.get(axis, 0.0) for index, axis in enumerate(("x", "y", "z")))
    delta_quat = math_utils.quat_from_euler_xyz(
        torch.tensor([rot_offset.get("roll", 0.0)]),
        torch.tensor([rot_offset.get("pitch", 0.0)]),
        torch.tensor([rot_offset.get("yaw", 0.0)]),
    )[0]
    x, y, z, w = math_utils.quat_mul(torch.tensor([base_pose["quat"]]), delta_quat.unsqueeze(0))[0].tolist()
    with Sdf.ChangeBlock():
        for prim in prims:
            if prim.IsValid():
                translate_attr = prim.GetAttribute("xformOp:translate")
                if translate_attr.IsValid():
                    translate_attr.Set(Gf.Vec3d(*position))
                orient_attr = prim.GetAttribute("xformOp:orient")
                if orient_attr.IsValid():
                    # Gf takes the real part first. Match the attribute's precision (Quatf or Quatd).
                    quat_type = Gf.Quatf if isinstance(orient_attr.Get(), Gf.Quatf) else Gf.Quatd
                    orient_attr.Set(quat_type(w, x, y, z))


def set_camera_focal_length(env, asset_cfg: SceneEntityCfg, focal_length):
    """Set one focal length on every matching camera prim."""
    camera = env.scene[asset_cfg.name]
    camera_prim_path = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    with Sdf.ChangeBlock():
        for prim in sim_utils.find_matching_prims(camera_prim_path):
            if prim.IsValid():
                focal_attr = prim.GetAttribute("focalLength")
                if focal_attr.IsValid():
                    focal_attr.Set(focal_length)
