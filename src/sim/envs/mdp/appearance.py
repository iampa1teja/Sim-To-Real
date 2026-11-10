"""Simulator appearance events."""
import torch
from pxr import Gf, Sdf, UsdShade
import isaaclab.sim as sim_utils
from ._common import _arm_spec

ROBOT_COLORS = {
    "orange": (0.876, 0.317, 0.132),
    "beige": (0.960784, 0.960784, 0.862745),
    "teal": (0.0, 0.8, 0.502),
    "white": (0.95, 0.95, 0.95),
    "black": (0.08, 0.08, 0.08),
}


def randomize_robot_color(env, env_ids, color_names: list[str] = list(ROBOT_COLORS.keys())):
    """Randomly set robot color from predefined palette on each reset."""
    idx = torch.randint(0, len(color_names), (1,), device="cpu").item()
    selected_name = color_names[idx]
    set_robot_color(env, ROBOT_COLORS[selected_name])
    return selected_name


def set_robot_color(env, rgb: tuple[float, float, float]):
    """Set color of the arm to a desired RGB value (USD material edit, independent of physics backend)."""
    with Sdf.ChangeBlock():
        robot = env.scene["robot"]
        material_path = _arm_spec(env).appearance.get("body_material")
        if not material_path:
            return
        material_prim_path = robot.cfg.prim_path + "/" + material_path.lstrip("/")
        for material_prim in sim_utils.find_matching_prims(material_prim_path):
            shader = UsdShade.Shader(material_prim)
            name = "diffuseColor" if shader.GetIdAttr().Get() == "UsdPreviewSurface" else "diffuse_color_constant"
            shader.GetInput(name).Set(Gf.Vec3f(*rgb))
