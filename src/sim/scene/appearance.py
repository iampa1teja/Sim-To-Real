"""Explicit scene lighting and optional arm appearance; no selected-scene globals."""

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def scene_content_root(asset_root, room_root=""):
    """Resolve a declared content root beneath the imported room asset."""
    return str(Sdf.Path(str(asset_root)).AppendPath(room_root)) if room_root else str(asset_root)


def _light_path(room_root, lighting, name):
    return scene_content_root(scene_content_root(room_root, lighting.get("root", "Lighting")), name)


def apply_scene_lighting(stage, room_root, lighting):
    """Author declared lights, preserving their names, intensity and USD transforms."""
    if not lighting:
        return
    if lighting.get("replace", True):
        room = stage.GetPrimAtPath(room_root)
        for prim in reversed(list(Usd.PrimRange(room))):
            if prim.HasAPI(UsdLux.LightAPI):
                stage.RemovePrim(prim.GetPath())
    lights = lighting.get("lights", {})
    if not isinstance(lights, dict):
        raise ValueError("scene.lighting.lights must contain named light tables")
    stage.DefinePrim(scene_content_root(room_root, lighting.get("root", "Lighting")), "Xform")
    for name, values in lights.items():
        path = _light_path(room_root, lighting, name)
        kind = values.get("type", "rect").lower()
        if kind in ("rect", "rectlight"):
            light = UsdLux.RectLight.Define(stage, path)
            light.CreateWidthAttr(values["width"])
            light.CreateHeightAttr(values["height"])
        elif kind in ("dome", "domelight"):
            light = UsdLux.DomeLight.Define(stage, path)
        elif kind in ("distant", "distantlight"):
            light = UsdLux.DistantLight.Define(stage, path)
            if "angle" in values:
                light.CreateAngleAttr(values["angle"])
        elif kind in ("sphere", "spherelight"):
            light = UsdLux.SphereLight.Define(stage, path)
            light.CreateRadiusAttr(values.get("radius", 0.5))
        else:
            raise ValueError(f"Unsupported scene light type {values.get('type')!r}")
        light.CreateIntensityAttr(values["intensity"])
        light.CreateNormalizeAttr(values.get("normalize", False))
        light.CreateColorAttr(Gf.Vec3f(*values.get("color", (1, 1, 1))))
        light.CreateEnableColorTemperatureAttr(values.get("enable_color_temperature", False))
        if "color_temperature" in values:
            light.CreateColorTemperatureAttr(values["color_temperature"])
        if "exposure" in values:
            light.CreateExposureAttr(values["exposure"])
        if "position" in values and "target" in values:
            transform = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*values["position"]), Gf.Vec3d(*values["target"]),
                Gf.Vec3d(*values.get("up", (0, 0, 1))),
            ).GetInverse()
            xform = UsdGeom.Xformable(light)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(transform)
        elif "euler_degrees" in values:
            from sim.scene.cameras import euler_xyz_degrees_to_xyzw

            x, y, z, w = euler_xyz_degrees_to_xyzw(values["euler_degrees"])
            xform = UsdGeom.Xformable(light)
            xform.ClearXformOpOrder()
            if "position" in values:
                xform.AddTranslateOp().Set(Gf.Vec3d(*values["position"]))
            xform.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def apply_robot_appearance(stage, robot_root, arm, look):
    """Delegate an explicitly supplied fitted look to the arm-folder hook."""
    from utils.arms.spec import load_arm_utils

    if not look:
        return
    look = look.get("appearance", look)
    if not look:
        return
    hook = getattr(load_arm_utils(arm), "apply_appearance", None)
    if hook is not None:
        hook(stage, robot_root, {**look, **arm.appearance})


def set_scene_light_exposure(env, env_ids, exposure):
    """Set exposure only on lights marked exposure_dr in this environment's scene."""
    from sim.envs.mdp._common import _env_prim_path, env_indices

    if not math.isfinite(float(exposure)):
        raise ValueError("Light exposure must be finite")
    scene = env.cfg.scene_spec
    lighting = scene.lighting
    selected = [name for name, light in lighting.get("lights", {}).items() if light.get("exposure_dr", False)]
    if not selected:
        return
    room = env.scene["room"]
    for index in env_indices(env, env_ids).tolist():
        root = scene_content_root(_env_prim_path(env, room, index), scene.room_root)
        for name in selected:
            path = _light_path(root, lighting, name)
            prim = env.sim.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError(f"Declared exposure-DR light not found: {path}")
            prim.GetAttribute("inputs:exposure").Set(float(exposure))


def randomize_scene_light_exposure(env, env_ids, exposure_range):
    """Draw one shared exposure, matching the original reset randomization cadence."""
    import isaaclab.utils.math as math_utils

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()
    set_scene_light_exposure(env, env_ids, exposure)
    return float(exposure)
