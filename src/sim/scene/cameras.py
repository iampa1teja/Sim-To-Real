"""Generic calibrated scene and arm-mounted cameras.

The USD OpenCV model uses the measured fx/fy and a +0.5 pixel-centre offset in
cx/cy for RTX sampling. Colour response is applied to observation copies.
"""
from dataclasses import asdict, is_dataclass
import math

from pxr import Gf, Sdf

from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.spawners.sensors import spawn_camera
from isaaclab.utils import configclass


def apply_opencv_intrinsics(prim, intrinsics):
    """Use RTX's native OpenCV model; USD aperture offsets alone are ignored."""
    # Isaac Lab's USD registry may omit this renderer schema. Author its
    # schema token and typed properties explicitly;
    # this is the same USD representation used by the native camera API.
    prim.AddAppliedSchema("OmniLensDistortionOpenCvPinholeAPI")
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole:"
    # RTX samples at half-integer raster coordinates; RealSense/OpenCV uses
    # integer pixel centers. Offset the lens center, not the measured K.
    prim.CreateAttribute("calibration:pixelCenterOffset", Sdf.ValueTypeNames.Float).Set(0.5)
    for key in ("fx", "fy", "cx", "cy"):
        value = float(intrinsics[key]) + (0.5 if key in ("cx", "cy") else 0.0)
        prim.CreateAttribute(prefix + key, Sdf.ValueTypeNames.Float).Set(value)
    prim.CreateAttribute(prefix + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(intrinsics["width"], intrinsics["height"]))
    # The connected D435I color stream reports zero distortion. Do not silently
    # reinterpret inverse Brown-Conrady coefficients as the forward OpenCV model.
    coefficients = intrinsics.get("coeffs", [0.0] * 5)
    if any(coefficients) and intrinsics.get("model") == "distortion.inverse_brown_conrady":
        raise ValueError("Nonzero inverse Brown-Conrady distortion requires conversion before use.")
    for index, key in enumerate(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")):
        prim.CreateAttribute(prefix + key, Sdf.ValueTypeNames.Float).Set(float(coefficients[index]) if index < len(coefficients) else 0.0)



@configclass
class CalibratedPinholeCameraCfg(PinholeCameraCfg):
    # fx, fy, cx, cy in pixels; width/height describe their calibration resolution.
    intrinsics: dict | None = None
    display_name: str | None = None
    # Per-camera residual colour response (see sim/scene/camera_response.py); None = raw render.
    response: dict | None = None


class CalibratedTiledCamera(TiledCamera):
    """TiledCamera that can apply the camera's fitted colour response to its RGB output."""

    def apply_response(self, images):
        """Apply the fitted response to an observation copy, never RTX buffers."""
        params = getattr(self.cfg.spawn, "response", None)
        if not params:
            return images
        if getattr(self, "_response", None) is None:
            from sim.scene.camera_response import CameraResponse
            self._response = CameraResponse(params, self.cfg.height, self.cfg.width, self.device)
        images[..., :3] = self._response(images[..., :3])
        return images


def quaternion_tuple(rotation):
    """Return XYZW for Isaac; USD/Gf itself stores the real component first."""
    quaternion = rotation.GetQuat()
    return (*quaternion.GetImaginary(), quaternion.GetReal())


def euler_xyz_degrees_to_xyzw(angles):
    x, y, z = angles
    rotation = (Gf.Rotation(Gf.Vec3d(1, 0, 0), x)
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), y)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), z))
    return quaternion_tuple(rotation)


def _values(spec):
    return asdict(spec) if is_dataclass(spec) else dict(spec)


def camera_rotation(spec):
    """Use explicit Euler calibration or a declared target/up camera frame."""
    values = _values(spec)
    if values.get("euler_degrees") is not None:
        return euler_xyz_degrees_to_xyzw(values["euler_degrees"])
    position, target = Gf.Vec3d(*values["position"]), Gf.Vec3d(*values["target"])
    up = Gf.Vec3d(*(values.get("up") or (0, 0, 1)))
    if (target - position).GetLength() < 1e-12 or Gf.Cross(target - position, up).GetLength() < 1e-12:
        raise ValueError("Camera target and up must define a non-degenerate camera frame")
    transform = Gf.Matrix4d().SetLookAt(position, target, up).GetInverse()
    return quaternion_tuple(transform.ExtractRotation())


def _spawn_named_camera(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = spawn_camera(
        prim_path, cfg.replace(intrinsics=None, response=None, display_name=None),
        translation=translation, orientation=orientation, **kwargs,
    )
    if cfg.intrinsics is not None:
        apply_opencv_intrinsics(prim, cfg.intrinsics)
    prim.SetDisplayName(cfg.display_name or prim.GetName())
    return prim


def build_camera_cfg(path, spec, renderer_cfg, *, data_types=None, response=None, rotation=None, display_name=None):
    """Build any fixed or arm camera from explicit manifest data."""
    values = _values(spec)
    calibration = values.get("calibration")
    metadata = values.get("metadata") or {}
    return TiledCameraCfg(
        class_type=CalibratedTiledCamera,
        prim_path=path,
        update_period=0.0,
        height=values["height"],
        width=values["width"],
        data_types=list(values.get("data_types", ("rgb",)) if data_types is None else data_types),
        renderer_cfg=renderer_cfg,
        spawn=CalibratedPinholeCameraCfg(
            func=_spawn_named_camera,
            intrinsics=({**calibration, "width": values["width"], "height": values["height"]}
                        if calibration else None),
            response=response or values.get("response") or None,
            display_name=display_name or values.get("display_name") or metadata.get("display_name"),
            focal_length=values["focal_length"],
            horizontal_aperture=values["horizontal_aperture"],
            vertical_aperture=values["vertical_aperture"],
            clipping_range=tuple(values.get("clipping_range", (0.01, 100.0))),
            f_stop=values.get("f_stop", 0.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(values["position"]), rot=rotation if rotation is not None else camera_rotation(values),
            convention="opengl",
        ),
    )


def sync_camera_intrinsics(env, env_ids):
    """Make changed camera focal lengths affect each authored OpenCV lens model."""
    import isaaclab.sim as sim_utils
    from sim.envs.mdp._common import env_indices

    indices = env_indices(env, env_ids).tolist()
    if not indices:
        return
    env_paths = list(env.scene.env_prim_paths)
    selected_paths = [Sdf.Path(env_paths[index]) for index in indices]
    all_paths = [Sdf.Path(path) for path in env_paths]
    for camera in env.scene.sensors.values():
        spawn = getattr(getattr(camera, "cfg", None), "spawn", None)
        calibration = getattr(spawn, "intrinsics", None)
        if calibration is None:
            continue
        nominal = float(spawn.focal_length)
        if not math.isfinite(nominal) or nominal <= 0:
            raise ValueError("Calibrated camera nominal focal length must be finite and positive")
        pattern = camera.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
        for prim in sim_utils.find_matching_prims(pattern):
            path = prim.GetPath()
            if any(path.HasPrefix(root) for root in all_paths) and not any(
                    path.HasPrefix(root) for root in selected_paths):
                continue
            focal = float(prim.GetAttribute("focalLength").Get())
            if not math.isfinite(focal) or focal <= 0:
                raise ValueError(f"Camera focal length must be finite and positive: {path}")
            scale = focal / nominal
            for axis in ("fx", "fy"):
                prim.GetAttribute("omni:lensdistortion:opencvPinhole:" + axis).Set(calibration[axis] * scale)
