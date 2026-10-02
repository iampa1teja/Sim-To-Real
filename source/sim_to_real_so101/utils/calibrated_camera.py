"""Preserve measured intrinsics in both RTX projection and sensor metadata."""
from isaaclab.sensors import TiledCamera
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from sim_to_real_so101.assets.real_setup import apply_opencv_intrinsics


@configclass
class CalibratedPinholeCameraCfg(PinholeCameraCfg):
    # fx, fy, cx, cy in pixels; width/height describe their calibration resolution.
    intrinsics: dict | None = None
    # Per-camera residual colour response (see utils/camera_response.py); None = raw render.
    response: dict | None = None

class CalibratedTiledCamera(TiledCamera):
    """Report the lens model's K instead of Isaac Lab's centered approximation.

    Optionally applies the camera's fitted colour response to its RGB output.
    """

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        params = getattr(self.cfg.spawn, "response", None)
        if not params or "rgba" not in self._data.output:
            return
        if getattr(self, "_response", None) is None:
            from sim_to_real_so101.utils.camera_response import CameraResponse
            self._response = CameraResponse(params, self.cfg.height, self.cfg.width, self.device)
        rgba = self._data.output["rgba"]
        rgba[env_ids, ..., :3] = self._response(rgba[env_ids, ..., :3])

    def _update_intrinsic_matrices(self, env_ids):
        super()._update_intrinsic_matrices(env_ids)
        for index in env_ids:
            prim = self._sensor_prims[index].GetPrim()
            if prim.GetAttribute("omni:lensdistortion:model").Get() != "opencvPinhole":
                continue
            prefix = "omni:lensdistortion:opencvPinhole:"
            width, height = prim.GetAttribute(prefix + "imageSize").Get()
            sy, sx = self.image_shape[0] / height, self.image_shape[1] / width
            offset = prim.GetAttribute("calibration:pixelCenterOffset").Get() or 0.0
            k = self._data.intrinsic_matrices[index]
            k.zero_()
            k[0, 0] = prim.GetAttribute(prefix + "fx").Get() * sx
            k[1, 1] = prim.GetAttribute(prefix + "fy").Get() * sy
            k[0, 2] = (prim.GetAttribute(prefix + "cx").Get() - offset) * sx
            k[1, 2] = (prim.GetAttribute(prefix + "cy").Get() - offset) * sy
            k[2, 2] = 1.0
