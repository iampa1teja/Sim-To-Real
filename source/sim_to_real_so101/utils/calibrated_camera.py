"""Preserve measured intrinsics in both RTX projection and sensor metadata."""
from isaaclab.sensors import TiledCamera
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from sim_to_real_so101.assets.real_setup import apply_opencv_intrinsics


@configclass
class CalibratedPinholeCameraCfg(PinholeCameraCfg):
    # fx, fy, cx, cy in pixels; width/height describe their calibration resolution.
    intrinsics: dict | None = None

class CalibratedTiledCamera(TiledCamera):
    """Report the lens model's K instead of Isaac Lab's centered approximation."""

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
