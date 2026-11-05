"""Room and wrist cameras: measured OpenCV lens plus an optional fitted colour response.

Isaac Lab 3.0 reads the authored OpenCV pinhole intrinsics (fx, fy, cx, cy) natively
(Camera._read_authored_opencv_intrinsics), so v1's _update_intrinsic_matrices override is gone
(3.0 no longer calls that method). apply_opencv_intrinsics authors cx and cy with +0.5 for RTX's
half-pixel sampling, so the K Isaac reports is 0.5 px off the measured K in cx and cy.
Rendering is unaffected; read the measured K from real_setup.json if you need it.
"""
from isaaclab.sensors import TiledCamera
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from sim.scene.real_setup import apply_opencv_intrinsics  # noqa: F401  (re-exported for env configs)


@configclass
class CalibratedPinholeCameraCfg(PinholeCameraCfg):
    # fx, fy, cx, cy in pixels; width/height describe their calibration resolution.
    intrinsics: dict | None = None
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
