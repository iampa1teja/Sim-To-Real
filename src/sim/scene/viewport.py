"""Configuration and lazy Kit viewport runtime for an existing calibrated camera."""
import logging

from isaaclab.utils import configclass
from isaaclab_visualizers.kit import KitVisualizerCfg

logger = logging.getLogger(__name__)


@configclass
class CalibratedKitVisualizerCfg(KitVisualizerCfg):
    """Select a scene camera without replacing its measured pose or lens settings."""

    class_type: str = "sim.scene.viewport:CalibratedKitVisualizer"
    camera_prim_path: str = ""


def __getattr__(name):
    if name != "CalibratedKitVisualizer":
        raise AttributeError(name)
    from pxr import UsdGeom
    from isaaclab.sim import SimulationContext
    from isaaclab_visualizers.kit import KitVisualizer

    class CalibratedKitVisualizer(KitVisualizer):
        """Keep the camera's authored calibration through Kit visualizer initialization."""

        def _ensure_simulation_app(self) -> None:
            super()._ensure_simulation_app()
            # The Kit launcher need not expose a SimulationApp singleton. In that
            # case the SDK misses --headless and would still construct a viewport.
            sim = SimulationContext.instance()
            if sim is not None and not sim.has_gui:
                self._runtime_headless = True

        def pumps_app_update(self) -> bool:
            # KitVisualizer.step() skips app.update() headlessly. Let the RTX
            # renderer pump it so camera frames advance in that mode too.
            return not self._runtime_headless and super().pumps_app_update()

        def _pause_headless_viewports(self) -> None:
            # Kit creates a default 1280x720 viewport even with its window disabled.
            # Camera render products have no ViewportWindow and remain active.
            try:
                from omni.kit.viewport.window import get_viewport_window_instances
            except ImportError:
                return
            for window in get_viewport_window_instances():
                window.viewport_api.updates_enabled = False

        def step(self, dt: float) -> None:
            if self._runtime_headless:
                # Startup docking callbacks can re-enable a hidden viewport later.
                self._pause_headless_viewports()
            super().step(dt)

        def _apply_cfg_camera_pose_if_configured(self) -> None:
            # Kit normally writes eye/lookat to the active camera during viewport setup.
            # That camera may already be our sensor, whose measured pose must stay intact.
            pass

        def _setup_initial_camera_view(self) -> None:
            # This hook runs after Kit creates the viewport and binds scene transforms.
            # Selecting the camera in a startup event happens before this lifecycle step.
            path = self.cfg.camera_prim_path
            prim = SimulationContext.instance().stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                raise ValueError(f"Calibrated viewport camera not found: {path}")
            if self.cfg.origin_type != "world":
                raise ValueError("A calibrated viewport camera requires origin_type='world'.")

            self._controlled_camera_path = path
            if self._runtime_headless:
                self._pause_headless_viewports()
                return
            if self._viewport_api is None:
                raise RuntimeError("Cannot select the calibrated camera: Kit has no active viewport.")
            self._viewport_api.set_active_camera(path)
            self._viewport_api.fill_frame = False
            resolution = (self.cfg.window_width, self.cfg.window_height)
            # Keep the widget's requested size in sync as well as its texture. Its
            # next layout event otherwise restores the previous viewport resolution.
            self._viewport_window.viewport_widget.set_resolution(resolution)
            self._viewport_api.set_texture_resolution(resolution)
            if str(self._viewport_api.get_active_camera()) != path:
                raise RuntimeError(f"Kit did not select the calibrated viewport camera: {path}")
            logger.info("Kit viewport uses calibrated camera: %s", path)


    CalibratedKitVisualizer.__qualname__ = "CalibratedKitVisualizer"
    globals()[name] = CalibratedKitVisualizer
    return CalibratedKitVisualizer
