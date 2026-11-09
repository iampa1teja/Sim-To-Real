"""Camera-selection lifecycle checks without importing the Isaac runtime."""

import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def visualizer_class(stage, has_gui=True):
    path = Path(__file__).resolve().parents[2] / "src/sim/scene/calibrated_viewport.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    class KitVisualizer:
        def _ensure_simulation_app(self):
            self._runtime_headless = self.cfg.headless

        def pumps_app_update(self):
            return True

        def step(self, dt):
            if not self._runtime_headless:
                self.app_updates += 1

        def initialize(self):
            self._apply_cfg_camera_pose_if_configured()
            self._setup_initial_camera_view()

        def _apply_cfg_camera_pose_if_configured(self):
            raise AssertionError("A calibrated camera must not receive an eye/lookat pose.")

    context = {
        "KitVisualizer": KitVisualizer,
        "SimulationContext": SimpleNamespace(instance=lambda: SimpleNamespace(stage=stage, has_gui=has_gui)),
        "UsdGeom": SimpleNamespace(Camera=object()),
        "logger": Mock(),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), context)
    return context[cls.name]


class CalibratedViewportTests(unittest.TestCase):
    def make_visualizer(self, headless=False):
        self.path = "/World/envs/env_0/realsense_camera_rgb"
        self.prim = Mock()
        self.prim.IsValid.return_value = True
        self.prim.IsA.return_value = True
        self.stage = Mock()
        self.stage.GetPrimAtPath.return_value = self.prim
        visualizer = visualizer_class(self.stage)()
        visualizer.cfg = SimpleNamespace(
            camera_prim_path=self.path, origin_type="world", window_width=640, window_height=480,
            headless=headless,
        )
        visualizer._runtime_headless = headless
        visualizer._viewport_api = None if headless else Mock()
        visualizer._viewport_window = None if headless else SimpleNamespace(viewport_widget=Mock())
        if not headless:
            visualizer._viewport_api.get_active_camera.return_value = self.path
        return visualizer

    def test_initialization_selects_native_camera_without_editing_it(self):
        visualizer = self.make_visualizer()
        visualizer.initialize()
        visualizer._viewport_api.set_active_camera.assert_called_once_with(self.path)
        visualizer._viewport_api.set_texture_resolution.assert_called_once_with((640, 480))
        visualizer._viewport_window.viewport_widget.set_resolution.assert_called_once_with((640, 480))
        self.assertFalse(visualizer._viewport_api.fill_frame)
        self.assertEqual(visualizer._controlled_camera_path, self.path)
        self.assertEqual([call[0] for call in self.prim.method_calls], ["IsValid", "IsA"])

    def test_headless_uses_existing_camera_without_viewport_api(self):
        visualizer = self.make_visualizer(headless=True)
        visualizer.initialize()
        self.assertEqual(visualizer._controlled_camera_path, self.path)

    def test_renderer_advances_camera_frames_in_both_modes(self):
        # RTX trusts pumps_app_update() and skips its own update when True.
        # The SDK visualizer's headless step returns without pumping the app.
        for headless in (False, True):
            with self.subTest(headless=headless):
                visualizer = self.make_visualizer(headless=headless)
                visualizer.app_updates = 0
                for _ in range(3):
                    visualizer.step(1 / 30)
                    if not visualizer.pumps_app_update():
                        visualizer.app_updates += 1
                self.assertEqual(visualizer.app_updates, 3)

    def test_windowless_launcher_gives_render_pump_to_camera(self):
        visualizer = visualizer_class(Mock(), has_gui=False)()
        visualizer.cfg = SimpleNamespace(headless=False)
        visualizer._ensure_simulation_app()
        self.assertTrue(visualizer._runtime_headless)
        self.assertFalse(visualizer.pumps_app_update())

    def test_desktop_launcher_keeps_viewport_render_pump(self):
        visualizer = visualizer_class(Mock(), has_gui=True)()
        visualizer.cfg = SimpleNamespace(headless=False)
        visualizer._ensure_simulation_app()
        self.assertFalse(visualizer._runtime_headless)
        self.assertTrue(visualizer.pumps_app_update())

    def test_explicit_headless_visualizer_stays_headless_with_desktop_app(self):
        visualizer = visualizer_class(Mock(), has_gui=True)()
        visualizer.cfg = SimpleNamespace(headless=True)
        visualizer._ensure_simulation_app()
        self.assertTrue(visualizer._runtime_headless)
        self.assertFalse(visualizer.pumps_app_update())

    def test_headless_pauses_only_window_render_products(self):
        windows = [SimpleNamespace(viewport_api=SimpleNamespace(updates_enabled=True)) for _ in range(2)]
        module = SimpleNamespace(get_viewport_window_instances=lambda: windows)
        visualizer = self.make_visualizer(headless=True)
        visualizer.app_updates = 0
        with patch.dict(sys.modules, {"omni.kit.viewport.window": module}):
            visualizer.step(1 / 30)
        self.assertTrue(all(not window.viewport_api.updates_enabled for window in windows))
        self.assertEqual(visualizer.app_updates, 0)

    def test_desktop_keeps_window_updates_enabled(self):
        windows = [SimpleNamespace(viewport_api=SimpleNamespace(updates_enabled=True))]
        module = SimpleNamespace(get_viewport_window_instances=lambda: windows)
        visualizer = self.make_visualizer()
        visualizer.app_updates = 0
        with patch.dict(sys.modules, {"omni.kit.viewport.window": module}):
            visualizer.step(1 / 30)
        self.assertTrue(windows[0].viewport_api.updates_enabled)
        self.assertEqual(visualizer.app_updates, 1)

    def test_widget_layout_keeps_calibrated_resolution(self):
        visualizer = self.make_visualizer()
        widget = visualizer._viewport_window.viewport_widget
        widget.full_resolution = (1280, 720)

        def set_resolution(value):
            widget.full_resolution = value

        widget.set_resolution.side_effect = set_resolution
        visualizer.initialize()
        # ViewportWidget reapplies its requested resolution on a layout event.
        visualizer._viewport_api.set_texture_resolution(widget.full_resolution)
        visualizer._viewport_api.set_texture_resolution.assert_called_with((640, 480))

    def test_missing_camera_is_reported(self):
        visualizer = self.make_visualizer()
        self.prim.IsValid.return_value = False
        with self.assertRaisesRegex(ValueError, "camera not found"):
            visualizer.initialize()

    def test_gui_missing_viewport_is_reported(self):
        visualizer = self.make_visualizer()
        visualizer._viewport_api = None
        with self.assertRaisesRegex(RuntimeError, "no active viewport"):
            visualizer.initialize()

    def test_rejected_camera_selection_is_reported(self):
        visualizer = self.make_visualizer()
        visualizer._viewport_api.get_active_camera.return_value = "/OmniverseKit_Persp"
        with self.assertRaisesRegex(RuntimeError, "did not select"):
            visualizer.initialize()


if __name__ == "__main__":
    unittest.main()
