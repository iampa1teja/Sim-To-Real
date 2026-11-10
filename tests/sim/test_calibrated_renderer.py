"""RGB/depth render-product setup contract without launching Kit."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def renderer_class(settings):
    source = Path(__file__).resolve().parents[2] / "src/sim/scene/renderer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "CalibratedRtxRenderer")

    class IsaacRtxRenderer:
        def create_render_data(self, spec):
            # The SDK updates this global setting independently for each camera.
            needs_color = any(kind in spec.cfg.data_types for kind in ("rgb", "rgba", "rgb_hdr"))
            settings.set("/rtx/sdg/force/disableColorRender", not needs_color)
            return spec.render_data

    namespace = {"IsaacRtxRenderer": IsaacRtxRenderer, "get_settings_manager": lambda: settings}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[cls.name]


class MixedCameraRenderingTests(unittest.TestCase):
    def make_renderer(self):
        class Settings(dict):
            def set(self, key, value):
                self[key] = value

        self.settings = Settings()
        return renderer_class(self.settings)()

    def setup_camera(self, renderer, outputs):
        spec = SimpleNamespace(cfg=SimpleNamespace(data_types=outputs), render_data=object())
        self.assertIs(renderer.create_render_data(spec), spec.render_data)

    def test_rgb_survives_depth_camera_created_after_it(self):
        renderer = self.make_renderer()
        for outputs in (["rgb"], ["depth"], ["instance_id_segmentation_fast"]):
            self.setup_camera(renderer, outputs)
            self.assertFalse(self.settings["/rtx/sdg/force/disableColorRender"])

    def test_rgb_camera_enables_color_after_depth_only_setup(self):
        renderer = self.make_renderer()
        self.setup_camera(renderer, ["depth"])
        self.assertTrue(self.settings["/rtx/sdg/force/disableColorRender"])
        self.setup_camera(renderer, ["rgb", "depth"])
        self.assertFalse(self.settings["/rtx/sdg/force/disableColorRender"])

    def test_depth_only_keeps_sdk_color_optimization(self):
        renderer = self.make_renderer()
        self.setup_camera(renderer, ["depth"])
        self.assertTrue(self.settings["/rtx/sdg/force/disableColorRender"])

    def test_other_color_buffers_are_preserved(self):
        for color in ("rgba", "rgb_hdr"):
            with self.subTest(color=color):
                renderer = self.make_renderer()
                self.setup_camera(renderer, [color])
                self.setup_camera(renderer, ["depth"])
                self.assertFalse(self.settings["/rtx/sdg/force/disableColorRender"])


if __name__ == "__main__":
    unittest.main()
