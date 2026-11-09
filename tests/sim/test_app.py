"""Prelaunch argument regression checks without importing Kit."""

import importlib.util
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[2] / "src/sim/app.py"
spec = importlib.util.spec_from_file_location("s2r_app", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rendering_kit_args = module.rendering_kit_args


class RenderingKitArgsTests(unittest.TestCase):
    def test_default_enables_dependency(self):
        self.assertEqual(rendering_kit_args(), "--enable omni.kit.context_menu")

    def test_preserves_existing_settings_and_extensions(self):
        original = "--enable other.extension --/renderer/multiGpu/enabled=false"
        self.assertEqual(rendering_kit_args(original), original + " --enable omni.kit.context_menu")

    def test_existing_dependency_is_not_added_twice(self):
        for original in ("--enable omni.kit.context_menu", "--enable=omni.kit.context_menu"):
            with self.subTest(original=original):
                self.assertEqual(rendering_kit_args(original), original)

    def test_similar_extension_name_does_not_satisfy_dependency(self):
        original = "--enable omni.kit.context_menu_extra"
        self.assertEqual(rendering_kit_args(original), original + " --enable omni.kit.context_menu")

    def test_repeated_preparation_is_idempotent(self):
        prepared = rendering_kit_args("--/app/window/width=640")
        self.assertEqual(rendering_kit_args(prepared), prepared)


if __name__ == "__main__":
    unittest.main()
