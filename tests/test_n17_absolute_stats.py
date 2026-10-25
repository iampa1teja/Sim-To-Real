"""Absolute training must not write unused relative-stat caches into read-only data."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "policies/groot_n17/train/launch_pick_place_n17.py"
SPEC = importlib.util.spec_from_file_location("n17_absolute_stats_test", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class SelectedStatisticsTest(unittest.TestCase):
    def config(self, *representations):
        actions = [SimpleNamespace(rep=SimpleNamespace(name=rep)) for rep in representations]
        return SimpleNamespace(data=SimpleNamespace(modality_configs={
            "new_embodiment": {"action": SimpleNamespace(action_configs=actions)}}))

    def test_absolute_groups_do_not_generate_relative_statistics(self):
        original = Mock(side_effect=AssertionError("must not write relative statistics"))
        factory = SimpleNamespace(generate_rel_stats=original)
        with patch.object(launcher.importlib, "import_module", return_value=factory):
            with launcher.selected_action_statistics(self.config("ABSOLUTE", "ABSOLUTE")):
                factory.generate_rel_stats("/readonly/dataset", SimpleNamespace(value="new_embodiment"))
        original.assert_not_called()
        self.assertIs(factory.generate_rel_stats, original)

    def test_relative_or_unknown_groups_preserve_upstream_generation(self):
        for representations, tag in ((["RELATIVE", "ABSOLUTE"], "new_embodiment"),
                                      (["ABSOLUTE"], "another_embodiment"), ([], "new_embodiment")):
            with self.subTest(representations=representations, tag=tag):
                original = Mock(return_value="generated")
                factory = SimpleNamespace(generate_rel_stats=original)
                with patch.object(launcher.importlib, "import_module", return_value=factory):
                    with launcher.selected_action_statistics(self.config(*representations)):
                        self.assertEqual(factory.generate_rel_stats("/dataset", tag), "generated")
                original.assert_called_once_with("/dataset", tag)
                self.assertIs(factory.generate_rel_stats, original)

    def test_restores_generation_after_training_failure(self):
        original = Mock()
        factory = SimpleNamespace(generate_rel_stats=original)
        with patch.object(launcher.importlib, "import_module", return_value=factory):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with launcher.selected_action_statistics(self.config("ABSOLUTE")):
                    raise RuntimeError("training failed")
        self.assertIs(factory.generate_rel_stats, original)
