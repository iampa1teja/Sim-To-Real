"""CPU checks for the small boundary around the untouched N1.7 launcher."""

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pick_place_n17_launcher", REPO / "scripts/launch_pick_place_n17.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def modality(horizon):
    return {"action": SimpleNamespace(delta_indices=list(range(horizon)))}


def fixture():
    return SimpleNamespace(
        training=SimpleNamespace(skip_weight_loading=False),
        model=SimpleNamespace(action_horizon=40, diffusion_model_cfg={"num_layers": 32}),
        data=SimpleNamespace(
            datasets=[SimpleNamespace(embodiment_tag="new_embodiment", dataset_paths=["prepared data"])],
            modality_configs={"new_embodiment": modality(16), "unused_unitree": modality(50)}))


class ScopeTests(unittest.TestCase):
    def test_selects_dataset_configs_without_mutating_registry_or_horizons(self):
        config = fixture()
        registry = config.data.modality_configs
        selected = registry["new_embodiment"]
        datasets = config.data.datasets
        launcher.scope_dataset_modalities(config)
        self.assertEqual(list(config.data.modality_configs), ["new_embodiment"])
        self.assertIs(config.data.modality_configs["new_embodiment"], selected)
        self.assertIs(config.data.datasets, datasets)
        self.assertEqual(list(registry), ["new_embodiment", "unused_unitree"])
        self.assertEqual(len(selected["action"].delta_indices), 16)
        self.assertEqual(config.model.action_horizon, 40)
        self.assertEqual(config.model.diffusion_model_cfg["num_layers"], 32)
        self.assertFalse(config.training.skip_weight_loading)

    def test_preserves_multiple_selected_tags_and_deduplicates(self):
        config = fixture()
        config.data.modality_configs["second_robot"] = modality(40)
        config.data.datasets.extend([
            SimpleNamespace(embodiment_tag=SimpleNamespace(value="second_robot")),
            SimpleNamespace(embodiment_tag="new_embodiment")])
        launcher.scope_dataset_modalities(config)
        self.assertEqual(list(config.data.modality_configs), ["new_embodiment", "second_robot"])

    def test_invalid_selections_or_skip_weights_fail_before_changing_config(self):
        for mode in ("empty", "missing", "null", "skip"):
            with self.subTest(mode=mode):
                config = fixture()
                original = config.data.modality_configs
                if mode == "empty":
                    config.data.datasets = []
                elif mode == "missing":
                    config.data.datasets[0].embodiment_tag = "unregistered"
                elif mode == "null":
                    config.data.datasets[0].embodiment_tag = None
                else:
                    config.training.skip_weight_loading = True
                with self.assertRaises(ValueError):
                    launcher.scope_dataset_modalities(config)
                self.assertIs(config.data.modality_configs, original)

    def test_actual_n17_validator_keeps_saved_40_and_so16_without_unused_50(self):
        groot = Path(os.environ.get("GROOT_N17_CHECKOUT", Path.home() / "Isaac-GR00T-N1.7"))
        source = groot / "gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py"
        if not source.is_file():
            self.skipTest("Pinned N1.7 source unavailable for the actual CPU validator")
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "validate_action_horizons")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        validate = namespace["validate_action_horizons"]
        saved_checkpoint = {"saved_pretrain": modality(40), "new_embodiment": modality(16)}
        config = fixture()
        with self.assertRaisesRegex(ValueError, "unused_unitree=50"):
            validate({**saved_checkpoint, **config.data.modality_configs}, 40)
        launcher.scope_dataset_modalities(config)
        merged = {**saved_checkpoint, **config.data.modality_configs}
        validate(merged, 40)
        self.assertEqual(set(merged), {"saved_pretrain", "new_embodiment"})
        self.assertEqual(len(merged["saved_pretrain"]["action"].delta_indices), 40)
        self.assertEqual(len(merged["new_embodiment"]["action"].delta_indices), 16)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="n17-launcher-test-")
        self.addCleanup(temporary.cleanup)
        self.groot = Path(temporary.name)
        self.official = self.groot / "gr00t/experiment/launch_finetune.py"
        self.official.parent.mkdir(parents=True)
        (self.groot / "gr00t/__init__.py").write_text("# CPU fixture\n")
        self.gr00t = ModuleType("gr00t")
        self.gr00t.__file__ = str(self.groot / "gr00t/__init__.py")
        self.experiment = ModuleType("gr00t.experiment.experiment")
        self.experiment.config = fixture()
        self.original_run = Mock(return_value="delegated")
        self.experiment.run = self.original_run
        self.offline_helper = ModuleType("gr00t_offline_cosmos")
        self.offline_helper.offline_cosmos_processor = Mock(return_value=contextlib.nullcontext())
        self.modules = {"gr00t": self.gr00t,
                        "gr00t.experiment": ModuleType("gr00t.experiment"),
                        "gr00t.experiment.experiment": self.experiment,
                        "gr00t_offline_cosmos": self.offline_helper}
        self.official.write_text('''
import os, sys
from gr00t.experiment.experiment import run
import gr00t.experiment.experiment as fixture_experiment
assert all(name not in os.environ for name in ('GROOT_SKIP_HF_MODEL_WEIGHTS', 'GROOT_HF_LOCAL_FIRST', 'PYTEST_CURRENT_TEST'))
fixture_experiment.forwarded = sys.argv[1:]
fixture_experiment.included_paths = sys.path[:]
fixture_experiment.result = run(fixture_experiment.config)
''')

    def invoke(self, forwarded=(), pin=launcher.PIN):
        with patch.dict(sys.modules, self.modules), \
                patch("subprocess.check_output", return_value=pin + "\n") as git:
            launcher.main(["--groot", str(self.groot), *forwarded])
        git.assert_called_once_with(["git", "-C", str(self.groot), "rev-parse", "HEAD"], text=True)

    def test_official_flags_and_selected_config_are_forwarded_unchanged(self):
        flags = ["--base-model-path", "model root", "--dataset-path", "prepared data",
                 "--embodiment-tag", "NEW_EMBODIMENT", "--no-tune-diffusion-model",
                 "--global-batch-size", "32", "--max-steps", "30"]
        prior_argv, prior_path, prior_cwd = sys.argv[:], sys.path[:], Path.cwd()
        with patch.dict(os.environ, {"GROOT_SKIP_HF_MODEL_WEIGHTS": "1",
                                     "GROOT_HF_LOCAL_FIRST": "1", "PYTEST_CURRENT_TEST": "test"}):
            self.invoke(flags)
        self.assertEqual(self.experiment.forwarded, flags)
        self.assertEqual(self.experiment.result, "delegated")
        self.original_run.assert_called_once_with(self.experiment.config)
        self.assertEqual(list(self.experiment.config.data.modality_configs), ["new_embodiment"])
        self.assertIs(self.experiment.run, self.original_run)
        self.assertEqual(sys.argv, prior_argv)
        self.assertEqual(sys.path, prior_path)
        self.assertEqual(Path.cwd(), prior_cwd)
        self.offline_helper.offline_cosmos_processor.assert_called_once_with()

    def test_runpy_entry_adds_project_helper_path_and_restores_it(self):
        project = REPO / "scripts/launch_pick_place_n17.py"
        flags = ["--model-profile", "unchanged fixture flag"]
        without_scripts = [path for path in sys.path if path != str(project.parent)]
        with patch.dict(sys.modules, self.modules), \
                patch("subprocess.check_output", return_value=launcher.PIN + "\n"), \
                patch.object(sys, "path", without_scripts[:]), \
                patch.object(sys, "argv", [str(project), "--groot", str(self.groot), *flags]):
            runpy.run_path(str(project), run_name="__main__")
            self.assertEqual(sys.path, without_scripts)
        self.assertEqual(self.experiment.forwarded, flags)
        self.assertIn(str(project.parent), self.experiment.included_paths)

    def test_wrong_pin_and_import_path_fail_before_running_training(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(pin="wrong revision")
        self.original_run.assert_not_called()
        self.gr00t.__file__ = "/tmp/other-checkout/gr00t/__init__.py"
        with self.assertRaisesRegex(RuntimeError, "GR00T imported from"):
            self.invoke()
        self.original_run.assert_not_called()

    def test_training_exception_and_system_exit_are_preserved_and_patch_restored(self):
        for error in (RuntimeError("load failed"), SystemExit(7)):
            with self.subTest(error=type(error).__name__):
                self.original_run.side_effect = error
                with self.assertRaises(type(error)) as caught:
                    self.invoke()
                if isinstance(error, SystemExit):
                    self.assertEqual(caught.exception.code, 7)
                else:
                    self.assertEqual(str(caught.exception), "load failed")
                self.assertIs(self.experiment.run, self.original_run)

    def test_skip_weight_loading_is_rejected_at_the_actual_run_boundary(self):
        self.experiment.config.training.skip_weight_loading = True
        with self.assertRaisesRegex(ValueError, "requires pretrained weight loading"):
            self.invoke(["--skip-weight-loading"])
        self.original_run.assert_not_called()
        self.assertIs(self.experiment.run, self.original_run)


if __name__ == "__main__":
    unittest.main()
