"""CPU coverage for the narrow offline processor and Docker server wrappers."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
        spec.loader.exec_module(module)
    return module


helper = load_module("offline_cosmos_test_helper", "gr00t_offline_cosmos.py")
server = load_module("offline_cosmos_test_server", "launch_gr00t_server_n17.py")


class ProcessorPathTests(unittest.TestCase):
    def setUp(self):
        self.processor_module = ModuleType("gr00t.model.gr00t_n1d7.processing_gr00t_n1d7")
        self.original = Mock(return_value="official processor")
        self.processor_module.build_processor = self.original
        self.utils = ModuleType("transformers.utils")
        self.utils.is_offline_mode = Mock(return_value=True)
        self.utils.cached_file = Mock(return_value="/cache/snapshots/pinned/config.json")
        self.modules = {self.processor_module.__name__: self.processor_module,
                        "transformers": ModuleType("transformers"), "transformers.utils": self.utils}

    def test_offline_cosmos_uses_revision_cache_snapshot_and_unchanged_loading_options(self):
        kwargs = {"cache_dir": "/explicit cache", "revision": "pinned", "local_files_only": False}
        before = kwargs.copy()
        with patch.dict(sys.modules, self.modules), helper.offline_cosmos_processor():
            actual = self.processor_module.build_processor(helper.COSMOS_REPO, kwargs)
        self.assertEqual(actual, "official processor")
        self.utils.cached_file.assert_called_once_with(
            helper.COSMOS_REPO, "config.json", cache_dir="/explicit cache", revision="pinned",
            local_files_only=True)
        self.original.assert_called_once_with("/cache/snapshots/pinned", kwargs)
        self.assertIs(self.original.call_args.args[1], kwargs)
        self.assertEqual(kwargs, before)
        self.assertIs(self.processor_module.build_processor, self.original)

    def test_other_model_names_and_online_loading_keep_the_official_path(self):
        for name, offline in [("another/model", True), (helper.COSMOS_REPO, False),
                              ("/local/Cosmos", True)]:
            with self.subTest(name=name, offline=offline):
                self.original.reset_mock()
                self.utils.is_offline_mode.return_value = offline
                kwargs = {"revision": "unchanged"}
                with patch.dict(sys.modules, self.modules), helper.offline_cosmos_processor():
                    self.processor_module.build_processor(name, kwargs)
                self.original.assert_called_once_with(name, kwargs)
        self.utils.cached_file.assert_not_called()

    def test_builder_is_restored_on_cache_or_processor_failure(self):
        for failing in (self.utils.cached_file, self.original):
            with self.subTest(failing=failing):
                failing.side_effect = RuntimeError("offline fixture failure")
                with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    with helper.offline_cosmos_processor():
                        self.processor_module.build_processor(helper.COSMOS_REPO, {})
                self.assertIs(self.processor_module.build_processor, self.original)
                failing.side_effect = None

    def test_actual_pinned_processor_uses_cache_without_hub_metadata_or_model_loading(self):
        checkout = os.environ.get("GROOT_N17_CHECKOUT")
        python = os.environ.get("GROOT_N17_PYTHON")
        if not checkout or not python:
            self.skipTest("Set GROOT_N17_CHECKOUT/PYTHON for the actual offline Cosmos CPU processor")
        code = '''
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from gr00t_model_profiles import N17_PIN, inspect_cosmos_cache
from gr00t_offline_cosmos import offline_cosmos_processor
assert subprocess.check_output(['git', '-C', sys.argv[2], 'rev-parse', 'HEAD'], text=True).strip() == N17_PIN
try:
    report = inspect_cosmos_cache()
except (OSError, ValueError):
    sys.exit(77)
module = importlib.import_module('gr00t.model.gr00t_n1d7.processing_gr00t_n1d7')
assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
original = module.build_processor
options = {'local_files_only': True, 'revision': report['revision']}
with patch('huggingface_hub.model_info', side_effect=AssertionError('Unexpected metadata lookup')) as info:
    with patch('transformers.AutoModel.from_pretrained', side_effect=AssertionError('Unexpected model allocation')) as model:
        with offline_cosmos_processor():
            processor = module.build_processor('nvidia/Cosmos-Reason2-2B', options)
            rendered = processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':'Pick up the blue cube'}]}], tokenize=False)
            assert 'Pick up the blue cube' in rendered
            assert processor.tokenizer(rendered)['input_ids']
            assert type(processor).__name__ == 'Qwen3VLProcessor'
        info.assert_not_called()
        model.assert_not_called()
assert module.build_processor is original
assert options == {'local_files_only': True, 'revision': report['revision']}
print('PASS actual pinned offline processor')
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
        result = subprocess.run([python, "-c", code, str(SCRIPTS), checkout],
                                text=True, capture_output=True, env=env)
        if result.returncode == 77:
            self.skipTest("Actual Cosmos cache unavailable; never download from this test")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS actual pinned offline processor", result.stdout)


class ServerWrapperTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="n17-server-test-")
        self.addCleanup(temporary.cleanup)
        self.groot = Path(temporary.name)
        self.official = self.groot / "gr00t/eval/run_gr00t_server.py"
        self.official.parent.mkdir(parents=True)
        (self.groot / "gr00t/__init__.py").write_text("# fixture\n")
        self.gr00t = ModuleType("gr00t")
        self.gr00t.__file__ = str(self.groot / "gr00t/__init__.py")
        self.official.write_text('''
import os, sys
import gr00t
assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
assert all(name not in os.environ for name in ('GROOT_SKIP_HF_MODEL_WEIGHTS','GROOT_HF_LOCAL_FIRST','PYTEST_CURRENT_TEST'))
gr00t.forwarded = sys.argv[1:]
gr00t.server_cwd = os.getcwd()
''')

    def invoke(self, flags=(), pin=server.N17_PIN):
        with patch.dict(sys.modules, {"gr00t": self.gr00t}), \
                patch("subprocess.check_output", return_value=pin + "\n"), \
                patch.object(server, "offline_cosmos_processor", return_value=contextlib.nullcontext()) as processor:
            server.main(["--groot", str(self.groot), *flags])
        return processor

    def test_official_server_flags_forwarded_and_process_paths_restored(self):
        flags = ["--model-path", "/workspace/model root", "--embodiment-tag", "NEW_EMBODIMENT", "--port", "5577"]
        previous = (sys.path[:], sys.argv[:], Path.cwd())
        with patch.dict(os.environ, {"GROOT_SKIP_HF_MODEL_WEIGHTS": "1"}):
            processor = self.invoke(flags)
        processor.assert_called_once_with()
        self.assertEqual(self.gr00t.forwarded, flags)
        self.assertEqual(self.gr00t.server_cwd, str(self.groot))
        self.assertEqual((sys.path, sys.argv, Path.cwd()), previous)

    def test_wrong_pin_and_import_root_rejected_before_server(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(pin="wrong revision")
        self.gr00t.__file__ = "/tmp/another-gr00t/gr00t/__init__.py"
        with self.assertRaisesRegex(RuntimeError, "GR00T imported from"):
            self.invoke()
        self.assertFalse(hasattr(self.gr00t, "forwarded"))

    def test_server_exit_status_and_path_restoration_preserved(self):
        self.official.write_text("raise SystemExit(9)\n")
        previous = (sys.path[:], sys.argv[:], Path.cwd())
        with self.assertRaises(SystemExit) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 9)
        self.assertEqual((sys.path, sys.argv, Path.cwd()), previous)


class DockerEntrypointTests(unittest.TestCase):
    def test_exact_official_python_server_redirect_preserves_all_flags(self):
        with tempfile.TemporaryDirectory(prefix="n17-entrypoint-test-") as temporary:
            root = Path(temporary)
            python = root / "python"
            python.write_text('#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ["ARGV_FILE"], "w").write(json.dumps(sys.argv[1:]))\n')
            python.chmod(0o755)
            output = root / "argv.json"
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"], ARGV_FILE=str(output))
            flags = ["--model-path", "/model root", "--embodiment-tag", "NEW_EMBODIMENT", "--port", "5577"]
            entrypoint = REPO / "docker/real/entrypoint.n17.sh"
            result = subprocess.run(["bash", str(entrypoint), "python", "/Isaac-GR00T/gr00t/eval/run_gr00t_server.py", *flags], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), ["/opt/so101/launch_gr00t_server_n17.py", "--groot", "/Isaac-GR00T", *flags])
            result = subprocess.run(["bash", str(entrypoint), "python", "-c", "unchanged generic Python"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), ["-c", "unchanged generic Python"])

    def test_generic_commands_keep_arguments_and_exit_code(self):
        entrypoint = REPO / "docker/real/entrypoint.n17.sh"
        result = subprocess.run(["bash", str(entrypoint), "/usr/bin/printf", "%s\n", "argument with spaces", "--port"], capture_output=True, text=True)
        self.assertEqual(result.stdout, "argument with spaces\n--port\n")
        self.assertEqual(result.returncode, 0)
        result = subprocess.run(["bash", str(entrypoint), "bash", "-c", "exit 7"])
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
