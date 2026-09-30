"""CPU training-launcher tests; every GPU/tool/model launch is a local fake."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/train_pick_place.sh"
PIN = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
HUB_PIN = "93a5a88f78a395939f784a5fe3685184913dcc60"


class TrainingScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="train-script-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.out = self.root / "output"
        self.base = self.root / "base"
        self.groot = self.root / "groot"
        self.dataset = self.root / "dataset"
        self.log = self.root / "tools.jsonl"
        self.hub_cache = self.root / "hf_hub"
        self.env = dict(os.environ)
        for key in ("GROOT", "BASE_MODEL", "DATASET", "OUT", "BATCH", "MAX_STEPS",
                    "SAVE_STEPS", "SAVE_TOTAL_LIMIT", "RESUME", "DATALOADER_NUM_WORKERS",
                    "DRY_RUN", "MEASURED_S_PER_STEP", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_HOME",
                    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                    "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
            self.env.pop(key, None)
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}",
                        TRAIN_TEST_LOG=str(self.log), GROOT=str(self.groot),
                        BASE_MODEL=str(self.base), DATASET=str(self.dataset), OUT=str(self.out),
                        HF_HUB_CACHE=str(self.hub_cache))
        # Run the actual stdlib preflight programs. Patch only disk capacity;
        # intercept the final launcher so Torch and model loading never occur.
        self.python_wrapper = f'''#!{sys.executable}
import json, os, subprocess, sys
from types import SimpleNamespace
args = sys.argv[1:]
with open(os.environ['TRAIN_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({{'tool': 'python', 'args': args,
        'offline': os.environ.get('HF_HUB_OFFLINE'),
        'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE'),
        'albumentations_update_disabled': os.environ.get('NO_ALBUMENTATIONS_UPDATE'),
        'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'transformers_cache': os.environ.get('TRANSFORMERS_CACHE'), 'cwd': os.getcwd(),
        'test_only_env': {{name: os.environ.get(name) for name in (
            'GROOT_SKIP_HF_MODEL_WEIGHTS', 'GROOT_HF_LOCAL_FIRST', 'PYTEST_CURRENT_TEST')}}}}) + '\\n')
if args and args[0] == '-':
    source = sys.stdin.read()
    if 'shutil.disk_usage' in source:
        import shutil
        shutil.disk_usage = lambda path: SimpleNamespace(
            free=int(os.environ.get('TRAIN_TEST_DISK_FREE', str(64 * 1024**3))))
    sys.argv = args
    exec(compile(source, '<actual-launcher-preflight>', 'exec'), {{'__name__': '__main__'}})
elif any(arg.endswith('/scripts/launch_pick_place_n17.py') for arg in args):
    print('FAKE training output; no model or GPU loaded', flush=True)
    sys.exit(int(os.environ.get('TRAIN_TEST_LAUNCH_EXIT', '0')))
else:
    os.execv({sys.executable!r}, [{sys.executable!r}, *args])
'''
        self.write_executable(self.bin / "python3", self.python_wrapper)
        self.write_executable(self.bin / "git", f'''#!{sys.executable}
import json, os, sys
with open(os.environ['TRAIN_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({{'tool': 'git', 'args': sys.argv[1:]}}) + '\\n')
print(os.environ.get('TRAIN_TEST_GROOT_PIN', {PIN!r}))
''')
        self.write_executable(self.bin / "nvidia-smi", f'''#!{sys.executable}
import json, os, sys
with open(os.environ['TRAIN_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({{'tool': 'nvidia-smi', 'args': sys.argv[1:]}}) + '\\n')
if '-x' in sys.argv:
    print(os.environ.get('TRAIN_TEST_GPU_XML', '<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>'))
else:
    print(os.environ.get('TRAIN_TEST_GPU_CSV', '100, 0'))
''')
        self.write_executable(self.bin / "hf", f'''#!{sys.executable}
import json, os, sys
with open(os.environ['TRAIN_TEST_LOG'], 'a') as stream:
    stream.write(json.dumps({{'tool': 'hf', 'args': sys.argv[1:]}}) + '\\n')
sys.exit(91)
''')

    def write_executable(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def run_script(self, *args, settings=None, defaults=False):
        env = dict(self.env)
        if defaults:
            for key in ("GROOT", "BASE_MODEL", "DATASET", "OUT"):
                env.pop(key, None)
        env.update(settings or {})
        return subprocess.run(["bash", str(SCRIPT), *args], env=env, cwd=self.root,
                              text=True, capture_output=True, timeout=10)

    def launch_args(self, result):
        line = next(line for line in result.stdout.splitlines() if line.startswith("Launch command: "))
        return shlex.split(line.removeprefix("Launch command: "))

    def fixture(self, *, cosmos_cache=True, cosmos_cache_root=None):
        """Tiny valid metadata/weights plus a signed prepared copy; no real model."""
        self.base.mkdir()
        modalities = {
            "video": {"modality_keys": ["room", "wrist"], "delta_indices": [0]},
            "state": {"modality_keys": ["single_arm", "gripper"], "delta_indices": [0]},
            "action": {"modality_keys": ["single_arm", "gripper"], "delta_indices": list(range(16)),
                       "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF",
                                           "format": "DEFAULT", "state_key": None}] * 2},
            "language": {"modality_keys": ["annotation.human.task_description"], "delta_indices": [0]},
        }
        files = {
            "config.json": {"model_type": "Gr00tN1d7", "action_horizon": 40},
            "processor_config.json": {"processor_class": "Gr00tN1d7Processor", "processor_kwargs": {
                "modality_configs": {"new_embodiment": modalities}, "max_action_horizon": 40}},
            "embodiment_id.json": {"new_embodiment": 10},
            "statistics.json": {"new_embodiment": {kind: {
                name: {stat: [1.0] * size for stat in ("min", "max", "mean", "std", "q01", "q99")}
                for name, size in (("single_arm", 5), ("gripper", 1))} for kind in ("state", "action")}},
            "model.safetensors.index.json": {"weight_map": {"fixture": "model-00001-of-00001.safetensors"}},
        }
        for name, value in files.items():
            (self.base / name).write_text(json.dumps(value))
        header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        (self.base / "model-00001-of-00001.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<f", 0))
        self.dataset.mkdir()
        (self.dataset / "so100_config.py").write_bytes((REPO / "scripts/so_arm_n17_config.py").read_bytes())
        (self.dataset / "meta").mkdir()
        (self.dataset / "meta/info.json").write_text('{"total_episodes": 1}')
        manifest = {name: hashlib.sha256((self.dataset / name).read_bytes()).hexdigest()
                    for name in ("so100_config.py", "meta/info.json")}
        (self.dataset / "prepared_sha256.json").write_text(json.dumps(manifest))
        (self.dataset / "preparation_report.json").write_text(json.dumps({
            "status": "passed", "model_profile": "so_arm_n17", "groot_revision": PIN}))
        self.write_executable(self.groot / ".venv/bin/python", self.python_wrapper)
        (self.groot / ".venv/bin/activate").write_text(
            f'export PATH={shlex.quote(str(self.groot / ".venv/bin"))}:"$PATH"\n')
        cuda = self.root / "cuda"
        self.write_executable(cuda / "bin/nvcc", "#!/bin/sh\nexit 0\n")
        self.env["CUDA_HOME"] = str(cuda)
        if cosmos_cache:
            repository = (cosmos_cache_root or self.hub_cache) / "models--nvidia--Cosmos-Reason2-2B"
            snapshot = repository / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            (repository / "refs").mkdir()
            (repository / "refs/main").write_text("a" * 40)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "preprocessor_config.json", "video_preprocessor_config.json"):
                (snapshot / name).write_text('{"fixture": true}')
            (snapshot / "chat_template.json").write_text('{"chat_template":"{{ messages }}"}')
            (snapshot / "model.safetensors").write_bytes(
                (self.base / "model-00001-of-00001.safetensors").read_bytes())

    def test_dry_run_defaults_are_v2_n17_and_frozen(self):
        result = self.run_script("--dry_run", defaults=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.launch_args(result)
        home = Path.home()
        expected = {"--base-model-path": home / "sim2real/models/nv_so_arm_n17",
                    "--dataset-path": REPO / "datasets/pick_place_v2_gr00t",
                    "--modality-config-path": REPO / "datasets/pick_place_v2_gr00t/so100_config.py",
                    "--output-dir": home / "sim2real/models/so101_pick_place_v2_nvinit",
                    "--embodiment-tag": "NEW_EMBODIMENT", "--max-steps": "10000",
                    "--save-steps": "1000", "--save-total-limit": "2", "--global-batch-size": "16"}
        self.assertEqual(args[0], str(home / "Isaac-GR00T-N1.7/.venv/bin/python"))
        self.assertEqual(args[2], str(REPO / "scripts/launch_pick_place_n17.py"))
        self.assertEqual(args[args.index("--groot") + 1], str(home / "Isaac-GR00T-N1.7"))
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], str(value))
        for flag in ("--no-tune-diffusion-model", "--no-tune-llm", "--no-tune-visual"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("hue") + 1], "0.02")
        self.assertNotIn("--resume-from-checkpoint", args)
        self.assertIn("1.08 h at 0.3902 s/step", result.stdout)
        self.assertIn("N1.7 batch-16 smoke; excludes loading and checkpoint saving", result.stdout)
        self.assertIn("Allocator: expandable_segments:True", result.stdout)
        self.assertEqual([call["tool"] for call in self.calls()], ["python"])
        self.assertFalse(self.out.exists())

    def test_environment_overrides_and_explicit_resume_flag(self):
        result = self.run_script(settings={"DRY_RUN": "1", "RESUME": "1", "MAX_STEPS": "30",
                                          "SAVE_STEPS": "30", "BATCH": "8", "SAVE_TOTAL_LIMIT": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.launch_args(result)
        self.assertIn("--resume-from-checkpoint", args)
        for flag, value in (("--max-steps", "30"), ("--global-batch-size", "8"),
                            ("--save-total-limit", "1"), ("--output-dir", str(self.out))):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertFalse(self.out.exists())

    def test_invalid_settings_and_arguments_never_create_output(self):
        settings = {"BATCH": "0", "MAX_STEPS": "-1", "SAVE_STEPS": "1.5", "SAVE_TOTAL_LIMIT": "0",
                    "RESUME": "2", "DRY_RUN": "yes", "DATALOADER_NUM_WORKERS": "-1"}
        for key, value in settings.items():
            with self.subTest(key=key):
                result = self.run_script(settings={"DRY_RUN": "1", key: value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(key, result.stderr)
                self.assertFalse(self.out.exists())
        result = self.run_script("--unsupported")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown argument", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_invalid_throughput_never_reaches_preflight(self):
        for value in ("0", "-1", "nan", "inf", "abc"):
            with self.subTest(value=value):
                result = self.run_script(settings={"MEASURED_S_PER_STEP": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("MEASURED_S_PER_STEP must be finite and positive", result.stderr)
                self.assertFalse(self.out.exists())
        self.assertTrue(all(call["tool"] == "python" for call in self.calls()))

    def test_missing_base_prints_exact_manual_download_without_downloading(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        command = (f"hf download nvidia/SO_ARM_Starter_Gr00tN17 --revision {HUB_PIN} "
                   "--include config.json --include processor_config.json "
                   "--include embodiment_id.json --include statistics.json "
                   "--include 'model-*.safetensors' --include model.safetensors.index.json "
                   f"--local-dir {shlex.quote(str(self.base))}")
        self.assertIn(command, result.stdout)
        printed_command = next(line for line in result.stdout.splitlines()
                               if line.startswith("hf download nvidia/SO_ARM_Starter_Gr00tN17 "))
        download_args = shlex.split(printed_command)
        includes = [download_args[index + 1] for index, arg in enumerate(download_args)
                    if arg == "--include"]
        self.assertEqual(includes, ["config.json", "processor_config.json", "embodiment_id.json",
                                    "statistics.json", "model-*.safetensors", "model.safetensors.index.json"])
        # The installed hf CLI consumes one pattern per --include occurrence.
        for index, arg in enumerate(download_args):
            if arg == "--include":
                self.assertTrue(download_args[index + 2].startswith("--"))
        self.assertIn("hf auth login", result.stdout)
        self.assertIn("hf download nvidia/Cosmos-Reason2-2B", result.stdout)
        self.assertIn("Missing BASE_MODEL", result.stderr)
        self.assertNotIn("hf", [call["tool"] for call in self.calls()])
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertFalse(self.base.exists())
        self.assertFalse(self.out.exists())

    def test_existing_checkpoints_require_resume_before_runtime(self):
        self.base.mkdir()
        checkpoint = self.out / "checkpoint-10000"
        checkpoint.mkdir(parents=True)
        marker = checkpoint / "preserve.txt"
        marker.write_text("original")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set RESUME=1", result.stderr)
        result = self.run_script(settings={"RESUME": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("N1.7 runtime missing", result.stderr)
        self.assertNotIn("Set RESUME=1", result.stderr)
        self.assertIn("--resume-from-checkpoint", self.launch_args(result))
        self.assertEqual(marker.read_text(), "original")

    def test_pinned_revision_is_checked_before_model_or_gpu(self):
        self.fixture()
        result = self.run_script(settings={"TRAIN_TEST_GROOT_PIN": "different"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("revision differs", result.stderr)
        self.assertEqual(next(call["args"] for call in self.calls() if call["tool"] == "git"),
                         ["-C", str(self.groot), "rev-parse", "HEAD"])
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertFalse(self.out.exists())

    def test_disk_guard_rejects_below_50_gib_before_gpu(self):
        self.fixture()
        result = self.run_script(settings={"TRAIN_TEST_DISK_FREE": str(50 * 1024**3 - 1)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Need at least 50 GiB free", result.stderr)
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertFalse(self.out.exists())

    def test_signed_dataset_mutation_stops_before_gpu(self):
        self.fixture()
        (self.dataset / "meta/info.json").write_text("changed since preparation")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Prepared checksum mismatch: meta/info.json", result.stderr)
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertFalse(self.out.exists())

    def test_pipeline_preserves_failure_and_logs_with_offline_allocator(self):
        self.fixture()
        result = self.run_script(settings={"TRAIN_TEST_LAUNCH_EXIT": "47",
                                          "TRAIN_TEST_DISK_FREE": str(50 * 1024**3),
                                          "NO_ALBUMENTATIONS_UPDATE": "0",
                                          "GROOT_SKIP_HF_MODEL_WEIGHTS": "1",
                                          "GROOT_HF_LOCAL_FIRST": "1",
                                          "PYTEST_CURRENT_TEST": "inherited test"})
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertIn("FAKE training output", (self.out / "train.log").read_text())
        launch = next(call for call in self.calls() if any(
            arg.endswith("/scripts/launch_pick_place_n17.py") for arg in call.get("args", [])))
        self.assertEqual(launch["offline"], "1")
        self.assertEqual(launch["transformers_offline"], "1")
        self.assertEqual(launch["albumentations_update_disabled"], "1")
        self.assertEqual(launch["allocator"], "expandable_segments:True")
        self.assertEqual(launch["transformers_cache"], str(self.hub_cache))
        self.assertEqual(launch["test_only_env"], {
            "GROOT_SKIP_HF_MODEL_WEIGHTS": None, "GROOT_HF_LOCAL_FIRST": None,
            "PYTEST_CURRENT_TEST": None})
        self.assertNotIn("--resume-from-checkpoint", launch["args"])
        self.assertEqual(len([call for call in self.calls() if call["tool"] == "nvidia-smi"]), 2)
        self.assertNotIn("hf", [call["tool"] for call in self.calls()])

    def test_compute_busy_gpu_never_creates_training_output(self):
        self.fixture()
        xml = "<nvidia_smi_log><gpu><processes><process_info><pid>123</pid>" \
              "<process_name>python</process_name><used_memory>100 MiB</used_memory>" \
              "</process_info></processes></gpu></nvidia_smi_log>"
        result = self.run_script(settings={"TRAIN_TEST_GPU_XML": xml})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU 0 is busy", result.stderr)
        self.assertIn("PID 123: python", result.stderr)
        self.assertFalse(self.out.exists())
        self.assertFalse(any(any(arg.endswith("/scripts/launch_pick_place_n17.py")
                                 for arg in call.get("args", [])) for call in self.calls()))

    def test_missing_cosmos_cache_stops_before_gpu_output_or_download(self):
        self.fixture(cosmos_cache=False)
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cosmos cache validation failed", result.stderr)
        self.assertIn("https://huggingface.co/nvidia/Cosmos-Reason2-2B", result.stderr)
        self.assertIn("hf auth login", result.stderr)
        self.assertIn("hf download nvidia/Cosmos-Reason2-2B", result.stderr)
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertNotIn("hf", [call["tool"] for call in self.calls()])
        self.assertFalse(self.out.exists())
        self.assertFalse(self.hub_cache.exists())

    def test_truncated_cosmos_weights_stop_before_gpu_output_or_download(self):
        self.fixture()
        weights = self.hub_cache / "models--nvidia--Cosmos-Reason2-2B/snapshots" / ("a" * 40) / "model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1])
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Incomplete weights: model.safetensors", result.stderr)
        self.assertNotIn("nvidia-smi", [call["tool"] for call in self.calls()])
        self.assertNotIn("hf", [call["tool"] for call in self.calls()])
        self.assertFalse(self.out.exists())

    def test_relative_legacy_cache_keeps_startup_path_after_runtime_chdir(self):
        self.fixture()
        result = self.run_script(settings={"TRANSFORMERS_CACHE": "hf_hub"})
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = next(call for call in self.calls() if any(
            arg.endswith("/scripts/launch_pick_place_n17.py") for arg in call.get("args", [])))
        self.assertEqual(launch["transformers_cache"], str(self.hub_cache))
        self.assertEqual(launch["cwd"], str(self.groot))

    def test_empty_legacy_cache_keeps_startup_directory_after_runtime_chdir(self):
        self.fixture(cosmos_cache_root=self.root)
        result = self.run_script(settings={"TRANSFORMERS_CACHE": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        launch = next(call for call in self.calls() if any(
            arg.endswith("/scripts/launch_pick_place_n17.py") for arg in call.get("args", [])))
        self.assertEqual(launch["transformers_cache"], str(self.root))
        self.assertEqual(launch["cwd"], str(self.groot))


class EmbeddedGpuGuardTests(unittest.TestCase):
    def run_guard(self, memory=100, utilization=0, processes=()):
        # Execute the exact embedded program with subprocess.run mocked; no
        # nvidia-smi binary or GPU is touched by these boundary checks.
        source = re.search(r"python - <<'PY'\n(.*?)\nPY\n", SCRIPT.read_text(), re.S).group(1)
        entries = "".join(f"<process_info><pid>{pid}</pid><process_name>{name}</process_name>"
                          f"<used_memory>{used} MiB</used_memory></process_info>"
                          for pid, name, used in processes)
        xml = f"<nvidia_smi_log><gpu><processes>{entries}</processes></gpu></nvidia_smi_log>"
        def fake_run(args, **kwargs):
            self.assertEqual(args[0], "nvidia-smi")
            output = xml if "-x" in args else f"{memory}, {utilization}\n"
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
        capture = io.StringIO()
        with patch("subprocess.run", side_effect=fake_run), contextlib.redirect_stdout(capture):
            exec(compile(source, "<actual-embedded-gpu-guard>", "exec"), {})
        return capture.getvalue()

    def test_accepts_small_known_desktop_baseline(self):
        output = self.run_guard(memory=128, utilization=10,
                                processes=((1, "/usr/lib/xorg/Xorg", 100),
                                           (2, "/usr/bin/xfce4-session", 28)))
        self.assertIn("GPU 0 available", output)

    def test_rejects_compute_and_excess_memory_or_utilization(self):
        cases = ({"processes": ((3, "python", 0),)},
                 {"memory": 513}, {"utilization": 11},
                 {"processes": ((4, "/usr/lib/xorg/Xorg", 129),)},
                 {"processes": ((5, "/usr/lib/xorg/Xorg", "N/A"),)})
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaisesRegex(SystemExit, "GPU 0 is busy"):
                self.run_guard(**settings)


if __name__ == "__main__":
    unittest.main()
