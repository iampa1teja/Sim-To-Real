"""Read-only offline cache validation using tiny CPU fixtures in test temp dirs."""

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts/gr00t_model_profiles.py"
spec = importlib.util.spec_from_file_location("cosmos_cache_profiles", MODULE)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)


def tiny_weights(path, name="weight"):
    header = json.dumps({name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))


class CosmosCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cosmos-cache-test-")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "hub"
        self.repository = self.cache / "models--nvidia--Cosmos-Reason2-2B"
        self.revision = "b" * 40
        self.snapshot = self.repository / "snapshots" / self.revision
        self.snapshot.mkdir(parents=True)
        (self.repository / "refs").mkdir()
        self.reference = self.repository / "refs/main"
        self.reference.write_text(self.revision)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "preprocessor_config.json", "video_preprocessor_config.json"):
            (self.snapshot / name).write_text('{"fixture": true}')
        (self.snapshot / "chat_template.json").write_text('{"chat_template":"{{ messages }}"}')
        self.weights = self.snapshot / "model.safetensors"
        tiny_weights(self.weights)

    def indexed(self):
        self.weights.unlink()
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for index, name in enumerate(names):
            tiny_weights(self.snapshot / name, f"weight{index}")
        (self.snapshot / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {f"weight{index}": name for index, name in enumerate(names)}}))
        return names

    def test_current_single_weights_cache_passes_without_modification(self):
        before = {str(path.relative_to(self.cache)): path.read_bytes()
                  for path in self.cache.rglob("*") if path.is_file()}
        report = profiles.inspect_cosmos_cache(self.cache)
        self.assertEqual(report["revision"], self.revision)
        self.assertEqual(report["weight_shards"], ["model.safetensors"])
        self.assertEqual(report["repo_id"], "nvidia/Cosmos-Reason2-2B")
        self.assertEqual(before, {str(path.relative_to(self.cache)): path.read_bytes()
                                 for path in self.cache.rglob("*") if path.is_file()})

    def test_cache_alias_precedence_matches_transformers_and_hub_defaults(self):
        env = {"HF_HOME": str(self.cache / "home"), "HF_HUB_CACHE": str(self.cache / "modern"),
               "HUGGINGFACE_HUB_CACHE": str(self.cache / "old_hub"),
               "TRANSFORMERS_CACHE": str(self.cache / "transformers"),
               "PYTORCH_TRANSFORMERS_CACHE": str(self.cache / "old_transformers"),
               "PYTORCH_PRETRAINED_BERT_CACHE": str(self.cache / "bert")}
        for key in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE",
                    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            self.assertEqual(profiles.transformers_hub_cache(env), Path(env[key]))
            env.pop(key)
        self.assertEqual(profiles.transformers_hub_cache(env), self.cache / "home/hub")
        self.assertEqual(profiles.transformers_hub_cache({"XDG_CACHE_HOME": str(self.cache)}),
                         self.cache / "huggingface/hub")
        self.assertEqual(profiles.transformers_hub_cache({}), Path.home() / ".cache/huggingface/hub")

    def test_hf_home_resolution_uses_its_hub_subdirectory(self):
        report = profiles.inspect_cosmos_cache(environ={"HF_HOME": str(self.cache.parent)})
        self.assertEqual(report["snapshot"], str(self.snapshot))

    def test_reference_and_snapshot_must_be_valid(self):
        for value in ("", "../escape", "/absolute", "c" * 40):
            with self.subTest(value=value):
                self.reference.write_text(value)
                with self.assertRaises(ValueError):
                    profiles.inspect_cosmos_cache(self.cache)
        self.reference.unlink()
        with self.assertRaises(FileNotFoundError):
            profiles.inspect_cosmos_cache(self.cache)

    def test_missing_empty_and_malformed_processor_files_are_rejected(self):
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
            path = self.snapshot / name
            original = path.read_bytes()
            for invalid in (None, b"", b"{"):
                with self.subTest(name=name, invalid=invalid):
                    if invalid is None:
                        path.unlink()
                    else:
                        path.write_bytes(invalid)
                    with self.assertRaises(ValueError):
                        profiles.inspect_cosmos_cache(self.cache)
                    path.write_bytes(original)

    def test_processor_template_is_required_even_with_tokenizer_template(self):
        (self.snapshot / "chat_template.json").unlink()
        (self.snapshot / "tokenizer_config.json").write_text('{"chat_template":"fixture"}')
        with self.assertRaisesRegex(ValueError, "Missing Cosmos processor chat template"):
            profiles.inspect_cosmos_cache(self.cache)
        (self.snapshot / "chat_template.jinja").write_text("fixture template")
        profiles.inspect_cosmos_cache(self.cache)

    def test_legacy_json_template_requires_actual_nonempty_payload(self):
        for document in ({"fixture": True}, {}, {"chat_template": ""}, {"chat_template": "  "},
                         {"chat_template": None}, {"chat_template": []}):
            with self.subTest(document=document):
                (self.snapshot / "chat_template.json").write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "nonempty chat_template"):
                    profiles.inspect_cosmos_cache(self.cache)

    def test_modern_cache_expands_variables_and_legacy_empty_is_preserved(self):
        env = {"HF_HUB_CACHE": "$HOME/custom_hub", "HOME": str(Path.home())}
        self.assertEqual(profiles.transformers_hub_cache(env), Path.home() / "custom_hub")
        env["TRANSFORMERS_CACHE"] = ""
        self.assertEqual(profiles.transformers_hub_cache(env), Path.cwd())
        env["TRANSFORMERS_CACHE"] = "relative-cache"
        self.assertEqual(profiles.transformers_hub_cache(env), Path.cwd() / "relative-cache")

    def test_resolver_matches_installed_pinned_hf_and_transformers_api(self):
        groot = Path(os.environ.get("GROOT_N17_CHECKOUT", Path.home() / "Isaac-GR00T-N1.7"))
        python = groot / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("Pinned N1.7 runtime unavailable for API comparison")
        code = '''
import importlib.util, pathlib, sys
import transformers
from transformers.utils.hub import TRANSFORMERS_CACHE
assert transformers.__version__ == '4.57.3'
spec = importlib.util.spec_from_file_location('cache_profiles', sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert module.transformers_hub_cache() == pathlib.Path(TRANSFORMERS_CACHE).resolve()
'''
        env = dict(os.environ)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                    "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE", "XDG_CACHE_HOME"):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1")
        for setting in ({"HF_HUB_CACHE": "$HOME/custom_hub"},
                        {"TRANSFORMERS_CACHE": "relative-cache"}, {"TRANSFORMERS_CACHE": ""},
                        {"PYTORCH_TRANSFORMERS_CACHE": ""}, {"HF_HOME": ""}):
            with self.subTest(setting=setting):
                result = subprocess.run([str(python), "-c", code, str(MODULE)], cwd=self.cache.parent,
                                        env={**env, **setting}, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_template_payload_matches_installed_processor_api(self):
        groot = Path(os.environ.get("GROOT_N17_CHECKOUT", Path.home() / "Isaac-GR00T-N1.7"))
        python = groot / ".venv/bin/python"
        if not python.is_file():
            self.skipTest("Pinned N1.7 runtime unavailable for processor API comparison")
        code = '''
import json, pathlib, sys
from transformers import Qwen3VLProcessor
snapshot = pathlib.Path(sys.argv[1]); path = snapshot / 'chat_template.json'
processor_dict, kwargs = Qwen3VLProcessor.get_processor_dict(str(snapshot), local_files_only=True)
assert processor_dict['chat_template'] == '{{ messages }}'
path.write_text(json.dumps({'fixture': True}))
try:
    Qwen3VLProcessor.get_processor_dict(str(snapshot), local_files_only=True)
except KeyError as error:
    assert error.args == ('chat_template',)
else:
    raise AssertionError('actual processor accepted a missing template payload')
path.unlink()
(snapshot / 'tokenizer_config.json').write_text(json.dumps({'chat_template':'embedded only'}))
processor_dict, kwargs = Qwen3VLProcessor.get_processor_dict(str(snapshot), local_files_only=True)
assert 'chat_template' not in processor_dict and 'chat_template' not in kwargs
'''
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([str(python), "-c", code, str(self.snapshot)], env=env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sharded_cache_checks_every_indexed_file_and_tensor(self):
        shards = self.indexed()
        self.assertEqual(profiles.inspect_cosmos_cache(self.cache)["weight_shards"], shards)
        path = self.snapshot / shards[1]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty Cosmos cache file"):
            profiles.inspect_cosmos_cache(self.cache)
        tiny_weights(path, "wrong_tensor")
        with self.assertRaisesRegex(ValueError, "Indexed tensors missing"):
            profiles.inspect_cosmos_cache(self.cache)

    def test_invalid_shard_names_and_empty_index_are_rejected(self):
        self.indexed()
        for mapping in ({}, {"weight": "../escape.safetensors"}, {"weight": "/absolute.safetensors"},
                        {"weight": "optimizer.pt"}, {"weight": None}):
            with self.subTest(mapping=mapping):
                (self.snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
                with self.assertRaises(ValueError):
                    profiles.inspect_cosmos_cache(self.cache)

    def test_truncated_and_invalid_weight_headers_are_rejected(self):
        original = self.weights.read_bytes()
        for invalid in (b"", original[:7], struct.pack("<Q", 0), original[:-1], original + b"extra"):
            with self.subTest(size=len(invalid)):
                self.weights.write_bytes(invalid)
                with self.assertRaises(ValueError):
                    profiles.inspect_cosmos_cache(self.cache)

    def test_standard_blob_symlinks_work_and_external_symlink_is_rejected(self):
        blob = self.repository / "blobs/weights"
        blob.parent.mkdir()
        blob.write_bytes(self.weights.read_bytes())
        self.weights.unlink()
        self.weights.symlink_to(blob)
        profiles.inspect_cosmos_cache(self.cache)
        self.weights.unlink()
        outside = self.cache.parent / "outside.safetensors"
        tiny_weights(outside)
        self.weights.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "leaves repository"):
            profiles.inspect_cosmos_cache(self.cache)

    def test_cli_missing_cache_is_actionable_and_offline(self):
        self.reference.unlink()
        result = subprocess.run([sys.executable, str(MODULE), "--cosmos-cache", "--hf-hub-cache", str(self.cache)],
                                text=True, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cosmos cache validation failed", result.stderr)
        self.assertIn("https://huggingface.co/nvidia/Cosmos-Reason2-2B", result.stderr)
        self.assertIn("hf auth login\nhf download nvidia/Cosmos-Reason2-2B", result.stderr)


if __name__ == "__main__":
    unittest.main()
