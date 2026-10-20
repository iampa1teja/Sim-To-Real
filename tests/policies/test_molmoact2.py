"""MolmoAct2 request handling and CPU-only action precision regressions."""
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]
sys.modules.setdefault("server_core", importlib.import_module("policy_servers.core"))
spec = importlib.util.spec_from_file_location("molmoact2_server", ROOT / "policies/molmoact2/server.py")


def client_policy(options=None):
    from sim_to_real_so101.policy_client.contract import load_client_policy
    from sim_to_real_so101.policy_client.discovery import load_manifest
    manifest = load_manifest(ROOT / "policies/molmoact2")
    return load_client_policy("molmoact2", manifest, options or {})


JOINT_ORDER = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
               "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def test_client_applies_official_degree_frame_and_inverse_actions():
    client = client_policy()
    state = np.array([0, -90, 90, 45, -30, 20], dtype=np.float32)
    expected = np.array([0, 180, 180, 45, -30, 20], dtype=np.float32)
    images = request()["images"]
    built = client.build_request(state, images, "Pick the cube")
    np.testing.assert_array_equal(built["state"], expected)
    decoded = client.parse_response({"actions": np.tile(expected, (30, 1))}, JOINT_ORDER)
    np.testing.assert_array_equal(decoded, np.tile(state, (30, 1)))
    np.testing.assert_array_equal(state, [0, -90, 90, 45, -30, 20])


def test_client_converts_normalized_units_and_preserves_gripper_percent():
    client = client_policy({"input_joint_units": "lerobot_normalized",
                            "arm_degrees_per_unit": "1.18,1,.95,.95,1.6"})
    normalized = np.array([50, -100, 100, 0, 25, 40], dtype=np.float32)
    expected = np.array([59, 190, 185, 0, 40, 40], dtype=np.float32)
    built = client.build_request(normalized, request()["images"], "Pick the cube")
    np.testing.assert_allclose(built["state"], expected, atol=1e-5)
    decoded = client.parse_response({"actions": expected[None]}, JOINT_ORDER)
    np.testing.assert_allclose(decoded[0], normalized, atol=1e-5)
    # A model shoulder-lift target of 180 degrees must command -90 in
    # LeRobot units, rather than +180 which would hit the opposite joint stop.
    np.testing.assert_allclose(client.parse_response(
        {"actions": [[0, 180, 180, 0, 0, 25]]}, JOINT_ORDER)[0],
        [0, -90, 90 / .95, 0, 0, 25], atol=1e-5)


def test_explicit_model_units_are_not_transformed_twice():
    client = client_policy({"input_joint_units": "model"})
    state = np.array([0, 189, 181, 60, -3, 1], dtype=np.float32)
    np.testing.assert_array_equal(client.build_request(state, request()["images"], "task")["state"], state)
    np.testing.assert_array_equal(client.parse_response({"actions": state[None]}, JOINT_ORDER), state[None])


def test_normalized_conversion_matches_lerobot_encoder_calibration():
    # Independently derive both public LeRobot representations from encoder
    # ticks. DEGREES is relative to the calibrated midpoint, even when that
    # midpoint differs from the motor's half-turn home (2048 ticks).
    lower = np.array([725, 831, 877, 888, 0, 2039], dtype=np.float64)
    upper = np.array([3411, 3326, 3109, 3245, 4095, 3514], dtype=np.float64)
    encoder = np.array([1900, 1200, 2800, 2400, 1600, 2500], dtype=np.float64)
    normalized = (encoder - lower) / (upper - lower) * 200 - 100
    normalized[-1] = (encoder[-1] - lower[-1]) / (upper[-1] - lower[-1]) * 100
    degrees = (encoder[:5] - (lower[:5] + upper[:5]) / 2) * 360 / 4095
    expected = np.concatenate((degrees, normalized[-1:]))
    expected[1] = 90 - degrees[1]
    expected[2] += 90
    client = client_policy({
        "input_joint_units": "lerobot_normalized",
        "arm_degrees_per_unit": ",".join(str(v) for v in (upper[:5] - lower[:5]) * 360 / 4095 / 200),
    })
    built = client.build_request(normalized, request()["images"], "Pick the cube")
    np.testing.assert_allclose(built["state"], expected, atol=2e-5)
    # An action target at the encoder midpoint must become normalized zero;
    # neither USD-limit midpoints nor the half-turn home belong in this frame.
    model_targets = np.stack((expected, [0, 90, 90, 0, 0, 75]))
    decoded = client.parse_response({"actions": model_targets}, JOINT_ORDER)
    np.testing.assert_allclose(decoded[0], normalized, atol=2e-5)
    np.testing.assert_allclose(decoded[1], [0, 0, 0, 0, 0, 75], atol=2e-5)


@pytest.mark.parametrize("scales", ["", "1,1,1,1", "1,1,1,1,0", "1,1,nan,1,1", "bad,1,1,1,1"])
def test_normalized_joint_units_require_valid_calibrated_scales(scales):
    with pytest.raises(ValueError, match="arm_degrees_per_unit"):
        client_policy({"input_joint_units": "lerobot_normalized", "arm_degrees_per_unit": scales})


def test_client_rejects_invalid_units_and_nonfinite_joints():
    with pytest.raises(ValueError, match="input_joint_units"):
        client_policy({"input_joint_units": "radians"})
    client = client_policy()
    with pytest.raises(ValueError, match="finite"):
        client.build_request(np.full(6, np.nan), request()["images"], "task")
    with pytest.raises(ValueError, match="finite"):
        client.parse_response({"actions": np.full((8, 6), np.inf)}, JOINT_ORDER)


def load_server():
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    return load_server()


class FakeModel:
    def __init__(self, chunk):
        self.chunk = chunk
        self.calls = []

    def predict_action(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(actions=self.chunk)


def ready(server, chunk):
    policy = server.MolmoAct2ServerPolicy()
    policy.model = FakeModel(chunk)
    policy.processor = object()
    return policy


def request():
    return {"state": np.arange(6, dtype=np.float32),
            "images": {"room": np.zeros((4, 5, 3), np.uint8), "wrist": np.ones((4, 5, 3), np.uint8)},
            "instruction": "Pick up the blue cube"}


@pytest.fixture
def no_gpu_torch(monkeypatch):
    """infer() only needs torch's context managers; use a stub when torch is not installed."""
    if importlib.util.find_spec("torch") is None:
        from contextlib import nullcontext
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
            inference_mode=nullcontext, autocast=lambda *a, **k: nullcontext(), bfloat16="bf16"))
    pytest.importorskip("PIL")


def test_infer_returns_float32_chunk_and_passes_model_arguments(server, no_gpu_torch):
    chunk = np.arange(48, dtype=np.float64).reshape(1, 8, 6)
    policy = ready(server, chunk)
    result = policy.infer(request())
    assert result["actions"].shape == (8, 6) and result["actions"].dtype == np.float32
    call = policy.model.calls[0]
    assert call["norm_tag"] == "so100_so101_molmoact2" and call["task"] == "Pick up the blue cube"
    assert len(call["images"]) == 2 and call["inference_action_mode"] == "continuous"
    np.testing.assert_array_equal(call["state"], np.arange(6, dtype=np.float32))


def test_decode_actions_validates_shape(server):
    assert server.decode_actions(np.zeros((16, 6))).shape == (16, 6)
    with pytest.raises(ValueError):
        server.decode_actions(np.zeros((16, 7)))


def test_bad_state_and_unloaded_policy_are_rejected(server):
    with pytest.raises(RuntimeError, match="not loaded"):
        server.MolmoAct2ServerPolicy().infer(request())


def test_info_reset_and_seed_validation(server):
    policy = ready(server, np.zeros((8, 6)))
    assert policy.info()["name"] == "molmoact2"
    assert policy.reset() == {"seed": None}
    for bad in (-1, 2**32, True, "1"):
        with pytest.raises(ValueError):
            server.seed_everything(bad)


def test_load_rejects_missing_checkpoint(server):
    policy = server.MolmoAct2ServerPolicy()
    with pytest.raises(ValueError):
        policy.load("", "cuda", {})
    with pytest.raises(FileNotFoundError):
        policy.load("/no/such/dir", "cuda", {})
    with pytest.raises(ValueError, match="dtype"):
        policy.load("/", "cuda", {"dtype": "int4"})


@pytest.mark.parametrize("extra_tokens", [["<im_start>", "<im_end>"], {"image_token": "<im_start>"}, None])
def test_load_processor_adapts_only_list_form_special_tokens(server, monkeypatch, extra_tokens):
    """The official checkpoint list must not reach 4.57's dict-only argument."""
    from types import ModuleType

    calls = []
    processor = object()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = SimpleNamespace(from_pretrained=lambda *a, **kw: (calls.append((a, kw)), processor)[1])
    tokenization = ModuleType("transformers.models.auto.tokenization_auto")
    tokenization.get_tokenizer_config = lambda checkpoint: {
        "extra_special_tokens": extra_tokens,
        "additional_special_tokens": ["<existing>"],
    }
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models.auto.tokenization_auto", tokenization)
    assert server.load_processor("checkpoint") is processor
    args, options = calls.pop()
    assert args == ("checkpoint",)
    assert options["trust_remote_code"] is True
    assert options["use_fast"] is True
    if isinstance(extra_tokens, list):
        assert options["extra_special_tokens"] == {}
        assert options["additional_special_tokens"] == ["<existing>", *extra_tokens]
    else:
        assert "extra_special_tokens" not in options
        assert "additional_special_tokens" not in options


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_loaded_policy_preserves_physical_target_precision_without_changing_weights(server, monkeypatch, dtype_name):
    """Exercise load -> native predict_action -> decode, without a GPU/model load."""
    from contextlib import nullcontext
    from types import ModuleType

    torch = pytest.importorskip("torch")

    class Stats:
        def unnormalize_action(self, actions, norm_tag):
            assert norm_tag == server.DEFAULT_NORM_TAG
            # Match the released normalizer's affine in float32 followed by
            # its cast back into the incoming trajectory dtype.
            return (actions.float() * 20 + 189.140625).to(actions.dtype)

    class Model(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.12345], dtype=dtype))
            self.stats = Stats()

        def _get_robot_stats(self):
            return self.stats

        def predict_action(self, **kwargs):
            normalized = torch.zeros((1, 30, 6), dtype=self.weight.dtype)
            return SimpleNamespace(actions=self.stats.unnormalize_action(normalized, kwargs["norm_tag"]).float())

    models = []

    def from_pretrained(checkpoint, **kwargs):
        model = Model(kwargs["dtype"])
        models.append(model)
        return model

    transformers = ModuleType("transformers")
    transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(server, "load_processor", lambda checkpoint: object())
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    policy = server.MolmoAct2ServerPolicy()
    policy.load("fake-checkpoint", "cpu", {"dtype": dtype_name})
    before = models[0].weight.detach().clone()
    result = policy.infer(request())
    np.testing.assert_array_equal(result["actions"], np.full((30, 6), 189.140625, dtype=np.float32))
    assert models[0].weight.dtype == getattr(torch, dtype_name)
    assert torch.equal(before, models[0].weight)
    assert policy.info()["action_denormalization_dtype"] == "float32"


def test_precision_hook_is_idempotent_and_scoped_to_one_models_stats(server):
    torch = pytest.importorskip("torch")

    class Stats:
        def unnormalize_action(self, action, norm_tag):
            return (action.float() + 189.140625).to(action.dtype)

    patched, untouched = Stats(), Stats()
    model = SimpleNamespace(_get_robot_stats=lambda: patched)
    server.preserve_action_denormalization_precision(model)
    wrapper = patched.unnormalize_action
    server.preserve_action_denormalization_precision(model)
    assert patched.unnormalize_action is wrapper
    values = torch.zeros((1, 6), dtype=torch.bfloat16)
    precise = patched.unnormalize_action(values, "tag")
    assert precise.dtype == torch.float32
    assert precise[0, 0].item() == 189.140625
    assert untouched.unnormalize_action(values, "tag").dtype == torch.bfloat16


def test_released_checkpoint_normalizer_retains_subdegree_targets_on_cpu(server):
    """Use the installed checkpoint's actual normalizer and stats, not its weights."""
    import ast
    import json
    import os
    from typing import Any, Mapping, Optional

    torch = pytest.importorskip("torch")
    checkpoint = Path(os.environ.get("MOLMOACT2_TEST_CHECKPOINT", "/external_storage/molmoact2/molmoact2_so101"))
    source_path, stats_path = checkpoint / "modeling_molmoact2.py", checkpoint / "norm_stats.json"
    if not source_path.is_file() or not stats_path.is_file():
        pytest.skip("Set MOLMOACT2_TEST_CHECKPOINT to run the released-normalizer regression")
    # Isolate this pure CPU class from the checkpoint's heavy model imports.
    tree = ast.parse(source_path.read_text())
    names = {"_to_array", "_to_mask", "_FeatureNormalizer"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"Any": Any, "Optional": Optional, "Mapping": Mapping, "np": np, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), "exec"), namespace)
    payload = json.loads(stats_path.read_text())
    metadata = payload["metadata_by_tag"][server.DEFAULT_NORM_TAG]
    normalizer = namespace["_FeatureNormalizer"].from_stats(metadata["action_stats"], payload["norm_mode"])
    stats = SimpleNamespace(unnormalize_action=lambda action, tag: normalizer.unnormalize(action))
    normalized = torch.linspace(-1, 1, 401).to(torch.bfloat16)[:, None].expand(-1, 6)
    original = stats.unnormalize_action(normalized, server.DEFAULT_NORM_TAG).float()
    expected = normalizer.unnormalize(normalized.float())
    assert (original - expected).abs().amax().item() > .4
    server.preserve_action_denormalization_precision(SimpleNamespace(_get_robot_stats=lambda: stats))
    actual = stats.unnormalize_action(normalized, server.DEFAULT_NORM_TAG)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def precision_checkpoint(tmp_path):
    """Tiny original-FP32 weights with the released model's module boundaries."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    class MolmoAct2RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2,), .1234567))

    class MolmoAct2RotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.tensor([1., .01234567]))
            self.original_inv_freq = self.inv_freq
            self.register_buffer("_pos_sin_cache", torch.ones(2), persistent=False)
            self.register_buffer("_pos_cos_cache", torch.ones(2), persistent=False)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_expert = torch.nn.Module()
            self.model.action_expert.action_embed = torch.nn.Linear(2, 2)
            self.model.action_expert.norm = torch.nn.LayerNorm(2)
            self.model.action_expert_depth_gate = torch.nn.Linear(2, 1)
            self.model.norm = MolmoAct2RMSNorm()
            self.model.rotary = MolmoAct2RotaryEmbedding()
            self.model.vision = torch.nn.Linear(2, 2)
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(.1234567)

    model = Model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    model.bfloat16()
    return model, original, tmp_path


@pytest.mark.parametrize("sharded", [False, True])
def test_mixed_action_profile_restores_original_fp32_values_and_keeps_backbone_bf16(server, precision_checkpoint, sharded):
    import json
    import torch
    from safetensors.torch import save_file

    model, original, directory = precision_checkpoint
    if sharded:
        weight_map = {}
        for index in range(2):
            shard = f"model-0000{index+1}-of-00002.safetensors"
            selected = {name: value for position, (name, value) in enumerate(original.items()) if position % 2 == index}
            save_file(selected, directory / shard)
            weight_map.update(dict.fromkeys(selected, shard))
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(original, directory / "model.safetensors")
    old_expert = model.model.action_expert.action_embed.weight.detach().float().clone()
    assert not torch.equal(old_expert, original["model.action_expert.action_embed.weight"])
    server.restore_float32_action_modules(model, str(directory))
    server.reset_float32_rotary_caches(model)
    for name, value in model.state_dict().items():
        if name.startswith("model.vision."):
            assert value.dtype == torch.bfloat16
            torch.testing.assert_close(value, original[name].bfloat16(), rtol=0, atol=0)
        else:
            assert value.dtype == torch.float32, name
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    assert model.model.action_expert.action_embed.weight.dtype == torch.float32
    assert model.model.rotary.original_inv_freq is model.model.rotary.inv_freq
    assert model.model.rotary._pos_sin_cache.numel() == 0
    assert model.model.rotary._pos_cos_cache.dtype == torch.float32


@pytest.mark.parametrize("corruption", ["missing", "shape"])
def test_mixed_action_restore_rejects_incompatible_checkpoints(server, precision_checkpoint, corruption):
    import torch
    from safetensors.torch import save_file

    model, original, directory = precision_checkpoint
    name = "model.action_expert.action_embed.weight"
    if corruption == "missing":
        del original[name]
    else:
        original[name] = torch.ones((5,))
    save_file(original, directory / "model.safetensors")
    with pytest.raises(ValueError, match="missing" if corruption == "missing" else "shape mismatch"):
        server.restore_float32_action_modules(model, str(directory))


def test_load_restores_action_precision_before_gpu_move_and_refreshes_rotary_alias_after(server, monkeypatch):
    from types import ModuleType

    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            return self

    model = Model()
    transformers = ModuleType("transformers")
    transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(server, "load_processor", lambda checkpoint: object())
    monkeypatch.setattr(server, "restore_float32_action_modules", lambda loaded, checkpoint: calls.append(("restore", checkpoint)))
    monkeypatch.setattr(server, "reset_float32_rotary_caches", lambda loaded: calls.append(("rotary", loaded)))
    monkeypatch.setattr(server, "preserve_action_denormalization_precision", lambda loaded: calls.append(("denorm", loaded)))
    policy = server.MolmoAct2ServerPolicy()
    policy.load("checkpoint", "cuda", {"dtype": "bfloat16", "action_dtype": "float32"})
    assert calls == [("restore", "checkpoint"), ("to", "cuda"), ("rotary", model), ("denorm", model)]
    assert policy.info()["dtype"] == "bfloat16"
    assert policy.info()["action_dtype"] == "float32"


@pytest.mark.parametrize("options", [{"action_dtype": "float16"}, {"dtype": "float32", "action_dtype": "bfloat16"}])
def test_load_rejects_unsupported_action_precision_before_loading_weights(server, options):
    with pytest.raises(ValueError, match="action_dtype"):
        server.MolmoAct2ServerPolicy().load("checkpoint", "cpu", options)
