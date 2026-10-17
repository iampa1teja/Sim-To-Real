"""MolmoAct2 plugin: request handling with a fake model (no torch, no weights)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
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
    pytest.importorskip("zmq")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server():
    try:
        module = load_server()
    except ModuleNotFoundError as error:  # server_core is mounted at /opt/so101 in the container
        pytest.skip(str(error))
    return module


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
