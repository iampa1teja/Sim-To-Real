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
