"""Policy reset seeding keeps Python, NumPy and Torch inference reproducible."""

import random

import numpy as np
import pytest

from core.policy_server.seeding import seed_everything


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5, "42"])
def test_rejects_invalid_seeds(seed):
    with pytest.raises(ValueError, match="uint32 integer"):
        seed_everything(seed)


def test_reseeding_reproduces_rngs_and_preserves_optional_cudnn_policy():
    import torch

    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        seed_everything(1984)
        first = random.random(), np.random.rand(3), torch.rand(3)
        seed_everything(1984)
        second = random.random(), np.random.rand(3), torch.rand(3)
        assert first[0] == second[0]
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
        seed_everything(1984, deterministic_cudnn=True)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
