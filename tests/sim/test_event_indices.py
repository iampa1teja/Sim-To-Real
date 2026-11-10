"""EventManager reset selectors preserve environment selection and integer indexing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_helper():
    path = Path(__file__).resolve().parents[2] / "src/sim/envs/mdp/resets.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "env_indices")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["env_indices"]


@pytest.mark.parametrize("selector,expected", [
    (None, [0, 1, 2, 3]), (slice(None), [0, 1, 2, 3]), (slice(1, 4, 2), [1, 3]),
    ([], []), ([3, 1], [3, 1]), (torch.tensor([2, 0]), [2, 0]),
])
def test_event_indices(selector, expected):
    ids = load_helper()(SimpleNamespace(num_envs=4, device="cpu"), selector)
    assert ids.tolist() == expected
    assert ids.dtype == torch.long
    values = torch.arange(4)
    assert values[ids].tolist() == expected
