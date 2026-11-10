"""Run the actual Warp solver kernels on CPU against the captured contact failure."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from sim.solver_numerics import install_mjwarp_precision_guard


def solve(hessian, gradient, *, guarded, done=False):
    from mujoco_warp._src import solver

    n = len(gradient)
    context = SimpleNamespace(
        grad=wp.array(np.asarray([gradient], dtype=np.float32), device="cpu"),
        h=wp.array(np.asarray([hessian], dtype=np.float32), device="cpu"),
        done=wp.array([done], dtype=wp.bool, device="cpu"),
        search=wp.full((1, n), value=17.0, device="cpu"),
        search_dot=wp.full(1, value=23.0, device="cpu"),
        newton_decrement=wp.full(1, value=29.0, device="cpu"),
    )
    model = SimpleNamespace(nv=n, block_dim=SimpleNamespace(update_gradient_cholesky=32))
    data = SimpleNamespace(nworld=1, nefc=wp.array([1], dtype=wp.int32, device="cpu"))
    function = solver._cholesky_factorize_solve
    if not guarded:
        function = inspect.unwrap(function)
    with wp.ScopedDevice("cpu"):
        function(model, data, context)
    return context.search.numpy()[0], context.search_dot.numpy()[0], context.newton_decrement.numpy()[0]


@pytest.fixture(autouse=True)
def install_guard(monkeypatch):
    from mujoco_warp._src import solver

    monkeypatch.setattr(solver, "_cholesky_factorize_solve", inspect.unwrap(solver._cholesky_factorize_solve))
    install_mjwarp_precision_guard()
    installed = solver._cholesky_factorize_solve
    install_mjwarp_precision_guard()
    assert solver._cholesky_factorize_solve is installed


def test_captured_contact_search_matches_double_precision_solution():
    fixture = json.loads((Path(__file__).parent / "fixtures/contact_hessian.json").read_text())
    hessian = np.asarray(fixture["hessian"])
    hessian = np.triu(hessian) + np.triu(hessian, 1).T
    gradient = np.asarray(fixture["gradient"])
    expected = -np.linalg.solve(hessian, gradient)
    actual, norm, decrement = solve(hessian, gradient, guarded=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(norm, expected @ expected, rtol=2e-6)
    np.testing.assert_allclose(decrement, -gradient @ expected, rtol=2e-6)


def test_valid_search_and_finished_world_are_bitwise_unchanged():
    hessian, gradient = [[4., 1.], [1., 3.]], [2., -5.]
    for done in (False, True):
        original = solve(hessian, gradient, guarded=False, done=done)
        repaired = solve(hessian, gradient, guarded=True, done=done)
        for actual, expected in zip(repaired, original):
            np.testing.assert_array_equal(actual, expected)


def test_lost_positive_definiteness_uses_finite_descent_for_existing_linesearch():
    gradient = np.array([3., 4.])
    direction, norm, decrement = solve([[1., 2.], [2., 1.]], gradient, guarded=True)
    np.testing.assert_array_equal(direction, [-3., -4.])
    assert gradient @ direction < 0
    assert norm == decrement == 25


def test_invalid_solver_inputs_are_not_hidden():
    direction, _, _ = solve([[1., 0.], [0., 1.]], [float("nan"), 1.], guarded=True)
    assert not np.isfinite(direction).all()
