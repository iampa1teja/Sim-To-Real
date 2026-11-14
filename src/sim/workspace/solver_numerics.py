"""Recover a failed MJWarp Newton search without changing the contact model.

The installed MJWarp float32 Cholesky can fail on a finite positive-definite
contact Hessian (captured condition number > 1e8). Retry that search in float64.
If float32 Hessian assembly has itself lost positive definiteness, use its
positive diagonal as a preconditioner. The existing objective, line search and
convergence checks still decide the step. Valid original searches are untouched.
"""
from functools import cache, wraps

import warp as wp


@wp.func
def _to_double(value: float):
    return wp.float64(value)


@wp.func
def _to_float(value: wp.float64):
    return float(value)


@wp.func
def _search_sums(gradient: wp.float64, solution: wp.float64):
    return wp.vec2d(solution * solution, gradient * solution)


@cache
def _double_search_kernel(size):
    @wp.kernel(module="unique", enable_backward=False, grid_stride=False)
    def kernel(gradient: wp.array2d[float], hessian: wp.array3d[float], done: wp.array[bool],
               search: wp.array2d[float], search_dot: wp.array[float], decrement: wp.array[float]):
        world = wp.tid()
        if done[world] or (wp.isfinite(search_dot[world]) and wp.isfinite(decrement[world])):
            return
        n = wp.static(size)
        matrix = wp.tile_map(_to_double, wp.tile_load(hessian[world], shape=(n, n)))
        grad = wp.tile_map(_to_double, wp.tile_load(gradient[world], shape=n))
        wp.tile_cholesky_inplace(matrix, fill_mode="upper")
        solution = wp.tile_cholesky_solve(matrix, grad, fill_mode="upper")
        sums = wp.tile_reduce(wp.add, wp.tile_map(_search_sums, grad, solution))[0]
        search_dot[world] = float(sums[0])
        decrement[world] = float(sums[1])
        wp.tile_store(search[world], wp.tile_map(_to_float, wp.tile_map(wp.mul, solution, wp.float64(-1.0))))

    return kernel


@wp.kernel
def _diagonal_search(nv: int, gradient: wp.array2d[float], hessian: wp.array3d[float], done: wp.array[bool],
                     search: wp.array2d[float], search_dot: wp.array[float], decrement: wp.array[float]):
    world = wp.tid()
    if done[world] or (wp.isfinite(search_dot[world]) and wp.isfinite(decrement[world])):
        return
    # Never hide invalid physical inputs or manufacture a descent direction from
    # a nonpositive diagonal. Those failures remain visible to the caller.
    valid = bool(True)
    for index in range(nv):
        diagonal = hessian[world, index, index]
        if not wp.isfinite(gradient[world, index]) or not wp.isfinite(diagonal) or diagonal <= 0.0:
            valid = False
    if not valid:
        return
    norm = wp.float64(0.0)
    descent = wp.float64(0.0)
    for index in range(nv):
        grad = wp.float64(gradient[world, index])
        direction = -grad / wp.float64(hessian[world, index, index])
        search[world, index] = float(direction)
        norm += direction * direction
        descent -= grad * direction
    search_dot[world] = float(norm)
    decrement[world] = float(descent)


def install_mjwarp_precision_guard():
    """Install once, before Newton captures its GPU graph; never edits SDK files.

Small dense systems retry Cholesky in double precision. Larger blocked systems
can use diagonal descent if their original solve fails. The hook deliberately
targets the factor/solve boundary, not integration, contacts or joint commands.
"""
    from mujoco_warp._src import solver

    original = solver._cholesky_factorize_solve
    if getattr(original, "_s2r_precision_guard", False):
        return

    @wraps(original)
    def guarded(model, data, context, *args, **kwargs):
        original(model, data, context, *args, **kwargs)
        if model.nv <= solver._BLOCK_CHOLESKY_DIM:
            wp.launch_tiled(
                _double_search_kernel(model.nv), dim=data.nworld,
                inputs=[context.grad, context.h, context.done, context.search,
                        context.search_dot, context.newton_decrement],
                block_dim=model.block_dim.update_gradient_cholesky,
            )
        wp.launch(
            _diagonal_search, dim=data.nworld,
            inputs=[model.nv, context.grad, context.h, context.done, context.search,
                    context.search_dot, context.newton_decrement],
        )

    guarded._s2r_precision_guard = True
    solver._cholesky_factorize_solve = guarded
