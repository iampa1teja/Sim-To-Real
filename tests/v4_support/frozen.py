"""``assert_frozen``: the immutability rules of contracts §1.3 and §3 (numpy + standard library)."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping

import numpy as np


def _check_value(value, where: str, depth: int, deep: bool) -> None:
    if isinstance(value, np.ndarray):
        if value.flags.writeable:
            raise AssertionError(f"{where}: numpy array is writeable")
        if not value.flags.c_contiguous:
            raise AssertionError(f"{where}: numpy array is not C-contiguous")
        if value.dtype.kind == "f" and value.dtype != np.float64:
            raise AssertionError(f"{where}: float array has dtype {value.dtype}, expected float64")
        return
    if isinstance(value, (list, set, bytearray)):
        raise AssertionError(f"{where}: mutable {type(value).__name__} (use a tuple / frozenset)")
    if isinstance(value, dict):
        raise AssertionError(f"{where}: plain dict (wrap it in types.MappingProxyType)")
    if isinstance(value, types.MappingProxyType):
        if deep:
            for key, item in value.items():
                _check_value(item, f"{where}[{key!r}]", depth + 1, deep)
        return
    if isinstance(value, Mapping):
        raise AssertionError(f"{where}: mapping of type {type(value).__name__} is not a MappingProxyType")
    if isinstance(value, tuple):
        if deep:
            for index, item in enumerate(value):
                _check_value(item, f"{where}[{index}]", depth + 1, deep)
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _assert_frozen(value, where, depth + 1, deep)


def _assert_frozen(obj, where: str, depth: int, deep: bool) -> None:
    if depth > 20:
        raise AssertionError(f"{where}: nesting too deep")
    params = getattr(type(obj), "__dataclass_params__", None)
    if params is None or not params.frozen:
        raise AssertionError(f"{where}: {type(obj).__name__} is not a frozen dataclass")
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        try:
            setattr(obj, field.name, value)
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError(f"{where}.{field.name}: assignment did not raise FrozenInstanceError")
        try:
            delattr(obj, field.name)
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError(f"{where}.{field.name}: deletion did not raise FrozenInstanceError")
        _check_value(value, f"{where}.{field.name}", depth, deep)


def assert_frozen(obj, *, deep: bool = True) -> None:
    """Raise ``AssertionError`` unless ``obj`` is deeply immutable per contracts §1.3 / §3.

    Checks, recursively through nested dataclasses, tuples and mapping proxies: the class is a
    ``@dataclass(frozen=True)``; assigning or deleting any field raises ``FrozenInstanceError``;
    no field value (at any depth) is a ``list``, ``set``, ``dict`` or other mutable mapping;
    mappings are ``types.MappingProxyType``; numpy arrays are read-only, C-contiguous and, when
    floating, ``float64``. Existing frozen dataclasses such as ``core.tasks.base.Pose`` pass.
    With ``deep=False`` the items inside tuple / mapping-proxy fields are not inspected (for
    contract types such as ``AuditReport.checks: tuple[dict, ...]``); nested dataclass fields are
    still checked.
    """
    _assert_frozen(obj, type(obj).__name__, 0, deep)
