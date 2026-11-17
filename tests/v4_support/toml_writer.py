"""A small TOML writer for synthetic arm.toml / task.toml files (stdlib ``tomllib`` only reads)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence


def _key(key: str) -> str:
    if key and all(ch.isalnum() or ch in "_-" for ch in key):
        return key
    return json.dumps(key)


def _scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if hasattr(value, "__fspath__"):
        return json.dumps(str(value))
    if hasattr(value, "item") and not isinstance(value, (Sequence, Mapping)):  # numpy scalar
        return _scalar(value.item())
    raise TypeError(f"Cannot write {type(value).__name__} to TOML")


def _inline(value) -> str:
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{_key(str(k))} = {_inline(v)}" for k, v in value.items()) + " }" if value else "{}"
    if isinstance(value, (list, tuple)) or (hasattr(value, "tolist") and not isinstance(value, str)):
        items = value.tolist() if hasattr(value, "tolist") else value
        return "[" + ", ".join(_inline(v) for v in items) + "]"
    return _scalar(value)


def _emit(table: Mapping, path: tuple[str, ...], lines: list[str]) -> None:
    plain = [(k, v) for k, v in table.items() if not isinstance(v, Mapping)]
    nested = [(k, v) for k, v in table.items() if isinstance(v, Mapping)]
    if path and (plain or not nested):
        lines.append("[" + ".".join(_key(p) for p in path) + "]")
    for key, value in plain:
        lines.append(f"{_key(str(key))} = {_inline(value)}")
    if plain or (path and not nested):
        lines.append("")
    for key, value in nested:
        _emit(value, path + (str(key),), lines)


def dump_toml(data: Mapping) -> str:
    """Serialise a nested mapping to TOML text that ``tomllib.loads`` reads back equal.

    Nested mappings become ``[a.b]`` tables (TOML insertion order is kept); mappings inside lists
    become inline tables (e.g. ``rules = [{ name = "x", if = ["a"] }]``); tuples and numpy arrays
    become arrays; ``Path`` values become strings. ``None`` is not representable and raises
    ``TypeError`` (omit the key instead). Example: ``dump_toml({"task": {"name": "fx", ...}})``.
    """
    lines: list[str] = []
    _emit(data, (), lines)
    return "\n".join(lines).rstrip() + "\n"
