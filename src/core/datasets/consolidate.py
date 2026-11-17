"""Deterministic, atomic consolidated sidecar tables."""

from pathlib import Path as _Path
import json as _json
import os as _os
import tempfile as _tempfile

import pyarrow as _pa
import pyarrow.parquet as _pq

from core.datasets.task_meta import (list_sidecars as _list_sidecars, read_sidecar as _read_sidecar,
                                    validate_sidecar as _validate_sidecar)
from core.generate.errors import SchemaError as _SchemaError
from core.utils.hashing import canonical_json as _canonical_json


_SCHEMA = _pa.schema([
    _pa.field("episode_index", _pa.int64(), nullable=False),
    _pa.field("num_frames", _pa.int64(), nullable=False),
    _pa.field("attempt_index", _pa.int64()), _pa.field("seed", _pa.uint64()),
    _pa.field("kind", _pa.string(), nullable=False),
    *[_pa.field(name, _pa.string()) for name in ("instruction", "failure_mode", "furthest_stage", "perturbation_type")],
    _pa.field("success", _pa.bool_()),
    *[_pa.field(name, _pa.string()) for name in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp", "source_episode_fp")],
    _pa.field("sidecar_json", _pa.string(), nullable=False),
])


def consolidate(root):
    sidecars = sorted((_read_sidecar(root, i) for i in _list_sidecars(root)), key=lambda sc: sc["episode_index"])
    rows = []
    for sc in sidecars:
        row = {name: sc[name] for name in ("episode_index", "num_frames", "kind", "instruction", "failure_mode", "furthest_stage", "success")}
        row.update(attempt_index=(sc["attempt"] or {}).get("index"), seed=(sc["attempt"] or {}).get("seed"),
                   perturbation_type=(sc["perturbation"] or {}).get("type"),
                   source_episode_fp=(sc["source"] or {}).get("episode_fp"), sidecar_json=_canonical_json(sc))
        row.update({name: (sc["fingerprints"] or {}).get(name) for name in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp")})
        rows.append(row)
    table = _pa.Table.from_pylist(rows, schema=_SCHEMA)
    path = _Path(root) / "task_meta" / "episodes.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = _tempfile.mkstemp(dir=path.parent, prefix=".episodes.", suffix=".tmp")
    _os.close(fd)
    try:
        _pq.write_table(table, name)
        with open(name, "rb") as stream:
            _os.fsync(stream.fileno())
        _os.replace(name, path)
    finally:
        _Path(name).unlink(missing_ok=True)
    return path


def read_consolidated(root):
    table = _pq.read_table(_Path(root) / "task_meta" / "episodes.parquet")
    rows = []
    for text in table.column("sidecar_json").to_pylist():
        try:
            sc = _json.loads(text)
        except (ValueError, TypeError) as exc:
            raise _SchemaError("Invalid consolidated sidecar JSON") from exc
        _validate_sidecar(sc)
        rows.append(sc)
    return rows
