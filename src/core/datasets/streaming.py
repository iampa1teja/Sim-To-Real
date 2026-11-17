"""Episode filtering and local iteration with lazy LeRobot streaming."""

from collections import Counter as _Counter
from pathlib import Path as _Path

import numpy as _np
import pyarrow as _pa
import pyarrow.compute as _pc
import pyarrow.parquet as _pq

from core.datasets.consolidate import read_consolidated as _read_consolidated
from core.datasets.task_meta import list_sidecars as _list_sidecars, read_sidecar as _read_sidecar
from core.generate.errors import SchemaError as _SchemaError
from core.generate.protocols import Kind as _Kind


def select_episodes(rows, *, kinds=None, success=None, exclude_holdout_of=None):
    allowed = None if kinds is None else {_Kind(kind) for kind in kinds}
    selected = set()
    for row in rows:
        if allowed is not None and _Kind(row["kind"]) not in allowed:
            continue
        if success is not None and row["success"] is not success:
            continue
        if exclude_holdout_of is not None:
            holdout = exclude_holdout_of.holdout
            objects = row["objects"].values()
            if any(obj["color"] in holdout.colors for obj in objects):
                continue
            counts = _Counter(obj["set"] for obj in objects if obj["set"] is not None)
            if any(count in holdout.counts for count in counts.values()):
                continue
        selected.add(row["episode_index"])
    return sorted(selected)


def make_streaming_dataset(repo_id, *, episodes=None, **kwargs):
    from lerobot.datasets import StreamingLeRobotDataset
    return StreamingLeRobotDataset(repo_id, episodes=None if episodes is None else [int(e) for e in episodes], **kwargs)


def _episode_tables(root):
    grouped = {}
    try:
        for path in sorted((_Path(root) / "data").rglob("*.parquet")):
            table = _pq.read_table(path)
            for index in sorted(set(table.column("episode_index").to_pylist())):
                part = table.filter(_pc.equal(table.column("episode_index"), index))
                grouped.setdefault(index, []).append(part)
        return {index: _pa.concat_tables(parts).sort_by([("frame_index", "ascending")])
                for index, parts in grouped.items()}
    except (KeyError, ValueError, _pa.ArrowException) as exc:
        raise _SchemaError(f"Invalid episode data: {exc}") from exc


def iter_episodes(root, *, episodes=None):
    root = _Path(root)
    sidecars = (_read_consolidated(root) if (root / "task_meta" / "episodes.parquet").is_file()
                else [_read_sidecar(root, i) for i in _list_sidecars(root)])
    by_index = {sc["episode_index"]: sc for sc in sidecars}
    selected = set(by_index) if episodes is None else {int(e) for e in episodes}
    if not selected <= by_index.keys():
        raise ValueError("Requested episode has no sidecar")
    tables = _episode_tables(root)
    if tables.keys() != by_index.keys():
        raise _SchemaError("Data episodes differ from sidecar episodes")
    for index, table in tables.items():
        if table.num_rows != by_index[index]["num_frames"]:
            raise _SchemaError(f"Frame count differs for episode {index}")
    for index in sorted(selected):
        table = tables[index]
        n = table.num_rows
        try:
            frames = {name: _np.asarray(table.column(name).to_pylist(), dtype=dtype).reshape(n, *shape)
                      for name, dtype, shape in (
                          ("action", _np.float32, (-1,)), ("observation.state", _np.float32, (-1,)),
                          ("label.stage", _np.int64, ()), ("label.imitate", _np.int64, ()),
                          ("label.ee_pose", _np.float32, (7,)), ("frame_index", _np.int64, ()))}
        except (KeyError, ValueError, TypeError) as exc:
            raise _SchemaError(f"Invalid frame columns for episode {index}") from exc
        yield by_index[index], frames
