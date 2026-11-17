"""Audit-gated, deterministic merges of generation shards."""

from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import json as _json

import pyarrow as _pa
import pyarrow.parquet as _pq

from core.datasets._audit_record import _atomic_text, _read_audit, _require_current
from core.datasets.streaming import _episode_tables
from core.datasets.task_meta import (list_sidecars as _list_sidecars, read_sidecar as _read_sidecar,
                                    write_sidecar as _write_sidecar)
from core.generate.errors import (ConfigError as _ConfigError, DuplicateError as _DuplicateError,
                                  GenerationError as _GenerationError, SchemaError as _SchemaError)
from core.generate.manifest import Manifest as _Manifest, read_plan_json as _read_plan_json
from core.utils.hashing import canonical_json as _canonical_json


@_dataclass(frozen=True)
class ShardSummary:
    root: _Path
    episodes: int
    episode_fps: tuple[str, ...]
    scenario_keys: tuple[str, ...]
    report: object

    def __post_init__(self):
        object.__setattr__(self, "root", _Path(self.root))
        object.__setattr__(self, "episode_fps", tuple(self.episode_fps))
        object.__setattr__(self, "scenario_keys", tuple(self.scenario_keys))

    @classmethod
    def from_root(cls, root):
        root = _Path(root)
        report = _read_audit(root)
        indices = _list_sidecars(root)
        if indices != list(range(len(indices))):
            raise _SchemaError("Shard sidecar indices must be contiguous from zero")
        fps, scenarios = [], []
        for index in indices:
            sc = _read_sidecar(root, index)
            fp = sc["fingerprints"]
            if fp is None or fp["episode_fp"] is None or fp["scenario_key"] is None:
                raise _SchemaError("Merging requires episode and scenario fingerprints")
            fps.append(fp["episode_fp"])
            if sc["source"] is None:
                scenarios.append(fp["scenario_key"])
        return cls(root, len(indices), tuple(fps), tuple(scenarios), report)


@_dataclass(frozen=True)
class MergePlan:
    index_maps: tuple[dict[int, int], ...]
    total: int

    def __post_init__(self):
        object.__setattr__(self, "index_maps", tuple(dict(m) for m in self.index_maps))


def _info(root):
    try:
        info = _json.loads((_Path(root) / "meta" / "info.json").read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError("Expected an info object")
        return info
    except (OSError, ValueError, UnicodeError) as exc:
        raise _SchemaError(f"Missing or invalid info.json: {root}") from exc


def _plan(shard):
    return _read_plan_json(shard.root) if (shard.root / "meta" / "generation_plan.json").exists() else None


def _order(shards):
    plans = [_plan(shard) for shard in shards]
    return sorted(range(len(shards)), key=lambda i: (plans[i]["shard"][0] if plans[i] is not None else 0,
                                                   shards[i].report.content_digest))


def _maps(shards):
    maps, offset = [None] * len(shards), 0
    for i in _order(shards):
        maps[i] = {old: offset + old for old in range(shards[i].episodes)}
        offset += shards[i].episodes
    return tuple(maps), offset


def _check_plan(plan, shards):
    maps, total = _maps(shards)
    if plan.total != total or len(plan.index_maps) != len(shards) or any(a != b for a, b in zip(plan.index_maps, maps)):
        raise ValueError("Merge plan does not match shards")


def plan_merge(shards):
    if not shards:
        raise ValueError("At least one shard is required")
    for shard in shards:
        _require_current(shard.root, shard.report)
    infos, plans = [_info(s.root) for s in shards], [_plan(s) for s in shards]
    for info in infos[1:]:
        if any(info.get(key) != infos[0].get(key) for key in ("codebase_version", "fps", "features", "robot_type")):
            raise _ConfigError("Shard info.json values differ")
    if any(p is None for p in plans) and not all(p is None for p in plans):
        raise _ConfigError("Generation plans must be present in all shards or none")
    if plans[0] is not None:
        for plan in plans[1:]:
            if any(plan[key] != plans[0][key] for key in ("task", "config_sha256", "task_toml_sha256", "arm_files_sha256")):
                raise _ConfigError("Generation plan hashes differ")
    episode_seen, scenario_seen, manifest_seen = set(), set(), set()
    for shard in shards:
        keys = {(line.task, line.kind, line.attempt_index)
                for line in _Manifest(shard.root / "meta" / "generation_manifest.jsonl").lines()}
        for values, seen, name in ((set(shard.episode_fps), episode_seen, "episode fingerprint"),
                                   (set(shard.scenario_keys), scenario_seen, "original scenario key"),
                                   (keys, manifest_seen, "manifest key")):
            if values & seen:
                raise _DuplicateError(f"Repeated {name} across shards")
            seen.update(values)
    maps, total = _maps(shards)
    return MergePlan(maps, total)


def merge_frames(plan, shards, dst_root, repo_id):
    _check_plan(plan, shards)
    dst_root = _Path(dst_root)
    if (dst_root / "meta" / "info.json").exists() or (dst_root / "data").exists() and any((dst_root / "data").iterdir()):
        raise _GenerationError("Destination already holds frame data")
    order = _order(shards)
    infos = [_info(s.root) for s in shards]
    if not all(info.get("codebase_version") == "v3.0-lite" for info in infos):
        from lerobot.datasets.aggregate import aggregate_datasets
        aggregate_datasets(repo_ids=[f"local/shard_{i}" for i in order], roots=[shards[i].root for i in order],
                           aggr_root=dst_root, aggr_repo_id=repo_id)
        if _info(dst_root).get("total_episodes") != plan.total:
            raise _GenerationError("Aggregated episode count differs from merge plan")
        return
    tables = [_episode_tables(shard.root) for shard in shards]
    for shard, grouped in zip(shards, tables):
        if set(grouped) != set(range(shard.episodes)):
            raise _SchemaError("Shard data episodes must be contiguous from zero")
    folder = dst_root / "data" / "chunk-000"
    folder.mkdir(parents=True, exist_ok=True)
    offset = 0
    for i in order:
        for old in range(shards[i].episodes):
            table = tables[i][old]
            new = plan.index_maps[i][old]
            for name, values in (("episode_index", [new] * table.num_rows), ("index", range(offset, offset + table.num_rows))):
                field = table.schema.field(name)
                table = table.set_column(table.schema.get_field_index(name), field, _pa.array(values, type=field.type))
            _pq.write_table(table, folder / f"file-{new:06d}.parquet")
            offset += table.num_rows
    info = dict(infos[order[0]], total_episodes=plan.total)
    _atomic_text(dst_root / "meta" / "info.json", _json.dumps(info, indent=2, sort_keys=True))


def apply_merge_meta(plan, shards, dst_root):
    _check_plan(plan, shards)
    dst_root = _Path(dst_root)
    manifest_path = dst_root / "meta" / "generation_manifest.jsonl"
    if _list_sidecars(dst_root) or manifest_path.exists():
        raise _GenerationError("Destination already holds sidecars or a manifest")
    lines, has_manifest = [], False
    for i in _order(shards):
        shard = shards[i]
        for old in range(shard.episodes):
            sc = _read_sidecar(shard.root, old)
            sc["episode_index"] = plan.index_maps[i][old]
            _write_sidecar(dst_root, sc)
        path = shard.root / "meta" / "generation_manifest.jsonl"
        has_manifest |= path.exists()
        for line in _Manifest(path).lines():
            payload = line.to_json()
            if payload["episode_index"] is not None:
                payload["episode_index"] = plan.index_maps[i][payload["episode_index"]]
            lines.append(_canonical_json(payload) + "\n")
    if has_manifest:
        _atomic_text(manifest_path, "".join(lines))
