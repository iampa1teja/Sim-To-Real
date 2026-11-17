"""Global deterministic attempt stream and shard-independent skip decisions."""

import itertools as _itertools
import math as _math
from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass

from core.generate.errors import SamplingError as _SamplingError, ScenarioSpaceExhausted as _ScenarioSpaceExhausted
from core.generate.fingerprint import scenario_key as _scenario_key
from core.generate.protocols import EpisodeSpec as _EpisodeSpec, Kind as _Kind, SkipReason as _SkipReason
from core.generate.seeds import derive_seed as _derive_seed

KINDS = (_Kind.SUCCESS, _Kind.FAILURE, _Kind.RECOVERY)


def attempt_seed(master_seed, task, kind, index) -> int:
    return _derive_seed(master_seed, task, _Kind(kind).value, index)


def _shard(shard, n):
    if type(n) is not int or type(shard) is not int or n < 1 or not 0 <= shard < n:
        raise ValueError("Shard must satisfy 0 <= shard < n, n >= 1")


def owns(index, shard, n) -> bool:
    _shard(shard, n)
    return index % n == shard


def shard_quota(total, shard, n) -> int:
    _shard(shard, n)
    if type(total) is not int or total < 0:
        raise ValueError("Total quota must be a nonnegative integer")
    return total // n + (1 if shard < total % n else 0)


@_dataclass(frozen=True)
class PlannedAttempt:
    task: str
    kind: _Kind
    index: int
    seed: int
    spec: _EpisodeSpec | None
    scenario_key: str | None
    skip: _SkipReason | None


def iter_plan(task, master_seed, spec_fn, *, unique_across_kinds=True):
    seen = set()
    for index in _itertools.count():
        for kind in KINDS:
            seed = attempt_seed(master_seed, task, kind, index)
            try:
                spec = spec_fn(kind, index, seed)
            except _SamplingError:
                yield PlannedAttempt(task, kind, index, seed, None, None, _SkipReason.SAMPLING_FAILED)
                continue
            key = _scenario_key(spec)
            mark = key if unique_across_kinds else (kind, key)
            skip = _SkipReason.SCENARIO_SEEN if mark in seen else None
            seen.add(mark)
            yield PlannedAttempt(task, kind, index, seed, spec, key, skip)


def shard_attempts(it, shard, n):
    _shard(shard, n)
    for attempt in it:
        if owns(attempt.index, shard, n):
            yield attempt


@_dataclass(frozen=True)
class CapacityReport:
    non_skipped: dict
    horizon: int


def check_capacity(task, master_seed, spec_fn, quotas, *, max_attempt_factor, max_consecutive_skips,
                   unique_across_kinds=True) -> CapacityReport:
    if not _math.isfinite(max_attempt_factor) or max_attempt_factor < 1 or max_consecutive_skips < 1:
        raise ValueError("Invalid capacity budget")
    quota = {kind: quotas.get(kind, quotas.get(kind.value, 0)) if isinstance(quotas, _Mapping)
             else getattr(quotas, kind.value) for kind in KINDS}
    if any(type(q) is not int or q < 0 for q in quota.values()):
        raise ValueError("Quotas must be nonnegative integers")
    budget = {kind: _math.ceil(max_attempt_factor * quota[kind]) for kind in KINDS}
    counts, skips = dict.fromkeys(KINDS, 0), dict.fromkeys(KINDS, 0)
    active = {kind for kind in KINDS if budget[kind] > 0}
    horizon = 0
    if not active:
        return CapacityReport(counts, horizon)
    for attempt in iter_plan(task, master_seed, spec_fn, unique_across_kinds=unique_across_kinds):
        kind = attempt.kind
        if kind not in active:
            continue
        horizon = attempt.index + 1
        if attempt.skip is None:
            counts[kind] += 1
            skips[kind] = 0
            if counts[kind] >= budget[kind]:
                active.remove(kind)
        else:
            skips[kind] += 1
            if skips[kind] >= max_consecutive_skips:
                if counts[kind] < quota[kind]:
                    raise _ScenarioSpaceExhausted(f"{task}: insufficient scenarios for {kind.value}")
                active.remove(kind)
        if not active:
            return CapacityReport(counts, horizon)
