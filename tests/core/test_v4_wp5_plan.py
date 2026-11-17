"""WP5: ``core.generate.plan`` — the deterministic attempt stream (contracts §8.1, §1.4)."""

import dataclasses
import itertools
import math
import unittest

import numpy as np

from core.generate.errors import SamplingError, ScenarioSpaceExhausted
from core.generate.fingerprint import scenario_key
from core.generate.plan import (KINDS, CapacityReport, PlannedAttempt, attempt_seed, check_capacity, iter_plan, owns,
                                shard_attempts, shard_quota)
from core.generate.protocols import Kind, SkipReason
from core.generate.seeds import derive_seed
from v4_support import factories as fx

TASK = "fx_pick"


def make_spec_fn(scenario_of, *, fail=lambda kind, index, seed: False, calls=None):
    """spec_fn whose scenario (cube_0 x position bin) is ``scenario_of(kind, index, seed)``.

    Specs built for the same scenario id have identical scenario_keys whatever the kind, index or seed.
    ``fail(kind, index, seed)`` true raises ``SamplingError``. ``calls`` (a list) records every call.
    """
    def spec_fn(kind, index, seed):
        if calls is not None:
            calls.append((kind, index, seed))
        if fail(kind, index, seed):
            raise SamplingError(f"no start for {kind} {index}")
        sid = int(scenario_of(kind, index, seed))
        return fx.make_episode_spec(
            kind, attempt_index=index, seed=seed,
            objects={"cube_0": fx.make_object_state(0.1 + 0.002 * sid, 0.05, 0.01, color="red"),
                     "box": fx.make_object_state(0.25, -0.05, 0.0)})
    return spec_fn


def unique_spec_fn():
    """Every (kind, index) is a new scenario."""
    return make_spec_fn(lambda kind, index, seed: 3 * index + KINDS.index(kind))


def take(it, count):
    return list(itertools.islice(it, count))


def up_to_index(it, last_index):
    """Items of an index-ordered attempt iterator with index <= last_index."""
    return list(itertools.takewhile(lambda a: a.index <= last_index, it))


def bounded(task, master_seed, spec_fn, *, last, **kw):
    """The global stream cut after attempt index ``last`` (finite input for shard_attempts)."""
    return itertools.islice(iter_plan(task, master_seed, spec_fn, **kw), 3 * (last + 1))


def oracle(task, master_seed, spec_fn, count, unique_across_kinds=True):
    """Straightforward re-statement of §8.1 used as the expected stream."""
    seen, out = set(), []
    for index in itertools.count():
        for kind in (Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY):
            if len(out) == count:
                return out
            seed = derive_seed(master_seed, task, kind.value, index)
            try:
                spec = spec_fn(kind, index, seed)
            except SamplingError:
                out.append((kind, index, seed, None, None, SkipReason.SAMPLING_FAILED))
                continue
            key = scenario_key(spec)
            mark = key if unique_across_kinds else (kind, key)
            skip = SkipReason.SCENARIO_SEEN if mark in seen else None
            seen.add(mark)
            out.append((kind, index, seed, spec, key, skip))


def as_tuple(a):
    return (a.kind, a.index, a.seed, a.spec, a.scenario_key, a.skip)


class SeedAndShardTests(unittest.TestCase):
    """WP5 §8.1/§1.4 KINDS, attempt_seed, owns, shard_quota."""

    def test_kinds(self):
        """§8.1 KINDS = (SUCCESS, FAILURE, RECOVERY)."""
        self.assertEqual(KINDS, (Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY))
        self.assertIsInstance(KINDS, tuple)

    def test_attempt_seed(self):
        """§1.4 attempt seed = derive_seed(master_seed, task, kind.value, attempt_index); Python int."""
        self.assertEqual(attempt_seed(0, "prim_reach_point", Kind.SUCCESS, 0), 11062146242513595739)
        for master in (0, 1984, 2**64 - 1):
            for kind in KINDS:
                for index in (0, 1, 17, 10**6):
                    with self.subTest(master=master, kind=kind, index=index):
                        value = attempt_seed(master, TASK, kind, index)
                        self.assertIs(type(value), int)
                        self.assertEqual(value, derive_seed(master, TASK, kind.value, index))
                        self.assertEqual(attempt_seed(master, TASK, kind.value, index), value)

    def test_owns(self):
        """§8.1 owns(index, shard, n) = index % n == shard; exactly one shard owns each index."""
        for n in (1, 2, 3, 5, 8):
            for index in range(60):
                owners = [s for s in range(n) if owns(index, s, n)]
                with self.subTest(n=n, index=index):
                    self.assertEqual(owners, [index % n])

    def test_shard_quota(self):
        """§8.1 shard_quota = total // n + (1 if shard < total % n else 0); quotas sum to the total."""
        for total in (0, 1, 2, 5, 7, 100, 4001):
            for n in (1, 2, 3, 4, 7, 16):
                quotas = [shard_quota(total, s, n) for s in range(n)]
                with self.subTest(total=total, n=n):
                    self.assertEqual(sum(quotas), total)
                    self.assertLessEqual(max(quotas) - min(quotas), 1)
                    self.assertEqual(quotas, [total // n + (1 if s < total % n else 0) for s in range(n)])
                    self.assertTrue(all(type(q) is int for q in quotas))

    def test_bad_shard_arguments(self):
        """§1.5 shard arguments outside 0 <= shard < n (n >= 1) raise ValueError."""
        for shard, n in ((0, 0), (1, 1), (-1, 2), (2, 2), (0, -1)):
            with self.subTest(shard=shard, n=n):
                with self.assertRaises(ValueError):
                    owns(0, shard, n)
                with self.assertRaises(ValueError):
                    shard_quota(10, shard, n)
                with self.assertRaises(ValueError):
                    list(shard_attempts(iter([]), shard, n))
        with self.assertRaises(ValueError):
            shard_quota(-1, 0, 1)


class IterPlanTests(unittest.TestCase):
    """WP5 §8.1 iter_plan."""

    def test_planned_attempt_fields(self):
        """§8.1 PlannedAttempt(task, kind, index, seed, spec, scenario_key, skip), frozen."""
        self.assertEqual([f.name for f in dataclasses.fields(PlannedAttempt)],
                         ["task", "kind", "index", "seed", "spec", "scenario_key", "skip"])
        a = take(iter_plan(TASK, 0, unique_spec_fn()), 1)[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            a.index = 5

    def test_order_seeds_and_specs(self):
        """§8.1 order (index, KINDS.index(kind)); seeds per §1.4; spec_fn(kind, index, seed) is called with them."""
        calls = []
        spec_fn = make_spec_fn(lambda kind, index, seed: 3 * index + KINDS.index(kind), calls=calls)
        items = take(iter_plan(TASK, 42, spec_fn), 30)
        self.assertEqual([(a.index, a.kind) for a in items], [(i, k) for i in range(10) for k in KINDS])
        for a in items:
            with self.subTest(index=a.index, kind=a.kind):
                self.assertEqual(a.task, TASK)
                self.assertIs(a.kind, Kind(a.kind))
                self.assertEqual(a.seed, derive_seed(42, TASK, a.kind.value, a.index))
                self.assertIsNone(a.skip)
                self.assertEqual(a.scenario_key, scenario_key(a.spec))
                self.assertEqual((a.spec.attempt_index, a.spec.seed, a.spec.kind), (a.index, a.seed, a.kind))
        self.assertEqual(calls[:30], [(a.kind, a.index, a.seed) for a in items])

    def test_iter_plan_is_infinite_and_lazy(self):
        """§8.1 iter_plan yields forever; it only calls spec_fn for what is consumed (plus at most one)."""
        calls = []
        it = iter_plan(TASK, 0, make_spec_fn(lambda k, i, s: 3 * i + KINDS.index(k), calls=calls))
        take(it, 5)
        self.assertLessEqual(len(calls), 6)
        self.assertEqual(take(it, 1)[0].index, 1)
        self.assertEqual(take(iter_plan(TASK, 0, unique_spec_fn()), 3000)[-1].index, 999)

    def test_matches_oracle_random(self):
        """§8.1 stream equals the restatement (first occurrence wins, later repeats skip scenario_seen)."""
        for k in range(6):
            rng = np.random.default_rng(k)
            space = int(rng.integers(3, 40))
            fail_mod = int(rng.integers(5, 13))
            spec_fn = make_spec_fn(lambda kind, index, seed, space=space: seed % space,
                                   fail=lambda kind, index, seed, m=fail_mod: seed % m == 0)
            for unique in (True, False):
                with self.subTest(k=k, space=space, unique=unique):
                    got = [as_tuple(a) for a in take(iter_plan(TASK, k, spec_fn, unique_across_kinds=unique), 150)]
                    self.assertEqual(got, oracle(TASK, k, spec_fn, 150, unique))
                    skips = {g[5] for g in got}
                    self.assertIn(SkipReason.SCENARIO_SEEN, skips)

    def test_scenario_seen_across_kinds(self):
        """§8.1 with unique_across_kinds a scenario of SUCCESS blocks FAILURE/RECOVERY at the same scenario; without it, it does not."""
        same_per_index = make_spec_fn(lambda kind, index, seed: index)
        got = take(iter_plan(TASK, 0, same_per_index), 9)
        self.assertEqual([a.skip for a in got], [None, SkipReason.SCENARIO_SEEN, SkipReason.SCENARIO_SEEN] * 3)
        got = take(iter_plan(TASK, 0, same_per_index, unique_across_kinds=False), 9)
        self.assertEqual([a.skip for a in got], [None] * 9)
        repeat_index = make_spec_fn(lambda kind, index, seed: KINDS.index(kind) + 3 * (index // 2))
        got = take(iter_plan(TASK, 0, repeat_index, unique_across_kinds=False), 12)
        self.assertEqual([a.skip for a in got], [None] * 3 + [SkipReason.SCENARIO_SEEN] * 3 + [None] * 3
                         + [SkipReason.SCENARIO_SEEN] * 3)

    def test_skip_payloads(self):
        """§8.1/§8.3 scenario_seen keeps spec and key (they go to the manifest); sampling_failed has neither."""
        spec_fn = make_spec_fn(lambda kind, index, seed: 0, fail=lambda kind, index, seed: index == 1)
        got = take(iter_plan(TASK, 0, spec_fn, unique_across_kinds=False), 9)
        seen = got[6]
        self.assertEqual((seen.index, seen.skip), (2, SkipReason.SCENARIO_SEEN))
        self.assertIsNotNone(seen.spec)
        self.assertEqual(seen.scenario_key, scenario_key(seen.spec))
        for a in got[3:6]:
            with self.subTest(kind=a.kind):
                self.assertEqual(a.skip, SkipReason.SAMPLING_FAILED)
                self.assertIsNone(a.spec)
                self.assertIsNone(a.scenario_key)
                self.assertEqual(a.seed, derive_seed(0, TASK, a.kind.value, 1))

    def test_other_exceptions_propagate(self):
        """§8.1 only SamplingError becomes a skip; other errors from spec_fn propagate."""
        def boom(kind, index, seed):
            raise RuntimeError("bug")
        with self.assertRaises(RuntimeError):
            take(iter_plan(TASK, 0, boom), 1)

    def test_pure_function_of_inputs(self):
        """§8.1 the stream is a pure function of (master seed, task, kind, index): restarts, global RNG state
        and interleaved iterators do not change it; master seed and task do."""
        spec_fn = make_spec_fn(lambda kind, index, seed: seed % 17)
        reference = [as_tuple(a) for a in take(iter_plan(TASK, 7, spec_fn), 120)]
        np.random.seed(123)
        np.random.random(10)
        a, b = iter_plan(TASK, 7, spec_fn), iter_plan(TASK, 7, spec_fn)
        interleaved_a, interleaved_b = [], []
        for _ in range(120):
            interleaved_a.append(as_tuple(next(a)))
            interleaved_b.append(as_tuple(next(b)))
        self.assertEqual(interleaved_a, reference)
        self.assertEqual(interleaved_b, reference)
        other_seed = [as_tuple(a) for a in take(iter_plan(TASK, 8, spec_fn), 120)]
        other_task = [as_tuple(a) for a in take(iter_plan("fx_other", 7, spec_fn), 120)]
        self.assertNotEqual([t[2] for t in other_seed], [t[2] for t in reference])
        self.assertNotEqual([t[2] for t in other_task], [t[2] for t in reference])

    def test_each_attempt_depends_only_on_its_own_position(self):
        """§8.1 attempt (kind, index) has the same seed and spec in any stream prefix and for any later quota."""
        spec_fn = make_spec_fn(lambda kind, index, seed: seed % 11)
        long = {(a.kind, a.index): as_tuple(a) for a in take(iter_plan(TASK, 3, spec_fn), 300)}
        for stop in (1, 2, 7, 50, 299):
            short = take(iter_plan(TASK, 3, spec_fn), stop)
            with self.subTest(stop=stop):
                self.assertEqual([as_tuple(a) for a in short], [long[(a.kind, a.index)] for a in short])


class ShardTests(unittest.TestCase):
    """WP5 §8.1 shard_attempts: shards partition the stream; skip decisions are global."""

    def test_partition(self):
        """§8.1 for several n: shards are pairwise disjoint and their union is the whole stream (same items)."""
        spec_fn = make_spec_fn(lambda kind, index, seed: seed % 23, fail=lambda kind, index, seed: seed % 9 == 0)
        last = 59
        whole = [as_tuple(a) for a in up_to_index(iter_plan(TASK, 5, spec_fn), last)]
        self.assertEqual(len(whole), 3 * (last + 1))
        for n in (1, 2, 3, 4, 7, 16):
            parts = [[as_tuple(a) for a in up_to_index(shard_attempts(bounded(TASK, 5, spec_fn, last=last), s, n), last)]
                     for s in range(n)]
            with self.subTest(n=n):
                positions = [(t[1], t[0]) for part in parts for t in part]
                self.assertEqual(len(positions), len(set(positions)))
                self.assertEqual(sorted(positions, key=lambda p: (p[0], KINDS.index(p[1]))),
                                 [(t[1], t[0]) for t in whole])
                merged = sorted((t for part in parts for t in part), key=lambda t: (t[1], KINDS.index(t[0])))
                self.assertEqual(merged, whole)
                for s, part in enumerate(parts):
                    self.assertTrue(all(t[1] % n == s for t in part))
                    self.assertEqual([t[1] for t in part], sorted(t[1] for t in part))

    def test_seen_scenario_owned_by_another_shard(self):
        """§8.1 a scenario first drawn at an index of shard 1 is skipped at a later index of shard 0."""
        spec_fn = make_spec_fn(lambda kind, index, seed: 1000 if index in (1, 2) else 3 * index + KINDS.index(kind),
                               fail=lambda kind, index, seed: kind is not Kind.SUCCESS)
        shard0 = [a for a in up_to_index(shard_attempts(bounded(TASK, 0, spec_fn, last=4), 0, 2), 4) if a.kind is Kind.SUCCESS]
        self.assertEqual([(a.index, a.skip) for a in shard0], [(0, None), (2, SkipReason.SCENARIO_SEEN), (4, None)])
        shard1 = [a for a in up_to_index(shard_attempts(bounded(TASK, 0, spec_fn, last=4), 1, 2), 4) if a.kind is Kind.SUCCESS]
        self.assertEqual([(a.index, a.skip) for a in shard1], [(1, None), (3, None)])

    def test_skips_identical_for_any_shard_count_and_resume_point(self):
        """§8.1 'skip decisions depend only on the index': the decision for each (index, kind) is the same for
        every shard count, and a resumed iterator (fresh iter_plan, consumed past a point) repeats it."""
        for k in range(3):
            rng = np.random.default_rng(50 + k)
            space = int(rng.integers(5, 30))
            spec_fn = make_spec_fn(lambda kind, index, seed, space=space: seed % space,
                                   fail=lambda kind, index, seed: seed % 7 == 0)
            last = 40
            decisions = {(a.index, a.kind): (a.skip, a.scenario_key) for a in up_to_index(iter_plan(TASK, k, spec_fn), last)}
            self.assertIn(SkipReason.SCENARIO_SEEN, {d[0] for d in decisions.values()})
            for n in (1, 2, 3, 5, 8):
                for s in range(n):
                    with self.subTest(k=k, n=n, shard=s):
                        got = {(a.index, a.kind): (a.skip, a.scenario_key)
                               for a in up_to_index(shard_attempts(bounded(TASK, k, spec_fn, last=last), s, n), last)}
                        self.assertEqual(got, {p: d for p, d in decisions.items() if p[0] % n == s})
            for resume_at in (0, 1, 13, 29):
                resumed = itertools.dropwhile(lambda a, r=resume_at: a.index < r, iter_plan(TASK, k, spec_fn))
                with self.subTest(k=k, resume_at=resume_at):
                    got = {(a.index, a.kind): (a.skip, a.scenario_key) for a in up_to_index(resumed, last)}
                    self.assertEqual(got, {p: d for p, d in decisions.items() if p[0] >= resume_at})

    def test_no_duplicate_non_skipped_scenarios_across_shards(self):
        """§8.1 (no-duplicates requirement) over all shards together, non-skipped attempts never repeat a scenario_key."""
        spec_fn = make_spec_fn(lambda kind, index, seed: seed % 31)
        for n in (1, 2, 3, 6):
            keys = []
            for s in range(n):
                keys += [a.scenario_key for a in up_to_index(shard_attempts(bounded(TASK, 1, spec_fn, last=80), s, n), 80)
                         if a.skip is None]
            with self.subTest(n=n):
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(len(keys), 31)


class CapacityTests(unittest.TestCase):
    """WP5 §8.1 check_capacity and ScenarioSpaceExhausted."""

    def test_report_fields(self):
        """§8.1 CapacityReport(non_skipped: dict[Kind, int], horizon: int)."""
        self.assertEqual([f.name for f in dataclasses.fields(CapacityReport)], ["non_skipped", "horizon"])

    def test_budget_and_horizon_without_skips(self):
        """§8.1 the walk counts up to ceil(max_attempt_factor × quota) non-skipped attempts per kind;
        horizon = 1 + the last attempt index examined."""
        from core.generate.config import Quotas
        report = check_capacity(TASK, 0, unique_spec_fn(), Quotas(success=4, failure=2, recovery=0),
                                max_attempt_factor=1.5, max_consecutive_skips=10)
        self.assertIsInstance(report, CapacityReport)
        self.assertEqual(report.non_skipped, {Kind.SUCCESS: 6, Kind.FAILURE: 3, Kind.RECOVERY: 0})
        self.assertEqual(report.horizon, 6)
        report = check_capacity(TASK, 0, unique_spec_fn(), Quotas(success=3, failure=3, recovery=3),
                                max_attempt_factor=3.0, max_consecutive_skips=10)
        self.assertEqual(report.non_skipped, {k: 9 for k in KINDS})
        self.assertEqual(report.horizon, 9)

    def test_quotas_as_mapping(self):
        """§8.1 quotas may also be given as a mapping keyed by Kind or kind value."""
        a = check_capacity(TASK, 0, unique_spec_fn(), {Kind.SUCCESS: 2, Kind.FAILURE: 1, Kind.RECOVERY: 1},
                           max_attempt_factor=2.0, max_consecutive_skips=5)
        b = check_capacity(TASK, 0, unique_spec_fn(), {"success": 2, "failure": 1, "recovery": 1},
                           max_attempt_factor=2.0, max_consecutive_skips=5)
        self.assertEqual(a, b)
        self.assertEqual(a.non_skipped, {Kind.SUCCESS: 4, Kind.FAILURE: 2, Kind.RECOVERY: 2})

    def test_zero_quotas(self):
        """§8.1 all-zero quotas need no attempts: horizon 0."""
        report = check_capacity(TASK, 0, unique_spec_fn(), {"success": 0, "failure": 0, "recovery": 0},
                                max_attempt_factor=3.0, max_consecutive_skips=5)
        self.assertEqual(report, CapacityReport(non_skipped={k: 0 for k in KINDS}, horizon=0))

    def test_exhausted_finite_space(self):
        """§1.5/§8.1 a finite scenario space smaller than the quota raises ScenarioSpaceExhausted."""
        small = make_spec_fn(lambda kind, index, seed: seed % 5)
        with self.assertRaises(ScenarioSpaceExhausted):
            check_capacity(TASK, 0, small, {"success": 10, "failure": 0, "recovery": 0}, max_attempt_factor=3.0,
                           max_consecutive_skips=50)

    def test_always_sampling_failed(self):
        """§8.1 a kind whose specs always fail to sample is exhausted after max_consecutive_skips."""
        spec_fn = make_spec_fn(lambda kind, index, seed: 3 * index + KINDS.index(kind),
                               fail=lambda kind, index, seed: kind is Kind.RECOVERY)
        with self.assertRaises(ScenarioSpaceExhausted):
            check_capacity(TASK, 0, spec_fn, {"success": 1, "failure": 1, "recovery": 1}, max_attempt_factor=2.0,
                           max_consecutive_skips=20)

    def test_across_kinds_blocking(self):
        """§8.1 with unique scenarios across kinds, a spec_fn that repeats the success scenario for the
        other kinds exhausts them; with unique_across_kinds=False it does not."""
        same = make_spec_fn(lambda kind, index, seed: index)
        with self.assertRaises(ScenarioSpaceExhausted):
            check_capacity(TASK, 0, same, {"success": 2, "failure": 2, "recovery": 0}, max_attempt_factor=1.0,
                           max_consecutive_skips=10)
        report = check_capacity(TASK, 0, same, {"success": 2, "failure": 2, "recovery": 0}, max_attempt_factor=1.0,
                                max_consecutive_skips=10, unique_across_kinds=False)
        self.assertEqual(report.non_skipped, {Kind.SUCCESS: 2, Kind.FAILURE: 2, Kind.RECOVERY: 0})

    def test_consecutive_skip_boundary(self):
        """§8.1 'after max_consecutive_skips consecutive skips': m leading skips exhaust a kind iff max <= m."""
        m = 6
        spec_fn = make_spec_fn(lambda kind, index, seed: 3 * index + KINDS.index(kind),
                               fail=lambda kind, index, seed: kind is Kind.FAILURE and index < m)
        quotas = {"success": 1, "failure": 2, "recovery": 0}
        for limit in (1, m - 1, m):
            with self.subTest(limit=limit), self.assertRaises(ScenarioSpaceExhausted):
                check_capacity(TASK, 0, spec_fn, quotas, max_attempt_factor=1.0, max_consecutive_skips=limit)
        for limit in (m + 1, m + 10):
            with self.subTest(limit=limit):
                report = check_capacity(TASK, 0, spec_fn, quotas, max_attempt_factor=1.0, max_consecutive_skips=limit)
                self.assertEqual(report.non_skipped[Kind.FAILURE], 2)
                self.assertEqual(report.horizon, m + 2)

    def test_quota_met_below_budget_is_enough(self):
        """§8.1 raise only if fewer than the quota are non-skipped: quota reached but budget not → no error."""
        three = make_spec_fn(lambda kind, index, seed: index if index < 3 else 0)
        report = check_capacity(TASK, 0, three, {"success": 2, "failure": 0, "recovery": 0}, max_attempt_factor=3.0,
                                max_consecutive_skips=10)
        self.assertEqual(report.non_skipped[Kind.SUCCESS], 3)
        with self.assertRaises(ScenarioSpaceExhausted):
            check_capacity(TASK, 0, three, {"success": 4, "failure": 0, "recovery": 0}, max_attempt_factor=3.0,
                           max_consecutive_skips=10)

    def test_budget_rounds_up(self):
        """§8.1 budget = ceil(max_attempt_factor × quota)."""
        for factor, quota in ((1.5, 3), (2.2, 5), (1.0, 7), (3.0, 1)):
            report = check_capacity(TASK, 0, unique_spec_fn(), {"success": quota, "failure": 0, "recovery": 0},
                                    max_attempt_factor=factor, max_consecutive_skips=10)
            with self.subTest(factor=factor, quota=quota):
                self.assertEqual(report.non_skipped[Kind.SUCCESS], math.ceil(factor * quota))


if __name__ == "__main__":
    unittest.main()
