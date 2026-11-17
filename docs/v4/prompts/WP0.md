# WP0 — generation protocols, errors and seeds (Codex prompt)

## Goal

Create the Isaac-free package `src/core/generate/` with the v4 exception hierarchy, the deterministic seed helpers and the shared protocol types (enums, frozen dataclasses with JSON round-trip, and the `GenEnv` / `EpisodeWriter` / `Expert` protocols). Every later work package imports these names, so they must match `docs/v4/contracts.md` exactly and make the WP0 tests pass.

## Files

Create exactly these four files and nothing else:

| File | Content |
|---|---|
| `src/core/generate/__init__.py` | Empty. A module docstring is allowed; no code, no re-exports. |
| `src/core/generate/errors.py` | The 11 classes of contracts §1.5 with exactly the listed bases and order (`ConfigError(GenerationError, ValueError)`, …). No imports from `core`. |
| `src/core/generate/seeds.py` | `derive_seed(*parts) -> int` (§1.4) and `rng_for(seed, purpose) -> np.random.Generator` = `np.random.default_rng(derive_seed(seed, purpose))` (§7.3). |
| `src/core/generate/protocols.py` | The 6 enums (§3.1), 13 dataclasses (§3.2), 3 protocols (§3.3), `to_json` / `from_json` (§3.4). |

Do NOT touch:
- anything under `tests/` (including `tests/conftest.py`, `tests/v4_support/`, `tests/core/test_v4_*.py`);
- `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`, `pyproject.toml`;
- any existing module, in particular `src/core/tasks/base.py` (`Pose` and `StepState` belong to existing code / WP1) and `src/sim/generate/dr_variants.py`;
- other packages' future modules (`core/generate/{plan,fingerprint,manifest,config,...}.py`, `core/tasks/*`, `core/arms/*`, `core/expert/*`, `core/datasets/*`). WP0 adds only the four files above.

## Read first

`docs/v4/contracts.md` sections §1.1 (quaternion order), §1.3 (dtypes), §1.4 (RNG / `derive_seed` golden values), §1.5 (errors), §1.6 (imports, Python 3.11), §2 (WP0 row of the module map), §3 (all of it), §13 (test conventions). The contract is the single source of truth; `docs/v4/design.md` is background only.

## Rules and pitfalls (the tests check all of these)

**Public surface.** `protocols.py` may define only the 22 public names of §2 (6 enums, 13 dataclasses, `GenEnv`, `EpisodeWriter`, `Expert`). `seeds.py` only `derive_seed` and `rng_for`; `errors.py` only the 11 errors. Every helper you need is private (leading underscore). Import `Pose` and `StepState` from `core.tasks.base`; do not define your own `Pose`.

**Imports.** Only the standard library, `numpy` and existing `core` modules. No `torch`, `scipy`, `yaml`, `pyarrow`, `random`, Isaac/`pxr`/`lerobot`. No global `np.random.*` calls. Python 3.11 syntax only (no `type X = ...`, no PEP 695 generics); the tests parse the files with `feature_version=(3, 11)` and run under Python 3.11.

**Enums.** `class X(str, Enum)` with exactly the members of §3.1 in that order. `RejectionReason` values are the lower-case member names. When writing to JSON always use `.value` (never `str(member)` / f-strings: their output differs between Python versions).

**Dataclass fields.** `@dataclass(frozen=True)`; field names, order and which fields have defaults exactly as in the §3.2 table (the tests compare `dataclasses.fields`). Note: `PerturbationSpec.onset_stage` and `StageProgress.regressed_stage` have **no** default; `ExpertAction` has no defaults at all. `PerturbationSpec.params` and `ExpertObs.signals` default to an empty read-only mapping (use `default_factory`).

**Immutability (`__post_init__`, using `object.__setattr__`).**
- Mapping fields (`objects`, `roles`, `goal`, `params`, `satisfied`, `signals`, `StepRecord.objects`) become `types.MappingProxyType` over a **copy** of the input — mutating the caller's dict afterwards must not change the instance.
- `goal` and `params` hold JSON values: freeze them **deeply** — nested lists/tuples become tuples, nested dicts become `MappingProxyType` (recursively). So `goal={"p": [1.0, 2.0]}` stores `(1.0, 2.0)`. This is what makes the JSON round-trip compare equal.
- Sequence fields (`initial_q`, `EpisodeResult.stage`, `.imitate`, `.segments`) become tuples.
- Array fields (`ExpertObs.q`, `ExpertAction.q_target`, `FrameLabels.ee_pose_wxyz`, `StepRecord.state_raw`, `.action_raw`) become `np.float64`, C-contiguous, **copied**, then `flags.writeable = False`. Accept lists and float32 / non-contiguous input. Raise `ValueError` for a wrong shape: joint vectors must be 1-D, `ee_pose_wxyz` must be `(7,)`.
- Enum-typed fields accept the enum or its value string and store the member (`EpisodeSpec(kind="failure").kind is Kind.FAILURE`).
- `tests/v4_support/frozen.py::assert_frozen` is the exact check applied to every instance (lists, dicts, writeable arrays anywhere inside → failure).

**Equality and hashing.** Instances must support `==`. The default dataclass `__eq__` raises on numpy arrays, so the four classes with array fields (`ExpertObs`, `ExpertAction`, `FrameLabels`, `StepRecord`) need a custom `__eq__` that compares arrays with shape + `np.array_equal` and returns a plain `bool` (use `eq=False` and define `__eq__`, set `__hash__ = None`). `EpisodeSpec`, `PerturbationSpec`, `StageProgress`, `ExpertObs` must be unhashable (`hash()` raises `TypeError`; mapping proxies give this for free).

**`EpisodeSpec` invariants (raise `ValueError` in `__post_init__`).** `kind is SUCCESS` ⇔ `perturbation is None`; `noise` is not `None` only when `kind is SUCCESS`; every value of `roles` names a key of `objects` whose `ObjectState.active` is true (an empty `roles` is fine). `expert_mode` is a `property` returning `"blind"` iff `kind is Kind.FAILURE`, else `"closed_loop"`.

**`ExpertObs.step_state()`.** Returns `core.tasks.base.StepState(objects={name: state.pose for active objects}, ee=ee, gripper=gripper_gap_m, signals=dict(signals))` and additionally `tcp=tcp` **only if** `StepState` already has a `tcp` field (check `dataclasses.fields(StepState)`). `StepState.tcp`/`joints` are added by WP1 later; your code must work before and after that change. The method takes one optional argument, `joint_names: Sequence[str] | None = None`. When it is given, its length must equal `len(q)` (else `ValueError`) and, if `StepState` has a `joints` field, `joints=dict(zip(joint_names, q))` (floats) is passed; when it is `None`, `joints` is not passed.

**JSON (§3.4).**
- `to_json()` returns plain builtins only — exact `dict`, `list`, `str`, `int`, `float`, `bool`, `None` (no enum members, no `np.float64`, no `mappingproxy`, no tuples). Keys are exactly the field names, plus `"schema": "episode_spec/1"` for `EpisodeSpec` and `"schema": "episode_result/1"` for `EpisodeResult` (no other class has a `schema` key). Enums → `.value`; tuples → lists; arrays → lists of Python `float`; `Pose` → `{"position": [3 floats], "quat_xyzw": [4 floats]}` (XYZW, as in memory); nested dataclasses → their `to_json()`; mappings → dicts (recursively). `seed` stays an exact Python `int` (values up to `2**64 − 1`; never go through float or numpy).
- `from_json(d)` is a `classmethod` (a classmethod inherited from a private mixin is fine). It must satisfy `X.from_json(json.loads(canonical_json(x.to_json()))) == x` for every class (`canonical_json` from `core.utils.hashing`; note it sorts keys, so do not rely on mapping order).
- `from_json` raises `core.generate.errors.SchemaError` for: a non-dict payload; any unknown key or any missing key — **including keys of defaulted fields** (the format is strict: `to_json` always writes every field) — at the top level and inside nested objects (`ObjectState`, `Pose`, …); a wrong or missing schema tag; any non-finite float anywhere (`json.loads` accepts `NaN`/`Infinity`, the tests pass such dicts directly); an unknown enum value (`EpisodeResult.reason` is looked up in `SkipReason` then `RejectionReason`; `None` stays `None`); a `Pose` whose position is not length 3 or quaternion not length 4; a `seed` that is not a JSON integer (string or float). Wrap the underlying `ValueError`/`TypeError`/`KeyError` as `SchemaError`. A payload that parses but violates the `EpisodeSpec` invariants may raise plain `ValueError` (SchemaError is also a ValueError, so either passes). `from_json` must not mutate its argument.

**Protocols (§3.3).** `typing.Protocol` + `@runtime_checkable`. Exact members, parameter names and annotations as in the contract code block: `GenEnv` has annotated attributes `fps: int`, `joint_names: tuple[str, ...]`, `camera_names: tuple[str, ...]` and methods `reset(self, spec)`, `step(self, q_target)`, `observe(self)`, `set_object_pose(self, name, pose)`, `render(self)`, `close(self)`; `EpisodeWriter` has `root: Path` and `episode_count`, `episode_lengths`, `add_frame(self, action_raw, state_raw, images, task, labels)`, `discard_episode`, `save_episode`, `finalize`; `Expert` has `reset(self, spec, obs)`, `act(self, obs, progress)` and `phase` / `done` as `@property`. Annotations must resolve with `typing.get_type_hints` (return types `ExpertObs`, `ExpertAction`, `int`, `list[int]` as in the contract). The tests use `isinstance` with structural stubs.

**Seeds.** `derive_seed` must reproduce the golden values `derive_seed(0, "prim_reach_point", "success", 0) == 11062146242513595739` and `derive_seed(1984, 3, "red") == 7801958555027104258`, equal `sim.generate.dr_variants.episode_seed` (do not edit that file), and return a Python `int`.

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp0_errors.py
tests/core/test_v4_wp0_seeds.py
tests/core/test_v4_wp0_enums.py
tests/core/test_v4_wp0_dataclasses.py
tests/core/test_v4_wp0_json.py
tests/core/test_v4_wp0_protocols.py
tests/core/test_v4_support_selftest.py      # already passes; must keep passing
```

They import `v4_support` (shared helpers in `tests/v4_support/`, put on `sys.path` by `tests/conftest.py`); `v4_support.factories` builds the instances the tests use — read it to see exactly how your constructors are called.

Run from the repo root with the host dependency group (Python 3.11):

```
uv run --group host pytest tests/core/test_v4_wp0_*.py tests/core/test_v4_support_selftest.py -q
```

Without uv: `PYTHONPATH=src python -m pytest tests/core/test_v4_wp0_*.py tests/core/test_v4_support_selftest.py -q` in an environment with numpy, pytest (and the rest of the `host` group for the full suite).

Full-suite check: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`. Baseline before WP0 (host group, Python 3.11): **32 passed, 8 skipped, 3 collection errors** that are unrelated to v4 (`tests/sim/test_app.py` needs the missing `src/sim/app.py`; `tests/test_code_explorer_*.py` need the missing `tools/code_explorer/`), plus the WP0 test files, which fail at collection with `ModuleNotFoundError: core.generate` until you implement them.

## Definition of done

1. All seven test files above pass (no skips, no xfails).
2. The full suite is no worse than the baseline: the same 3 pre-existing collection errors at most, and every previously passing test still passes.
3. Only the four files listed under "Files" were created; nothing else changed.

## If a test seems wrong

If a test appears to contradict `docs/v4/contracts.md`, or cannot be satisfied together with another test, **stop and report** the test name, the contract clause and your reasoning. Do not edit, skip or weaken any test, and do not change `tests/v4_support/`.
