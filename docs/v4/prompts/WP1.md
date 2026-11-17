# WP1 — task contract v2: TaskSpec additions, StageTracker, GenericTask, predicates, start sampler (Codex prompt)

## Goal

Extend the Isaac-free task layer so a task can be described entirely in `task.toml`: object sets, palette, roles, hold-outs and instruction templates in `TaskSpec`; a `StageTracker` with ever-latched and current stage flags; rule-based failure modes and `Task.outcome`; a TOML-driven `GenericTask` with a predicate registry; and a deterministic start sampler with injected reachability. Existing tasks (`tasks/pick_place_cube`) and existing callers must keep working unchanged. Make the WP1 tests pass.

## Files

| File | Change |
|---|---|
| `src/core/tasks/base.py` | Change: `StepState.tcp` / `.joints`; `TaskSpec` new fields; new `ObjectSetSpec`, `HoldoutSpec`, `CheckContext`, `FailureRule`, `Outcome`, `StageTracker`; new `Task` methods; `load_task_spec` reads and validates the new tables. |
| `src/core/tasks/generic.py` | New: `GenericTask`, `PREDICATES`, `register_predicate`, `resolve_ref`. |
| `src/core/tasks/sampling.py` | New: `StartSample`, `sample_start`, `park_positions`, `render_instruction`. |

`src/core/tasks/loader.py` needs no change (the brief's "loader changes" are the `load_task_spec` changes in `base.py`); change it only if a test requires it.

Do NOT touch: anything under `tests/` (including `tests/v4_support/`, `tests/conftest.py`, `tests/fixtures/`), `docs/`, `tasks/` (WP10 rewrites `pick_place_cube`), `arms/`, `scenes/`, `configs/`, `pyproject.toml`, `src/core/generate/*` (WP0), other packages' modules.

## Read first

`docs/v4/contracts.md` §1.1 (frames, quaternions XYZW in memory, object origins), §1.4 (RNG), §1.5 (errors), §1.6 (imports), §2 (WP1 row), §3.2 (`ObjectState`, `EpisodeSpec`, `StageProgress`, `ExpertObs.step_state`), **§4 (all of it — the main spec)**, §13, §14 (decisions 5–11). `tasks/pick_place_cube/__init__.py` and `benchmark.py::StageTrace` show the behaviour the rules must reproduce.

## Interfaces and the decisions taken where the contract is open

Everything below is pinned by the tests. Where this list adds to the contract it is a decision you must follow.

### base.py

- **Imports / cycles.** `core.generate.protocols` imports `core.tasks.base`, so `base.py` must not import `protocols` at module level: import `StageProgress` (and anything else from protocols) inside the functions that need it. Importing `core.generate.errors` at module level is fine. `sampling.py` may import `base` at module level; `base.py` imports `sampling` lazily inside methods.
- **`StepState`** fields, in order: `objects, ee, gripper, signals={}, tcp: Pose | None = None, joints: Mapping[str, float] | None = None`.
- **`ObjectSetSpec(name, prefix, count_max, template: ObjectSpec)`** frozen dataclass; `slot_names()`; `template` is `ObjectSpec.from_toml(<set name>, <set table minus prefix/count_max>, task_dir)` (so `template.name` is the set name and its params do **not** contain `set`/`slot`).
- **`HoldoutSpec(colors: tuple[str, ...] = (), counts: tuple[int, ...] = ())`** frozen dataclass.
- **`TaskSpec`** keeps its 7 fields and appends, in this order: `object_sets={}`, `palette={}`, `roles={}`, `holdout=HoldoutSpec()`, `instruction_templates=()` (use `default_factory`). Positional construction with 7 arguments must still work.
- **`load_task_spec`** (all new errors are `ConfigError`; the old `ValueError` / `FileNotFoundError` cases stay as they are):
  - Object sets: `prefix` defaults to the set name; `count_max` must be an `int` (not bool/float/str) ≥ 1. Slots are added to `spec.objects` **after all plain objects**, sets in TOML order, each slot an `ObjectSpec` copied from the template with `name=<slot>` and `params = {**template.params, "set": <set name>, "slot": i}`.
  - Name clashes: plain object names, set names and all slot names must be pairwise distinct.
  - Palette: TOML order kept. A value is either 3 numbers each in [0, 1] (stored as a tuple of floats) or the string `"from_setup:<key>"` (non-empty key); anything else is a `ConfigError`. **`from_setup` is resolved when the value is read, not at load** (loading must work without a `real_setup.json`): make `palette` a read-only `Mapping` (a private class) whose `__getitem__` returns `tuple(float(c) for c in load_setup()["object_materials"][key]["color"])`. Iterating keys must not read the setup. It must compare equal to `{}` when empty.
  - Roles: each value is a plain object name or a set name (not a slot name).
  - Holdout: only keys `colors`, `counts`. Colours must be palette names. Counts must lie in the target set's count range (and require a target set).
  - `instruction_template.texts`: a non-empty list of strings → `instruction_templates`. Placeholders (parse with `string.Formatter().parse`) must be exactly `{<role>.color}` or `{<role>.name}` with `<role>` in `[task.roles]`, or `{goal.<key>}` with `<key>` in `[task.start.goal]`. Any other field (`{target}`, `{target.size}`, `{}`, `{0}`, unknown role/goal key) is a `ConfigError`.
  - `instruction` keeps its old default (name with underscores → spaces); it is not derived from the templates.
  - `[task.failures].rules`: a list of tables with a string `name` and optional `if` / `unless` lists of strings, no other keys; the last rule must have empty `if` and `unless`.
  - `[task.start]` validation (only what is present; old layouts such as top-level `xy_min` / `xy_max` keep loading): see the sampler section. Do the start validation in `sampling.py` (private helper) and call it from `load_task_spec`.
- **`Task`**: `stages` and `observe` stay abstract (`Task.__abstractmethods__ == {"stages", "observe"}`); `sample_start(rng)` becomes concrete and returns `{name: state.pose for active objects of self.sample_episode_start(rng)}`. New members exactly as §4.2:
  - `sample_episode_start(self, rng, *, reach_fn=None, home_q=None, joint_limits=None, instruction_rng=None, joint_names=None, arm_joints=None) -> StartSample` = `sample_start(self.spec, rng, ...)` with the same keywords (the extra keywords beyond §4.2 are a decision, see "Contract deviations").
  - `observe_checks(state, ctx)` default `self.observe(state)`; `make_tracker()`; `hold_until` = `dict(raw["stages"]["hold_until"])` or `{}`; `failure_rules` = tuple of `FailureRule` from `raw["failures"]["rules"]` (`if` → `if_`) or `()`; `failure_names` = rule names in order.
  - `failure_mode(flags, holding, success)` exactly §4.2. Missing flag names are false. The name `"holding"` in `if`/`unless` always takes the `holding` argument, even if `flags` contains `"holding"`.
  - `outcome(tracker, final_checks, final_state) -> Outcome` exactly §4.2: flags = `tracker.flags()` plus the final values of every key of `final_checks` that is not a stage; `holding` = `final_checks.get("holding", final_state.signals.get("holding", False))`; `success` uses `[task.stages].success_requires_final` (default true); `furthest_stage = tracker.furthest_stage`.
  - `PickPlaceTask` overrides `sample_start` and `failure_mode`; both must keep working untouched.
- **`CheckContext`** frozen dataclass, fields `roles, start, goal, objects, active` (in that order); store mappings as `MappingProxyType` copies and `active` as `frozenset`. `from_spec(spec, task_spec)`: `roles = spec.roles`, `start = {name: state.pose for every object in spec.objects}` (inactive ones included), `goal = spec.goal`, `objects = task_spec.objects` (only `.objects` of `task_spec` is read), `active = {names with state.active}`.
- **`FailureRule(name, if_=(), unless=())`**, **`Outcome(success, failure_mode, furthest_stage)`**: frozen dataclasses, those fields in that order.
- **`StageTracker(stages, hold_until={})`** exactly §4.3. Attributes `stage_index`, `furthest_index` (both −1 initially), `regressions` (list of `(frame, stage)`), `flags()` returns a fresh `dict` of every stage → ever flag, `furthest_stage` property (`None` when −1). `update(checks)` returns `StageProgress(frame=t, stage_index, furthest_index, satisfied={k: bool(v) for every key of checks}, regressed_stage)` where t counts calls from 0. Missing keys are false. Only the first regressing stage (in stage order) fires per frame; the regression test runs after step 1 of the same frame. `hold_until` entries naming unknown stages raise `ValueError`.

### generic.py

- `PREDICATES: dict[str, Callable[[StepState, CheckContext, Mapping], bool]]`, pre-filled with the 13 predicates of §4.4 (including `nested_in`, §14.5). `register_predicate(name)` returns a decorator that stores the function and returns it unchanged.
- **Predicate calling convention:** `PREDICATES[name](state, ctx, params)` where `params` is the check table without `check` and `negate`, with defaults **not** filled in and refs **unresolved** (the predicate resolves them with `resolve_ref` and applies the §4.4 defaults). Return a plain `bool`.
- **`all` / `any`:** `GenericTask` evaluates dependencies first and calls them with `params["of"]` replaced by the list of already-computed **boolean values** (`PREDICATES["all"](s, ctx, {"of": [True, False]}) is False`).
- **`resolve_ref(ref, ctx)`**:
  - `"@role"` → `ctx.roles[role]` (an object name);
  - `"@role.inner"` → `MappingProxyType({"inner": <object name>})` (the rectangle itself depends on the container's current pose, so predicates compute it from the state); the object must have `params.inner_half_size`;
  - `"@goal.<key>(.<part>)*"` → `ctx.goal[key]`, then each part indexes a sequence (integer) or a mapping (string key), e.g. `"@goal.point.2"`, `"@goal.joint.angles.1"`, `"@goal.joint.joint"`;
  - a literal object name in `ctx.objects` → itself;
  - a literal list/tuple → `tuple(ref)`.
  - Anything unresolvable (unknown role, object, goal key, index; `.inner` on a non-container; any other `.suffix`) is a `ConfigError`. A non-string, non-sequence argument is a `TypeError`.
- **Geometry conventions** (§1.1, §4.4): `p(o)` = `state.objects[o].position` (root origin); `tcp` = `state.tcp or state.ee`; dims = `size` or `params["size_outer"]` (else zeros); `h` = dims[2]; `w` = min(dims[0], dims[1]); an object is bottom-origin iff it has no `size` and `params.get("origin", "center") == "bottom"`; yaw from the quaternion as in §8.2.
- **Predicate decisions** (beyond the §4.4 table):
  - A predicate whose object is absent from `state.objects` (inactive / parked) returns `False` (before `negate`).
  - `ee_near` requires exactly one of `object` / `point`; `point` is a literal `[x, y, z]` or a goal ref.
  - `object_in_region` regions: rect `{xy_min, xy_max}` (shrunk by `margin_m` on every side), polar `{r_m, bearing_deg}` (r range shrunk by `margin_m`, bearing = `degrees(atan2(y, x))` unshrunk), `"@goal.<key>"` resolving to an `{xy_min, xy_max}` mapping, or `"@<role>.inner"`. Inner: express `xy(o) − xy(c)` in the container's yaw frame and require `|local| ≤ inner_half_size − margin_m` per axis. The z test (only for the container case, when `check_z`): `bottom(o) − (p(c).z + params.floor_z) ≤ z_tol_m`, with `bottom(o) = p(o).z − h(o)/2` for centre origin and `p(o).z` for bottom origin. There is no lower z bound.
  - `objects_in_region`: `set` is a literal set name; members = objects in `ctx.objects` with `params["set"] == set`, in `ctx.active`, not in `{resolve_ref(r) for r in exclude}`.
  - `object_on_object`: expected `z(o) = z(b) + top(b) + bottom_offset(o)` with `top(b) = h(b)/2` (centre origin) or `h(b)` (bottom origin) and `bottom_offset(o) = h(o)/2` (centre) or `0` (bottom). (The contract's "drop the respective /2" is read physically: a bottom-origin object's origin sits on the base's top.)
  - `nested_in`: `max_rise_m` required (`ConfigError` at build when missing).
  - `joint_near`: `joint` is a USD name or a goal ref (if the ref resolves to a mapping, use its `"joint"` entry); `value_rad` is a number or a goal ref. Returns `False` when `state.joints` is `None` or lacks the joint.
  - `gap_near.value_m` may be a number or a goal ref. `object_displaced.axes` is any subset string of `"xyz"`.
- **`GenericTask(spec)`** (`ConfigError` for every problem, raised in `__init__`):
  - `[task.stages].order` must exist and be non-empty; `stages` property returns it as a tuple.
  - every stage has a check; every check's `check` is in `PREDICATES`; required parameters present (`object_lifted/held/object_displaced: object`, `object_displaced: min_m`, `object_in_region: object, region`, `objects_in_region: set, region`, `object_on_object` / `nested_in: object, base` (+ `max_rise_m`), `joint_near: joint, value_rad`, `gap_near: value_m`, `signal: name`, `all/any: of`);
  - **static ref check** against the TaskSpec (no episode yet): `@role` must be a key of `spec.roles`; `@role.inner` must be bound to an object (or a set whose template) has `inner_half_size`; `@goal.<key>` must be a key of `[task.start.goal]`; literal object names must be in `spec.objects`. Checked params: `object`, `base`, `point`, `region` (when a string), each entry of `exclude`, and `joint` / `value_rad` / `value_m` when they start with `@`. `objects_in_region.set` must be in `spec.object_sets`;
  - `of` entries are existing check names; no cycles (self-reference included);
  - `hold_until`: keys and values are stages and the value comes later in `order`;
  - failure rule `if` / `unless` names are checks, stages or `"holding"`.
  - `observe(state)` raises `TypeError`. `observe_checks(state, ctx)` returns a `dict` with **every** check (stage and helper), plain `bool`s, evaluating `all`/`any` after their dependencies whatever the TOML order, applying `negate` last.

### sampling.py

- **`StartSample`** frozen dataclass with the §4.5 fields; `objects` / `roles` as `MappingProxyType` copies, `goal` deep-frozen (lists → tuples, dicts → `MappingProxyType`, numpy scalars → Python floats), `initial_q` a tuple of floats or `None`. It must pass `v4_support.assert_frozen`. Equality (`==`) between samples from the same seed must hold.
- **`park_positions(names)`** → `{name_i: (0.1 * i, 0.0, -2.0)}` in the given order. The sampler parks inactive slots at `park_positions(list(task_spec.objects))[name]` with identity orientation, `active=False`, `color=None`.
- **`render_instruction(template, roles, colors, goal)`**: `{r.color}` → `colors[roles[r]]`; `{r.name}` → `roles[r]` with a trailing `_<digits>` removed and `_` → space (`"cube_3"` → `"cube"`, `"small_box"` → `"small box"`); `{goal.k}` → float: `f"{v:.2f}"`, sequence: items formatted the same way joined by `", "`, anything else `str(v)`.
- **`sample_start(task_spec, rng, *, reach_fn=None, home_q=None, joint_limits=None, instruction_rng=None, joint_names=None, arm_joints=None)`**. One Generator `rng` for everything except the template choice. Draw order and exact calls:
  1. Only if role `"target"` is bound to an object set (the *target set*): `lo, hi = start.count` (default `[count_max, count_max]`); `allowed = [c for c in range(lo, hi + 1) if c not in holdout.counts]`; `n = allowed[rng.integers(len(allowed))]`.
  2. Only if the target set exists and the palette is non-empty: `pal = [c for c in palette if c not in holdout.colors]` (palette key order); `target = pal[rng.integers(len(pal))]`; `distractors = rng.choice([c for c in pal if c != target], n - 1, replace=not distinct_colors)` — **always called, also when n − 1 = 0**. With an empty palette no colour draws happen and colours are `None`.
  3. Only if the target set exists: `slots = rng.permutation(count_max)[:n]`; `slots[0]` is the target slot with the target colour, `slots[1:]` get the distractor colours in order; the other slots are inactive. `roles["target"]` = the target slot name. Sets not bound to `"target"` keep all slots active with colour `None`; a role bound to such a set binds its slot 0. Plain objects are always active with colour `None`.
  4. Placement in this order: the plain objects with a `[task.start.place.<obj>]` entry (key order), then for each set entry (key order) its **active** slots in ascending slot index. Each object: up to `max_tries` (default 200) attempts; each attempt draws xy (rect: `rng.uniform(xy_min, xy_max)` → 2 values; polar: `r = rng.uniform(*r_m)`, `b = rng.uniform(*bearing_deg)` in degrees, `xy = r·(cos b, sin b)`), then `yaw = radians(rng.uniform(*yaw_deg))` (always drawn, default `[0, 0]`). Accept iff the constraints of §4.5 hold against every already placed active object:
     - a *container* is an object with `params.inner_half_size`; between two non-containers: xy centre distance ≥ `min_separation_m` (default 0.035); if either is a container: the distance between the two yawed footprint rectangles (dims[0] × dims[1], centred at the xy) ≥ `clearance_m` (default 0.01; 0 when they overlap);
     - if `reach_fn` is given and the object is in `reachable`: `reach_fn(Pose((x, y, centre_z + 0.06), (1.0, 0.0, 0.0, 0.0)))` is true, where `centre_z` = resting z + h/2 for bottom-origin objects (resting z itself for centre origin). **Decision: "TCP pointing down" is passed as `quat_xyzw = (1, 0, 0, 0)`; reach_fn implementations interpret it as approach axis −z.** Every attempt counts against `max_tries`.
     - Exhaustion raises `SamplingError`.
     - Resting pose: z = `table_z_m + h/2` (centre origin) or `table_z_m` (bottom origin); orientation yaw only, `quat_xyzw = (0, 0, sin(yaw/2), cos(yaw/2))`.
     - Active plain objects without a place entry are fixed at `ObjectSpec.position` with yaw `radians(rotation_euler_degrees[2])`, are placed before everything else, and act as obstacles; no draws.
  5. Goals in `[task.start.goal]` key order (values are JSON-like; tests read them as sequences/mappings of Python floats):
     - `point {region, z_m}` → `[x, y, z]`: xy as in step 4 then `z = rng.uniform(*z_m)`; if `"@goal.<key>"` is in `reachable` and `reach_fn` is given, retry (up to `max_tries`, else `SamplingError`) until `reach_fn(Pose((x, y, z), (1, 0, 0, 0)))`.
     - `points {count, region, z_m, min_separation_m=0}` → list of `count` points drawn like `point`, re-drawn until the 3-D distance to every earlier point is ≥ `min_separation_m` (`SamplingError` after `max_tries`).
     - `grid {points}` → `points[rng.integers(len(points))]` as a list of floats.
     - `gap {gap_m}` → `rng.uniform(*gap_m)`.
     - `region {center_region, half_size_m: [hx, hy]}` → centre drawn as an xy in `center_region`, result `{"xy_min": [cx − hx, cy − hy], "xy_max": [cx + hx, cy + hy]}` (`half_size_m` are the fixed half extents, not a range).
     - `joint {joints = "arm" | [names], fraction = [0.1, 0.9], count = 2}` → candidates = `arm_joints` for `"arm"`, else the list; `joint = candidates[rng.integers(len)]`; `f = rng.uniform(fraction[0], fraction[1], size=count)`; `angles = [lo + f_k (hi − lo)]` with `lo, hi = joint_limits[0][i], joint_limits[1][i]`, `i = joint_names.index(joint)`. Result `{"joint": joint, "angles": [...]}`. Missing `joint_names` / `joint_limits` (or `arm_joints` for `"arm"`) is a `ValueError` at call time.
  - `reachable` default: `["@target"]` if role `target` exists, else `[]`. Entries: `@role`, a literal object name, or `@goal.<key>`.
  - `initial_q`: `"home"` (default) → `tuple(home_q)` or `None` when `home_q` is `None`; a table `{joint = rad}` → `home_q` with those entries replaced at `joint_names.index(joint)` (`ValueError` if `home_q`/`joint_names` is missing or a joint is unknown).
  - `held` (a ref) → `held_object` = the resolved object name (it is placed normally; the env ignores its pose).
  - Instruction: if there are templates, `k = instruction_rng.integers(len(templates))` when `instruction_rng` is given, else `k = 0` (no draw from `rng`); render with `colors = {name: colour for active objects with a colour}`. Without templates, `instruction = task_spec.instruction`.
- **Load-time start validation** (`ConfigError`, called from `load_task_spec`): `start.count` requires a target set and `1 ≤ lo ≤ hi ≤ count_max`; at least one allowed count; holdout colours in the palette, holdout counts within `[lo, hi]` (and none without a target set); `len(pal) ≥ count_max` when `distinct_colors` (default true) and the palette is non-empty; every `start.place` key is a plain object or a set (not a slot) and has a `region`; a rect region has exactly `xy_min`, `xy_max` (2 values each, min ≤ max), a polar region exactly `r_m`, `bearing_deg` (lo ≤ hi, r ≥ 0); `reachable` entries resolve (role in `[task.roles]`, object in `spec.objects`, or `@goal.<key>` with a key of `[task.start.goal]`) and `held` resolves to a role or an object; each goal has a known `type`, and `point`/`points` have a valid `region`, `region` a valid `center_region`.

## Contract deviations (decisions to report back into contracts.md)

1. `sample_start` and `Task.sample_episode_start` take two extra keyword-only parameters, `joint_names` (USD names in joint-vector order, matching `home_q` and the `joint_limits` arrays) and `arm_joints` (the `"arm"` candidates, i.e. `[arm.expert].arm_joints`). The contract signature cannot map `initial_q = {joint = rad}` or `joints = "arm"` without them. `Task.sample_episode_start` also forwards `joint_limits` and `instruction_rng`.
2. `@role.inner` resolves to `{"inner": <container name>}`; `@goal.<key>.<part>…` allows any depth and string parts.
3. `object_on_object` origin handling and the `object_in_region` bottom-origin z test as described above.
4. `reachable` may contain `@goal.<key>` (used by point goals); pregrasp orientation `(1, 0, 0, 0)`.

## Tests

Acceptance tests (do not edit):

```
tests/core/test_v4_wp1_taskspec.py
tests/core/test_v4_wp1_stage_tracker.py
tests/core/test_v4_wp1_predicates.py
tests/core/test_v4_wp1_generic_task.py
tests/core/test_v4_wp1_sampling.py
```

They use `tests/v4_support` (`sandbox`, `write_task`, the §4.6 example tables `contract_color_task_table` / `contract_reach_task_table`, `make_pose`, `assert_frozen`) and the fixture tasks `tests/fixtures/v4_wp1/tasks/{fx_pick,fx_reach,fx_color}`. `fx_pick` is a GenericTask version of `pick_place_cube` whose six failure rules must reproduce `StageTrace.finish`; the tests extract `StageTrace` from `tasks/pick_place_cube/benchmark.py` with `ast`.

```
uv run --group host pytest tests/core/test_v4_wp1_*.py -q
# without uv:
PYTHONPATH=src python -m pytest tests/core/test_v4_wp1_*.py -q
```

WP0 (`src/core/generate/{errors,seeds,protocols}.py`) must be in place first; the WP1 tests import it (`ConfigError`, `SamplingError`, `StageProgress`, `ObjectState`, `ExpertObs.step_state(joint_names=...)`).

## Must keep working

- All WP0 tests (`tests/core/test_v4_wp0_*.py`) and every previously passing test in `tests/`. Full suite: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`; only the pre-existing collection errors (missing `src/sim/app.py`, missing `tools/code_explorer/`) and other wave-1 packages' not-yet-implemented modules may error.
- `tasks/pick_place_cube` loads and behaves exactly as before (`PickPlaceTask.sample_start`, `.failure_mode`, `.observe`).
- **Known conflict:** `tests/core/test_v4_support_selftest.py::TaskFolderTests::test_contract_tasks_load_as_specs` asserts `set(spec.objects) == {"small_box"}` for the colour-cube example, but §4.1 requires the six `cube_i` slots in `spec.objects`. This selftest will fail once WP1 is implemented; it is being corrected in `tests/v4_support` by the coordinator. Do not edit it and do not work around §4.1 to keep it green; mention it in your report.

## Rules

- Do not edit, skip, xfail or weaken any test, and do not change `tests/v4_support/` or `tests/conftest.py`.
- Public names: only those listed in §2 for WP1 (plus the existing ones in `base.py`); every helper is private (leading underscore).
- Imports in the new/changed modules: standard library, `numpy`, existing `core` modules, `core.generate.errors` (module level) and `core.generate.protocols` (lazily in `base.py`). No `torch`, `scipy`, `yaml`, Isaac, `random`, or global `np.random`. Python 3.11 syntax.
- If a test seems to contradict `docs/v4/contracts.md` or another test, stop and report the test name, the clause and your reasoning.

## Definition of done

1. The five WP1 test files pass (no skips, no xfails) together with the WP0 tests.
2. The full suite is no worse than before apart from the known selftest conflict above.
3. Only the three source files listed under "Files" changed.
