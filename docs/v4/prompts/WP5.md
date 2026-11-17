# WP5 — generation plan, fingerprints, manifest, config and DR appearance (Codex prompt)

## Goal

Add five Isaac-free modules to `src/core/generate/` that make generation reproducible and duplicate-free: the deterministic attempt stream and shard split (`plan.py`), the content fingerprints (`fingerprint.py`), the append-only generation manifest with crash reconcile and `generation_plan.json` (`manifest.py`), the YAML run config (`config.py`) and the DR appearance draws (`appearance.py`). The owner's hard requirement is **no duplicate episodes**: every rule below that looks pedantic (quantisation, sign canonicalisation, global skip decisions, torn-line handling, never re-executing a recorded attempt) exists for that requirement. Follow `docs/v4/contracts.md` §8 exactly; where the contract leaves a choice open, this prompt fixes it and the tests pin it.

## Files

Create exactly these files:

| File | Content |
|---|---|
| `src/core/generate/plan.py` | §8.1: `KINDS`, `attempt_seed`, `shard_quota`, `owns`, `PlannedAttempt`, `iter_plan`, `shard_attempts`, `check_capacity`, `CapacityReport` |
| `src/core/generate/fingerprint.py` | §8.2: `quantize`, `scenario_key`, `spec_fp`, `traj_fp`, `appearance_fp`, `episode_fp` |
| `src/core/generate/manifest.py` | §8.3–§8.5: `ManifestLine`, `Manifest`, `reconcile`, `ReconcileResult`, `write_plan_json`, `read_plan_json` |
| `src/core/generate/config.py` | §8.5: `Quotas`, `NoiseConfig`, `DRConfig`, `GenerateConfig`, `load_generate_config` |
| `src/core/generate/appearance.py` | §8.6: `AppearanceDraw`, `draw_appearance`, `dr_copy_sidecar` |

and make one change: add `pyyaml` to the `host` dependency group in `pyproject.toml` (contracts §14 recommends it for WP5; it currently arrives only through `draccus`).

Already committed with the tests (do not edit): `tests/fixtures/fingerprints/golden.json` (the §13 WP5 fixture; the contract table names this path) and `tests/fixtures/v4_wp5/contract_example.yaml` (the §8.5 YAML example verbatim).

Do NOT touch: anything under `tests/`, `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`; the WP0 files (`src/core/generate/{__init__,errors,protocols,seeds}.py`); `src/core/utils/hashing.py`; `src/sim/**` (in particular `sim/generate/dr_variants.py`); any other package's modules. `scripts/plan_generation.py` / `core/scripts/plan_generation.py` belong to WP7, not here.

## Read first

`docs/v4/contracts.md` §1.1 (quaternion sign rule), §1.3, §1.4, §1.5, §1.6, §2 (WP5 row), §3 (the WP0 types you consume), §4.1 and §4.5 step 2 (palette, hold-outs, recolour), §8 (all of it), §9.1 (sidecar keys used by reconcile and DR), §13, §14 decisions 12, 13, 22. Read `tests/v4_support/factories.py` and `tests/v4_support/stubs.py` to see how specs, results and writers are built in the tests.

## Module rules (tested)

- Public names: each module defines exactly its names above (classes, functions, and the module-level constant `KINDS`). Every other module-level name starts with `_` (including constants such as a manifest path or a term list). Methods on the public classes are fine.
- Imports: standard library, `numpy`, `core.*` only; `yaml` only in `config.py`. Allowed `core` imports: WP0 (`core.generate.errors`, `.protocols`, `.seeds`), the other four WP5 modules, and existing code (`core.utils.hashing`, `core.tasks.base`). Do not import `core.tasks.generic`, `core.tasks.sampling`, `core.arms`, `core.datasets`, `core.expert`, `sim`, `random`, `torch`, `pyarrow`, Isaac or LeRobot. Importing `plan`, `fingerprint`, `manifest` and `appearance` must not load `yaml` (so `manifest.py` and `appearance.py` must not import `config.py` at module level).
- RNG: only `np.random.default_rng(seed)`; no global `np.random.*` calls.
- Python 3.11 syntax. No SO-101 string literals (`"so101"`, joint names, `"jaw"`, `"gripper"`, …) in these modules.
- Hashing: always `canonical_json` / `json_sha256` from `core.utils.hashing`, never `json.dumps` with other settings, for anything that is hashed.

## plan.py (§8.1)

- `KINDS = (Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY)` (a tuple).
- `attempt_seed(master_seed, task, kind, index) -> int` = `derive_seed(master_seed, task, Kind(kind).value, index)`; accepts a `Kind` or its value string.
- `owns(index, shard, n)`, `shard_quota(total, shard, n)` exactly as in §8.1. Shard arguments must satisfy `n >= 1` and `0 <= shard < n`, and `total >= 0`; otherwise raise `ValueError` (also from `shard_attempts`, eagerly or on first `next`).
- `PlannedAttempt`: `@dataclass(frozen=True)` with fields `task, kind, index, seed, spec, scenario_key, skip` in that order. A `scenario_seen` skip **keeps** `spec` and `scenario_key` (the manifest records them, §8.3); a `sampling_failed` skip has `spec = scenario_key = None`.
- `iter_plan(task, master_seed, spec_fn, *, unique_across_kinds=True)`: an infinite, lazy generator over `index = 0, 1, …` and, inside each index, `KINDS` order. For each position: `seed = attempt_seed(...)`; call `spec_fn(kind, index, seed)` (kind is the `Kind` member); `SamplingError` → skip `sampling_failed`; any other exception propagates. Otherwise compute `scenario_key(spec)`; the mark is the key (or `(kind, key)` when `unique_across_kinds=False`); if the mark was already produced by an earlier **non-skipped** position the attempt is a `scenario_seen` skip, else it is non-skipped and the mark is recorded. Decisions:
  - `spec_fn` is called for every position, whatever the quotas, and the seen-set covers the **whole global stream**, not a shard's subset. That is what makes skip decisions identical for every shard count and every resume point: a scenario first drawn at an index owned by shard 1 is a `scenario_seen` skip at a later index owned by shard 0.
  - The first occurrence wins even if it is later rejected or never executed (quota met); the scenario is consumed.
  - No hidden state: two interleaved iterators over the same arguments yield identical items; the global numpy RNG is never used.
- `shard_attempts(it, shard, n)`: yields the items of any iterable whose `index % n == shard`, in order (skips included).
- `CapacityReport`: `@dataclass(frozen=True)`, fields `non_skipped: dict` (all three `Kind` keys, plain `dict`) and `horizon: int`.
- `check_capacity(task, master_seed, spec_fn, quotas, *, max_attempt_factor, max_consecutive_skips, unique_across_kinds=True)`. Decisions (the contract text is short):
  - `quotas` is a `config.Quotas` or a mapping keyed by `Kind` or kind value (missing kinds = 0). The extra keyword `unique_across_kinds` mirrors the config flag (without it the check would be wrong for configs that set it false). Import `Quotas` lazily or duck-type it (`getattr(q, "success")`), never import `config` at module level.
  - `budget[k] = ceil(max_attempt_factor × quota[k])`. Walk `iter_plan` in order. A kind is *active* while `budget[k] > 0` and it is neither done nor exhausted. Positions of inactive kinds are ignored. For an active kind: a non-skipped attempt increments `non_skipped[k]` and resets its consecutive-skip run; reaching `budget[k]` marks it done. A skip increments the run; when the run **reaches** `max_consecutive_skips` the kind is exhausted: raise `ScenarioSpaceExhausted` if `non_skipped[k] < quota[k]`, otherwise just stop counting it. Consecutive skips are counted per kind (other kinds' positions neither add to nor reset the run).
  - The walk ends when no kind is active. `horizon` = 1 + the largest attempt index examined for an active kind (0 when every quota is 0).
  - Enforcing `AttemptBudgetExceeded` and `ScenarioSpaceExhausted` while a shard runs is WP7's loop, not this module.

## fingerprint.py (§8.2)

- `quantize(x, step)`: for a `np.ndarray` return `np.floor(np.asarray(x, np.float64) / step + 0.5).astype(np.int64)`; for a scalar (Python or numpy, float32 included) convert with `float(x)` **first** and return the Python `int` `math.floor(v / step + 0.5)` (so a float32 scalar quantizes exactly like the Python float of the same value; halves round up, never banker's rounding; `-0.0` gives `0`). Non-finite values and `step <= 0` (or non-finite) raise `ValueError`; a `bool` raises `TypeError`.
- Quaternion sign (§1.1): before using a quaternion `(x, y, z, w)` negate it if `w < 0`, or if `w == 0` and the first non-zero of `x, y, z` is negative.
- `scenario_key(spec)`: the §8.2 payload with these decisions:
  - `"kind"`: `EpisodeSpec` carries no set membership, so the kind is the object name with a trailing `_<digits>` removed (`re.sub(r"_\d+$", "", name)`): `cube_3 → "cube"` (the set prefix), `small_box → "small_box"`. Two same-kind, same-colour objects that swap poses are therefore the same scenario, and slot names never matter.
  - `"role"`: the lexicographically smallest role whose value is that object, else `null`.
  - `"color"`: `ObjectState.color`; `"p"`: `[quantize(float(c), 1e-3) for c in position]`; `"yaw"`: `quantize(atan2(2(wz + xy), 1 − 2(y² + z²)), π/180)` of the sign-canonical quaternion.
  - Only active objects; the list is sorted by `canonical_json` of each entry.
  - `"goal"`: the goal converted to plain JSON with every **float** (Python or numpy floating) replaced by `quantize(v, 1e-3)`; `bool`/`np.bool_` stay bools, ints (and numpy ints) stay ints, strings and `None` unchanged, lists/tuples → lists, mappings → dicts (recursively). Note an int `1` and a float `1.0` therefore differ; that is intended.
  - `"q0"`: `[quantize(float(q), π/360) for q in spec.initial_q]`; `"held"`: `spec.held_object is not None`; plus `"v": 1`, `"task"`, `"arm"`, `"scene"`. Nothing else (no kind, attempt index, seed, instruction, perturbation, noise, max_steps, fps).
- `spec_fp(spec)`: `d = spec.to_json()`, remove `"attempt_index"` and `"seed"` (keep `"schema"` and everything else), then the same recursive float quantization with step `1e-4`; a dict whose key set is exactly `{"position", "quat_xyzw"}` is a pose: canonicalise its quaternion sign before quantizing. Hash `{"v": 1, "spec": <that>}`.
- `traj_fp(action_raw, state_raw, objects0)`: both arrays (lists, float32 or float64) must be 2-D with equal shapes and finite, else `ValueError`. `Q` exactly as in §8.2 (`"<i4"`, C order). `objects0` maps names to a `Pose` (its `.position` is used) or a length-3 sequence/array (else `ValueError`); each coordinate is `quantize(float(c), 1e-4)`. `"shape"` holds Python ints.
- `appearance_fp(colors, instruction, light_exposure, robot_color)`: payload exactly as §8.2; `colors` is used as given (callers pass the active objects); `light_exposure` → `quantize(float(x), 1e-3)` or `null`.
- `episode_fp(traj_fp, appearance_fp)`: both arguments must be 64-character lower-case hex strings (else `ValueError`); `hashlib.sha256((traj_fp + appearance_fp).encode("ascii")).hexdigest()`.
- Golden values: `tests/fixtures/fingerprints/golden.json` holds inputs (specs as `EpisodeSpec.to_json()`), the canonical payloads and the expected hex. If your output differs, your payload differs from the table; never "fix" the fixture.

## manifest.py (§8.3–§8.5)

**`ManifestLine`** — `@dataclass(frozen=True)` with fields in table order without `v`: `task, kind, attempt_index, seed, shard, status, reason, episode_index, scenario_key, spec_fp, traj_fp, appearance_fp, episode_fp, success, failure_mode, furthest_stage, perturbation, num_frames, wall_s` (`wall_s: float = 0.0` is the only default). `to_json()` returns plain builtins: `{"v": 1, ...all fields}` with enums as `.value`, `shard` as `[i, n]`. `from_json(d)` (classmethod) requires exactly the 20 keys and `v == 1`. `__post_init__` coerces `kind`/`status` (`Kind`, `AttemptStatus`), `reason` (`SkipReason` first, then `RejectionReason`), `shard` → tuple, `wall_s` → float, and validates; every violation raises `SchemaError` (from the constructor and from `from_json`):
- `task` non-empty str; `attempt_index` int ≥ 0; `seed` int (not bool, not float) in `[0, 2**64)`; `shard` `[i, n]` ints with `0 ≤ i < n`; the five fingerprints `null` or 64-char lower-case hex; `success` bool or null; `failure_mode`, `furthest_stage`, `perturbation` str or null; `num_frames` int ≥ 0 or null; `episode_index` int ≥ 0 or null; `wall_s` finite number.
- accepted: `reason` null, `episode_index`, `num_frames` and all five fingerprints non-null; non-accepted: `episode_index` null;
- rejected: `reason` is a `RejectionReason`;
- skipped: `reason` is a `SkipReason`; `traj_fp`, `appearance_fp`, `episode_fp` null; `sampling_failed` ⇒ `scenario_key` and `spec_fp` null; `scenario_seen` ⇒ both non-null.

Extra classmethod `ManifestLine.from_result(result: EpisodeResult, *, shard, perturbation: str | None = None, wall_s: float = 0.0)`: fields from the result and its `Fingerprints`; `num_frames` is `None` for skipped results.

**`Manifest(path)`** (`path` is the `.jsonl` file; WP7 passes `<root>/meta/generation_manifest.jsonl`):
- `lines() -> list[ManifestLine]` reads the file every call (`[]` if it does not exist). Split the bytes on `"\n"`: a trailing piece without `"\n"` is torn and ignored. Among complete lines, the **last** one is also torn if it is not valid JSON. Any other line that is not valid JSON (an empty line included) raises `SchemaError`; a valid-JSON line that fails `ManifestLine.from_json` raises `SchemaError` wherever it is. Two lines with the same `(task, kind, attempt_index)` raise `SchemaError`.
- `append(line)`: validate (accepts a `ManifestLine`, or a dict via `from_json`), scan the file as above (so a corrupt file raises before anything is written), raise `SchemaError` if a line for the same `(task, kind, attempt_index)` exists (file unchanged), create the parent directory, truncate the file to the end of the last good line if a torn tail exists, then open in append-binary mode, write `canonical_json(line.to_json()) + "\n"`, `flush()`, `os.fsync(fileno)` (call it as `os.fsync`; a test spies on it). Earlier bytes are never rewritten.
- `accepted()` = accepted lines in file order; `last_index(kind)` = the largest `attempt_index` of any line of that kind, `None` if there is none.

**`reconcile(root, writer, manifest) -> ReconcileResult`** with `ReconcileResult` a frozen dataclass `(rebuilt_lines: int, deleted_sidecars: list[int], next_position: tuple[int, int])`. Sidecars are `<root>/task_meta/episode_{index:06d}.json` (WP4's `sidecar_path` layout; reimplement the path privately, WP5 may not import WP4); only names matching `^episode_\d{6}\.json$` count. **All checks run before any change; on `ResumeError` nothing is modified.** Order:
1. `lines = manifest.lines()`; accepted `episode_index` values in file order must be exactly `[0, 1, …, M−1]`, else `ResumeError`.
2. `E = writer.episode_count()`; require `M ≤ E ≤ M + 1`, else `ResumeError`.
3. Every index `< E` must have a sidecar, else `ResumeError` (rule 5).
4. If `E = M + 1`: read sidecar `E−1` and build its line — `task`, `kind`, `num_frames`, `success`, `failure_mode`, `furthest_stage` from the sidecar; `attempt_index`, `seed`, `shard` from `attempt.{index, seed, shard}`; the five fingerprints from `fingerprints`; `perturbation` = `perturbation["type"]` or null; `status = accepted`, `reason = null`, `episode_index = E−1`, `wall_s = 0.0`. `ResumeError` if the file is not valid JSON, `attempt` or `fingerprints` is null/missing, its `episode_index` is not `E−1`, the line would be invalid, or a line for the same `(task, kind, attempt_index)` already exists.
5. Now mutate: append the rebuilt line (if any), then delete every sidecar with index `≥ E` (orphans; returned sorted).
6. `next_position`: positions are `(attempt_index, KINDS.index(kind))`. It is the position immediately after the largest recorded position over all lines (after the rebuild): `(i, k + 1)` if `k < 2`, else `(i + 1, 0)`; `(0, 0)` with no lines. This is §8.4 rule 6 in a shard-independent form: the loop resumes at the first position it owns that is `≥ next_position`. Earlier positions without a line are kinds whose quota was already met (no line is written for those), which stay met, so nothing is ever executed twice.

**`write_plan_json(root, plan: Mapping) -> dict`** and **`read_plan_json(root) -> dict`** (`<root>/meta/generation_plan.json`):
- Validation (`SchemaError`): exactly the §8.5 keys; `v == 1`; `config_sha256`, `task_toml_sha256` 64-char hex; `config` a mapping; `master_seed` int in `[0, 2**64)`; `git_commit`, `task`, `arm`, `scene`, `created_at` non-empty strings; `shard` `[i, n]` with `0 ≤ i < n`; `arm_files_sha256` a mapping of str → hex; `quotas` exactly `{success, failure, recovery}` ints ≥ 0; `holdout` exactly `{colors: [str], counts: [int]}`.
- `write_plan_json`: validate first (nothing written on error). If the file does not exist, write it atomically (temp file in `meta/` + flush + `os.fsync` + `os.replace`; `json.dump(indent=2, sort_keys=True)`; no `*.tmp` left) and return the plan as plain JSON. If it exists (resume): read and validate it; a difference in `config_sha256`, `task_toml_sha256`, `arm_files_sha256`, `master_seed` or `shard` raises `ResumeError`; a different `git_commit` only emits `warnings.warn(..., RuntimeWarning)`; the existing file is kept unchanged (its `created_at` survives) and returned. Other keys are not compared.
- `read_plan_json`: read, validate, return; `FileNotFoundError` if absent.

## config.py (§8.5)

- Dataclasses as in §8.5, frozen, fields in that order; `GenerateConfig.quotas` and `perturbation_weights` are `types.MappingProxyType` (or `None` for weights); `tasks` and `DRConfig.terms` tuples; `shard` a tuple; `dst_root` a `Path`. Instances pass `v4_support.assert_frozen`.
- Top-level keys are exactly those of the §8.5 example. Required: `arm`, `scene`, `tasks`, `master_seed`, `quotas`, `dst_root`, `repo_id`. Defaults: `per_task_quotas: {}`, `shard: "0/1"`, `fps: 30`, `max_attempt_factor: 3.0`, `max_consecutive_skips: 1000`, `scenario_unique_across_kinds: true`, `noise: {}`, `dr: {}`, `perturbation_weights: null`, `cameras: true`. A missing required key or any unknown key (top level, inside `quotas`, `per_task_quotas.<task>`, `noise`, `dr`) raises `ConfigError`. Hold-outs are not a config key (§14 decision 8).
- Values (`ConfigError` otherwise; bools are never accepted as ints): `arm`, `scene`, `repo_id` non-empty strings; `tasks` a non-empty list of distinct non-empty strings (a bare string is rejected); `master_seed` int in `[0, 2**64)`; `quotas` needs all three keys, ints ≥ 0; `per_task_quotas` maps tasks listed in `tasks` to partial quota tables whose missing keys inherit from `quotas`; `shard` the string `"i/n"` (or a 2-list of ints) with `n ≥ 1`, `0 ≤ i < n`; `dst_root` a string (or `Path`); `repo_id` may use only the `{task}` placeholder and must contain it when there is more than one task (otherwise two datasets would share a repo); `fps` int ≥ 1; `max_attempt_factor` a finite number ≥ 1.0 stored as `float`; `max_consecutive_skips` int ≥ 1; `scenario_unique_across_kinds`, `cameras`, `dr.recolor` bools; `noise.fraction` and `noise.theta` in `[0, 1]`, `noise.sigma_rad` and `noise.max_abs_rad` ≥ 0 (all finite, stored as float; missing keys take the `NoiseConfig` defaults); `dr.copies` int ≥ 0, `dr.terms` a list of distinct names from `("light", "robot_color")` (camera terms are not fingerprinted by §8.2, so they are rejected), `dr.seed` int in `[0, 2**64)` (missing keys take the `DRConfig` defaults); `perturbation_weights` null or a non-empty mapping of str → finite float ≥ 0 (type names are validated against the task by WP6, not here). A YAML syntax error or a non-mapping document raises `ConfigError`; a missing file raises `FileNotFoundError`. Parse with `yaml.safe_load`.
- `overrides`: top-level keys replacing the YAML values before validation; a `None` value means "not given" (so CLI defaults can be passed straight through); an unknown key raises `ConfigError`.
- `quotas` = `{task: per-task Quotas}` for every task in `tasks`, in `tasks` order.
- Extra method `GenerateConfig.to_json() -> dict` returns the resolved config, exactly: `{"arm", "scene", "tasks": [..], "master_seed", "quotas": {task: {"success", "failure", "recovery"}}, "shard": [i, n], "dst_root": str(dst_root), "repo_id", "fps", "max_attempt_factor": float, "max_consecutive_skips", "scenario_unique_across_kinds", "noise": {"fraction", "theta", "sigma_rad", "max_abs_rad"}, "dr": {"copies", "terms": [..], "recolor", "seed"}, "perturbation_weights": dict | None, "cameras"}`. WP7 writes it as `generation_plan.json["config"]`. `sha256` = `json_sha256(to_json() without "shard" and "dst_root")`, so all shards of one run agree and every other setting (including quotas and per-task overrides, after merging) changes it; equivalent spellings (`3` vs `3.0`, key order, a per-task override equal to the global quota) give the same hash.
- `dataset_root(task)` = `dst_root / task`, `dataset_repo_id(task)` = `repo_id.format(task=task)`; both raise `ValueError` for a task not in `tasks`.

## appearance.py (§8.6)

`draw_appearance` must not import WP1, so it duck-types `task_spec` and reads only: `task_spec.palette` (mapping; its key order is the palette order), `task_spec.holdout.colors`, `task_spec.instruction_templates` (tuple of str) and `task_spec.raw.get("start", {}).get("distinct_colors", True)`. The real WP1 `TaskSpec` has these attributes; the tests use a `SimpleNamespace` stand-in.

`parent` is a v2 sidecar (§9.1). Read `objects` (name → `{set, color, active, …}`), `roles`, `instruction`, `fingerprints.traj_fp`, `fingerprints.appearance_fp` and optionally `goal` (absent in §9.1 sidecars; used only for `{goal.*}` placeholders). `ValueError` if `fingerprints` or its `traj_fp` is null (legacy v1 parents cannot be DR'd here), if a term is not `"light"`/`"robot_color"`, if `"robot_color"` is a term and `robot_colors` is empty, or if `exposure_range[0] > exposure_range[1]`.

Algorithm (all names sorted, so the result never depends on dict order — a parent read back through canonical JSON gives the same draw):
1. `names` = sorted active object names; `base` = their parent colours.
2. Recolour group (only if `recolor`, the palette minus hold-outs `pal` is non-empty, a `"target"` role exists and its object has a non-null colour): `[target] + sorted(other active objects whose "set" equals the target's non-null set)`. Objects outside the group keep their colour (a box stays `null`).
3. `excluded = set(taken) ∪ {parent appearance_fp}` — the parent's own fingerprint is always excluded even if the caller forgot it. `taken` is never mutated.
4. For `r = 0 … max_redraws − 1`: `seed = derive_seed(dr_seed, traj_fp, copy_index, r)`, `rng = np.random.default_rng(seed)`, then in this order: (a) recolour per §4.5 step 2 — target colour `pal[rng.integers(len(pal))]`, distractors `rng.choice(pal minus target, len(group) − 1, replace=not distinct_colors)` assigned in group order (`GenerationError` if distinct colours are needed and the palette is too small); (b) if there are templates, `template = templates[rng.integers(len(templates))]` and render it, else keep the parent's instruction; (c) `light = float(rng.uniform(lo, hi))` if `"light"` in terms else `None`; (d) `robot = robot_colors[rng.integers(len(robot_colors))]` if `"robot_color"` in terms else `None`; (e) `fp = appearance_fp(colors over names, instruction, light, robot)`. The first `fp` not in `excluded` wins. If none, raise `GenerationError` (D13: a copy with no effective change collides with its parent).
5. Rendering (a private copy of §4.5 `render_instruction`, since WP1 may not be imported): `{<role>.color}` → the new colour of `roles[role]`, `{<role>.name}` → the object name `roles[role]`, `{goal.<key>}` → `str(goal[key])`; an unknown placeholder, a role with no colour or a missing goal key raises `GenerationError`.

`AppearanceDraw`: frozen; `colors` stored as a `MappingProxyType` copy (name → colour or `None`, every active object of the parent); `seed` is the winning draw seed; `appearance_fp` is the winning `fp`.

`dr_copy_sidecar(parent, draw, *, episode_index, source_repo_id) -> dict`: a deep copy of `parent` (the parent is never mutated) with `episode_index` set; each `objects[name]["color"]` from `draw.colors`; `instruction = draw.instruction`; `appearance = {"light_exposure", "robot_color"}` from the draw; `fingerprints = {scenario_key, spec_fp, traj_fp}` kept from the parent plus `appearance_fp = draw.appearance_fp` and `episode_fp = episode_fp(traj_fp, draw.appearance_fp)`; `source = {"repo_id": source_repo_id, "episode_index": parent["episode_index"], "traj_fp": parent traj_fp, "episode_fp": parent episode_fp}`; `generator = {"name": "dr_replay", "git_commit": parent's git_commit (or "unknown")}`; `legacy_v1 = False`. Everything else (per-frame arrays, `attempt`, `roles`, `kind`, …) is unchanged. The result is plain JSON.

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp5_fingerprint.py   # golden values, quantisation, -0.0, float32, sign, dict order
tests/core/test_v4_wp5_plan.py          # attempt stream, shards, global skips, capacity
tests/core/test_v4_wp5_manifest.py      # ManifestLine, append-only file, torn lines, reconcile, plan json
tests/core/test_v4_wp5_config.py        # contract example, defaults, overrides, sha256, rejections
tests/core/test_v4_wp5_appearance.py    # DR draws and copy sidecars
tests/core/test_v4_wp5_surface.py       # public names, imports, 3.11 syntax, no SO-101 literals
```

Run from the repo root with the host dependency group (Python 3.11):

```
uv run --group host pytest tests/core/test_v4_wp5_*.py tests/core/test_v4_wp0_*.py tests/core/test_v4_support_selftest.py -q
```

Without uv: `PYTHONPATH=src python -m pytest tests/core/test_v4_wp5_*.py tests/core/test_v4_wp0_*.py tests/core/test_v4_support_selftest.py -q` in an environment with numpy, pyyaml and pytest.

## What must keep working

- All WP0 tests and `tests/core/test_v4_support_selftest.py`.
- The full suite must be no worse than before: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors` — same pre-existing collection errors at most (see `docs/v4/prompts/WP0.md`), every previously passing test still passing. Other packages' tests that fail only because their modules do not exist yet are not your concern.
- `sim/generate/dr_variants.py` is unchanged (its v1 `plan_episodes` output is golden for WP11).

## Rules

- Do not edit, skip, xfail or weaken any test, and do not change `tests/v4_support/`, `tests/conftest.py` or the fixtures.
- Create only the files listed under "Files" (plus the one-line `pyproject.toml` change).
- If a test appears to contradict `docs/v4/contracts.md` or this prompt, or two tests cannot both pass, **stop and report** the test name, the clause and your reasoning instead of working around it.
