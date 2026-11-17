# WP6 — perturbations, labels and episode specs (Codex prompt)

## Goal

Add three Isaac-free modules to `src/core/generate/`:
- `perturbations.py`: the perturbation taxonomy (expert errors A, world disturbances B), the validated, deterministic draw of a `PerturbationSpec` from `[task.perturbations]`, the group-B `WorldPerturber` and group-C OU noise.
- `labels.py`: per-frame labels (stage, imitate, ee pose), the dataset label columns, phases, segments, recovery frames and the outcome-based accept/reject decision.
- `specs.py`: `build_spec`, which turns `(task, kind, attempt_index, seed)` into an `EpisodeSpec`.

The labels are the training signal and the acceptance table decides what enters the dataset. Labels always follow the **observed** outcome (`Task.outcome`, WP1), never the intended kind. An attempt whose outcome does not match its kind is rejected with the contract's reason; it is never relabelled. Follow `docs/v4/contracts.md` §7 exactly. Where the contract leaves a choice open, this prompt fixes it and the tests pin it.

## Files

Create exactly these files:

| File | Public names |
|---|---|
| `src/core/generate/perturbations.py` | `PERTURBATION_TYPES`, `PerturbationType`, `draw_perturbation`, `WorldPerturber`, `ou_noise` |
| `src/core/generate/labels.py` | `build_segments`, `build_imitate`, `accept` (§7.2), plus four helpers decided here: `build_phases`, `build_recovery`, `frame_labels`, `label_columns` |
| `src/core/generate/specs.py` | `build_spec` |

Do NOT touch:
- anything under `tests/` (including `tests/v4_support/`, `tests/conftest.py`, `tests/fixtures/`);
- `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`, `pyproject.toml`;
- WP0 (`core/generate/{__init__,errors,protocols,seeds}.py`), WP1 (`core/tasks/*`), WP4 (`core/datasets/*`) and WP5 (`core/generate/{plan,fingerprint,manifest,config,appearance}.py`), and any other module.

## Read first

`docs/v4/contracts.md`:
- §1.4 (RNG, sub-seeds), §1.5 (errors), §1.6 (imports), §2 (WP6 row);
- §3.1–3.2 (`Kind`, `SegmentLabel`, `RejectionReason`, `PerturbationGroup`, `PerturbationSpec`, `NoiseSpec`, `EpisodeSpec` invariants, `StageProgress`, `FrameLabels`, `Segment`, `RecoveryInfo`);
- §4.2–4.3 (`Task.outcome`, `StageTracker`), §4.5 (`sample_start`), §4.6 (the `perturbations` row and both example tables), §6.1–6.2 (expert steps, re-entry);
- **§7 (all of it, the main spec)**;
- §8.5 (`NoiseConfig`, `perturbation_weights`), §9.1 and §9.3 (sidecar label keys, `LABEL_FEATURES`), §11 (support rule), §11.1 (loop steps 3 and 5), §13, §14 decisions 2, 14, 15, 16 and 17;
- §15 (clarifications; WP1 sampler keywords `joint_names` / `arm_joints`).

Also read `docs/v4/prompts/WP1.md` for the `GenericTask` / `TaskSpec` / `StartSample` behaviour you build on.

## Module rules (tested)

- **Public names.** Each module defines exactly its names above (classes, functions and the constant `PERTURBATION_TYPES`). Every other module-level name starts with `_`.
- **Imports.** Standard library, `numpy` and `core.*` only.
  - Allowed `core` imports: WP0 (`core.generate.errors`, `.protocols`, `.seeds`), WP1 / existing `core.tasks.*`, `core.utils.*`, and `core.generate.perturbations` (from `specs.py`).
  - Forbidden: `core.arms`, `core.datasets` (so `labels.py` restates the three `LABEL_FEATURES` dtypes rather than importing them; a test checks that they agree), `core.expert`, every WP5 module (duck-type `NoiseConfig`), `scipy`, `yaml`, `pyarrow`, `torch`, `random`, `sim`, Isaac and LeRobot.
  - Importing the three modules must not load any of those (checked in a subprocess).
- **RNG.** Use only the `np.random.Generator` you are given (or `rng_for`). Never call the global `np.random.*`.
- **Syntax and literals.** Python 3.11 syntax. No SO-101 string literals (`"so101"`, joint names, `"jaw"`, `"gripper"`, …).

## perturbations.py (§7.1)

### `PerturbationType` and `PERTURBATION_TYPES`

`PerturbationType` is a frozen dataclass:

```python
@dataclass(frozen=True)
class PerturbationType:
    name: str
    group: PerturbationGroup
    kinds: frozenset[Kind]
    params: Mapping[str, tuple | str | None]
    def allowed_stages(self, task) -> tuple[str, ...]
```

- `kinds` is stored as a `frozenset`.
- `params` maps each parameter name, in declared order, to its default: a `(lo, hi)` range tuple, a string, or `None` for a required parameter. Store it as a `MappingProxyType`.
- Instances must pass `v4_support.assert_frozen`.

`PERTURBATION_TYPES` is a `MappingProxyType` from name to `PerturbationType`, in the order of the §7.1 table. The parameters, their order and their defaults are a decision of this prompt. `delay_steps` is listed for every type whose trigger uses it.

| name | group | kinds | params (declared order; `None` = required) |
|---|---|---|---|
| `grasp_offset` | expert | F, R | `xy_m` None, `z_m` (0.0, 0.0), `yaw_rad` (0.0, 0.0) |
| `early_close` | expert | F, R | `delay_steps` None |
| `partial_close` | expert | F, R | `extra_gap_m` (0.004, 0.012) |
| `early_release` | expert | F, R | `delay_steps` None |
| `place_offset` | expert | F, R | `xy_m` None |
| `wrong_target` | expert | F, R | — |
| `target_offset` | expert | F, R | `xy_m`, `z_m`, `joint_rad`, `gap_m`, all (0.0, 0.0) |
| `stall` | expert | F | `delay_steps` (0, 0), `steps` None |
| `object_shift` | world | F, R | `object` "@target", `delay_steps` (0, 0), `xy_m` None, `yaw_rad` (0.0, 0.0) |
| `object_drop` | world | F, R | `delay_steps` (0, 0), `xy_jitter_m` (0.0, 0.0) |
| `container_shift` | world | F, R | `object` "@container", `delay_steps` (0, 0), `xy_m` None |

### `allowed_stages(task)`

Returns the stages, in task stage order, that an entry's `stages` may name. `task` is a WP1 `Task`; read `task.stages` and `task.spec.raw` (`expert.steps`, `checks`).

- `wrong_target`: `()`.
- `target_offset`: the stages that appear as `stage` on an expert step that also has a `target`, `q` or `gap_m` key.
- `object_shift`: the stages strictly before the `stage` of the first `grasp` step that has a `stage`. If there is no such step, every stage.
- `container_shift`: the same rule with the first `release` step ("any stage before `placed`").
- `object_drop`: the stages whose check implies holding. A check implies holding if any of these holds:
  - its name is `holding`;
  - it is a non-negated `held` check;
  - it is a non-negated `all` check with at least one `of` entry that implies holding (recursively; guard against cycles).

  `any` checks and negated checks never imply holding.
- Every other type: all task stages. For the step-triggered types (`grasp_offset`, `early_close`, `partial_close`, `early_release`, `place_offset`), `onset_stage` is recorded in the spec and the sidecar, but the trigger is the step rule of §7.1. This is why the §4.6 example may write `early_release` with `stages = ["lifted"]`.

### Validation (`ConfigError`)

`draw_perturbation` validates the **whole** `[task.perturbations]` table on every call, for every kind including `Kind.SUCCESS`, before any draw. This lets WP7 call it once up front so a bad table fails at load and not mid-run (§1.5). Each item below is a `ConfigError`:

- `[task.perturbations]` is not a table, or has a key other than `failure` / `recovery`.
- A list entry is not a table, or has no string `type`.
- An unknown type, or a type whose `kinds` do not include the list's kind (`stall` under `recovery`).
- A bad `weight` (default 1.0): it must be an `int` or `float`, not a bool, finite and ≥ 0.
- A bad `stages`: if present it must be a non-empty list of str, and a subset of `allowed_stages(task)`. `wrong_target` therefore accepts no `stages` at all.
- A missing `stages` for `stall`, `object_shift`, `object_drop` or `container_shift`; their trigger needs an onset stage.
- An unknown parameter key. The keys other than `type`, `weight` and `stages` must be declared parameters of the type, so `delay_steps` on `grasp_offset` is an error.
- A missing required parameter.
- A bad range parameter. It must be a 2-element list/tuple `[lo, hi]` with `lo ≤ hi`. In addition:
  - `delay_steps` and `steps` must be Python ints (`type(v) is int`);
  - every other range parameter must be finite numbers (not bool);
  - `xy_m`, `xy_jitter_m`, `extra_gap_m` and `delay_steps` must have `lo ≥ 0`;
  - `steps` must have `lo ≥ 1`.
- A bad string parameter (`object`). It must be a `str`. `"@role"` needs the role in `task.spec.roles`; any other string must be a key of `task.spec.objects`.
- A missing task feature that the type's trigger reads:
  - `grasp_offset` and `partial_close` need a `grasp` step, `early_close` an `approach` step, `early_release` a `transport` step and `place_offset` a `place` step;
  - `target_offset` needs a non-empty `allowed_stages`;
  - `object_drop` needs a `target` role;
  - `wrong_target` needs the `target` role bound to an object set (`task.spec.object_sets`) whose `[task.start].count` upper bound (default `count_max`) is ≥ 2. So `fx_pick`'s single plain cube is an error.

At draw time:

- `weights_override` (the §8.5 `perturbation_weights`) is validated:
  - every key must be a `PERTURBATION_TYPES` name, otherwise `ConfigError`;
  - every value must be finite, ≥ 0 and not a bool or str, otherwise `ConfigError`;
  - a known type that the task does not list is ignored, because one config serves several tasks.
- Drawing `FAILURE` or `RECOVERY` when that list is missing or empty is a `ConfigError` (§14.2: every task has both).
- The kind's weights must sum to more than 0 after the override, otherwise `ConfigError`.

### `draw_perturbation(task, kind, rng, weights_override=None, *, start=None) -> PerturbationSpec | None`

`start=` is an addition to the contract signature. It accepts anything with `objects` (name → `ObjectState`) and `roles`, such as a WP1 `StartSample` or an `EpisodeSpec`.

- `kind` may be a `Kind` or its value. `Kind.SUCCESS` returns `None` after validation and makes **no** draw (`rng` state unchanged).
- **Applicability (per start).** `wrong_target` needs at least 2 active members of the target role's set. Count them as the objects of `start.objects` that are active and whose `task.spec.objects[name].params["set"]` equals `task.spec.roles["target"]`. An entry that is not applicable gets weight 0, so it is never drawn. With `start=None` every entry is applicable. If no entry with positive weight is left, raise `SamplingError`; `build_spec` propagates it and the plan records the attempt as `sampling_failed`.
- **Exact draw order.** This is pinned so specs are reproducible across implementations:
  1. `w` = float array of entry weights (the override replaces the weight of every entry of that type; inapplicable entries are 0). Then `i = int(rng.choice(len(w), p=w / w.sum()))`. Always call it, even for a single entry.
  2. If the entry has `stages`: `onset_stage = str(rng.choice(stages))`. Otherwise `onset_stage = None` and there is no draw.
  3. For each declared parameter in declared order, using the entry's value or else the default:
     - a string is copied (no draw);
     - `delay_steps` / `steps` is drawn as `int(rng.integers(lo, hi + 1))`;
     - any other parameter is drawn as `float(rng.uniform(lo, hi))`;
     - immediately after `xy_m`, draw `dir_rad = float(rng.uniform(0, 2π))`. Only `xy_m` gets a direction; `object_drop`'s jitter direction is drawn by `WorldPerturber`.

     Default ranges are drawn too (`uniform(0, 0)` → 0.0).
- **Result.** `PerturbationSpec(type, group, onset_stage, delay_steps, params)`:
  - `delay_steps` is the drawn value, or 0 when the type has no `delay_steps` parameter;
  - `params` holds every other declared parameter plus `dir_rad`. Ints stay Python `int`, floats are Python `float`.

  Examples:
  - `grasp_offset` → `{xy_m, dir_rad, z_m, yaw_rad}`
  - `stall` → `{steps}`
  - `object_shift` → `{object, xy_m, dir_rad, yaw_rad}`
  - `wrong_target` → `{}`

### `WorldPerturber(spec, task_spec, rng)` (group B)

Arguments:
- `spec` is the `EpisodeSpec`.
- `task_spec` is the WP1 `TaskSpec`. Read `raw["stages"]["order"]`, `raw["start"].get("table_z_m", 0.0)` and `objects` (`ObjectSpec.size` / `params`).
- `rng` is the perturber stream.

Attributes:
- `applied: bool` (initially `False`);
- `onset_frame: int | None`.

`due(progress, t) -> bool`:
- Always `False` when `spec.perturbation` is `None` or group A, or once applied.
- **Latch.** The onset stage latches on the first call where `progress.furthest_index >= stages.index(onset_stage)`. This uses the ever-reached index, so a later regression of the current stage does not postpone the onset. Remember that frame as `t_latch`.
- `due` returns `True` for every `t ≥ t_latch + spec.perturbation.delay_steps` until `apply` has been called.
- **`object_drop`.** If `progress.satisfied.get("holding", False)` is false on the first frame where it would be due ("not applied if nothing is held then"), it is never due afterwards and stays not applied.

`apply(env, obs) -> Pose`:
- Compute the new pose, call `env.set_object_pose(name, pose)` exactly once, set `applied = True` and `onset_frame = t + 1`, where t is the frame `due` last returned `True` for (§11.1 step 5, D14). Return the pose.
- Calling it before `due` returned `True` is a `RuntimeError`.
- A target object that is missing or inactive in `obs.objects` is an `EnvError`.
- The object is `params["object"]`: `"@role"` resolves through `spec.roles`, any other string is a literal name. For `object_drop` the object is `spec.roles["target"]`.

New poses (current pose taken from `obs.objects[name].pose`, XYZW):
- **`object_shift` / `container_shift`.**
  - Position: `(x + xy_m·cos dir, y + xy_m·sin dir, z)`.
  - Orientation: `q_z(yaw_rad) ⊗ q`, a world-frame rotation about +z; `yaw_rad` is 0 for `container_shift`.
- **`object_drop`.** Draw one `d = rng.uniform(0, 2π)` in `apply`. Set `xy = current xy + xy_jitter_m·(cos d, sin d)`.
  - The support z starts at `table_z_m` and is the maximum over every other active object in `obs.objects` that is in `task_spec.objects`. Each object uses its yawed frame (yaw from its quaternion, `atan2(2(wz + xy), 1 − 2(y² + z²))`), with dims = `size` or `params.size_outer`.
  - A container (`params.inner_half_size`) whose inner rectangle contains the xy gives its floor, `z_c + params.floor_z`.
  - Otherwise, an object whose footprint `|local| ≤ dims[:2]/2` contains the xy gives its top: `z + h` for bottom origin (no `size` and `origin = "bottom"`), else `z + h/2`.
  - The dropped object's z is `support + h/2` (centre origin) or `support` (bottom origin).
  - Orientation: yaw only, with the current yaw kept and roll/pitch 0. This is the §11 rule, with "container floor beats container top when inside the inner rectangle".

### `ou_noise(noise, arm_mask, steps, rng) -> np.ndarray`

Decision: the second argument is a **boolean mask** of length J (true for the arm joints that get noise). It is not an integer `J_arm`, because the gripper is not necessarily the last joint (§1.2). An `int` raises `TypeError`.

- Returns a `float64`, C-contiguous array of shape `(steps, J)`. `n[0] = 0`. Columns where the mask is false stay 0.
- For `t = 0 … steps − 2`, the arm columns are `n[t+1] = clip(n[t] − θ·n[t] + σ·rng.standard_normal(k), −max_abs_rad, max_abs_rad)`. Call `standard_normal` once per step, with `k` = the number of arm columns.
- `steps = 0` gives shape `(0, J)`; `steps = 1` gives a row of zeros.

## labels.py (§7.2)

- `build_imitate(injected) -> tuple[bool, ...]`: `imitate[t] = not injected[t]`, as Python bools; accepts lists, tuples and numpy arrays.
  - Imitate is false only on group-A window frames. Recovery frames, OU-noise frames and group-B onset frames keep `True` (D14).
- `frame_labels(stage: int, injected: bool, ee: Pose) -> FrameLabels`: `FrameLabels(stage, not injected, [x, y, z, qw, qx, qy, qz])`. The ee pose (XYZW in memory) goes to disk order as `float64 (7,)`; there is no sign canonicalisation (§1.1, D17, §11.1 step 3).
- `label_columns(labels: FrameLabels) -> dict[str, np.ndarray]`: the per-frame dataset columns, keys in `LABEL_FEATURES` order (§9.3). The arrays are fresh and never alias `labels`. A test checks the dtypes and shapes against `core.datasets.task_meta.LABEL_FEATURES`.
  - `"label.stage"`: `int64 (1,)`
  - `"label.imitate"`: `int64 (1,)`, 0 or 1 (D16)
  - `"label.ee_pose"`: `float32 (7,)`
- `build_phases(phases) -> list[dict]`: the sidecar `phases` value (§9.1), the run-length list `[{"name", "start", "end"}]` over the per-frame phase names (ints, half-open, adjacent runs have different names). `[]` gives `[]`. A name that is not a non-empty `str` is a `ValueError`.
- `build_recovery(reentries, stage, stages) -> RecoveryInfo | None`. `reentries` is `ScriptedExpert.reentries`, a list of `(frame, resumed_stage)`; `stage` is the per-frame label. It returns `None` without re-entries. Otherwise:
  - `detected_frame` = the first re-entry frame;
  - `resumed_stage` = the **last** re-entry's stage;
  - `retries` = the number of re-entries;
  - `back_on_plan_frame` = the first `t ≥` the last re-entry frame with `stage[t] ≥ stages.index(last resumed_stage)`, or N = `len(stage)` if there is none (§7.2).
- `build_segments(num_frames, kind, *, onset, injected_end, detected, back_on_plan, phases) -> tuple[Segment, ...]`, as the §7.2 table. Decisions:
  - **Failure.** `p_end = injected_end`, or `onset + 1` when `injected_end is None`. This is the group-B rule: the loop passes `None` for B and the window end (N if still open) for A.
  - **Recovery.** `detected = None` → N; `back_on_plan = None` → N.
  - **No onset.** `onset is None` (perturbation not applied) gives a single nominal segment for any kind; the attempt is rejected anyway but `EpisodeResult` still carries segments.
  - **Clamping.** Each boundary is clamped to `[previous boundary, N]`, in order. Zero-length segments are dropped and adjacent segments are not merged. `phase = phases[start]`.
  - **Errors.** `num_frames < 1` or `len(phases) != num_frames` → `ValueError`; a negative boundary → `ValueError`; a non-int boundary (bools included) → `TypeError`. `kind` may be a value string.
- `accept(kind, *, success, applied, reentered, reentered_before_onset) -> RejectionReason | None`: exactly the §7.2 table, rows evaluated in order. The `env_error`, `invalid_state` and `duplicate_trajectory` rows are the loop's job (WP7). `kind` may be a value string. A recovery that does not recover is **rejected** (`recovery_failed`), not relabelled as a failure.

## specs.py (§7.3)

```python
def build_spec(task, kind, attempt_index, seed, *, arm, scene, home_q, reach_fn=None, noise=None,
               weights_override=None, fps=30, joint_names=None, joint_limits=None, arm_joints=None) -> EpisodeSpec
```

`joint_names`, `joint_limits` and `arm_joints` are additions. They are passed through to WP1's `sample_episode_start` (§15), which needs them for `initial_q` tables and joint goals.

1. `start = task.sample_episode_start(rng_for(seed, "start"), reach_fn=reach_fn, home_q=home_q, joint_limits=joint_limits, instruction_rng=rng_for(seed, "instruction"), joint_names=joint_names, arm_joints=arm_joints)`. `SamplingError` propagates.
2. `perturbation = draw_perturbation(task, kind, rng_for(seed, "perturbation"), weights_override, start=start)`.
3. Noise (success only, and only when `noise` is not `None`): if `rng_for(seed, "noise").random() < noise.fraction`, set `NoiseSpec(theta, sigma_rad, max_abs_rad)`. `noise` is duck-typed (WP5's `NoiseConfig`; do not import it). Other kinds make no noise draw and get `None`.
4. Build the `EpisodeSpec`:
   - `task=task.name`, `arm`, `scene`, `kind`, `attempt_index`, `seed`;
   - `objects`, `roles`, `goal`, `instruction` and `held_object` from the start;
   - `initial_q` = `start.initial_q`, or `tuple(home_q)` when that is `None`;
   - `max_steps` = `[task.expert].max_steps` (default 600);
   - `fps`.

## Contract additions to report back into contracts.md

1. `labels.py` gains `build_phases`, `build_recovery`, `frame_labels` and `label_columns`. The WP7 loop, the writers and the WP8 recorder writer all need these conversions, so they live in one place.
2. `draw_perturbation(..., *, start=None)` and the per-start applicability rule; `SamplingError` when nothing is applicable.
3. `ou_noise` takes an arm mask, not `J_arm`.
4. `build_spec` takes `joint_names`, `joint_limits` and `arm_joints`.
5. `PerturbationType` fields and `allowed_stages`, the parameter defaults table, the exact draw order, the stage rules ("grasp stage" = the stage of the first `grasp` step; "before placed" = before the first `release` step's stage; "implies holding" as defined above), and the requirement for `stages` on stage-triggered types.
6. `WorldPerturber` attributes `applied` / `onset_frame`; `apply` calls `set_object_pose` itself; "latched" means `furthest_index`; for `object_drop`, "held" means `progress.satisfied["holding"]` and the jitter direction comes from the perturber rng.
7. `build_segments`: `injected_end=None` means `onset + 1`, `None` boundaries mean N, and boundaries are clamped.

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp6_perturbation_types.py   # taxonomy table, allowed stages, validator (ConfigError)
tests/core/test_v4_wp6_draw.py                 # kinds, ranges, determinism, coverage, exact draw order, overrides, start applicability
tests/core/test_v4_wp6_world.py                # WorldPerturber onset rule and poses, ou_noise
tests/core/test_v4_wp6_labels.py               # imitate, frame labels, label columns, phases, recovery frames, segments, accept table
tests/core/test_v4_wp6_sidecar.py              # label builders + Task.outcome -> WP4 validate_sidecar; observed-outcome rejections
tests/core/test_v4_wp6_specs.py                # build_spec streams, invariants, noise decision
tests/core/test_v4_wp6_surface.py              # public names, imports, 3.11 syntax, no SO-101 literals
```

The tests build tasks from the §4.6 example tables (`v4_support.tasks.contract_color_task_table` / `contract_reach_task_table`) and from the WP1 fixture `tests/fixtures/v4_wp1/tasks/fx_pick/task.toml` inside a `sandbox`. They need WP0, WP1 and WP4 (`core.datasets.task_meta.validate_sidecar`, `LABEL_FEATURES`) in place.

Run from the repo root (Python 3.11, host group):

```
uv run --group host pytest tests/core/test_v4_wp6_*.py -q
# without uv:
PYTHONPATH=src python -m pytest tests/core/test_v4_wp6_*.py -q
```

The whole set runs in about 4 s.

## What must keep working

- All WP0, WP1, WP4 and WP5 tests (`tests/core/test_v4_wp{0,1,4,5}_*.py`, `tests/sim/test_v4_wp4_*.py`) and `tests/core/test_v4_support_selftest.py` (apart from its known WP1 conflict).
- The full suite must be no worse than before: `PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`. Other packages' tests that fail only because their modules do not exist yet are not your concern.

## Rules

- Do not edit, skip, xfail or weaken any test, and do not change `tests/v4_support/`, `tests/conftest.py` or the fixtures.
- Create only the three files listed under "Files".
- If a test appears to contradict `docs/v4/contracts.md` or this prompt, or two tests cannot both pass, **stop and report** the test name, the clause and your reasoning instead of working around it.
