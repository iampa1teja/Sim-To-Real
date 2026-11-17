<!-- Interface contracts for v4. Precedence: docs/platform_v4.md > this file > docs/v4/design.md. Tests are written from this file. -->

# v4 interface contracts

Every name, signature, field, unit, default and error type that crosses a work-package (WP) boundary is fixed here. A WP may add private helpers (leading underscore) but no other public names. Where this file and `docs/v4/design.md` differ, this file wins; deviations are listed in section 14.

## 1. Conventions

### 1.1 Units, frames, quaternions

| Quantity | Unit | Notes |
|---|---|---|
| length, position | metres | TOML keys carry `_m` |
| angle | radians | degrees only in TOML keys ending `_deg` |
| time | frames (int) at `fps` (default 30) | durations in TOML carry `_steps` |
| joint values in datasets (`action`, `observation.state`) | raw LeRobot units: arm joints −100…100, gripper 0…100 | float32 on disk |
| colour | linear RGB, each 0…1 | |

- **Frame:** every pose that crosses an interface is in the robot base-link frame (`[arm.links].base`). The table is the plane `z = [task.start].table_z_m` in that frame. Sidecar `reference_frame` = `f"robot base link ({base})"` (equal to today's `SIDECAR_FRAME` for SO-101).
- **Quaternions:** XYZW in memory (`core.tasks.base.Pose.quat_xyzw`, Isaac Lab 3.0, `ee_frame_state`). WXYZ on disk (sidecar, kinematics.json, TOML, parquet, dataset columns). On-disk keys always end `_wxyz`. A 7-vector is `[x, y, z, qx, qy, qz, qw]` in memory and `[x, y, z, qw, qx, qy, qz]` on disk. Fingerprints canonicalise sign so that `qw ≥ 0`.
- **Object position** is the object's root origin: the geometric centre for `size` objects (cuboids), and per `params.origin` (`"center"` default, or `"bottom"`) for USD objects. `white_box` and the new boxes and bowls use `origin = "bottom"`.
- **ee vs TCP:** `ee` is the pose of `[arm.links].ee` (what `ee_frame_state` reports, what is stored as `ee_pose`). `tcp` = `ee ∘ [arm.kinematics].tcp_*`, the point between the jaw tips. Predicates and skills use the TCP.

### 1.2 Joint order

- The **joint vector** `q` (length J) is in `[arm.joints].names` order (USD names). Index `i` pairs with `[arm.lerobot].names[i]` (`joint_name_pairs`), so raw dataset vectors use the same index order. The gripper is the entry named `[arm.gripper].joint`.
- The native Isaac articulation order never crosses an interface. `IsaacGenEnv` maps it with `find_joints(names, preserve_order=True)`.
- Kinematic chain joints are a subset of `q` (`[arm.expert].arm_joints`). Chain APIs take and return full-length `q`.
- New code reads joint names from `load_arm_spec(arm)` at call time. It never relies on the import-time constants `FOLLOWER_JOINTS`, `ARM_NAME`, `TASK_NAME` or `SETUP_PATH`, which are frozen at first import.

### 1.3 dtypes

- CPU computation and every protocol array: `np.float64`, C-contiguous, shape as documented. Arrays inside frozen dataclasses are set read-only (`flags.writeable = False`).
- Dataset writes: `action` and `observation.state` float32 `(J,)`; label columns as in §9.3; images uint8 `(H, W, 3)`.
- Indices and counts: Python `int`. Seeds are `int` in `[0, 2**64)`, stored as JSON integers and as arrow `uint64`.

### 1.4 RNG

- Only `np.random.Generator` objects created by `np.random.default_rng(seed: int)`. No global `np.random`, and no `random` module in new code (existing v1 `dr_variants` keeps `random.Random`).
- `core.generate.seeds.derive_seed(*parts) -> int` = `int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big")`. This is exactly `dr_variants.episode_seed(seed, src_index, color) == derive_seed(seed, src_index, color)`. Golden values: `derive_seed(0, "prim_reach_point", "success", 0) == 11062146242513595739` and `derive_seed(1984, 3, "red") == 7801958555027104258`.
- Attempt seed: `derive_seed(master_seed, task, kind.value, attempt_index)`.
- Sub-seeds: `derive_seed(attempt_seed, purpose)` with `purpose ∈ {"start", "instruction", "perturbation", "noise", "expert"}`. Each purpose has its own Generator, so adding draws for one purpose never changes another.
- DR sub-seeds are defined in §8.6.

### 1.5 Errors (`src/core/generate/errors.py`, WP0)

| Class | Bases | Raised by |
|---|---|---|
| `GenerationError` | `Exception` | base of all below |
| `ConfigError` | `GenerationError, ValueError` | invalid task.toml / arm.toml tables, expert program, YAML config, unknown predicate or skill, bad `@` reference (raised at load time, not mid-episode) |
| `SchemaError` | `GenerationError, ValueError` | sidecar, manifest, plan, kinematics.json or `from_json` validation |
| `KinematicsError` | `GenerationError` | chain construction (cycle, missing joint, bad axis). IK never raises for an unreachable target. |
| `SamplingError` | `GenerationError` | start sampler exceeded `max_tries` (the loop turns it into skip `sampling_failed`) |
| `ScenarioSpaceExhausted` | `GenerationError` | plan cannot supply its quota |
| `AttemptBudgetExceeded` | `GenerationError` | executed attempts > `max_attempt_factor × quota` |
| `ResumeError` | `GenerationError` | reconcile finds an inconsistency it may not repair |
| `DuplicateError` | `GenerationError` | merge collision |
| `AuditFailed` | `GenerationError` | merge or upload without a passing, current audit |
| `EnvError` | `GenerationError` | a GenEnv cannot honour a spec (unknown object, bad q length) |

Plain `ValueError` / `TypeError` are used for wrong argument shapes or types in pure functions.

### 1.6 Imports, Python, packages

- Python ≥ 3.11 (the Isaac image runs 3.11; the host may be newer). No 3.12+ syntax or stdlib APIs.
- `src/core/**` never imports `isaaclab`, `isaacsim`, `omni`, `carb`, `pxr`, `warp` or `lerobot` at module level. Container-only functions in `src/core/datasets/{merge,streaming,upload}.py` may import `lerobot`, `huggingface_hub` or `fsspec` lazily inside a function.
- `torch` is not imported by any new `src/core` module. It stays only in the existing `core/arms/mapping.py` and in `src/sim`.
- Allowed third-party imports in new `src/core` modules: `numpy`, `scipy`, `pyarrow` (only `core/datasets/*`), `yaml` (only `core/generate/config.py`). The host group does not list PyYAML directly; it arrives through `draccus` (section 14).
- `src/sim` modules that CPU tests import (`scripted_generate`, `extract_kinematics`, `dr_variants`) keep Isaac imports inside functions, following the `dr_replay.py` pattern.

## 2. Module map

"May import" lists the WPs (besides existing code) whose modules a package may import. Anything else needed across a boundary comes from WP0.

| WP | Wave | Modules (new unless marked) and public names | May import |
|---|---|---|---|
| WP0 | 0 | `core/generate/__init__.py` (empty); `core/generate/protocols.py`: `Kind`, `SegmentLabel`, `AttemptStatus`, `SkipReason`, `RejectionReason`, `PerturbationGroup`, `ObjectState`, `PerturbationSpec`, `NoiseSpec`, `EpisodeSpec`, `StageProgress`, `ExpertObs`, `ExpertAction`, `FrameLabels`, `StepRecord`, `Segment`, `RecoveryInfo`, `Fingerprints`, `EpisodeResult`, `GenEnv`, `EpisodeWriter`, `Expert`; `core/generate/errors.py` (§1.5); `core/generate/seeds.py`: `derive_seed`, `rng_for`; `tests/conftest.py`, `tests/v4_support/__init__.py` | existing `core.tasks.base.Pose` |
| WP1 | 1 | `core/tasks/base.py` (change): `TaskSpec` fields, `ObjectSetSpec`, `HoldoutSpec`, `FailureRule`, `CheckContext`, `StageTracker`, `Outcome`, new `Task` methods; `core/tasks/generic.py`: `GenericTask`, `PREDICATES`, `register_predicate`, `resolve_ref`; `core/tasks/sampling.py`: `StartSample`, `sample_start`, `park_positions`, `render_instruction` | WP0 |
| WP2 | 1 | `core/arms/kinematics.py`: `JointDef`, `KinematicChain`, `IKWeights`, `IKResult`, `load_kinematics_json`; `core/arms/gripper.py`: `Gripper`; `core/arms/raw_mapping.py`: `RawMapping`; `core/arms/mapping.py` (change): `JointMapping.to_raw_np`, `.clamp_radians_np`, `.as_raw_mapping`; `core/arms/joint_limits.py` (change); `arms/so101/arm.toml` (add tables); `sim/generate/extract_kinematics.py`: `build_kinematics`, `main` | WP0 |
| WP4 | 1 | `core/datasets/__init__.py`; `core/datasets/task_meta.py`: `SIDECAR_V2_DIR`, `LABEL_FEATURES`, `sidecar_path`, `write_sidecar`, `read_sidecar`, `load_episode`, `v1_to_v2`, `validate_sidecar`, `list_sidecars`; `sim/recording/episode_metadata.py` (change: wrappers); `sim/recording/replay_source.py` (change) | WP0 |
| WP5 | 1 | `core/generate/plan.py`: `KINDS`, `attempt_seed`, `shard_quota`, `owns`, `PlannedAttempt`, `iter_plan`, `shard_attempts`, `check_capacity`, `CapacityReport`; `core/generate/fingerprint.py`: `quantize`, `scenario_key`, `spec_fp`, `traj_fp`, `appearance_fp`, `episode_fp`; `core/generate/manifest.py`: `ManifestLine`, `Manifest`, `reconcile`, `ReconcileResult`, `write_plan_json`, `read_plan_json`; `core/generate/config.py`: `Quotas`, `NoiseConfig`, `DRConfig`, `GenerateConfig`, `load_generate_config`; `core/generate/appearance.py`: `AppearanceDraw`, `draw_appearance`, `dr_copy_sidecar` | WP0 |
| WP3 | 2 | `core/expert/__init__.py`; `motion.py`: `MotionLimits`, `step_toward_joint`, `cartesian_waypoint`; `skills.py`: `SKILLS`, `Skill`; `program.py`: `ProgramStep`, `ExpertProgram`, `parse_program`; `machine.py`: `ScriptedExpert` | WP0, WP1, WP2 |
| WP6 | 2 | `core/generate/perturbations.py`: `PERTURBATION_TYPES`, `PerturbationType`, `draw_perturbation`, `WorldPerturber`, `ou_noise`; `labels.py`: `build_segments`, `build_imitate`, `accept`; `specs.py`: `build_spec` | WP0, WP1 |
| WP12 | 2 | `core/datasets/consolidate.py`: `consolidate`, `read_consolidated`; `merge.py`: `ShardSummary`, `MergePlan`, `plan_merge`, `apply_merge_meta`, `merge_frames` (container); `upload.py`: `StorageClient`, `HubStorage`, `BucketStorage`, `UploadItem`, `UploadPlan`, `plan_upload`, `run_upload`; `streaming.py`: `select_episodes`, `make_streaming_dataset` (container); `core/scripts/merge_shards.py`, `core/scripts/upload_dataset.py` (`main`) | WP0, WP4, WP5 |
| WP7 | 3 | `core/generate/fake_env.py`: `FakeEnv`, `SimulatedCrash`; `core/datasets/frames.py`: `LiteDatasetWriter`, `read_frames`, `MemoryWriter`; `core/generate/loop.py`: `run_episode`, `run_generation`, `GenerationSummary`, `make_expert_factory`; `core/datasets/audit.py`: `AuditReport`, `audit_dataset`, `write_audit_report`, `read_audit_report`, `content_digest`; `core/scripts/audit_dataset.py`, `core/scripts/plan_generation.py` (`main`); `tests/v4_support/harness.py` | WP0–WP6, WP12 |
| WP8 | 4 | `sim/envs/generic_task_cfg.py`: `apply_teleop`, `apply_generate`; `sim/envs/mdp/objects.py`: `set_object_pose_base`, `set_object_color`, `object_poses_in_base`, `park_all`; `sim/generate/isaac_gen_env.py`: `IsaacGenEnv`, `RecorderEpisodeWriter`; `sim/generate/scripted_generate.py`: `build_parser`, `main`; `sim/recording/lerobot_recorder.py` (change); `cli/paths.py` rows; `pyproject.toml` scripts; `configs/generate/smoke.yaml`; SO-101 literals removed from `sim/scene/arm.py`, `sim/scene/contact_materials.py`, `sim/eval/grasp_diagnostics.py`, `tasks/pick_place_cube/env_cfg.py` | all core WPs |
| WP9 | 4 | `tasks/prim_*/` (13 folders), `configs/generate/primitives.yaml` | WP1, WP8 (`env_cfg.py` only) |
| WP10 | 5 | `tasks/pick_place_cube/` (change), `tasks/stack_two_cubes/`, `tasks/pick_place_color_cube/` (+ `usd/Small-Box-Physics.usda`), configs | WP1, WP8 |
| WP11 | 5 | `sim/generate/dr_replay.py`, `dr_variants.py`, `sim/recording/replay_scene.py` (change) | WP4, WP5, WP8 |
| WP13 | 5 | `tasks/stack_bowls/` (+ procedural bowl USDA), `configs/generate/stack_bowls.yaml` | WP1, WP8 |

## 3. WP0 types (`src/core/generate/protocols.py`)

All dataclasses are `@dataclass(frozen=True)`. Mapping-valued fields are wrapped in `types.MappingProxyType` and sequence fields are converted to tuples in `__post_init__`. Instances are therefore comparable with `==` but not hashable (`spec_fp` is the identity).

### 3.1 Enums (`class X(str, Enum)`; the value is the on-disk string)

| Enum | Values |
|---|---|
| `Kind` | `SUCCESS="success"`, `FAILURE="failure"`, `RECOVERY="recovery"` |
| `SegmentLabel` | `NOMINAL="nominal"`, `PERTURBATION="perturbation"`, `FAILURE="failure"`, `RECOVERY="recovery"` |
| `AttemptStatus` | `ACCEPTED="accepted"`, `REJECTED="rejected"`, `SKIPPED="skipped"` |
| `SkipReason` | `SCENARIO_SEEN="scenario_seen"`, `SAMPLING_FAILED="sampling_failed"` |
| `RejectionReason` | `EXPERT_FAILED`, `UNPLANNED_REGRESSION`, `PERTURBATION_NOT_APPLIED`, `PERTURBATION_INEFFECTIVE`, `NO_REGRESSION`, `RECOVERY_FAILED`, `DUPLICATE_TRAJECTORY`, `INVALID_STATE`, `ENV_ERROR` (each value is the lower-case name) |
| `PerturbationGroup` | `EXPERT="expert"` (A), `WORLD="world"` (B) |

### 3.2 Dataclasses

| Class | Fields (type, default) |
|---|---|
| `ObjectState` | `pose: Pose`; `active: bool = True`; `color: str \| None = None` (palette name) |
| `PerturbationSpec` | `type: str`; `group: PerturbationGroup`; `onset_stage: str \| None`; `delay_steps: int = 0`; `params: Mapping[str, float \| int \| str \| tuple] = {}` |
| `NoiseSpec` | `theta: float`; `sigma_rad: float`; `max_abs_rad: float` (OU on arm joints only) |
| `EpisodeSpec` | `task: str`; `arm: str`; `scene: str`; `kind: Kind`; `attempt_index: int`; `seed: int`; `objects: Mapping[str, ObjectState]` (every object of the expanded `TaskSpec.objects`, inactive ones with `active=False` and their park pose); `roles: Mapping[str, str]` (role → object name); `goal: Mapping[str, Any]` (JSON values, §4.6); `instruction: str`; `initial_q: tuple[float, ...]` (J, rad); `held_object: str \| None = None`; `perturbation: PerturbationSpec \| None = None`; `noise: NoiseSpec \| None = None`; `max_steps: int = 600`; `fps: int = 30`. Property `expert_mode -> Literal["closed_loop", "blind"]` is `"blind"` iff `kind is Kind.FAILURE`. Invariants (`ValueError`): success ⇔ `perturbation is None`; `noise` only when `kind is Kind.SUCCESS`; each role value is an active object. |
| `StageProgress` | `frame: int`; `stage_index: int` (−1 = none); `furthest_index: int`; `satisfied: Mapping[str, bool]` (instantaneous values of all checks); `regressed_stage: str \| None` (set on the frame a regression fires) |
| `ExpertObs` | `frame: int`; `q: np.ndarray (J,)` measured rad; `ee: Pose`; `tcp: Pose`; `gripper_gap_m: float`; `objects: Mapping[str, ObjectState]`; `signals: Mapping[str, bool] = {}`. Method `step_state(joint_names: Sequence[str] | None = None) -> StepState` = `StepState(objects={n: s.pose for active n}, ee=ee, gripper=gripper_gap_m, signals=signals, tcp=tcp)`; when `joint_names` is given (length must equal `len(q)`, else `ValueError`) it also sets `joints=dict(zip(joint_names, q))`, otherwise `joints` stays `None` |
| `ExpertAction` | `q_target: np.ndarray (J,)` rad, clamped to limits; `phase: str`; `done: bool`; `injected: bool` (frame inside a group-A error window); `reentered: bool`; `resumed_stage: str \| None` |
| `FrameLabels` | `stage: int`; `imitate: bool`; `ee_pose_wxyz: np.ndarray (7,)` |
| `StepRecord` | `frame: int`; `state_raw: np.ndarray (J,)`; `action_raw: np.ndarray (J,)`; `labels: FrameLabels`; `phase: str`; `injected: bool`; `reentered: bool`; `objects: Mapping[str, Pose]` (active objects) |
| `Segment` | `label: SegmentLabel`; `start: int`; `end: int` (half-open); `phase: str \| None = None` (phase active at `start`) |
| `RecoveryInfo` | `detected_frame: int`; `resumed_stage: str`; `retries: int`; `back_on_plan_frame: int` |
| `Fingerprints` | `scenario_key: str`; `spec_fp: str`; `traj_fp: str \| None`; `appearance_fp: str \| None`; `episode_fp: str \| None` (lower-case hex SHA-256) |
| `EpisodeResult` | `task: str`; `kind: Kind`; `attempt_index: int`; `seed: int`; `status: AttemptStatus`; `reason: SkipReason \| RejectionReason \| None`; `episode_index: int \| None`; `num_frames: int`; `success: bool \| None`; `failure_mode: str \| None`; `furthest_stage: str \| None`; `stage: tuple[int, ...]`; `imitate: tuple[bool, ...]`; `segments: tuple[Segment, ...]`; `perturbation_onset_frame: int \| None`; `perturbation_applied: bool`; `recovery: RecoveryInfo \| None`; `fingerprints: Fingerprints` |

### 3.3 Protocols (`typing.Protocol`, `runtime_checkable`)

```python
class GenEnv(Protocol):
    fps: int
    joint_names: tuple[str, ...]          # [arm.joints].names
    camera_names: tuple[str, ...]         # () allowed
    def reset(self, spec: EpisodeSpec) -> ExpertObs: ...      # frame 0; poses, colours, initial_q, held_object applied
    def step(self, q_target: np.ndarray) -> ExpertObs: ...    # one control step (decimation inside)
    def observe(self) -> ExpertObs: ...                       # no stepping; frame unchanged
    def set_object_pose(self, name: str, pose: Pose) -> None: ...  # teleport, zero velocity; detaches if held
    def render(self) -> dict[str, np.ndarray]: ...            # {camera: uint8 (H, W, 3)} for camera_names
    def close(self) -> None: ...

class EpisodeWriter(Protocol):
    root: Path
    def episode_count(self) -> int: ...                       # committed episodes
    def episode_lengths(self) -> list[int]: ...
    def add_frame(self, action_raw: np.ndarray, state_raw: np.ndarray, images: Mapping[str, np.ndarray],
                  task: str, labels: FrameLabels) -> None: ...
    def discard_episode(self) -> None: ...                    # drops the open buffer; no-op if empty
    def save_episode(self) -> int: ...                        # commits; returns index == episode_count() before the call
    def finalize(self) -> None: ...

class Expert(Protocol):
    def reset(self, spec: EpisodeSpec, obs: ExpertObs) -> None: ...
    def act(self, obs: ExpertObs, progress: StageProgress) -> ExpertAction: ...
    @property
    def phase(self) -> str: ...
    @property
    def done(self) -> bool: ...
```

Call order: `GenEnv.reset` → `Expert.reset` → per frame `act` → `add_frame` → `step` (→ optional `set_object_pose` + `observe`). At the end, either `save_episode` or `discard_episode`. Calling `act` before `reset` raises `RuntimeError`.

### 3.4 JSON round-trip

Each dataclass has `to_json() -> dict` and `@classmethod from_json(d) -> Self`. Enums become their value, tuples become lists, arrays become nested lists of float, `Pose` becomes `{"position": [3], "quat_xyzw": [4]}` (in-memory types keep XYZW), and mappings become dicts. `EpisodeSpec.to_json()` includes `"schema": "episode_spec/1"` and `EpisodeResult.to_json()` includes `"schema": "episode_result/1"`. `from_json` raises `SchemaError` on unknown or missing keys, a wrong schema tag, or non-finite floats. The required property is `X.from_json(json.loads(canonical_json(x.to_json()))) == x`.

## 4. Task contract v2

### 4.1 `TaskSpec` additions (all defaulted; existing positional construction is unchanged)

| Field | Type | Default | Source |
|---|---|---|---|
| `object_sets` | `dict[str, ObjectSetSpec]` | `{}` | `[task.object_sets.*]` |
| `palette` | `dict[str, tuple[float, float, float]]` | `{}` | `[task.palette]`, TOML order kept |
| `roles` | `dict[str, str]` | `{}` | `[task.roles]` (role → object or set name) |
| `holdout` | `HoldoutSpec` | empty | `[task.holdout]` |
| `instruction_templates` | `tuple[str, ...]` | `()` | `[task.instruction_template]` |

- `ObjectSetSpec(name, prefix, count_max, template: ObjectSpec)`.
- `slot_names(self) -> tuple[str, ...]` returns `f"{prefix}_{i}"` for i in `0 … count_max−1`.
- `load_task_spec` adds every slot to `spec.objects` as an `ObjectSpec` with `params["set"] = name` and `params["slot"] = i`, in set order and after the plain objects.
- Name clashes raise `ConfigError`.
- `HoldoutSpec(colors: tuple[str, ...] = (), counts: tuple[int, ...] = ())`.
- A palette value may be `"from_setup:<object_materials key>"`; it resolves through `load_setup()` when read.

### 4.2 `Task` additions (`core/tasks/base.py`)

- `StepState` gains `tcp: Pose | None = None`.
- `stages` and `observe` stay abstract. `sample_start` becomes concrete: it returns the active-object poses of `self.sample_episode_start(rng)`. `PickPlaceTask` still loads unchanged.

```python
@dataclass(frozen=True)
class CheckContext:
    roles: Mapping[str, str]; start: Mapping[str, Pose]; goal: Mapping[str, Any]
    objects: Mapping[str, ObjectSpec]; active: frozenset[str]
    @classmethod
    def from_spec(cls, spec: EpisodeSpec, task_spec: TaskSpec) -> "CheckContext"   # start = spec.objects poses

@dataclass(frozen=True)
class FailureRule:
    name: str; if_: tuple[str, ...] = (); unless: tuple[str, ...] = ()

@dataclass(frozen=True)
class Outcome:
    success: bool; failure_mode: str | None; furthest_stage: str | None

class Task:
    def observe_checks(self, state: StepState, ctx: CheckContext) -> dict[str, bool]   # default: self.observe(state)
    def make_tracker(self) -> StageTracker              # StageTracker(self.stages, self.hold_until)
    @property
    def hold_until(self) -> dict[str, str]              # [task.stages].hold_until, default {}
    @property
    def failure_rules(self) -> tuple[FailureRule, ...]  # [task.failures].rules, default ()
    @property
    def failure_names(self) -> tuple[str, ...]
    def failure_mode(self, flags: Mapping[str, bool], holding: bool, success: bool) -> str | None
    def outcome(self, tracker: StageTracker, final_checks: Mapping[str, bool], final_state: StepState) -> Outcome
    def sample_episode_start(self, rng: np.random.Generator, *, reach_fn=None, home_q=None) -> StartSample
```

**`failure_mode` default.**
1. If `success`, return `None`.
2. Otherwise evaluate `failure_rules` in order. A rule matches iff every name in `if_` is true and every name in `unless` is false. The pseudo-flag `"holding"` takes the `holding` argument.
3. Return the first match's `name`. With no rules, return `None`.

The keyword in TOML is `if`, stored as `if_`. The last rule must have empty `if` and `unless`; otherwise `ConfigError`. `PickPlaceTask` keeps the same 3-argument signature.

**`outcome`.**
- `flags` = `tracker.flags()` (ever-latched stage flags) plus the final-frame values of non-stage checks.
- `holding` = `final_checks.get("holding", final_state.signals.get("holding", False))`.
- `success` = `flags[stages[-1]]` and, if `[task.stages].success_requires_final` (default `true`), also `final_checks[stages[-1]]`.
- `furthest_stage` = `tracker.furthest_stage`.

### 4.3 StageTracker

```python
class StageTracker:
    def __init__(self, stages: Sequence[str], hold_until: Mapping[str, str] = {}): ...
    def update(self, checks: Mapping[str, bool]) -> StageProgress     # exactly once per frame, frames 0..N-1
    stage_index: int; furthest_index: int; regressions: list[tuple[int, str]]
    def flags(self) -> dict[str, bool]          # ever-latched (StageTrace-compatible)
    @property
    def furthest_stage(self) -> str | None
```

Per frame t, `update` does the following:
1. For every stage s, set `ever[s] |= checks[s]` and `current[s] |= checks[s]`.
2. **Regression.** Find the first stage s (in order) with `hold_until[s] = h` such that `current[s]` is true, `checks[s]` is false and `current[h]` is false. If one exists, clear `current` for s and every later stage, append `(t, s)` to `regressions`, and set `regressed_stage = s`.
3. `stage_index` = the largest i with `current[stages[0..i]]` all true, else −1. `furthest_index` is the same over `ever`, so it is monotone non-decreasing.

The per-frame label `stage[t]` is `stage_index` after the update at frame t. Missing keys in `checks` count as false. `StageTrace.finish` (benchmark) becomes `TASK.failure_mode(stage_flags, holding, success)` in WP10.

### 4.4 GenericTask and predicates (`core/tasks/generic.py`)

- `GenericTask(spec: TaskSpec)` builds from `[task.stages]`, `[task.checks.*]` and `[task.failures]`. Validation (`ConfigError`):
  - every stage has a check;
  - every check names a known predicate;
  - references resolve against roles, objects and goal keys;
  - `all`/`any` dependencies have no cycles.
- `observe_checks` evaluates checks in dependency order; `observe(state)` raises `TypeError` (a context is required).
- `PREDICATES: dict[str, Callable[[StepState, CheckContext, Mapping[str, Any]], bool]]`. `register_predicate(name)` is a decorator.
- Every check accepts `negate: bool = false`.
- `p(o)` is the object position, `tcp` is `state.tcp or state.ee`, and `h(o)` / `w(o)` are the object height and the minimum horizontal size (from `size` or `params.size_outer`).
- `resolve_ref(ref, ctx)` syntax:
  - `"@role"` → object name;
  - `"@role.inner"` → the container's inner rectangle;
  - `"@goal.key"` or `"@goal.key.i"` → a goal value;
  - a literal object name;
  - a literal list.

| Predicate | Parameters (default) | True iff |
|---|---|---|
| `ee_near` | `object` or `point`; `offset_m=[0,0,0]`; `tol_m=0.02` | `‖tcp − (target + offset)‖ ≤ tol_m` |
| `object_lifted` | `object`; `min_lift_m=0.01` | `p(o).z − start(o).z ≥ min_lift_m` |
| `object_in_region` | `object`; `region` (rect `{xy_min, xy_max}`, polar `{r_m, bearing_deg}`, or `"@c.inner"`); `margin_m=0`; `check_z=true`; `z_tol_m=0.01` | xy inside the region shrunk by `margin_m` (inner rect in the container's yaw frame, half extents `params.inner_half_size`). If `check_z` and the region is a container: `p(o).z − h(o)/2 − (p(c).z + params.floor_z) ≤ z_tol_m`. |
| `objects_in_region` | `set`; `region`; `exclude=[]`; `min_count=1`; `margin_m=0`; `check_z=true`; `z_tol_m=0.01` | at least `min_count` active members of `set`, not in `exclude`, satisfy `object_in_region` |
| `object_on_object` | `object`; `base`; `xy_tol_m=0.01`; `z_tol_m=0.005` | `‖xy(o) − xy(b)‖ ≤ xy_tol_m` and `|z(o) − (z(b) + h(b)/2 + h(o)/2)| ≤ z_tol_m` (centre origins; bottom origins drop the respective `/2`) |
| `nested_in` | `object`; `base`; `xy_tol_m=0.01`; `max_rise_m` (required) | `‖xy(o) − xy(b)‖ ≤ xy_tol_m` and `0 ≤ z(o) − z(b) ≤ max_rise_m` |
| `held` | `object`; `max_dist_m=0.03`; `gap_low_m=0.006`; `gap_high_m=0.004` | `w(o) − gap_low_m ≤ state.gripper ≤ w(o) + gap_high_m` and `‖tcp − p(o)‖ ≤ max_dist_m` |
| `joint_near` | `joint` (USD name or `"@goal.joint"`); `value_rad`; `tol_rad=0.02` | `|state.joints[joint] − value| ≤ tol_rad` (`StepState.joints`, see note below) |
| `gap_near` | `value_m`; `tol_m=0.003` | `|state.gripper − value_m| ≤ tol_m` |
| `object_displaced` | `object`; `min_m`; `max_m=inf`; `axes="xy"` | `min_m ≤ ‖Δp(o)[axes]‖ ≤ max_m`, Δ measured from `start` |
| `signal` | `name` | `state.signals.get(name, False)` |
| `all` / `any` | `of: list[str]` (check names) | conjunction / disjunction |

Note for `joint_near`: `StepState` also gains `joints: Mapping[str, float] | None = None` (USD name → rad), which `ExpertObs.step_state(joint_names=...)` fills when the caller passes the arm's USD joint names (the generation loop does; see §3.2).

### 4.5 Start sampler (`core/tasks/sampling.py`)

```python
@dataclass(frozen=True)
class StartSample:
    objects: Mapping[str, ObjectState]; roles: Mapping[str, str]; goal: Mapping[str, Any]
    instruction: str; initial_q: tuple[float, ...] | None; held_object: str | None

def sample_start(task_spec: TaskSpec, rng: np.random.Generator, *, reach_fn: Callable[[Pose], bool] | None = None,
                 home_q: Sequence[float] | None = None, joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
                 instruction_rng: np.random.Generator | None = None) -> StartSample
def park_positions(names: Sequence[str]) -> dict[str, tuple[float, float, float]]   # name_i -> (0.1*i, 0.0, -2.0)
def render_instruction(template: str, roles: Mapping[str, str], colors: Mapping[str, str], goal: Mapping) -> str
```

**Draw order** (one Generator, stream `"start"`):
1. **Count.** `allowed = [c for c in range(lo, hi + 1) if c not in holdout.counts]`; `n = allowed[rng.integers(len(allowed))]`.
2. **Colours.** `pal = [c for c in palette if c not in holdout.colors]`. The target colour is `pal[rng.integers(len(pal))]`. Distractors are `rng.choice(pal minus target, n − 1, replace=not distinct_colors)`.
3. **Slots.** `slots = rng.permutation(count_max)[:n]`. `slots[0]` is the target and gets the target colour; the rest get the distractor colours in order.
4. **Placement.** Place the objects in `[task.start.place]` key order (plain objects first, then set members in slot order). Each object uses rejection sampling up to `max_tries`. Constraints: inside its region, xy centre distance ≥ `min_separation_m` from every placed movable object, footprint clearance ≥ `clearance_m` from every placed container, and `reach_fn(pregrasp)` true for objects listed in `reachable`. `pregrasp` = object centre + `[0, 0, 0.06]` with the TCP pointing down.
5. **Goal entries** in TOML order (§4.6).

Rules:
- Exhausting `max_tries` raises `SamplingError`. Resting z = `table_z_m + h/2` (centre origin) or `table_z_m` (bottom origin). Orientation is yaw only.
- Inactive slots get `park_positions` and `active=False`.
- `held_object` objects are placed by the env at the TCP; their sampled pose is ignored.
- The instruction is `render_instruction(templates[k], …)`, with k drawn from `instruction_rng`. Placeholders are `{<role>.color}`, `{<role>.name}` and `{goal.<key>}`; any other placeholder is a `ConfigError` at load time.
- Validation at load (`ConfigError`): `len(pal) ≥ count_max` when `distinct_colors`; at least one allowed count; holdout names exist in the palette and the count range.

### 4.6 task.toml tables (all under `[task]`)

| Table / key | Type | Meaning |
|---|---|---|
| `object_sets.<set>` | `prefix: str = <set>`, `count_max: int ≥ 1`, plus any `ObjectSpec` keys (`size`, `usd`, `mass`, `rigid`, …) | expands to slots |
| `palette` | `<name> = [r, g, b]` or `"from_setup:<key>"` | colour names |
| `roles` | `<role> = "<object or set>"` | a set role binds the sampled target slot |
| `holdout` | `colors: [str]`, `counts: [int]` | never drawn by generation or DR |
| `instruction_template` | `texts: [str]` (≥ 1) | one is chosen per episode |
| `start` | `table_z_m: float = 0.0`; `max_tries: int = 200`; `count: [int, int]` (set of the target role); `distinct_colors: bool = true`; `min_separation_m: float = 0.035`; `clearance_m: float = 0.01`; `reachable: [ref] = ["@target"]` (if a target role exists); `initial_q: "home" \| {joint = rad} = "home"`; `held: ref \| absent` | |
| `start.place.<object\|set>` | `region` (rect `{xy_min: [2], xy_max: [2]}` or polar `{r_m: [2], bearing_deg: [2]}`); `yaw_deg: [2] = [0, 0]` | where an object or set starts |
| `start.goal.<key>` | `type`, one of: `point {region, z_m: [2]}` → `[x, y, z]`; `points {count, region, z_m, min_separation_m}` → `[[x, y, z], …]`; `grid {points: [[x, y, z], …]}` → one entry (finite space); `joint {joints: "arm" \| [names], fraction: [2] = [0.1, 0.9], count: int = 2}` → `{"joint": name, "angles": [rad, …]}`; `gap {gap_m: [2]}` → float; `region {center_region, half_size_m: [2]}` → `{"xy_min", "xy_max"}` | goal values in `EpisodeSpec.goal` |
| `stages` | `order: [str]`; `hold_until: {stage = stage} = {}`; `success_requires_final: bool = true` | |
| `checks.<name>` | `check: str` (predicate) + parameters + `negate` | stage or helper check; `holding` is a reserved helper name |
| `failures` | `rules: [{name, if = [str], unless = [str]}]` | §4.2 |
| `expert` | §6.1 | |
| `perturbations` | `failure: [entry]`, `recovery: [entry]`; entry = `{type, weight = 1.0, stages = [str], <param> = [lo, hi]}` | §7 |
| `success` (existing) | unchanged; FakeEnv reads `min_lift`, `confirm_steps` | |

**Worked example: `tasks/pick_place_color_cube/task.toml`.** Numeric values are first values to tune in WP10; the palette and hold-outs need owner review.

```toml
[task]
name = "pick_place_color_cube"
default_variant = "teleop"
[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"
[task.objects.small_box]
usd = "usd/Small-Box-Physics.usda"
origin = "bottom"
size_outer = [0.05, 0.05, 0.035]
inner_half_size = [0.02, 0.02]
floor_z = 0.002
[task.object_sets.cubes]
prefix = "cube"
count_max = 6
size = [0.02, 0.02, 0.02]
mass = 0.008
[task.palette]
red = [0.8, 0.05, 0.05]
green = [0.05, 0.6, 0.1]
blue = "from_setup:blue_cube"
yellow = [0.9, 0.8, 0.05]
orange = [0.95, 0.4, 0.02]
purple = [0.4, 0.1, 0.6]
pink = [0.95, 0.45, 0.65]
[task.roles]
target = "cubes"
container = "small_box"
[task.holdout]
colors = ["purple"]
counts = [4]
[task.instruction_template]
texts = ["Pick up the {target.color} cube and place it in the box"]
[task.start]
count = [2, 6]
min_separation_m = 0.035
clearance_m = 0.01
reachable = ["@target", "@container"]
[task.start.place.small_box]
region = { xy_min = [0.18, -0.12], xy_max = [0.28, 0.12] }
[task.start.place.cubes]
region = { xy_min = [0.12, -0.18], xy_max = [0.30, 0.18] }
yaw_deg = [0.0, 90.0]
[task.stages]
order = ["reached", "grasped", "lifted", "over_box", "placed", "success"]
hold_until = { reached = "grasped", grasped = "placed", lifted = "placed", over_box = "placed" }
[task.checks.reached]
check = "ee_near"
object = "@target"
tol_m = 0.02
[task.checks.holding]
check = "held"
object = "@target"
[task.checks.grasped]
check = "all"
of = ["holding"]
[task.checks.raised]
check = "object_lifted"
object = "@target"
min_lift_m = 0.02
[task.checks.lifted]
check = "all"
of = ["holding", "raised"]
[task.checks.above_box]
check = "object_in_region"
object = "@target"
region = "@container.inner"
check_z = false
[task.checks.over_box]
check = "all"
of = ["holding", "above_box"]
[task.checks.in_box]
check = "object_in_region"
object = "@target"
region = "@container.inner"
[task.checks.released]
check = "held"
object = "@target"
negate = true
[task.checks.placed]
check = "all"
of = ["in_box", "released"]
[task.checks.ee_clear]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true
[task.checks.success]
check = "all"
of = ["placed", "ee_clear"]
[task.checks.distractor_in_box]
check = "objects_in_region"
set = "cubes"
exclude = ["@target"]
region = "@container.inner"
[task.failures]
rules = [
  { name = "wrong_object_placed", if = ["distractor_in_box"] },
  { name = "placed_not_confirmed", if = ["placed"] },
  { name = "never_reached", unless = ["reached"] },
  { name = "missed_grasp", unless = ["grasped"] },
  { name = "timeout_holding", if = ["holding"] },
  { name = "dropped_before_box", unless = ["over_box"] },
  { name = "dropped_outside" },
]
[task.expert]
max_steps = 600
steps = [
  { skill = "move_joint", q = "home" },
  { skill = "set_gripper", gap_m = "open" },
  { skill = "approach", target = "@target", offset_m = [0, 0, 0.06], stage = "reached" },
  { skill = "grasp", target = "@target", stage = "grasped" },
  { skill = "lift", height_m = 0.08, stage = "lifted" },
  { skill = "transport", target = "@container", height_m = 0.06, stage = "over_box" },
  { skill = "place", target = "@container" },
  { skill = "release", stage = "placed" },
  { skill = "retreat", dz_m = 0.08, stage = "success" },
]
[task.perturbations]
failure = [
  { type = "grasp_offset", weight = 2, xy_m = [0.008, 0.02] },
  { type = "early_release", stages = ["lifted"], delay_steps = [5, 40] },
  { type = "place_offset", xy_m = [0.025, 0.05] },
  { type = "wrong_target" },
  { type = "object_drop", stages = ["lifted", "over_box"], xy_jitter_m = [0.0, 0.02] },
]
recovery = [
  { type = "grasp_offset", xy_m = [0.008, 0.02] },
  { type = "object_shift", stages = ["reached"], xy_m = [0.02, 0.05], yaw_rad = [-0.5, 0.5] },
  { type = "object_drop", stages = ["lifted"], xy_jitter_m = [0.0, 0.02] },
  { type = "container_shift", stages = ["lifted"], xy_m = [0.02, 0.05] },
]
```

**Primitive example: `tasks/prim_reach_point/task.toml`.**

```toml
[task]
name = "prim_reach_point"
[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"
[task.instruction_template]
texts = ["Move the gripper to the point"]
[task.start.goal.point]
type = "point"
region = { r_m = [0.15, 0.32], bearing_deg = [-60.0, 60.0] }
z_m = [0.03, 0.20]
[task.stages]
order = ["reached", "success"]
[task.checks.reached]
check = "ee_near"
point = "@goal.point"
tol_m = 0.01
[task.checks.success]
check = "all"
of = ["reached"]
[task.failures]
rules = [{ name = "never_reached", unless = ["reached"] }, { name = "left_target" }]
[task.expert]
steps = [
  { skill = "move_joint", q = "home" },
  { skill = "reach", target = "@goal.point", stage = "reached" },
  { skill = "wait", steps = 10, stage = "success" },
]
[task.perturbations]
failure = [{ type = "target_offset", xy_m = [0.03, 0.08] }, { type = "stall", stages = ["reached"], steps = [600, 600] }]
recovery = [{ type = "target_offset", xy_m = [0.03, 0.08] }]
```

The `__init__.py` is `from core.tasks.base import load_task_spec`, `from core.tasks.generic import GenericTask`, `TASK = GenericTask(load_task_spec("<name>"))`. The `env_cfg.py` is `from sim.envs.generic_task_cfg import apply_teleop, apply_generate  # noqa: F401`.

## 5. Arm contract

### 5.1 arm.toml additions

| Table.key | Type | SO-101 value |
|---|---|---|
| `links.moving_jaw` | str | `"jaw"` |
| `links.moving_jaw_mesh` | str | `"moving_jaw_so101_v1"` |
| `kinematics.file` | str (relative to the arm dir) | `"kinematics.json"` |
| `kinematics.ee_link` | str = `links.ee` | `"gripper"` |
| `kinematics.tcp_position_m` | float[3], in the ee frame | measured in WP2 from `gripper_geometry.gripper_tip_local_offset` |
| `kinematics.tcp_quat_wxyz` | float[4] | `[1, 0, 0, 0]` |
| `kinematics.approach_axis` | float[3] unit, TCP frame, pointing out of the jaws | container-measured |
| `kinematics.closing_axis` | float[3] unit, ⟂ approach | container-measured |
| `gripper.joint` | str (USD) | `"Jaw"` |
| `gripper.open_rad` / `gripper.closed_rad` | float | container-measured |
| `gripper.opening_table` | `[[angle_rad, gap_m], …]`, ≥ 2 rows, angles strictly increasing, gaps strictly monotone | container-measured |
| `gripper.max_object_width_m` | float | `0.045` |
| `expert.arm_joints` | [str] (USD, chain joints) | the five non-jaw joints |
| `expert.home_rad` | `{joint = rad}`, optional | default `[arm.initial_state.joint_positions]` |
| `expert.max_joint_speed_rad_s` | float, or `{joint = float}` | `1.5` |
| `expert.max_gripper_speed_rad_s` | float | `3.0` |
| `expert.max_ee_speed_m_s` | float | `0.15` |
| `expert.ik_position_weight` / `ik_axis_weight` / `ik_roll_weight` | float ≥ 0 | `1.0` / `0.3` / `0.1` |
| `expert.ik_tol_pos_m` / `ik_tol_rot_rad` | float | `0.0005` / `0.02` |

The SO-101 values marked container-measured are committed by WP2 together with the extraction run. Until then CPU tests use fixtures (§13).

### 5.2 kinematics.json (schema_version 1)

```json
{"schema_version": 1, "arm": "so101", "base_link": "base",
 "source": {"usd": "usd/SO-ARM101-USD.usd", "usd_sha256": "hex", "git_commit": "hex|unknown"},
 "joints": [{"name": "Rotation", "type": "revolute", "parent": "base", "child": "shoulder",
             "axis": [0, 0, 1], "parent_frame": {"position": [3], "quat_wxyz": [4]},
             "child_frame": {"position": [3], "quat_wxyz": [4]}, "limits_rad": [lo, hi]}]}
```

- `type` is `"revolute"` only.
- `axis` is a unit basis vector, from USD `axis` X, Y or Z.
- `parent_frame` holds `localPos0`/`localRot0` and `child_frame` holds `localPos1`/`localRot1`.
- Link transform: `T_child = T_parent · F(parent_frame) · R(axis, q) · F(child_frame)⁻¹`, exactly as `tests/sim/check_grasp_stability.py::_forward_kinematics` computes it.
- `extract_kinematics.build_kinematics(joints: list[dict], base_link: str, arm: str, source: dict) -> dict` is pure. Its input rows hold `name, parent, child, axis ("X"|"Y"|"Z"), pos0, rot0_wxyz, pos1, rot1_wxyz, lower_deg, upper_deg` (USD limits are degrees).
- `main()` opens USD lazily and writes the file with `write_json_atomic`.

### 5.3 KinematicChain (`core/arms/kinematics.py`)

```python
@dataclass(frozen=True)
class IKWeights: position: float = 1.0; axis: float = 0.3; roll: float = 0.1
@dataclass(frozen=True)
class IKResult:
    q: np.ndarray; success: bool; position_error_m: float; orientation_error_rad: float
    axis_error_rad: float; roll_error_rad: float; iterations: int

class KinematicChain:
    q_names: tuple[str, ...]; chain_joints: tuple[str, ...]; lower: np.ndarray; upper: np.ndarray   # (J,) rad
    @classmethod
    def from_arm(cls, name: str | None = None) -> "KinematicChain"     # arm.toml + kinematics.json
    @classmethod
    def from_dict(cls, data: dict, *, q_names, chain_joints, ee_link, tcp_position_m, tcp_quat_wxyz,
                  approach_axis, closing_axis) -> "KinematicChain"
    def fk(self, q, frame: Literal["tcp", "ee"] = "tcp") -> np.ndarray          # (4, 4) base frame
    def fk_pose(self, q, frame="tcp") -> Pose
    def link_poses(self, q) -> dict[str, np.ndarray]
    def tcp_from_ee(self, ee: Pose) -> Pose
    def jacobian(self, q, frame="tcp") -> np.ndarray     # (6, J): rows [v; ω] in base frame; non-chain columns 0
    def ik(self, target: Pose, seed_q, *, weights: IKWeights | None = None, lower=None, upper=None,
           max_iters: int = 200, tol_pos_m: float = 5e-4, tol_rot_rad: float = 0.02) -> IKResult
```

IK uses damped least squares on the residual `[w_p (p − p*); w_a (a × a*); w_r · roll]`, where a is the approach axis and roll is the signed angle about a* between the closing axes. It falls back to `scipy.optimize.least_squares` with bounds when DLS stalls.

- The result is always within `[max(lower, self.lower), min(upper, self.upper)]`. Non-chain joints are copied from `seed_q`.
- `success` = `position_error_m ≤ tol_pos_m`, and `axis_error_rad ≤ tol_rot_rad` when `w_a > 0`, and `roll_error_rad ≤ tol_rot_rad` when `w_r > 0`.
- An unreachable target returns `success=False` and does not raise. NaN input raises `ValueError`.

**Tolerances the tests assert:**

| Property | Tolerance |
|---|---|
| `ik(fk(q))` position, random in-limit q, seed = q perturbed by ±0.3 rad | ≤ 2e-4 m |
| `ik(fk(q))` orientation (full weights, fixture 6-DoF) | ≤ 1e-3 rad |
| Jacobian vs central differences (h = 1e-6) | ≤ 1e-5 |
| `fk` against fixture golden matrices | ≤ 1e-9 |

### 5.4 Gripper, RawMapping, limits

```python
@dataclass(frozen=True)
class Gripper:
    joint: str; index: int; open_rad: float; closed_rad: float
    angles_rad: tuple[float, ...]; gaps_m: tuple[float, ...]; max_object_width_m: float
    @classmethod
    def from_arm(cls, name=None) -> "Gripper"
    @classmethod
    def from_table(cls, table: Mapping, q_names: Sequence[str]) -> "Gripper"
    def gap_from_angle(self, rad: float) -> float      # piecewise linear, clamped to the table ends
    def angle_from_gap(self, gap_m: float) -> float    # inverse; round trip ≤ 1e-9 inside the table

@dataclass(frozen=True)
class RawMapping:                                      # core/arms/raw_mapping.py, numpy only
    names: tuple[str, ...]                             # LeRobot names, joint-vector order
    mins_deg: np.ndarray; maxs_deg: np.ndarray; clamp_mins_deg: np.ndarray; clamp_maxs_deg: np.ndarray
    gripper_mask: np.ndarray
    @classmethod
    def from_tables(cls, names, mapping: Mapping, runtime: Mapping, gripper_joint: str) -> "RawMapping"
    @classmethod
    def from_setup(cls, setup: Mapping | None = None, *, arm: str | None = None) -> "RawMapping"
    def to_radians(self, raw) -> np.ndarray        # same formula as JointMapping.to_radians, clamps to runtime
    def to_raw(self, rad) -> np.ndarray            # same formula as JointMapping.to_raw, no clamp
    def clamp_radians(self, rad) -> np.ndarray
    @property
    def lower_rad(self) -> np.ndarray
    @property
    def upper_rad(self) -> np.ndarray
```

- `JointMapping` gains `to_raw_np`, `clamp_radians_np` and `as_raw_mapping() -> RawMapping`. They must agree with the torch methods to 1e-5 in float32.
- `runtime_joint_limits(setup=None, *, arm=None)` and the mapping lookup use `setup["arms"][arm]["runtime_joint_limits" | "sim_joint_mapping"]` when `setup` has `"arms"` containing `arm`; otherwise they use today's top-level keys. Joint names come from `load_arm_spec(arm)`.
- Tests assert: `to_raw(to_radians(x)) == x` within 1e-6 for x whose mapped angle lies inside runtime limits; `to_radians(to_raw(r)) == r` within 1e-9 for r inside limits.
- **Expert joint limits** = `[max(runtime, USD), min(runtime, USD)]` per joint, in radians.

## 6. Expert

### 6.1 Program schema (`[task.expert]`)

| Key | Type | Default |
|---|---|---|
| `max_steps` | int | 600 |
| `settle_steps` | int | 10 |
| `max_retries` | int | 2 |
| `steps` | list of steps | required |

Every step has `skill` (required), `name: str = skill` (the phase name), `stage: str | None` (the stage this step must achieve; it must be in `stages.order`) and `budget: int` (steps; default per skill). Parameters are below. Refs use §4.4 syntax. `offset_m` is in the base frame.

| Skill | Parameters (default) | Terminates when | Budget |
|---|---|---|---|
| `move_joint` | `q: "home" \| {joint = rad} \| "@goal.x"`; `tol_rad=0.01` | max \|q_meas − target\| ≤ tol over commanded joints | 150 |
| `set_gripper` | `gap_m: float \| "open" \| "closed" \| "@goal.x"`; `tol_m=0.002` | \|gap − target\| ≤ tol, or the gap changed < 1e-4 m for 5 steps (blocked) | 60 |
| `reach` | `target` (point or object ref); `offset_m=[0,0,0]`; `orientation="down" \| "keep"`; `yaw="object" \| "keep" \| float`; `tol_m=0.005` | ‖tcp − goal‖ ≤ tol | 200 |
| `approach` | as `reach`, with defaults `offset_m=[0,0,0.06]`, `orientation="down"`, `yaw="object"` | as `reach` | 200 |
| `grasp` | `target`; `z_offset_m=0`; `close_gap_m=0.0`; `dwell_steps=10` | descent reached and gripper closed or blocked, then `dwell_steps` | 150 |
| `lift` | `height_m=0.08` (relative) | ‖tcp − goal‖ ≤ 0.005 | 120 |
| `transport` | `target` (object or region); `height_m=0.06` above the target top | ‖tcp − goal‖ ≤ 0.005 | 250 |
| `place` | `target`; `offset_m=[0,0,0]`; `clearance_m=0.005` | TCP at target top + h(held)/2 + clearance, within 0.003 | 150 |
| `release` | `gap_m="open"`; `dwell_steps=10` | gap reached and dwell done | 60 |
| `retreat` | `dz_m=0.08` | reached within 0.005 | 120 |
| `push` | `target`; `goal` (region or point ref); `standoff_m=0.03`; `depth_m=0.0` | the target's object is in the goal region, or budget | 300 |
| `touch` | `target`; `contact_tol_m=0.004` | tcp within `contact_tol_m` of the object top | 200 |
| `sweep` | `joint: str \| "@goal.joint"`; `angles: list \| "@goal.angles"`; `tol_rad=0.01` | each angle reached in turn | 300 |
| `wait` | `steps` | after `steps` | `steps` |

`parse_program(table, *, task: Task, arm_spec) -> ExpertProgram` raises `ConfigError` for an unknown skill or key, an unknown role, goal key or stage, a non-positive budget, or a `grasp` without a target.

"yaw = object" picks the object yaw (mod π/2) closest to the current TCP roll.

### 6.2 ScriptedExpert (`core/expert/machine.py`)

```python
@dataclass(frozen=True)
class MotionLimits:
    max_joint_speed_rad_s: np.ndarray; max_gripper_speed_rad_s: float; max_ee_speed_m_s: float
    @classmethod
    def from_arm(cls, name=None) -> "MotionLimits"

class ScriptedExpert:                     # implements Expert
    def __init__(self, program: ExpertProgram, chain: KinematicChain, gripper: Gripper, *,
                 lower: np.ndarray, upper: np.ndarray, limits: MotionLimits, fps: int,
                 stages: Sequence[str], weights: IKWeights | None = None): ...
    reentries: list[tuple[int, str]]      # (frame, resumed_stage)
```

**Modes** (from `spec.expert_mode`):
- `closed_loop`: references are resolved from the current `obs` at every step.
- `blind`: every object reference is resolved once from the `obs` given to `reset` and never re-read, and the expert never re-enters.

**Detection** (closed-loop only; at most `max_retries` re-entries, after which the expert acts as blind):
1. `progress.regressed_stage is not None`;
2. a step with `stage = s` terminates (or runs out of budget) while `s` is not satisfied per `progress.stage_index`;
3. a `wrong_target` perturbation is active and the current `grasp` step ends.

**Re-entry.**
- `resumed_stage` = `stages[progress.stage_index + 1]`.
- The step index is 1 + the index of the last stage-tagged step before the first step tagged `resumed_stage`, or 0 if there is none.
- The re-entry is reported on the frame it happens (`reentered=True`). The first action after it starts the re-entered step.

**Completion.** `done` becomes true after the last step terminates, or after a budget overrun on the last step. Once done, the expert holds the last target.

**Output rules (tests assert):**
- the output is clamped to `[lower, upper]`;
- `|q_target[t] − q_target[t−1]| ≤ max_speed / fps + 1e-9` per joint (gripper uses `max_gripper_speed_rad_s`), and `q_target[−1] := obs.q` at reset;
- the Cartesian goal advances ≤ `max_ee_speed_m_s / fps` per step;
- `phase` is non-empty on every step;
- identical observations give identical outputs;
- every skill terminates within its budget on a kinematic stub that tracks targets exactly.

## 7. Perturbations and labels

### 7.1 Types (`core/generate/perturbations.py`)

`draw_perturbation(task: Task, kind: Kind, rng, weights_override: Mapping[str, float] | None = None) -> PerturbationSpec | None`:
- Returns `None` for `Kind.SUCCESS`.
- Otherwise it chooses an entry of `[task.perturbations].<kind>` by weight, then `onset_stage = rng.choice(entry.stages)` (if given), then each range parameter. `*_steps` uses `rng.integers(lo, hi + 1)`; floats use `rng.uniform(lo, hi)`. A direction angle `dir_rad ~ U[0, 2π)` is added for each `xy_m`.
- The validator (`ConfigError`) checks the type, that the kind is allowed, and that `stages` is a subset of the type's allowed stages and of the task's stages.

| Type | Group | Kinds | Trigger (onset frame) | Parameters (TOML ranges) | Injected window (group A) |
|---|---|---|---|---|---|
| `grasp_offset` | A | F, R | first frame of the first `grasp` step | `xy_m`, `z_m=[0,0]`, `yaw_rad=[0,0]` | that grasp step |
| `early_close` | A | F, R | first `approach` step start + `delay_steps` | `delay_steps` | until that step ends |
| `partial_close` | A | F, R | first `grasp` step | `extra_gap_m=[0.004, 0.012]` (close target = w + extra) | grasp step |
| `early_release` | A | F, R | first `transport` step start + `delay_steps` | `delay_steps` | until gap ≥ open − tol, or step end |
| `place_offset` | A | F, R | first `place` step | `xy_m` | place and the following release step |
| `wrong_target` | A | F, R | frame 0 (needs a set role and count ≥ 2; the expert binds a random distractor drawn with the perturbation rng) | — | until re-entry or episode end |
| `target_offset` | A | F, R | first step tagged `onset_stage` (default: first stage-tagged step with a target, q or gap) | `xy_m`, `z_m=[0,0]`, `joint_rad`, `gap_m` (applied according to skill) | that step |
| `stall` | A | F | first frame with `onset_stage` latched + `delay_steps` | `steps` | `steps` frames (holds the last target) |
| `object_shift` | B | F, R | first frame with `onset_stage` latched (allowed: stages before any grasp stage) + `delay_steps` | `object="@target"`, `xy_m`, `yaw_rad` | — |
| `object_drop` | B | F, R | first frame with `onset_stage` latched (allowed: stages whose check implies holding) + `delay_steps`; not applied if nothing is held then | `xy_jitter_m` | — |
| `container_shift` | B | F, R | as `object_shift` (any stage before `placed`) | `object="@container"`, `xy_m` | — |

Group A is executed inside `ScriptedExpert`, which reads `spec.perturbation`.

Group B is executed by `WorldPerturber(spec, task_spec, rng)`:
- `.due(progress, t) -> bool`; `.apply(env, obs) -> Pose`.
- The loop calls `.apply` after `env.step` at frame t; the onset frame is `t + 1`.
- New pose for `object_shift` / `container_shift`: the current xy + `xy_m·(cos dir, sin dir)`, yaw + `yaw_rad`, z unchanged.
- New pose for `object_drop`: the held object at its current xy + jitter, resting on the support below (§11).

**Noise (group C):** `ou_noise(spec.noise, J_arm, steps, rng) -> np.ndarray (steps, J)`.
- `n[0] = 0`; `n[t+1] = clip(n[t] − θ·n[t] + σ·N(0, 1), −max_abs, max_abs)`.
- The gripper column is 0.
- It is added to `q_target` before the final clamp, on success episodes that carry a `NoiseSpec`.
- Noise frames keep `imitate=True`.

### 7.2 Labels and acceptance (`core/generate/labels.py`)

```python
def build_imitate(injected: Sequence[bool]) -> tuple[bool, ...]            # imitate[t] = not injected[t]
def build_segments(num_frames: int, kind: Kind, *, onset: int | None, injected_end: int | None,
                   detected: int | None, back_on_plan: int | None, phases: Sequence[str]) -> tuple[Segment, ...]
def accept(kind: Kind, *, success: bool, applied: bool, reentered: bool,
           reentered_before_onset: bool) -> RejectionReason | None
```

**Segments.** Zero-length segments are dropped. The result is contiguous, non-overlapping and covers `[0, N)`. Each segment's `phase` = `phases[start]`.

| Kind | Segments |
|---|---|
| success | `nominal [0, N)` |
| failure | `nominal [0, onset)`, `perturbation [onset, p_end)`, `failure [p_end, N)`. For A, p_end = the injected window end; for B, p_end = onset + 1. |
| recovery | `nominal [0, onset)`, `perturbation [onset, detected)`, `recovery [detected, back_on_plan)`, `nominal [back_on_plan, N)` |

`back_on_plan` = the first t ≥ the last re-entry frame with `stage[t] ≥ index(last resumed_stage)`; N if there is none. `detected` = the first re-entry frame.

**Acceptance table.** Rows are evaluated in order. Exceptions from `GenEnv` give `env_error`; a non-finite q, pose or action gives `invalid_state`; both are checked first. After acceptance, a `traj_fp` already accepted in this shard gives `duplicate_trajectory`.

| Intended | Condition | Result |
|---|---|---|
| success | outcome failure | `expert_failed` |
| success | any re-entry | `unplanned_regression` |
| success | outcome success | accept success |
| failure / recovery | perturbation not applied | `perturbation_not_applied` |
| failure / recovery | re-entry before onset | `unplanned_regression` |
| failure | outcome success | `perturbation_ineffective` |
| failure | outcome failure | accept failure (`failure_mode` from `Task.outcome`) |
| recovery | outcome failure | `recovery_failed` |
| recovery | success, no re-entry | `no_regression` |
| recovery | success, ≥ 1 re-entry | accept recovery |

### 7.3 Spec construction (`core/generate/specs.py`)

```python
def build_spec(task: Task, kind: Kind, attempt_index: int, seed: int, *, arm: str, scene: str,
               home_q: Sequence[float], reach_fn=None, noise: NoiseConfig | None = None,
               weights_override=None, fps: int = 30) -> EpisodeSpec      # raises SamplingError
```

- The start comes from `rng_for(seed, "start")`, the instruction from `"instruction"`, the perturbation from `"perturbation"`, and the noise decision from `"noise"`. The noise decision is `rng.random() < noise.fraction` → `NoiseSpec`.
- `max_steps` = `[task.expert].max_steps`.
- `rng_for(seed, purpose) = np.random.default_rng(derive_seed(seed, purpose))`.

## 8. Plan, fingerprints, manifest

### 8.1 Attempt stream (`core/generate/plan.py`)

- `KINDS = (Kind.SUCCESS, Kind.FAILURE, Kind.RECOVERY)`.
- The total order is `(attempt_index, KINDS.index(kind))`.
- `iter_plan(task, master_seed, spec_fn, *, unique_across_kinds=True) -> Iterator[PlannedAttempt]` yields every (index, kind) in that order, forever, with skip decisions applied:
  - `spec_fn(kind, index, seed) -> EpisodeSpec` raising `SamplingError` gives a skip with `sampling_failed`;
  - a `scenario_key` already yielded earlier (keyed by `(kind, key)` when `unique_across_kinds=False`) gives a skip with `scenario_seen`.
- Skip decisions depend only on the index, so every shard and every resume reproduces them.
- `PlannedAttempt(task, kind, index, seed, spec | None, scenario_key | None, skip: SkipReason | None)`.
- `owns(index, shard, n) = index % n == shard`. `shard_attempts(it, shard, n)` filters on `owns`.
- `shard_quota(total, shard, n) = total // n + (1 if shard < total % n else 0)`.
- A shard runs its owned attempts in order and skips kinds whose quota is met.
  - It raises `AttemptBudgetExceeded` when executed, non-skipped attempts of a kind exceed `ceil(max_attempt_factor × quota)`.
  - It raises `ScenarioSpaceExhausted` after `max_consecutive_skips` consecutive skips.
- `check_capacity(task, master_seed, spec_fn, quotas, *, max_attempt_factor, max_consecutive_skips) -> CapacityReport(non_skipped: dict[Kind, int], horizon: int)` walks the plan up to `ceil(max_attempt_factor × quota)` per kind. It raises `ScenarioSpaceExhausted` if fewer than the quota are non-skipped.

### 8.2 Fingerprints (`core/generate/fingerprint.py`)

- `quantize(x, step) = floor(x / step + 0.5)` as Python `int` (numpy: `np.floor(x / step + 0.5).astype(np.int64)`).
- All hashes are `json_sha256(payload)` from `core.utils.hashing`, except `episode_fp`.
- `yaw(q)` = `atan2(2(w·z + x·y), 1 − 2(y² + z²))`.

| Fingerprint | Payload (canonical JSON) |
|---|---|
| `scenario_key(spec)` | `{"v": 1, "task", "arm", "scene", "objects": sorted([{"kind": set name or object name, "role": role or null, "color": str or null, "p": [quantize(x, 1e-3) ×3], "yaw": quantize(yaw, π/180)} for active objects], key=canonical_json), "goal": goal with every float quantized at 1e-3, "q0": [quantize(q, π/360)…], "held": bool}`. Slot names, kind, instruction, seed and perturbation are excluded. |
| `spec_fp(spec)` | `{"v": 1, "spec": spec.to_json()}` without `attempt_index` and `seed`, with every float quantized at 1e-4 (ints and bools unchanged) |
| `traj_fp(action_raw, state_raw, objects0)` | `{"v": 1, "shape": [N, J], "action": sha256(Q(action)), "state": sha256(Q(state)), "objects0": {name: [quantize(c, 1e-4) for c in position]}}`, where `Q(a) = np.floor(np.asarray(a, np.float32).astype(np.float64) * 100 + 0.5).astype("<i4").tobytes()` and `sha256` is a hex digest. `objects0` holds the frame-0 positions of the active objects. |
| `appearance_fp(colors, instruction, light_exposure, robot_color)` | `{"v": 1, "colors": {active name: color or null}, "instruction": str, "light": quantize(exposure, 1e-3) or null, "robot_color": str or null}`. Originals have `light = robot_color = null`. |
| `episode_fp(traj_fp, appearance_fp)` | `hashlib.sha256((traj_fp + appearance_fp).encode("ascii")).hexdigest()` |

WP5 commits `tests/fixtures/fingerprints/golden.json` (inputs plus expected hex) computed from this table. Sensitivity tests move a value by 0.4·step (no change) and by 1.0·step (change), away from bin edges.

### 8.3 Manifest (`<root>/meta/generation_manifest.jsonl`)

Each line is `canonical_json(line) + "\n"`, followed by flush and `os.fsync`.

| Key | Type |
|---|---|
| `v` | 1 |
| `task`, `kind` | str |
| `attempt_index` | int |
| `seed` | int |
| `shard` | [int, int] |
| `status` | `AttemptStatus` |
| `reason` | `SkipReason` \| `RejectionReason` \| null |
| `episode_index` | int \| null (accepted only) |
| `scenario_key`, `spec_fp` | str \| null (null for `sampling_failed`) |
| `traj_fp`, `appearance_fp`, `episode_fp` | str \| null (null when skipped) |
| `success` | bool \| null |
| `failure_mode`, `furthest_stage`, `perturbation` (type) | str \| null |
| `num_frames` | int \| null |
| `wall_s` | float (ignored by every comparison) |

`Manifest(path)` has `.lines()`, `.append(line)`, `.accepted()` and `.last_index(kind)`.

- A final line without `"\n"`, or one that is not valid JSON, is a torn line. It is dropped on read and truncated before the next append.
- An invalid non-final line raises `SchemaError`.

### 8.4 Commit order and reconcile

Per accepted attempt: `write_sidecar` (atomic) → `writer.save_episode()` → manifest line. Rejected and skipped attempts write only the manifest line, after `discard_episode`.

`reconcile(root, writer, manifest) -> ReconcileResult(rebuilt_lines: int, deleted_sidecars: list[int], next_position: tuple[int, int])`, with E = `writer.episode_count()`, M = the number of accepted lines and S = the sidecar indices:
1. The `episode_index` values of accepted lines must be exactly `0..M−1`; otherwise `ResumeError`.
2. Require `M ≤ E ≤ M + 1`; otherwise `ResumeError`.
3. If `E = M + 1`, sidecar `E−1` must exist (otherwise `ResumeError`); its line is rebuilt from it with `wall_s = 0.0`.
4. Sidecars with index ≥ E are deleted (orphans).
5. A missing sidecar with index < E raises `ResumeError`.
6. Generation continues at the first owned (index, kind) in plan order without a line.

A line, once written, is never executed again. An unrecorded attempt is re-executed and gives identical results.

### 8.5 generation_plan.json and config

`<root>/meta/generation_plan.json` is written with `write_json_atomic`:

```json
{"v": 1, "config_sha256": "hex", "config": {...}, "master_seed": 0, "git_commit": "hex", "shard": [0, 1],
 "task": "name", "task_toml_sha256": "hex", "arm": "so101", "arm_files_sha256": {"arm.toml": "hex", "kinematics.json": "hex"},
 "scene": "rpd_meeting_room", "quotas": {"success": 0, "failure": 0, "recovery": 0},
 "holdout": {"colors": [], "counts": []}, "created_at": "ISO-8601"}
```

On resume, a different `config_sha256`, `task_toml_sha256`, `arm_files_sha256`, `master_seed` or `shard` raises `ResumeError`. A different `git_commit` prints a warning only.

```python
@dataclass(frozen=True)
class Quotas:
    success: int; failure: int; recovery: int
@dataclass(frozen=True)
class NoiseConfig:
    fraction: float = 0.5; theta: float = 0.15; sigma_rad: float = 0.01; max_abs_rad: float = 0.02
@dataclass(frozen=True)
class DRConfig:
    copies: int = 2; terms: tuple[str, ...] = ("light", "robot_color"); recolor: bool = True; seed: int = 0
@dataclass(frozen=True)
class GenerateConfig:
    arm: str; scene: str; tasks: tuple[str, ...]; master_seed: int; quotas: Mapping[str, Quotas]   # per task
    shard: tuple[int, int]; dst_root: Path; repo_id: str; fps: int; max_attempt_factor: float
    max_consecutive_skips: int; scenario_unique_across_kinds: bool; noise: NoiseConfig; dr: DRConfig
    perturbation_weights: Mapping[str, float] | None; cameras: bool; sha256: str
    def dataset_root(self, task: str) -> Path        # dst_root / task
    def dataset_repo_id(self, task: str) -> str      # repo_id.format(task=task)
def load_generate_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> GenerateConfig
```

`sha256` = `json_sha256` of the resolved config without `shard` and `dst_root`. Unknown keys raise `ConfigError`. `shard` `"i/n"` requires `0 ≤ i < n`.

```yaml
# configs/generate/pick_place_color_cube.yaml
arm: so101
scene: rpd_meeting_room
tasks: [pick_place_color_cube]
master_seed: 0
quotas: {success: 4000, failure: 1000, recovery: 1000}
per_task_quotas: {}                # task: {success, failure, recovery}
shard: "0/1"
dst_root: /data/v4
repo_id: "local/{task}"
fps: 30
max_attempt_factor: 3.0
max_consecutive_skips: 1000
scenario_unique_across_kinds: true
noise: {fraction: 0.5, theta: 0.15, sigma_rad: 0.01, max_abs_rad: 0.02}
perturbation_weights: null         # {type: weight} overrides task.toml weights
cameras: true
dr: {copies: 2, terms: [light, robot_color], recolor: true, seed: 0}
```

### 8.6 DR appearance (`core/generate/appearance.py`)

```python
@dataclass(frozen=True)
class AppearanceDraw:
    colors: Mapping[str, str]; instruction: str; light_exposure: float | None; robot_color: str | None
    seed: int; appearance_fp: str
def draw_appearance(parent: Mapping, *, task_spec: TaskSpec, terms: Sequence[str], exposure_range: tuple[float, float],
                    robot_colors: Sequence[str], recolor: bool, dr_seed: int, copy_index: int,
                    taken: set[str], max_redraws: int = 100) -> AppearanceDraw
def dr_copy_sidecar(parent: Mapping, draw: AppearanceDraw, *, episode_index: int, source_repo_id: str) -> dict
```

- Draw seed = `derive_seed(dr_seed, parent["fingerprints"]["traj_fp"], copy_index, r)` for r = 0, 1, …. The first draw whose `appearance_fp` is not in `taken` (the parent's and earlier siblings' fingerprints) wins; otherwise `GenerationError`.
- Recolour follows §4.5 step 2 over the same active slots: the target colour is unique, and no holdout colour is used.
- The instruction is re-rendered from the template.
- The copy keeps `traj_fp` and `scenario_key`, sets `source`, and recomputes `appearance_fp` and `episode_fp`.

## 9. Sidecar v2, parquet, dataset columns

### 9.1 `<root>/task_meta/episode_NNNNNN.json` (`core/datasets/task_meta.py`)

| Key | Type |
|---|---|
| `schema_version` | 2 |
| `episode_index`, `num_frames`, `fps` | int |
| `task`, `arm`, `scene` | str (null only when `legacy_v1`) |
| `reference_frame` | `"robot base link (<base>)"` |
| `units`, `angle_units` | `"m"`, `"rad"` |
| `joint_names` | [str] (USD) or null |
| `objects` | `{name: {set: str \| null, size_m: [3], color: str \| null, active: bool, role: str \| null, position: [N][3], orientation_wxyz: [N][4]}}`, active objects only |
| `ee_pose` | `{position: [N][3], orientation_wxyz: [N][4]}` \| null |
| `instruction` | str \| null |
| `roles` | {role: name} |
| `kind` | `Kind` |
| `success` | bool \| null |
| `failure_mode`, `furthest_stage` | str \| null |
| `stages` | [str] |
| `stage` | [N] int \| null |
| `imitate` | [N] bool |
| `phases` | [{name, start, end}] (run-length) |
| `segments` | [{label, start, end, phase}] |
| `perturbation` | `PerturbationSpec.to_json()` + `onset_frame`, `applied` \| null |
| `recovery` | `RecoveryInfo.to_json()` \| null |
| `noise` | `NoiseSpec.to_json()` \| null |
| `attempt` | `{index, seed, shard: [2], spec_fp}` \| null |
| `fingerprints` | `{scenario_key, spec_fp, traj_fp, appearance_fp, episode_fp}` \| null |
| `appearance` | `{light_exposure: float \| null, robot_color: str \| null}` |
| `source` | `{repo_id, episode_index, traj_fp, episode_fp}` \| null |
| `generator` | `{name: "scripted_expert" \| "teleop" \| "dr_replay", git_commit}` |
| `legacy_v1` | bool |

Positions and quaternions are rounded to 6 decimals.

```python
SIDECAR_V2_DIR = "task_meta"
def sidecar_path(root, index) -> Path
def write_sidecar(root, payload: dict) -> Path          # validate, then tmp + fsync + os.replace
def read_sidecar(root, index) -> dict                   # v2 only; validates
def load_episode(root, index, *, task="pick_place_cube", arm=None, scene=None) -> dict
    # task_meta/ first, else pick_place_meta/ mapped by v1_to_v2
def v1_to_v2(v1: dict, *, task, arm, scene) -> dict
def validate_sidecar(payload: dict, *, num_frames: int | None = None, fps: int | None = None) -> None  # SchemaError
def list_sidecars(root) -> list[int]
```

**v1 → v2 mapping.**
- `objects = {"blue_cube": {set: null, size_m: cube_size_m, color: "blue", active: true, role: "target", position, orientation_wxyz}}`.
- `roles = {"target": "blue_cube", "container": "white_box"}`.
- `kind = "success"`, `success = null`, `stage = null`, `ee_pose = null`, `instruction = null`.
- `imitate = [true] * N`, `segments = [nominal 0..N]`, `phases = []`.
- `perturbation`, `recovery`, `noise`, `attempt`, `fingerprints` and `source` are null.
- `generator.name = "teleop"`, `legacy_v1 = true`; the remaining keys are copied.

**Validator rules (`SchemaError`):**
- all keys present with the types above;
- every per-frame array has length `num_frames`;
- all floats are finite;
- quaternion norms are within 1e-3 of 1;
- `stage` values are in `[−1, len(stages))`;
- segments and phases are contiguous and non-overlapping, cover `[0, N)` and have `start < end`;
- `kind = success` ⇒ `perturbation` is null;
- `kind = recovery` ⇒ `recovery` is non-null;
- `success == (failure_mode is None)` unless `success` is null;
- a fingerprint is null only if `legacy_v1`;
- `num_frames` and `fps` match the arguments when given.

`episode_metadata.save_cube_trajectory` and `validate_cube_trajectories` are unchanged in behaviour (v1 still written by teleop). `replay_source.SourceEpisode` gains `objects: dict[str, np.ndarray (N, 7) disk order]`, `colors`, `roles` and `meta: dict`, and keeps `poses_base` (the target role object). `load_source` uses `load_episode`.

### 9.2 Consolidated parquet (`core/datasets/consolidate.py`, WP12)

`consolidate(root) -> Path` writes `task_meta/episodes.parquet`, one row per sidecar, sorted by `episode_index`. `read_consolidated(root) -> list[dict]` returns sidecars equal to `read_sidecar` output.

| Column | Arrow type |
|---|---|
| `episode_index`, `num_frames`, `attempt_index` | int64 (`attempt_index` nullable) |
| `seed` | uint64, nullable |
| `kind`, `instruction`, `failure_mode`, `furthest_stage`, `perturbation_type` | string (nullable) |
| `success` | bool, nullable |
| `scenario_key`, `spec_fp`, `traj_fp`, `appearance_fp`, `episode_fp`, `source_episode_fp` | string, nullable |
| `sidecar_json` | string (`canonical_json` of the full sidecar) |

### 9.3 Extra dataset columns (`task_meta.LABEL_FEATURES`)

```python
LABEL_FEATURES = {
  "label.stage":   {"dtype": "int64",   "shape": (1,), "names": ["stage"]},
  "label.imitate": {"dtype": "int64",   "shape": (1,), "names": ["imitate"]},   # 0 or 1
  "label.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
}
```

`LeRobotRecorder(…, extra_features: dict | None = None)` merges them into the features. `make_sim_frame(action, observation, visual_buffers, extra: Mapping[str, np.ndarray] | None = None)` adds them to the frame. The default `None` keeps today's features byte for byte.

## 10. Audit, merge, upload, streaming

### 10.1 Frames and audit (WP7)

`core/datasets/frames.py`:

```python
def read_frames(root) -> dict[int, dict[str, np.ndarray]]
    # episode_index -> {"action", "observation.state", "label.stage", "label.imitate", "label.ee_pose", "frame_index"}
```

`read_frames` uses pyarrow over `root/data/**/*.parquet` (LeRobot v3, list or fixed-size-list columns) and works on both real and lite datasets.

```python
@dataclass(frozen=True)
class AuditReport:
    v: int; root: str; passed: bool; episodes: int; checks: tuple[dict, ...]   # {name, passed, details: [str] (≤ 50)}
    content_digest: str; fingerprints_digest: str; created_at: str
def audit_dataset(root, *, source_root=None, check_manifest: bool = True) -> AuditReport
def content_digest(root) -> str        # json_sha256(tree_sha256 over meta/, data/, videos/, task_meta/, minus meta/audit_report.json)
def write_audit_report(root, report) -> Path   # meta/audit_report.json
def read_audit_report(root) -> AuditReport
```

Checks (names fixed):
- `sidecars_valid`
- `counts_match`: dataset episodes = sidecars = manifest accepted lines (if a manifest exists)
- `labels_match`: the label columns equal the sidecar `stage`, `imitate` and `ee_pose` within float32 precision
- `fingerprints_recomputed`: traj from the data parquet and sidecar `objects` frame 0; appearance and episode from the sidecar
- `episode_fp_unique`
- `traj_fp_unique_originals`
- `scenario_key_unique_originals`
- `holdout_absent`: originals and copies
- `dr_sources_exist`: copies only, when `source_root` is given; `source.traj_fp` is in the source
- `dr_appearance_distinct`: a copy differs from its parent and its siblings

`passed` = all checks passed. `fingerprints_digest` = `json_sha256(sorted episode_fp)`.

### 10.2 Merge (WP12)

```python
@dataclass(frozen=True)
class ShardSummary:
    root: Path; episodes: int; episode_fps: tuple[str, ...]; scenario_keys: tuple[str, ...]; report: AuditReport
@dataclass(frozen=True)
class MergePlan:
    index_maps: tuple[dict[int, int], ...]; total: int
def plan_merge(shards: Sequence[ShardSummary]) -> MergePlan
    # new = offset_k + old, offsets cumulative in the given order
def apply_merge_meta(plan, shards, dst_root) -> None
    # renumbered sidecars (episode_index rewritten) and a concatenated manifest
def merge_frames(plan, shards, dst_root, repo_id) -> None   # container: LeRobot aggregate
```

`plan_merge` raises `AuditFailed` if a shard report has not passed or is stale (its `content_digest` differs). It raises `DuplicateError` on any `episode_fp` or `scenario_key` collision between shards.

### 10.3 Upload (WP12)

```python
class StorageClient(Protocol):
    def list_files(self) -> dict[str, str | None]        # remote path -> sha256 (None if unknown)
    def upload_file(self, local: Path, remote: str) -> None
    def finalize(self, message: str) -> None             # Hub: no-op after per-file commits; bucket: write _COMPLETE

class HubStorage:      # injected api with list_repo_files, get_paths_info, upload_file (huggingface_hub signatures)
    def __init__(self, api, repo_id: str, *, revision: str = "main", repo_type: str = "dataset"): ...
class BucketStorage:   # injected fs with find(prefix), info(path) -> {"sha256"?}, put_file(local, remote)
    def __init__(self, fs, prefix: str): ...

@dataclass(frozen=True)
class UploadItem:
    local: Path; remote: str; sha256: str; size: int
@dataclass(frozen=True)
class UploadPlan:
    items: tuple[UploadItem, ...]; skipped: tuple[str, ...]
def plan_upload(root, storage: StorageClient, *, allow_overwrite: bool = False) -> UploadPlan
def run_upload(plan, storage, *, message: str) -> None
```

- `plan_upload` raises `AuditFailed` if: `meta/audit_report.json` is missing; it has not passed; its `content_digest` differs from the current one; or `task_meta/episodes.parquet` is missing.
- It raises `GenerationError` if: the root contains `*.tmp`, an `mp4/` preview tree or an open episode buffer; or a remote file differs in sha256 without `allow_overwrite`.
- Files whose remote sha256 matches are skipped (resumable).
- Order: `data/`, `videos/`, `task_meta/`, `meta/` (with `meta/audit_report.json` last), then `finalize`.
- Per-episode sidecar JSON files are excluded (the parquet replaces them) unless there are fewer than 1000 episodes.

### 10.4 Streaming (WP12)

- `select_episodes(rows: Sequence[dict], *, kinds=None, success=None, exclude_holdout_of: TaskSpec | None = None) -> list[int]` is pure and runs over `read_consolidated` rows.
- `make_streaming_dataset(repo_id, *, episodes=None, **kwargs)` is container-only and imports LeRobot lazily.

## 11. FakeEnv and MemoryWriter (WP7, `core/generate/fake_env.py`, `core/datasets/frames.py`)

```python
class FakeEnv:                       # implements GenEnv; deterministic, no randomness
    def __init__(self, task: Task, chain: KinematicChain, gripper: Gripper, *, fps: int = 30,
                 alpha: float = 0.5, cameras: Mapping[str, tuple[int, int]] = {}, push_radius_m: float = 0.01,
                 crash_at_step: int | None = None): ...
class SimulatedCrash(Exception): ...  # raised by crash hooks; never caught by run_generation
```

Dynamics, applied per `step(q_target)` in this order:
1. **Joints.** `q ← q + alpha·(clip(q_target, chain.lower, chain.upper) − q)`. If an object is attached, the gripper joint cannot pass `gripper.angle_from_gap(w)`; it stops there.
2. **Pose.** The TCP and ee come from `chain.fk`, and `gap = gripper.gap_from_angle(q_g)`.
3. **Attached object.** Its pose is `T_tcp · T_rel` (`T_rel` is frozen at attach).
4. **Attach.** Applies if nothing is attached, the gap decreased this step, `gap ≤ w + 0.004`, and the object centre is within the jaw box: `|xy offset in TCP frame| ≤ w/2` and `|z offset| ≤ h/2`. The nearest such active object attaches and the gap is set to `w`.
5. **Detach.** Applies if attached and `gap > w + 0.002`. The object drops to support: z = the highest support under its xy + h/2 (centre origin). Supports are the table (`table_z_m`), the top of any active object whose footprint contains the xy, and a container's floor (`z_c + floor_z`) when the xy is inside its inner rectangle. Yaw is kept; roll and pitch are set to 0.
6. **Push.** Applies to each active non-attached, non-container object with the TCP inside its footprint (in object yaw) expanded by `push_radius_m` and TCP z in `[bottom, top]`. Its xy moves by the TCP xy displacement of this step; z and yaw are unchanged. Container walls and object-object collisions are not modelled.

Other behaviour:
- **`reset(spec)`** sets the poses and colours from `spec`, sets `q = spec.initial_q` and `held_object` attached at the TCP with gap `w`, and returns frame 0.
- **`set_object_pose`** sets the pose exactly (no drop) and detaches.
- **Inactive objects** report `active=False` at their park position.
- **`signals`** are always provided:
  - `holding` = attached;
  - `grasped` = attached;
  - `lifted` = attached and z − start z ≥ `[task.success].min_lift` (default 0.01);
  - `placed` = not attached and the `@container` role inner-region test holds;
  - `success` = `placed` for `[task.success].confirm_steps` (default 25) consecutive frames.
- **`render()`** returns `np.zeros((h, w, 3), np.uint8)` per camera.
- **`crash_at_step`** raises `SimulatedCrash` on that global step count.

```python
class MemoryWriter:                  # implements EpisodeWriter, persists to root in the "lite" layout
    def __init__(self, root: Path, *, fps: int, joint_names: tuple[str, ...],
                 crash: Literal["before_save", "after_save"] | None = None, crash_on_episode: int | None = None): ...
```

- **Lite layout.** `meta/info.json` holds `{"codebase_version": "v3.0-lite", "fps", "total_episodes", "features"}` and is replaced atomically. That replace is the commit point.
- Episode data goes to `data/chunk-000/file-{index:06d}.parquet` with columns `action`, `observation.state` (list<float32>), `episode_index`, `frame_index`, `index` (int64), `timestamp` (float32), `task` (string) and the `LABEL_FEATURES` columns.
- On open, files with index ≥ `total_episodes` are deleted.
- `crash="after_save"` commits and then raises `SimulatedCrash`; `"before_save"` raises before writing.

### 11.1 Loop (`core/generate/loop.py`)

```python
def run_episode(spec: EpisodeSpec, task: Task, env: GenEnv, expert: Expert, writer: EpisodeWriter,
                mapping: RawMapping, *, noise_rng=None, perturber_rng=None) -> tuple[EpisodeResult, list[StepRecord]]
def run_generation(cfg: GenerateConfig, task_name: str, env: GenEnv, writer: EpisodeWriter, *,
                   expert_factory: Callable[[EpisodeSpec], Expert], mapping: RawMapping,
                   spec_fn: Callable[[Kind, int, int], EpisodeSpec], git_commit: str = "unknown") -> GenerationSummary
def make_expert_factory(task: Task, arm: str, *, setup: Mapping | None = None) -> Callable[[EpisodeSpec], Expert]
@dataclass(frozen=True)
class GenerationSummary:
    accepted: Mapping[Kind, int]; rejected: Mapping[str, int]; skipped: int; attempts: int
```

**Per frame t:**
1. `checks = task.observe_checks(obs.step_state(), ctx)`; `prog = tracker.update(checks)`.
2. `act = expert.act(obs, prog)`; `target = clamp(act.q_target + noise[t])`.
3. `writer.add_frame(mapping.to_raw(target), mapping.to_raw(obs.q), env.render(), spec.instruction, FrameLabels(prog.stage_index, not act.injected, ee_wxyz))`.
4. Stop after this frame if the expert has been `done` for `settle_steps` frames or `t = max_steps − 1`.
5. `obs = env.step(target)`; if a world perturbation is due: `env.set_object_pose(...)`; `obs = env.observe()`.

**Invariant:** `mapping.to_radians(action_raw[t])` equals the executed target within 1e-6 rad.

## 12. Isaac side (WP8, WP11)

```python
class IsaacGenEnv:   # sim/generate/isaac_gen_env.py; implements GenEnv
    def __init__(self, env, task: Task, *, arm: str, mapping: RawMapping, chain: KinematicChain, gripper: Gripper,
                 cameras: Mapping[str, Mapping[str, int]], palette_rgb: Mapping[str, tuple[float, float, float]]): ...
class RecorderEpisodeWriter:   # wraps LeRobotRecorder(extra_features=LABEL_FEATURES); discard -> clear_episode_buffer
    def __init__(self, recorder): ...
```

`IsaacGenEnv` behaviour:
- `reset` calls `env.reset()` and `park_all`. It then teleports the active objects with `set_object_pose_base`, sets colours with `set_object_color`, writes `initial_q` as the joint state, and attaches `held_object` by teleport and closing.
- `step` sends `q_target` as an absolute joint-position action in arm.toml order. At construction it asserts that the action term has scale 1 and offset 0.
- `ee` comes from `ee_frame_state`, `tcp` from `chain.tcp_from_ee`, and `signals` from the task observation group (`subtask_terms`) when present.

`generic_task_cfg.apply_teleop(cfg, task)` / `apply_generate(cfg, task)`:
- They build every object in `task.spec.objects`: `CuboidCfg` for `size`, `UsdFileCfg` for `usd`.
- Park offsets come from `park_positions(names)`, and every object gets a park reset event.
- The contact sensor sits at `{ENV_REGEX_NS}/Robot/<links.moving_jaw>`, filtered to all movable objects.
- `apply_generate` has no terminations, and `episode_length_s` covers `max_steps`.

`scripted_generate.build_parser()` arguments: `--config_path` (required), `--task`, `--shard i/n`, `--dst_root`, `--master_seed`, `--dry_run`, `--limit_attempts`, `--disable_fabric`. AppLauncher arguments are added only after `--dry_run` is handled. `--dry_run` prints the capacity report and quotas without importing Isaac.

| CLI name (`SIM_PROGRAMS`) | Console script | Target |
|---|---|---|
| `generate` | `scripted_generate` | `sim.generate.scripted_generate:main` |
| `generate-plan` | `plan_generation` | `core.scripts.plan_generation:main` |
| `arm-kinematics` | `extract_kinematics` | `sim.generate.extract_kinematics:main` |
| `dataset-audit` | `audit_dataset` | `core.scripts.audit_dataset:main` |
| `dataset-merge` | `merge_shards` | `core.scripts.merge_shards:main` |
| `dataset-upload` | `upload_dataset` | `core.scripts.upload_dataset:main` |

CPU tests may assert the following, using AST parsing or `sys.modules` stubs only:
- the `SIM_PROGRAMS` rows and pyproject scripts above, and that the target functions exist;
- `build_parser` parsing;
- the `--dry_run` path with `isaaclab` absent;
- that `run_generation` is called with an `IsaacGenEnv` and a `RecorderEpisodeWriter`;
- that no SO-101 literal appears in `src/core/expert`, `src/core/generate`, `src/sim/generate/{isaac_gen_env,scripted_generate}.py` or `src/sim/envs/generic_task_cfg.py`. The literals are `"so101"`, the six USD joint names, the six LeRobot names, `"jaw"`, `"gripper"` and `"moving_jaw_so101_v1"`.

WP11:
- `dr_variants.plan_episodes` keeps its signature and output for v1 sources (golden).
- New `plan_episodes_v2(sources: Sequence[dict sidecar], *, task_spec, cfg: DRConfig, exposure_range, robot_colors, include_originals) -> list[PlannedEpisodeV2(dst_index, src_index, draw: AppearanceDraw | None)]` uses `draw_appearance`.
- `SceneReplayer.pose_frame(state_raw, poses_by_name: Mapping[str, np.ndarray (7,) disk order])`.
- `make_env` disables every event term whose function is `park_object` or a task spawn event.

## 13. Test conventions

- **Framework:** `unittest.TestCase` classes, collected and run by pytest (`uv run pytest tests/core tests/sim -q` from the repo root, host group). This matches every existing test, and pytest is the runner the plan names. No pytest fixtures or markers, and no `hypothesis` (not installed). Property tests loop over `np.random.default_rng(k)` for fixed k with `self.subTest`.
- **Files:** `tests/core/test_<module>.py` for `src/core`, and `tests/sim/test_<module>_v4.py` for `src/sim`. Basenames must be unique across `tests/` (there are no `__init__.py` files). The WP prefix goes in the class docstring.
- **Shared support:** WP0 adds `tests/conftest.py` (docstring only, so `tests/` lands on `sys.path`) and the `tests/v4_support/` package. Its helpers are `temp_env(**vars)`, a context manager patching `os.environ` for `S2R_ARMS_DIR`, `S2R_ARM`, `S2R_SCENE_DIR`, `S2R_TASKS_DIR` and `S2R_TASK`; `copy_fixture(name, dst)`; and `assert_frozen(obj)`. `harness.py` (WP7) has `generate_small(task_names, kinds, quotas, shards, kill_points) -> Path`.
- **Fixtures** live under `tests/fixtures/`:

| Fixture | Owner |
|---|---|
| `arms/fx_arm6/` (6-DoF + gripper, hand-written `kinematics.json`) | WP2 |
| `arms/fx_arm5/` (SO-101-like 5 + jaw) | WP2 |
| `scenes/fx_scene/real_setup.json` (an `arms` block with mapping and limits for both arms) | WP2 |
| `tasks/fx_pick/`, `tasks/fx_reach/`, `tasks/fx_color/` (`fx_` prefix, because the `load_task` cache is keyed by name) | WP1 |
| `sidecars/v1/pick_place_meta/episode_000000.json` | WP4 |
| `fingerprints/golden.json` | WP5 |

- **Synthetic folders:** copy fixtures into a `TemporaryDirectory` and point the env vars there with `temp_env`. Never mutate `arms/`, `tasks/` or `scenes/`. Call APIs with an explicit `arm=` or `name=`; never rely on import-time constants.
- **Red before green:** new modules are imported at the top of each test file, unguarded. Before implementation the tests fail with `ImportError` / `AttributeError`; they are never skipped. `skipUnless` is allowed only for container checks.
- **Container-only checks** are `tests/sim/check_*.py` scripts (not collected), or `unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run inside s2r-sim")` classes following `tests/sim/test_contact_materials.py`.
- **Isaac-free sim tests** follow `tests/sim/test_env_prim_path.py` (exec selected AST nodes) or `sys.modules` stubs.
- **Main gate:** WP7 `tests/core/test_no_duplicates.py` runs the harness on the fixture tasks. WP9, WP10 and WP13 add `tests/core/test_generation_{primitives,cube_tasks,stack_bowls}.py`, running the same harness on their real tasks with `fx_arm5`. Together these cover all 17 tasks × 3 kinds.

## 14. Decisions and open points

**Decisions** (D = deviation from `design.md`):

1. Errors are a small hierarchy in `core/generate/errors.py`. `ConfigError` and `SchemaError` subclass `ValueError`, so existing `ValueError` expectations hold.
2. Owner override: every task has failure and recovery. Primitives need perturbations without grasps, so I added `target_offset` (A) and `stall` (A, failure-only). (D)
3. The numpy mapping lives in a new torch-free `core/arms/raw_mapping.py`, because `mapping.py` imports torch at module level. `JointMapping` gets thin numpy methods. (D)
4. `plan_generation.py` moves from WP5 to WP7, because it needs `build_spec` (WP6), which needs the sampler (WP1). WP5's plan takes an injected `spec_fn`. (D)
5. `nested_in` is in WP1's registry, not WP13. (D)
6. Reachability in the sampler is an injected `reach_fn`, so WP1 does not depend on WP2.
7. Per-object placement tables `[task.start.place.*]` replace the top-level `region` / `yaw_deg`, so the box position can be randomised. WP10 must remove `PickPlaceTask.sample_start` (it reads the old top-level keys) and its `failure_mode` (the base rules reproduce it). (D)
8. Hold-outs live in `task.toml` `[task.holdout]`, not in the YAML config, and are not overridable. (D from design §3)
9. Provisional palette and hold-outs for `pick_place_color_cube`: 7 colours, hold out `purple` and count `4`.
10. StageTracker uses both ever-latched flags (furthest stage, `failure_mode`, StageTrace-compatible) and current flags with `hold_until` regression (per-frame label and re-entry).
11. Failure rules are flag-only. State-dependent failures such as `wrong_object_placed` are expressed as helper checks.
12. `scenario_key` excludes slot names, kind and instruction, and includes colours.
13. `appearance_fp` has no `"nominal"` marker. Originals have null light and robot colour, so a DR copy with no effective change collides and is redrawn.
14. Group-B onset is the frame after the step on which it fires. The imitate flag is false only on group-A windows; noise frames keep `imitate = true`.
15. The success kind rejects any re-entry (`unplanned_regression`) to keep labels clean.
16. `label.imitate` is int64 0/1, because LeRobot bool feature support is unverified. Label columns use the `label.` prefix so policies do not treat them as observations.
17. `ee_pose` and `label.ee_pose` are the ee link (as `ee_frame_state`), not the TCP, in WXYZ disk order.
18. `pick_place_cube` keeps its contact-sensor signals. FakeEnv synthesises those signals so the task runs on CPU.
19. `prim_waypoint_path` uses a fixed waypoint `count` per task.toml (design said 2–4 per episode), because stage lists are static.
20. A test runner `tests/conftest.py` is added so shared helpers in `tests/v4_support/` can be imported.
21. MemoryWriter persists a lite LeRobot-v3 layout so that kill and resume and the audit run on the host.
22. DR appearance drawing is pure core code (WP5 `appearance.py`), so WP7's main test can run fake DR before WP11.

**Inconsistencies found:**
- design.md assumes primitives are success-only; the plan and owner say all kinds.
- design.md puts hold-outs in config keys; the prompt lists `[task.holdout]`.
- design.md lists `nested_in` under WP13, but the plan's WP1 registry includes "nested in".
- The WP5 plan script depends on WP1 and WP6.
- `SIDECAR_FRAME` hard-codes `"robot base link (base)"`, and `tasks/pick_place_cube/eval.py::load_starts` rejects any other frame. For a non-SO-101 arm, v2 readers must accept `"robot base link (<base>)"`, while v1 readers stay strict.
- `core.arms.joint_calibration` reads the arm at import, so a synthetic arm cannot change `FOLLOWER_JOINTS` in-process.
- `pyproject` host group lacks PyYAML, which is available only transitively through `draccus`. I recommend adding it in WP5.
- `JointMapping.to_radians_np` already exists; design.md only mentions adding `to_raw_np` and the clamp.
- The WP7 main test cannot cover 17 tasks before WP9/10/13 exist; it is split as in §13.

**Not decided (owner or container):**
- the SO-101 TCP offset, approach and closing axes, gripper table and speed limits (container);
- the table height in the base frame (`table_z_m`; default 0.0 to verify);
- per-filter contact forces with N cubes under Newton;
- the LeRobot v0.6.1 merge, aggregate and streaming APIs;
- the bowl geometry;
- the primitive semantics (WP9 review);
- the Hub organisation and visibility;
- whether `label.*` columns break any existing training prepare script.

## 15. Clarifications after wave 1 (binding; override earlier text where they differ)

These were settled while the wave-1 tests were written. The per-package prompts in `docs/v4/prompts/` carry the full detail; the tests are the final word.

**WP0**
- `ExpertObs.step_state(joint_names=None)`: `joints` is filled only when joint names are passed (§3.2, §4.4).
- Nested values in `goal` and `params` are frozen deeply (lists become tuples, dicts become read-only mappings). Mappings that went through JSON come back with sorted keys; nothing may depend on mapping order.
- `from_json` is strict: every key required, unknown keys rejected, `SchemaError` on any violation.

**WP1**
- `sample_start` and `Task.sample_episode_start` take two extra keyword arguments, `joint_names` and `arm_joints`; `joint_limits` is `(lower, upper)` in `joint_names` order.
- `TaskSpec.palette` resolves `from_setup:` values lazily, so a task loads without a scene.
- Predicates are called as `(state, ctx, params)` with raw parameters; `all` / `any` receive already-computed booleans; a predicate on an inactive object is false.
- `@role.inner` resolves to a container marker; the rectangle is computed from the current state.
- `object_on_object` expected height: `z(b) + (h(b)/2 or h(b)) + (h(o)/2 or 0)`, depending on each object's origin (centre or bottom).
- `reachable` may list `@goal.<key>`; the pregrasp orientation passed to the reachability callback is `(1, 0, 0, 0)` (WXYZ).
- `[task.start]` problems are `ConfigError` at load, only for keys that are present; old layouts keep loading. `core/tasks/loader.py` is not changed.

**WP4**
- Sidecar keys are exact at every level; only object names, `roles` and `perturbation.params` are free-form. `roles` values are not checked against `objects`.
- `phases` may be `[]`; a non-empty list must cover `[0, N)`. Segments are non-empty, ordered, contiguous.
- Fingerprints are 64 lower-case hex characters; all five are required unless `legacy_v1` is true. `generator.git_commit` may be null.
- Integers must be real JSON integers (no bool, no `2.0`).
- The `replay_source.py` change (`SourceEpisode.objects/colors/roles/meta`, loading through `load_episode`) is **moved from WP4 to WP11**.

**WP2**
- `load_kinematics_json` raises `SchemaError` for format problems; `KinematicChain.from_dict` raises only `KinematicsError`.
- IK keeps iterating below the default tolerances so that `ik(fk(q))` meets §5.3; a check (position, axis, roll) is on for any weight above zero; non-chain joints are copied from the seed.
- `RawMapping` mirrors `JointMapping.from_setup`; float32 agreement means 1e-5 rad and 1e-4 raw units.
- `load_arm_spec` validates only the v4 keys that are present and injects no defaults. For SO-101, `tcp_position_m`, `approach_axis`, `closing_axis`, `open_rad`, `closed_rad` and `opening_table` are measured in the container and absent until then; `arms/so101/kinematics.json` is generated there too.
- `joint_limits.py` no longer imports `joint_calibration` at import time.
- Open for WP3: a public name for the "expert joint limits" (runtime limits intersected with USD limits, §5.4).

**WP5**
- `scenario_key` object kind = object name without a trailing `_<digits>`; role = the smallest role naming the object.
- Skip decisions are global: the seen-set spans the whole stream, the first occurrence wins even on another shard.
- Quaternions are canonicalised to `qw >= 0` before hashing. Ints and bools are never quantised (`1` and `1.0` hash differently).
- `check_capacity(..., unique_across_kinds=True)`; `GenerateConfig.to_json()` and `ManifestLine.from_result()` exist. Enforcing `AttemptBudgetExceeded` during a run is WP7's job.
- Manifest: a schema-invalid final line is `SchemaError` (not torn); a duplicate `(task, kind, attempt_index)` is `SchemaError`. Reconcile validates everything before changing any file; `next_position` is the position after the last recorded line.
- `draw_appearance` reads the task spec by attribute and always excludes the parent's fingerprint.
- `pyyaml` joins the `host` dependency group.
- Open for WP11: DR copies of episodes whose instruction template uses `{goal.*}` (sidecar v2 has no `goal`; WP5 raises `GenerationError`), and legacy v1 parents without `traj_fp`.

**Fixtures**
- Shared fixtures stay at the §13 paths: `tests/fixtures/arms/fx_arm{5,6}/`, `tests/fixtures/scenes/fx_scene/`, `tests/fixtures/fingerprints/golden.json`. Package-private ones are under `tests/fixtures/v4_wpN/` (fixture tasks in `v4_wp1/tasks/`, the v1 sidecar in `v4_wp4/v1_root/`).

**WP6** (see `docs/v4/prompts/WP6.md` for the full list)
- `specs.py` (`build_spec`) is part of WP6; it takes `joint_names`, `joint_limits`, `arm_joints` and passes them to the sampler.
- The perturbation table is validated on every `draw_perturbation` call, including for success. A type that needs a feature the task lacks is `ConfigError`; a type not applicable to one start (for example `wrong_target` with one active cube) gets weight 0 via the `start=` argument, and `SamplingError` if nothing is left.
- World perturbations trigger on the furthest stage reached; the onset frame is the frame after the trigger; `object_drop` applies only while holding.
- `ou_noise` takes a boolean arm mask (the gripper is not always the last joint).
- A recovery that does not recover is rejected as `recovery_failed`, never relabelled as a failure.
- New public label helpers: `build_phases`, `build_recovery`, `frame_labels`, `label_columns`.

**WP12** (see `docs/v4/prompts/WP12.md` for the full list)
- Merge order is canonical, not the listed order: shards are sorted by (shard index from `generation_plan.json`, audit content digest), so the merged dataset does not depend on how shards are listed. This replaces the ordering sentence in §10.2.
- Audit record that WP7 must produce exactly: `meta/audit_report.json` with the eight §10.1 fields; `content_digest = json_sha256({relative path: sha256})` over every file under `meta/`, `data/`, `videos/`, `task_meta/` except `meta/audit_report.json`. Merge and upload refuse when the digest does not match the current files.
- Merge refuses with `ConfigError` on differing `info.json` or plan hashes and with `DuplicateError` on a repeated manifest key, episode fingerprint or original scenario key; DR copies may share their parent's scenario key. The merged dataset carries no audit report (it must be audited again).
- Upload: a remote file with unknown hash is uploaded again; `_COMPLETE` is written last; a dirty tree (`*.tmp`, leftover image folders) is refused.
- Added: `ShardSummary.from_root`, `streaming.iter_episodes`.
- To verify in the container before relying on them: the LeRobot v0.6.1 aggregate and streaming APIs (names, episode order, `label.*` columns kept). Known limits to revisit: one Hub commit per file (rate limits), and buckets that report no sha256 (full re-send on resume).

**WP3** (see `docs/v4/prompts/WP3.md` for the full list)
- Expert joint limits: `core.expert.motion.expert_joint_limits(arm=None, *, setup=None) -> (lower, upper)`, the per-joint intersection of runtime and USD limits; `ConfigError` if they do not overlap. `MotionLimits` lives in `motion.py`.
- Detection rule 2 (stage check lost) is not applied to `approach` steps; this resolves the §4.6 colour-cube example, where `approach` is tagged `reached` but stops 6 cm short.
- `ExpertProgram` carries `objects`, `approach_axis` and `closing_axis`; `parse_program(..., joint_limits=None)` enables the reachability checks.
- Every step emits between 1 and `budget` actions; a re-entry frame's action already belongs to the re-entered step; after `max_retries` the expert stops detecting.
- Expert-side perturbations are executed by the expert, on the first execution of a step only; `wrong_target` draws from `rng_for(seed, "expert")`; a grasp started by a re-entry re-opens the gripper first.
- Per-frame motion uses a private damped-least-squares step, not `KinematicChain.ik` (30–200 ms per call).

**WP7** (see `docs/v4/prompts/WP7.md` for the full list)
- `run_episode` only adds frames; the caller saves or discards. Accepted results carry `episode_index = None` until the loop commits them.
- `traj_fp` uses the frame-0 object positions rounded to 6 decimals, exactly as the sidecar stores them, so the audit recomputes the same value.
- `run_generation` walks the shard's plan from the start; recorded positions are not re-executed but update the counters, so a resumed run equals an uninterrupted one (data, sidecars, `info.json` byte for byte; manifest except `wall_s`; plan except `created_at`).
- Each attempt is a pure function of (master seed, task, kind, index). With rejections, different shard counts accept different sets of attempts, so "1 shard equals 3 shards" holds per attempt, not per dataset.
- New public names: `loop.make_spec_fn`, `loop.home_q`, `audit.CHECKS`; `frames.LiteDatasetWriter` is the writer, `MemoryWriter` adds crash hooks. Images are not stored in the lite layout.
- The no-duplicates gate is `tests/core/test_no_duplicates.py`; its harness is `tests/v4_support/harness.py` (`run_no_duplicates_suite`, `generate_small`, `fake_dr`, `assert_no_duplicates`). WP9, WP10 and WP13 call it for their tasks on `fx_arm5`.
- WP7 fixture tasks live in `tests/fixtures/v4_wp7/tasks/` (`fx7_grip`, `fx7_reach`, `fx7_pick`, `fx7_color`), with the table at z = −0.05 m because the fx arms reach a down-pointing TCP only for z ≈ −0.06…0.10 m.

**WP8** (see `docs/v4/prompts/WP8.md`)
- `generate --dry_run` prints exactly what `generate-plan` prints for the resolved config; `--shard`, `--dst_root` and `--master_seed` are config overrides; `--limit_attempts` defaults to 0 (no limit).
- `LeRobotRecorder(extra_features=None)` keeps today's features; `make_sim_frame(..., extra=None)` keeps today's frame; an unknown extra key raises `KeyError`.
- The SO-101 literal scan excludes docstrings and mapping keys (`spec["gripper"]` names the `[arm.gripper]` table).
- Container check scripts are written after WP8 lands, against the implemented `IsaacGenEnv`.

**WP11** (see `docs/v4/prompts/WP11.md`)
- `PlannedEpisodeV2` gains `copy_index` (None for originals). Originals come first, then the copies of each source in order.
- `robot_color_names()` reads `ROBOT_COLORS` from `sim/envs/mdp/resets.py` with `ast`, so the dry run shows exactly the draws of the real run; `exposure_range(task_spec)` defaults to (−3.0, 1.0).
- New helpers `sidecar_v2`, `build_manifest_v2`, `print_plan_v2`; `dr_replay` gains `--copies` and `--no_recolor` and takes the v2 path when the source has `task_meta/`.
- `load_source` gains keyword arguments `task`, `arm`, `scene` (passed to `load_episode`); `SourceEpisode` gains `objects`, `colors`, `roles`, `meta`.

**WP10** (see `docs/v4/prompts/WP10.md`)
- CPU tests of real tasks run on `fx_arm5l` (`tests/fixtures/v4_wp10/`): fx_arm5 scaled ×2.2 and turned −90° about z, matching the SO-101 base-frame conventions (faces −y, table at z = 0). `harness.prepare_sandbox` copies it and merges its setup block.
- `pick_place_cube` keeps `PickPlaceTask` (signals), gains `[task.stages].order` without `hold_until`, `[task.failures]`, `[task.roles]`, `[task.start]`, `[task.expert]`, `[task.perturbations]` and the `generate` variant; `reached` uses the TCP when the state has one; `_over_box` no longer imports `eval.py`. `object_drop` and `object_shift` recoveries are not available to it (signal stages); its recovery uses `grasp_offset`.
- `stack_two_cubes`: roles `target = red_cube`, `base = blue_cube`. `pick_place_color_cube`: 7-colour palette, hold-out `purple` and count 4, 4 cm-inner box with randomised pose, `wrong_object_placed`.
- Two WP1 assertions accept both the pre- and post-WP10 state of `pick_place_cube` (variants, `sample_start`).

**WP13** (see `docs/v4/prompts/WP13.md`)
- Proposed bowl geometry (owner to confirm): small bowl 4 × 4 × 2.5 cm, large bowl 8 × 8 × 3.5 cm with inner half size 3 cm, both bottom-origin procedural USDA with convex collision pieces. Success via `nested_in`.

**WP9** (see `docs/v4/prompts/WP9.md`)
- Owner-approved primitive meanings, with two adaptations for scenario variety: `prim_calibration_reach` visits three points drawn from a fixed 9-point grid; `prim_gripper_gap` sets the width at a random reach point. `prim_place_at` grasps the cube from the table (a held start breaks the expert's place height).
- Arm-only tasks have small scenario spaces when `initial_q` is fixed; `generate-plan` reports the capacity before a run.
