# WP4 — episode sidecar v2 (Codex prompt)

## Goal

Create the Isaac-free module `src/core/datasets/task_meta.py`: the writer, reader and validator of the per-episode sidecar `<root>/task_meta/episode_NNNNNN.json` (schema version 2), a reader that maps the existing v1 sidecars `<root>/pick_place_meta/episode_NNNNNN.json` into the same v2 form, file naming and listing, and the `LABEL_FEATURES` table of the per-frame dataset label columns. Then make `src/sim/recording/episode_metadata.py` reuse the new atomic-write code while keeping its v1 output byte for byte (teleop keeps writing v1 and must not notice). Every later package that writes or reads sidecars (WP5 appearance, WP7 loop and audit, WP11 DR replay, WP12 consolidate/merge) goes through these functions, so they must match `docs/v4/contracts.md` §9 exactly and make the WP4 tests pass.

## Files

| File | Action |
|---|---|
| `src/core/datasets/__init__.py` | Create. A module docstring only; no code, no re-exports. |
| `src/core/datasets/task_meta.py` | Create. Public names exactly: `SIDECAR_V2_DIR`, `LABEL_FEATURES`, `sidecar_path`, `write_sidecar`, `read_sidecar`, `load_episode`, `v1_to_v2`, `validate_sidecar`, `list_sidecars`. Everything else private (leading underscore). |
| `src/sim/recording/episode_metadata.py` | Change: `save_cube_trajectory` delegates its temp-file/fsync/replace write to a private helper in `task_meta` (a thin wrapper). Same public names, signatures and output bytes. |

Do NOT touch:
- anything under `tests/` (including `tests/conftest.py`, `tests/v4_support/`, `tests/fixtures/`);
- `docs/`, `arms/`, `tasks/`, `scenes/`, `configs/`, `pyproject.toml`;
- `src/core/generate/*` (WP0; import from it, do not change it), `src/sim/recording/replay_source.py` and every other module. The `replay_source.SourceEpisode` / `load_source` change listed for WP4 in contracts §2 and §9.1 is **not** part of this prompt (it is scheduled separately).

## Read first

`docs/v4/contracts.md` §1.1 (frames, WXYZ on disk), §1.3, §1.5 (`SchemaError`), §1.6 (imports), §2 (WP4 row), §3.2–3.4 (`PerturbationSpec`, `RecoveryInfo`, `NoiseSpec`, `Fingerprints` and their `to_json`/`from_json`), §7.2 (segments), §8.4 (commit order: the sidecar is written first, atomically), **§9.1 and §9.3 (all of it)**, §13, §14 ("v1 readers stay strict"). The existing `src/sim/recording/episode_metadata.py` shows the v1 format.

## Interfaces and the decisions taken where the contract is open

Signatures are those of §9.1 (`load_episode(root, index, *, task="pick_place_cube", arm=None, scene=None)`, `v1_to_v2(v1, *, task, arm, scene)`, `validate_sidecar(payload, *, num_frames=None, fps=None)`). `root` may be `str` or `Path`. `LABEL_FEATURES` is exactly the §9.3 dict (shapes are tuples, keys in that order).

**Naming.**
- `sidecar_path(root, index) -> Path` = `Path(root) / "task_meta" / f"episode_{index:06d}.json"` (indices ≥ 10⁶ simply get more digits). It accepts `int` and numpy integers; `bool`, `float` and `str` raise `TypeError`; a negative index raises `ValueError`. It touches no files.
- `list_sidecars(root) -> list[int]`: v2 only (`task_meta/`), sorted ascending, Python ints. A name counts only if it is a regular file whose name equals `f"episode_{i:06d}.json"` for the integer it encodes (so `episode_7.json`, `episode_0000011.json`, `*.tmp`, other extensions and folders are ignored). Missing `task_meta/` → `[]`. v1 files are never listed.

**Write (`write_sidecar(root, payload) -> Path`).**
1. `validate_sidecar(payload)` first; on `SchemaError` nothing is written.
2. Round every position and quaternion component (`objects[*].position`, `objects[*].orientation_wxyz`, `ee_pose.position`, `ee_pose.orientation_wxyz`) with Python `round(v, 6)`. No other value is rounded (e.g. `size_m`, `light_exposure` are kept exactly). The caller's dict is never mutated.
3. Serialise with `json` and `allow_nan=False` (byte layout is free; `sort_keys=True` recommended). Write to a temp file **in `task_meta/` itself** whose name does not look like a sidecar (e.g. `tempfile.NamedTemporaryFile(dir=..., prefix=".", suffix=".tmp", delete=False)`), `flush`, `os.fsync`, then `os.replace` onto `sidecar_path(root, payload["episode_index"])`. Create `task_meta/` if needed. Return that path.
4. On any failure (including in `os.fsync` or `os.replace`) remove the temp file and re-raise; an existing sidecar stays intact. Call `os.fsync` and `os.replace` as attributes of the `os` module at call time (the tests patch `os.fsync` / `os.replace`).

**Read.**
- `read_sidecar(root, index) -> dict`: v2 only. Missing file → `FileNotFoundError`. Invalid JSON (e.g. truncated), a non-object, a v1 payload, any validator failure, or `payload["episode_index"] != index` → `SchemaError`. Returns the plain parsed JSON (`dict`/`list`/`str`/`int`/`float`/`bool`/`None`), so `read_sidecar(root, i) == payload` after `write_sidecar(root, payload)` for any valid payload whose poses are already 6-decimal (and equals the rounded payload otherwise).
- `load_episode(...)`: if `task_meta/episode_N.json` exists, return `read_sidecar(root, N)` (the `task`/`arm`/`scene` arguments are then unused). Otherwise read `pick_place_meta/episode_N.json` and return `v1_to_v2(v1, task=task, arm=arm, scene=scene)`; a v1 file that is not JSON, or whose `episode_index != N`, → `SchemaError`. Neither file → `FileNotFoundError`.

**v1 → v2 (`v1_to_v2`).** Exactly the §9.1 mapping list, with these choices:
- Input must contain the 8 keys the v1 writer emits (`episode_index`, `reference_frame`, `units`, `fps`, `num_frames`, `cube_size_m`, `position`, `orientation_wxyz`); a missing key → `SchemaError`; unknown extra keys are ignored.
- Strict v1: `reference_frame` must equal `"robot base link (base)"` and `units` `"m"` (else `SchemaError`).
- Output: `schema_version 2`; `episode_index`, `num_frames`, `fps`, `reference_frame` copied; `units "m"`, `angle_units "rad"`, `joint_names null`; `objects = {"blue_cube": {"set": null, "size_m": cube_size_m, "color": "blue", "active": true, "role": "target", "position": ..., "orientation_wxyz": ...}}` (values copied, not rounded again); `roles {"target": "blue_cube", "container": "white_box"}`; `kind "success"`, `success`/`failure_mode`/`furthest_stage` null, `stages []`, `stage null`, `ee_pose null`, `instruction null`, `imitate [true]*N`, `phases []`, `segments [{"label": "nominal", "start": 0, "end": N, "phase": null}]`; `perturbation`, `recovery`, `noise`, `attempt`, `fingerprints`, `source` null; `appearance {"light_exposure": null, "robot_color": null}`; `generator {"name": "teleop", "git_commit": null}`; `legacy_v1 true`; `task`/`arm`/`scene` from the arguments.
- The output is validated before it is returned (so short arrays, NaN, a non-unit quaternion, a `cube_size_m` that is not length 3, or `num_frames == 0` → `SchemaError`), shares no list with the input, and the input is not mutated.

**Validator (`validate_sidecar`) — raises `SchemaError`, returns `None`, never mutates.** The §9.1 rule list, made precise:
- *Keys:* the payload is a `dict` with exactly the 33 keys of the §9.1 table. Nested objects have exactly their listed keys: object entries `{set, size_m, color, active, role, position, orientation_wxyz}`; `ee_pose {position, orientation_wxyz}`; phases `{name, start, end}`; segments `{label, start, end, phase}`; `perturbation` = the keys of `PerturbationSpec.to_json()` plus `onset_frame`, `applied`; `recovery` = `RecoveryInfo.to_json()` keys; `noise` = `NoiseSpec.to_json()` keys; `attempt {index, seed, shard, spec_fp}`; `fingerprints {scenario_key, spec_fp, traj_fp, appearance_fp, episode_fp}`; `appearance {light_exposure, robot_color}`; `source {repo_id, episode_index, traj_fp, episode_fp}`; `generator {name, git_commit}`. Free-form mappings: the object names, `roles`, and `perturbation.params`.
- *Types:* an "int" is a JSON integer (`type(v) is int`; `bool` and `2.0` are rejected); a "float" is an `int` or `float` (not `bool`); bools are `bool`; strings are `str`. `schema_version` is exactly the integer `2`. `episode_index ≥ 0`, `num_frames ≥ 1`, `fps ≥ 1`. `task`/`arm`/`scene` are `str`, or null only when `legacy_v1` is true. `reference_frame` fully matches `robot base link \((.+)\)` (any non-empty base name). `units == "m"`, `angle_units == "rad"`. `joint_names` is a list of str or null. `roles` maps str → str (values are **not** checked against `objects`: the v1 mapping names `white_box`, which is not an object). `instruction`, `failure_mode`, `furthest_stage` str or null; `success` bool or null; `stages` list of str; `legacy_v1` bool.
- *Objects:* `objects` is a dict (may be empty); each entry has `active` exactly `true` (active objects only), `set`/`color`/`role` str or null, `size_m` 3 finite numbers.
- *Per-frame arrays:* `objects[*].position` and `ee_pose.position` are N rows of 3 numbers; `orientation_wxyz` N rows of 4 numbers with `|‖q‖ − 1| ≤ 1e-3`; `stage` (if not null) N ints in `[−1, len(stages))`; `imitate` N bools. N = `num_frames`.
- *Finite:* every float anywhere in the payload (including `perturbation.params`, `noise`, `appearance`) is finite.
- *Segments* are a non-empty list, in order, each `start < end` (ints), the first starting at 0, each starting where the previous ended, the last ending at N; labels are `SegmentLabel` values; `phase` str or null. *Phases* follow the same coverage rule with `name` str, except that `[]` (no phase information, as in v1) is allowed.
- *Enums:* `kind` ∈ `Kind` values; `generator.name` ∈ {`scripted_expert`, `teleop`, `dr_replay`}; `generator.git_commit` str or null.
- *Embedded WP0 objects:* the `perturbation` dict without `onset_frame`/`applied` must pass `PerturbationSpec.from_json`; `recovery` must pass `RecoveryInfo.from_json`; `noise` must pass `NoiseSpec.from_json`. Convert any `ValueError`/`TypeError`/`KeyError` they raise into `SchemaError`. `onset_frame` is an int in `[0, N)` or null; `applied` a bool.
- *Attempt* (or null): `index` int ≥ 0; `seed` int in `[0, 2**64)`; `shard` `[i, n]` ints with `0 ≤ i < n`; `spec_fp` a fingerprint string.
- *Fingerprint strings* are 64 lower-case hex characters. If `legacy_v1` is false, `fingerprints` is non-null and all five values are fingerprint strings; if `legacy_v1` is true, `fingerprints` may be null and any of its values may be null. `source` (or null): `repo_id` str, `episode_index` int ≥ 0, `traj_fp`/`episode_fp` fingerprint string or null.
- *Appearance:* `light_exposure` finite number or null; `robot_color` str or null.
- *Cross-field:* `kind == "success"` ⇒ `perturbation is None`; `kind == "recovery"` ⇒ `recovery` is not None; if `success` is not null, `success == (failure_mode is None)`; `num_frames == num_frames` argument and `fps == fps` argument when given.
- *Not enforced* (tests check these are accepted): a failure sidecar without a perturbation or with a recovery block; `success` null with any `failure_mode`; empty `objects`/`roles`; `stage` null with non-empty `stages`; unrounded poses (the writer rounds).

**`episode_metadata.py`.** `SIDECAR_DIR = "pick_place_meta"`, `SIDECAR_FRAME = "robot base link (base)"`, `save_cube_trajectory(root, episode_index, poses, cube_size, fps)` and `validate_cube_trajectories(root, episode_lengths, fps)` keep their behaviour exactly: same payload, key order and `json.dump(..., allow_nan=False)` default separators (the tests compare bytes with a committed fixture), `pick_place_meta/` is created before serialising (a NaN pose raises `ValueError` and leaves the folder empty), no temp file survives a failure, the old file survives a failed `os.replace`, and the resume check raises `ValueError` on the same conditions as today. The only change: the temp-file/fsync/replace code moves into a private helper of `task_meta` that both writers use (for example `_write_text_atomic(path, text)`). Add no public names here.

**Imports (contracts §1.6).** `task_meta.py` imports only the standard library, `numpy` (optional), `pyarrow` (not needed) and `core.*` (`core.generate.errors`, `core.generate.protocols`); never `sim`, Isaac, `torch`, `lerobot`, `yaml`, `scipy`. `episode_metadata.py` may import `core.datasets.task_meta`. Both modules must import in a fresh interpreter with `isaaclab`, `isaacsim`, `omni`, `carb`, `pxr`, `warp`, `lerobot` and `torch` blocked (the tests check this with a subprocess). Python 3.11 syntax only.

## Tests

Acceptance tests (already written; do not edit):

```
tests/core/test_v4_wp4_task_meta.py          # naming, round trip, atomic write, listing, validator, surface
tests/core/test_v4_wp4_v1_sidecar.py         # v1 fixture → v1_to_v2 / load_episode
tests/sim/test_v4_wp4_v1_writer_reader.py    # save_cube_trajectory output → load_episode
tests/sim/test_v4_wp4_episode_metadata.py    # regression of the v1 writer/resume check (passes before and after)
```

Fixtures: `tests/fixtures/v4_wp4/v1_root/pick_place_meta/episode_00000{0,1}.json`, written by the current `save_cube_trajectory`. The tests build v2 payloads with `v4_support.factories` (`make_episode_spec`, `make_recovery_info`, `make_noise`, `make_fingerprints`), so WP0 (`src/core/generate/`) must be in place.

Run from the repo root (Python 3.11, host group):

```
uv run --group host pytest tests/core/test_v4_wp4_*.py tests/sim/test_v4_wp4_*.py -q
```

Without uv: `PYTHONPATH=src python -m pytest tests/core/test_v4_wp4_*.py tests/sim/test_v4_wp4_*.py -q`.

Before WP4, `test_v4_wp4_episode_metadata.py` passes (15 tests) and the other three fail at collection with `ModuleNotFoundError: core.datasets`.

## Definition of done

1. All four test files pass (no skips, no xfails).
2. The full suite (`PYTHONPATH=src python -m pytest tests -q --continue-on-collection-errors`) is no worse than before: every previously passing test still passes, in particular `tests/sim/test_v4_wp4_episode_metadata.py`, the WP0 tests and `tests/core/test_v4_support_selftest.py`; no new collection errors.
3. Only the three files under "Files" were created or changed. Teleop (`sim/teleop/lerobot_agent.py`), `sim/recording/rerecording.py`, `sim/recording/replay_source.py`, `sim/generate/dr_replay.py` and `sim/generate/calibration_replay.py` keep working unchanged (they use the v1 names only).

## If a test seems wrong

If a test appears to contradict `docs/v4/contracts.md`, or cannot be satisfied together with another test, **stop and report** the test name, the contract clause and your reasoning. Do not edit, skip or weaken any test, and do not change `tests/v4_support/` or the fixtures.
