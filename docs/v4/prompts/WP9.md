# WP9 — the 13 primitive tasks (Codex prompt)

## Goal

Add the 13 primitive tasks as task folders, so the scripted generator produces successes, failures and recoveries for each. Meanings were approved by the owner; two were adapted so each primitive has enough distinct start scenarios for the no-duplicates rule (noted below).

| Task | Meaning |
|---|---|
| `prim_joint_sweep` | one arm joint passes 3 random angles in order, then the gripper is set to a random width |
| `prim_gripper_gap` | at a random point, the gripper opens to a random width *(adapted: the random point adds start variety; a width alone gives ~41 scenarios)* |
| `prim_reach_point` | the TCP goes to a random free-space point |
| `prim_test_reach` | the TCP goes 4 cm beside the cube without moving it |
| `prim_calibration_reach` | the TCP visits three points of a fixed 9-point calibration grid, in order *(adapted: a fixed path from a fixed start is one scenario)* |
| `prim_waypoint_path` | the TCP passes 3 random waypoints, ≥ 6 cm apart, in order |
| `prim_touch_object` | the TCP touches the cube's top and withdraws; the cube does not move |
| `prim_nudge` | the cube is pushed 0.5–5 cm into an area just in front of it |
| `prim_push_to_region` | the cube is pushed into a random target area |
| `prim_lift_test` | the cube is grasped, raised about 1 cm and set down |
| `prim_grasp_lift` | the cube is grasped and held at least 6 cm up |
| `prim_place_at` | the cube is picked up and set down inside a random target spot *(the expert takes its place height from the held object's start pose, so an episode cannot start with the cube already held)* |
| `prim_regrasp_after_drop` | the cube is lifted, let go, and grasped and lifted again |

## Files

For each task `tasks/<name>/`: `task.toml` (appendix), `__init__.py` (`TASK = GenericTask(load_task_spec("<name>"))`), `env_cfg.py` (re-export `apply_teleop`, `apply_generate` from `sim.envs.generic_task_cfg`). Plus `configs/generate/primitives.yaml` listing all 13 (`arm: so101`, `scene: rpd_meeting_room`; quotas are an owner decision, proposal 1000 / 500 / 500).

Do NOT touch `tests/`, `docs/`, `src/` or other tasks.

## Read first

`docs/v4/contracts.md` §4.4–§4.6, §6.1 (skills), §7.1 (perturbation types; `target_offset` needs a stage-tagged step with a `target`, `q` or `gap_m`; `stall` and world types need `stages`; `early_close` needs an `approach` step, `early_release` a `transport` step), §15. Prompts `WP3.md`, `WP6.md`, `WP7.md`, `WP10.md`.

## Acceptance

```
uv run --group host pytest tests/core/test_v4_wp9_primitives.py tests/core/test_generation_primitives.py -q
```

pass (≥ 9/10 successes per primitive in FakeEnv on the test arm, the behaviour checks of each meaning, and the no-duplicates gate on all 13), and every earlier test still passes. Container: expert success ≥ 90 % per primitive on the SO-101. Do not edit, skip or weaken tests; if one contradicts the contract, stop and report.

## Lessons from proving the tests (keep them)

- A failure perturbation must make the *final* outcome fail: offsetting an early step of a multi-point path is undone by later steps, so offset the last stage (`stages = ["c2"]`, `["w2"]`) or use `stall` on a stage that must be followed by more work.
- A recovery needs a regression the closed-loop expert detects: either a `target_offset` on a stage-tagged step (the step ends but its check is false) or a world perturbation after a stage that is in `hold_until`.
- Do not put a stage in `hold_until` if its check naturally turns false during the next step (pushing moves the TCP away from "above the cube").
- Two stages with the same check latch together; give each stage a distinct check (the regrasp uses a higher lift threshold than the first lift).
- No `{goal.*}` placeholders in instructions (DR cannot re-render them).

## Appendix — the task tables the tests were proven against

### `tasks/prim_joint_sweep/task.toml`

```toml
[task]
name = "prim_joint_sweep"
instruction = "Sweep one arm joint through the given angles"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.start]
table_z_m = 0.0

[task.start.goal.sweep]
type = "joint"
joints = "arm"
fraction = [0.2, 0.8]
count = 3

[task.start.goal.gap]
type = "gap"
gap_m = [0.005, 0.045]

[task.stages]
order = ["started", "swept", "set", "success"]

[task.checks.started]
check = "joint_near"
joint = "@goal.sweep.joint"
value_rad = "@goal.sweep.angles.0"
tol_rad = 0.02

[task.checks.swept]
check = "joint_near"
joint = "@goal.sweep.joint"
value_rad = "@goal.sweep.angles.2"
tol_rad = 0.02

[task.checks.set]
check = "gap_near"
value_m = "@goal.gap"
tol_m = 0.003

[task.checks.success]
check = "all"
of = ["set"]

[task.failures]
rules = [{ name = "never_started", unless = ["started"] }, { name = "sweep_incomplete", unless = ["swept"] }, { name = "wrong_gap" }]

[task.expert]
max_steps = 400
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "sweep", joint = "@goal.sweep.joint", angles = "@goal.sweep.angles", stage = "swept" }, { skill = "set_gripper", gap_m = "@goal.gap", stage = "set" }, { skill = "wait", steps = 5, stage = "success" }]

[task.perturbations]
failure = [{ type = "stall", stages = ["started"], steps = [400, 400] }, { type = "target_offset", gap_m = [0.008, 0.015] }]
recovery = [{ type = "target_offset", gap_m = [0.008, 0.015] }]
```

### `tasks/prim_gripper_gap/task.toml`

```toml
[task]
name = "prim_gripper_gap"
instruction = "Open the gripper to the requested width"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.start]
table_z_m = 0.0

[task.start.goal.gap]
type = "gap"
gap_m = [0.005, 0.045]

[task.start.goal.point]
type = "point"
z_m = [0.06, 0.15]
region = { r_m = [0.22, 0.34], bearing_deg = [-140.0, -40.0] }

[task.stages]
order = ["opened", "success"]

[task.checks.opened]
check = "gap_near"
value_m = "@goal.gap"
tol_m = 0.003

[task.checks.success]
check = "all"
of = ["opened"]

[task.failures]
rules = [{ name = "wrong_gap", unless = ["opened"] }, { name = "lost_gap" }]

[task.expert]
max_steps = 300
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "reach", target = "@goal.point" }, { skill = "set_gripper", gap_m = "@goal.gap", stage = "opened" }, { skill = "wait", steps = 5, stage = "success" }]

[task.perturbations]
failure = [{ type = "target_offset", gap_m = [0.008, 0.015] }]
recovery = [{ type = "target_offset", gap_m = [0.008, 0.015] }]
```

### `tasks/prim_reach_point/task.toml`

```toml
[task]
name = "prim_reach_point"
instruction = "Move the gripper to the point"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.start]
table_z_m = 0.0

[task.start.goal.point]
type = "point"
z_m = [0.03, 0.15]
region = { r_m = [0.22, 0.36], bearing_deg = [-140.0, -40.0] }

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
rules = [{ name = "never_reached", unless = ["reached"] }, { name = "left_point" }]

[task.expert]
max_steps = 300
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "reach", target = "@goal.point", stage = "reached" }, { skill = "wait", steps = 10, stage = "success" }]

[task.perturbations]
failure = [{ type = "target_offset", xy_m = [0.03, 0.06] }]
recovery = [{ type = "target_offset", xy_m = [0.03, 0.06] }]
```

### `tasks/prim_test_reach/task.toml`

```toml
[task]
name = "prim_test_reach"
instruction = "Move the gripper next to the cube without touching it"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [-0.15, -0.36]
xy_max = [0.15, -0.22]

[task.stages]
order = ["beside", "success"]

[task.checks.beside]
check = "ee_near"
object = "@target"
offset_m = [0.0, 0.04, 0.0]
tol_m = 0.01

[task.checks.moved]
check = "object_displaced"
object = "@target"
min_m = 0.003

[task.checks.untouched]
check = "all"
of = ["beside"]

[task.checks.success]
check = "all"
of = ["beside"]

[task.failures]
rules = [{ name = "touched_object", if = ["moved"] }, { name = "never_beside", unless = ["beside"] }, { name = "left_point" }]

[task.expert]
max_steps = 300
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "reach", target = "@target", offset_m = [0.0, 0.04, 0.0], stage = "beside" }, { skill = "wait", steps = 10, stage = "success" }]

[task.perturbations]
failure = [{ type = "target_offset", xy_m = [0.025, 0.05] }]
recovery = [{ type = "target_offset", xy_m = [0.025, 0.05] }]
```

### `tasks/prim_calibration_reach/task.toml`

```toml
[task]
name = "prim_calibration_reach"
instruction = "Visit the three calibration points in order"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.start]
table_z_m = 0.0

[task.start.goal.c0]
type = "grid"
points = [[-0.12, -0.24, 0.07], [0.0, -0.24, 0.07], [0.12, -0.24, 0.07], [-0.12, -0.3, 0.07], [0.0, -0.3, 0.07], [0.12, -0.3, 0.07], [-0.12, -0.36, 0.07], [0.0, -0.36, 0.07], [0.12, -0.36, 0.07]]

[task.start.goal.c1]
type = "grid"
points = [[-0.12, -0.24, 0.07], [0.0, -0.24, 0.07], [0.12, -0.24, 0.07], [-0.12, -0.3, 0.07], [0.0, -0.3, 0.07], [0.12, -0.3, 0.07], [-0.12, -0.36, 0.07], [0.0, -0.36, 0.07], [0.12, -0.36, 0.07]]

[task.start.goal.c2]
type = "grid"
points = [[-0.12, -0.24, 0.07], [0.0, -0.24, 0.07], [0.12, -0.24, 0.07], [-0.12, -0.3, 0.07], [0.0, -0.3, 0.07], [0.12, -0.3, 0.07], [-0.12, -0.36, 0.07], [0.0, -0.36, 0.07], [0.12, -0.36, 0.07]]

[task.stages]
order = ["c0", "c1", "c2", "success"]

[task.checks.c0]
check = "ee_near"
point = "@goal.c0"
tol_m = 0.01

[task.checks.c1]
check = "ee_near"
point = "@goal.c1"
tol_m = 0.01

[task.checks.c2]
check = "ee_near"
point = "@goal.c2"
tol_m = 0.01

[task.checks.success]
check = "all"
of = ["c2"]

[task.failures]
rules = [{ name = "missed_c0", unless = ["c0"] }, { name = "missed_c1", unless = ["c1"] }, { name = "missed_c2", unless = ["c2"] }, { name = "left_grid" }]

[task.expert]
max_steps = 500
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "reach", target = "@goal.c0", stage = "c0" }, { skill = "reach", target = "@goal.c1", stage = "c1" }, { skill = "reach", target = "@goal.c2", stage = "c2" }, { skill = "wait", steps = 10, stage = "success" }]

[task.perturbations]
failure = [{ type = "target_offset", stages = ["c2"], xy_m = [0.03, 0.06] }]
recovery = [{ type = "target_offset", xy_m = [0.03, 0.06] }]
```

### `tasks/prim_waypoint_path/task.toml`

```toml
[task]
name = "prim_waypoint_path"
instruction = "Move the gripper through the waypoints in order"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.start]
table_z_m = 0.0

[task.start.goal.points]
type = "points"
count = 3
z_m = [0.04, 0.14]
min_separation_m = 0.06
region = { r_m = [0.22, 0.36], bearing_deg = [-140.0, -40.0] }

[task.stages]
order = ["w0", "w1", "w2", "success"]

[task.checks.w0]
check = "ee_near"
point = "@goal.points.0"
tol_m = 0.01

[task.checks.w1]
check = "ee_near"
point = "@goal.points.1"
tol_m = 0.01

[task.checks.w2]
check = "ee_near"
point = "@goal.points.2"
tol_m = 0.01

[task.checks.success]
check = "all"
of = ["w2"]

[task.failures]
rules = [{ name = "missed_w0", unless = ["w0"] }, { name = "missed_w1", unless = ["w1"] }, { name = "missed_w2", unless = ["w2"] }, { name = "left_path" }]

[task.expert]
max_steps = 500
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "reach", target = "@goal.points.0", stage = "w0" }, { skill = "reach", target = "@goal.points.1", stage = "w1" }, { skill = "reach", target = "@goal.points.2", stage = "w2" }, { skill = "wait", steps = 10, stage = "success" }]

[task.perturbations]
failure = [{ type = "target_offset", stages = ["w2"], xy_m = [0.03, 0.06] }]
recovery = [{ type = "target_offset", xy_m = [0.03, 0.06] }]
```

### `tasks/prim_touch_object/task.toml`

```toml
[task]
name = "prim_touch_object"
instruction = "Touch the top of the cube and pull back"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [-0.15, -0.36]
xy_max = [0.15, -0.22]

[task.stages]
order = ["touched", "withdrawn", "success"]

[task.checks.touched]
check = "ee_near"
object = "@target"
offset_m = [0.0, 0.0, 0.01]
tol_m = 0.006

[task.checks.away]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.checks.withdrawn]
check = "all"
of = ["away"]

[task.checks.moved]
check = "object_displaced"
object = "@target"
min_m = 0.005

[task.checks.success]
check = "all"
of = ["withdrawn"]

[task.failures]
rules = [{ name = "never_touched", unless = ["touched"] }, { name = "pushed_object", if = ["moved"] }, { name = "stayed_on_object" }]

[task.expert]
max_steps = 400
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "closed" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.05] }, { skill = "touch", target = "@target", stage = "touched" }, { skill = "retreat", dz_m = 0.06, stage = "withdrawn" }, { skill = "wait", steps = 5, stage = "success" }]

[task.perturbations]
failure = [{ type = "stall", stages = ["touched"], steps = [400, 400] }]
recovery = [{ type = "target_offset", xy_m = [0.025, 0.05] }]
```

### `tasks/prim_nudge/task.toml`

```toml
[task]
name = "prim_nudge"
instruction = "Nudge the cube a little"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube.region]
xy_min = [-0.10, -0.335]
xy_max = [0.10, -0.32]

[task.start.goal.area]
type = "region"
center_region = { xy_min = [-0.10, -0.28], xy_max = [0.10, -0.28] }
half_size_m = [0.15, 0.025]

[task.stages]
order = ["above", "in_place", "success"]

[task.checks.above]
check = "ee_near"
object = "@target"
offset_m = [0.0, 0.0, 0.05]
tol_m = 0.015

[task.checks.in_goal]
check = "object_in_region"
object = "@target"
region = "@goal.area"
check_z = false

[task.checks.nudged_amount]
check = "object_displaced"
object = "@target"
min_m = 0.005
max_m = 0.05

[task.checks.in_place]
check = "all"
of = ["in_goal", "nudged_amount"]

[task.checks.away]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.checks.success]
check = "all"
of = ["in_place", "away"]

[task.failures]
rules = [{ name = "not_in_goal", unless = ["in_place"] }, { name = "stayed_on_object" }]

[task.expert]
max_steps = 500
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "closed" }, { skill = "reach", target = "@target", offset_m = [0, 0, 0.05], stage = "above" }, { skill = "push", target = "@target", goal = "@goal.area", stage = "in_place" }, { skill = "retreat", dz_m = 0.06 }, { skill = "wait", steps = 5, stage = "success" }]

[task.perturbations]
failure = [{ type = "object_shift", stages = ["above"], xy_m = [0.05, 0.08] }, { type = "stall", stages = ["above"], steps = [500, 500] }]
recovery = [{ type = "target_offset", stages = ["above"], xy_m = [0.03, 0.05] }]
```

### `tasks/prim_push_to_region/task.toml`

```toml
[task]
name = "prim_push_to_region"
instruction = "Push the cube into the marked area"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube.region]
xy_min = [-0.15, -0.38]
xy_max = [0.15, -0.34]

[task.start.goal.area]
type = "region"
center_region = { xy_min = [-0.12, -0.30], xy_max = [0.12, -0.26] }
half_size_m = [0.03, 0.03]

[task.stages]
order = ["above", "in_place", "success"]

[task.checks.above]
check = "ee_near"
object = "@target"
offset_m = [0.0, 0.0, 0.05]
tol_m = 0.015

[task.checks.in_goal]
check = "object_in_region"
object = "@target"
region = "@goal.area"
check_z = false

[task.checks.in_place]
check = "all"
of = ["in_goal"]

[task.checks.away]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.checks.success]
check = "all"
of = ["in_place", "away"]

[task.failures]
rules = [{ name = "not_in_goal", unless = ["in_place"] }, { name = "stayed_on_object" }]

[task.expert]
max_steps = 500
settle_steps = 5
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "closed" }, { skill = "reach", target = "@target", offset_m = [0, 0, 0.05], stage = "above" }, { skill = "push", target = "@target", goal = "@goal.area", stage = "in_place" }, { skill = "retreat", dz_m = 0.06 }, { skill = "wait", steps = 5, stage = "success" }]

[task.perturbations]
failure = [{ type = "object_shift", stages = ["above"], xy_m = [0.05, 0.08] }, { type = "stall", stages = ["above"], steps = [500, 500] }]
recovery = [{ type = "target_offset", stages = ["above"], xy_m = [0.03, 0.05] }]
```

### `tasks/prim_lift_test/task.toml`

```toml
[task]
name = "prim_lift_test"
instruction = "Lift the cube a little and set it down"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [-0.15, -0.36]
xy_max = [0.15, -0.22]

[task.stages]
order = ["reached", "grasped", "tested", "set_down", "success"]

[task.checks.reached]
check = "ee_near"
object = "@target"
tol_m = 0.02

[task.checks.grasped]
check = "all"
of = ["holding"]

[task.checks.raised]
check = "object_lifted"
object = "@target"
min_lift_m = 0.007

[task.checks.tested]
check = "all"
of = ["holding", "raised"]

[task.checks.low]
check = "object_lifted"
object = "@target"
min_lift_m = 0.004
negate = true

[task.checks.set_down]
check = "all"
of = ["released", "low"]

[task.checks.success]
check = "all"
of = ["set_down", "clear"]

[task.checks.holding]
check = "held"
object = "@target"

[task.checks.released]
check = "held"
object = "@target"
negate = true

[task.checks.clear]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.failures]
rules = [{ name = "not_set_down", if = ["tested"] }, { name = "never_reached", unless = ["reached"] }, { name = "missed_grasp", unless = ["grasped"] }, { name = "slipped" }]

[task.expert]
max_steps = 600
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "open" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.06], stage = "reached" }, { skill = "grasp", target = "@target", stage = "grasped" }, { skill = "lift", height_m = 0.01, stage = "tested" }, { skill = "release", stage = "set_down" }, { skill = "retreat", dz_m = 0.06, stage = "success" }]

[task.perturbations]
failure = [{ type = "stall", stages = ["tested"], steps = [600, 600] }]
recovery = [{ type = "grasp_offset", xy_m = [0.008, 0.02] }]
```

### `tasks/prim_grasp_lift/task.toml`

```toml
[task]
name = "prim_grasp_lift"
instruction = "Pick up the cube and hold it"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [-0.15, -0.36]
xy_max = [0.15, -0.22]

[task.stages]
order = ["reached", "grasped", "lifted", "success"]

[task.stages.hold_until]
reached = "grasped"
grasped = "lifted"

[task.checks.reached]
check = "ee_near"
object = "@target"
tol_m = 0.02

[task.checks.grasped]
check = "all"
of = ["holding"]

[task.checks.raised]
check = "object_lifted"
object = "@target"
min_lift_m = 0.06

[task.checks.lifted]
check = "all"
of = ["holding", "raised"]

[task.checks.success]
check = "all"
of = ["lifted"]

[task.checks.holding]
check = "held"
object = "@target"

[task.checks.released]
check = "held"
object = "@target"
negate = true

[task.checks.clear]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.failures]
rules = [{ name = "never_reached", unless = ["reached"] }, { name = "missed_grasp", unless = ["grasped"] }, { name = "not_lifted", unless = ["lifted"] }, { name = "dropped" }]

[task.expert]
max_steps = 600
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "open" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.06], stage = "reached" }, { skill = "grasp", target = "@target", stage = "grasped" }, { skill = "lift", height_m = 0.08, stage = "lifted" }, { skill = "wait", steps = 15, stage = "success" }]

[task.perturbations]
failure = [{ type = "grasp_offset", weight = 2, xy_m = [0.008, 0.02] }, { type = "early_close", delay_steps = [0, 5] }, { type = "object_drop", stages = ["lifted"], xy_jitter_m = [0.0, 0.02] }]
recovery = [{ type = "grasp_offset", xy_m = [0.008, 0.02] }, { type = "object_shift", stages = ["reached"], xy_m = [0.02, 0.05], yaw_rad = [-0.5, 0.5] }]
```

### `tasks/prim_place_at/task.toml`

```toml
[task]
name = "prim_place_at"
instruction = "Place the cube on the marked spot"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [0.05, -0.38]
xy_max = [0.20, -0.22]

[task.start.goal.spot]
type = "region"
center_region = { xy_min = [-0.22, -0.36], xy_max = [-0.08, -0.22] }
half_size_m = [0.02, 0.02]

[task.stages]
order = ["reached", "grasped", "over_spot", "placed", "success"]

[task.stages.hold_until]
reached = "grasped"
grasped = "placed"

[task.checks.reached]
check = "ee_near"
object = "@target"
tol_m = 0.02

[task.checks.grasped]
check = "all"
of = ["holding"]

[task.checks.at_spot]
check = "object_in_region"
object = "@target"
region = "@goal.spot"
check_z = false

[task.checks.over_spot]
check = "all"
of = ["holding", "at_spot"]

[task.checks.on_table]
check = "object_lifted"
object = "@target"
min_lift_m = 0.005
negate = true

[task.checks.placed]
check = "all"
of = ["at_spot", "released"]

[task.checks.success]
check = "all"
of = ["placed", "clear"]

[task.checks.holding]
check = "held"
object = "@target"

[task.checks.released]
check = "held"
object = "@target"
negate = true

[task.checks.clear]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.failures]
rules = [{ name = "placed_not_confirmed", if = ["placed"] }, { name = "timeout_holding", if = ["holding"] }, { name = "dropped_before_spot", unless = ["over_spot"] }, { name = "missed_spot" }]

[task.expert]
max_steps = 600
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "open" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.06], stage = "reached" }, { skill = "grasp", target = "@target", stage = "grasped" }, { skill = "lift", height_m = 0.06 }, { skill = "transport", target = "@goal.spot", height_m = 0.06, stage = "over_spot" }, { skill = "place", target = "@goal.spot" }, { skill = "release", stage = "placed" }, { skill = "retreat", dz_m = 0.06, stage = "success" }]

[task.perturbations]
failure = [{ type = "place_offset", xy_m = [0.05, 0.08] }, { type = "early_release", stages = ["grasped"], delay_steps = [0, 10] }]
recovery = [{ type = "grasp_offset", xy_m = [0.008, 0.02] }, { type = "object_shift", stages = ["reached"], xy_m = [0.02, 0.05], yaw_rad = [-0.5, 0.5] }]
```

### `tasks/prim_regrasp_after_drop/task.toml`

```toml
[task]
name = "prim_regrasp_after_drop"
instruction = "Pick up the cube, let it go, and pick it up again"
default_variant = "teleop"

[task.variants]
teleop = "env_cfg:apply_teleop"
generate = "env_cfg:apply_generate"

[task.objects.cube]
size = [0.02, 0.02, 0.02]
mass = 0.008

[task.roles]
target = "cube"

[task.start]
table_z_m = 0.0
reachable = ["@target"]

[task.start.place.cube]
yaw_deg = [0.0, 90.0]

[task.start.place.cube.region]
xy_min = [-0.15, -0.36]
xy_max = [0.15, -0.22]

[task.stages]
order = ["reached", "grasped", "lifted", "dropped", "regrasped", "success"]

[task.checks.reached]
check = "ee_near"
object = "@target"
tol_m = 0.02

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

[task.checks.dropped]
check = "all"
of = ["released"]

[task.checks.high]
check = "object_lifted"
object = "@target"
min_lift_m = 0.05

[task.checks.regrasped]
check = "all"
of = ["holding", "high"]

[task.checks.success]
check = "all"
of = ["regrasped"]

[task.checks.holding]
check = "held"
object = "@target"

[task.checks.released]
check = "held"
object = "@target"
negate = true

[task.checks.clear]
check = "ee_near"
object = "@target"
tol_m = 0.04
negate = true

[task.failures]
rules = [{ name = "never_reached", unless = ["reached"] }, { name = "missed_grasp", unless = ["grasped"] }, { name = "never_dropped", unless = ["dropped"] }, { name = "missed_regrasp" }]

[task.expert]
max_steps = 600
steps = [{ skill = "move_joint", q = "home" }, { skill = "set_gripper", gap_m = "open" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.06], stage = "reached" }, { skill = "grasp", target = "@target", stage = "grasped" }, { skill = "lift", height_m = 0.025, stage = "lifted" }, { skill = "release", stage = "dropped" }, { skill = "set_gripper", gap_m = "open" }, { skill = "approach", target = "@target", offset_m = [0, 0, 0.06] }, { skill = "grasp", target = "@target" }, { skill = "lift", height_m = 0.07, stage = "regrasped" }, { skill = "wait", steps = 10, stage = "success" }]

[task.perturbations]
failure = [{ type = "stall", stages = ["dropped"], steps = [600, 600] }]
recovery = [{ type = "grasp_offset", xy_m = [0.008, 0.02] }]
```
