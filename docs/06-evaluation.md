# Evaluation on a fixed benchmark

A benchmark fixes object starts, raw arm start states, the saved joint mapping,
scene geometry and per-episode policy seeds. Run different policies against the
same set and schedule. Changing a seed, horizon, calibration or scene is a new
experimental condition, not automatically a paired comparison.

## Capture the current scene

Start the simulator container and create a successful dataset following
[data generation](08-data-generation.md), or use your own recorded dataset with
object sidecars and saved mapping. The example below uses
`/workspace/datasets/pick_place_scripted` from that guide. Capture geometry from
the exact arm/scene/task/variant that evaluation will use:

```bash
./s2r check --arm so101 --scene rpd_meeting_room --task pick_place_cube \
  --variant eval --unit rpd_robot \
  --scene-json /workspace/s2r/outputs/benchmarks/meeting-room.json
```

`--scene-json` is a simulator-container path. It requires one selected
composition and a full runtime check; it cannot be combined with `--static`.
The snapshot contains the live robot base, table plane/outline, named object
poses/sizes and anchor frames. Do not substitute an old snapshot after changing
the mount, table, object geometry or scene assets.

## Build and retain a set

For a first wiring check, use only the recorded starts. Setting the three sampled
counts to zero still includes the `train` split from the input episodes:

```bash
./s2r eval-set --dataset /workspace/datasets/pick_place_scripted \
  --scene_json /workspace/s2r/outputs/benchmarks/meeting-room.json \
  --out /workspace/s2r/outputs/benchmarks/pick-place-demo.json \
  --arm so101 --task pick_place_cube --seed 1984 \
  --n_id 0 --n_ood 0 --n_yaw 0
```

For a real benchmark collect varied, valid starts first, then generate nonzero
`--n_id`, `--n_ood` and `--n_yaw` counts with a sensible `--min_dist_m` and OOD
margin. A single demonstration does not establish a useful training region or
OOD distribution. The task's `start_is_valid` rejects invalid placements; do not
relax that rule just to reach a requested count.

The schema-2 set contains `task`, `arm`, ordered `joint_names`, `joint_mapping`,
named `objects` per start, and an `eval_set_id` content hash. Raw
`arm_start_state` values are interpreted through the **saved** mapping. Keep the
JSON immutable once results use it. Store published sets under the task's
`eval_sets/` folder, retaining source provenance. The shipped frozen sets may
describe an older measured mount; their conversion to schema 2 did not replace
their geometry.

## Serve and run a wiring evaluation

The `example` policy repeats the current joint state. It exercises serving,
transport, resets, cameras and reporting without a downloaded model. It is
expected to fail the manipulation task.

```bash
mkdir -p outputs/example-models
./s2r serve --policy example --checkpoint unused \
  --models-dir outputs/example-models
```

Create `configs/eval/local_example.yaml` with:

```yaml
arm: so101
scene: rpd_meeting_room
task: pick_place_cube
variant: eval
unit: rpd_robot
policy:
  name: example
  checkpoint: wiring
eval:
  dataset: /workspace/datasets/pick_place_scripted
  eval_set: /workspace/s2r/outputs/benchmarks/pick-place-demo.json
  splits: [train]
  repeats: 1
  seed: 1984
  robot_start: recorded
  action_horizon: 8
  episode_length_s: 1
  render_warmup: 0
  results_json: /workspace/s2r/outputs/benchmarks/example-results.json
  rename_map:
    realsense_rgb: room
    wrist_cam: wrist
run:
  gui: false
  rerun: false
```

```bash
./s2r --dry-run eval --config_path configs/eval/local_example.yaml
./s2r eval --config_path configs/eval/local_example.yaml
```

All eval paths are inside the simulator container. `rename_map` maps simulator
sensor names to the policy manifest's camera keys. Check them when using another
scene or wrist camera. A policy's fixed action dimension must match the selected
arm. Per-policy options belong in its client/server plugin, not in generic eval.

For a trained policy, serve the actual checkpoint as described in
[training](05-training.md), update `policy.name`, the checkpoint label, matching
client options and camera map, and choose an adequate episode duration. Serving
loads weights; changing only the YAML checkpoint label does not switch a running
server. Use a fresh result path for every independent run.

## Resume and compare

Interrupted fixed-set runs can resume from their own results file:

```bash
./s2r eval --config_path configs/eval/local_example.yaml --resume
```

Resume checks saved configuration and schedule rather than silently changing the
benchmark. A completed report has `complete: true`. Reports include the embedded
eval set, attempt stages, failure modes, start IDs, repeats, policy seeds and
per-split summaries. `--save_videos` records camera evidence; `--video_encoder
libx264` uses CPU encoding. Videos do not replace task signals or final outcomes.

After running two comparable checkpoints into separate result files:

```bash
./s2r compare /workspace/s2r/outputs/benchmarks/model-a.json \
  /workspace/s2r/outputs/benchmarks/model-b.json \
  --task pick_place_cube --repeat 0
```

Comparison requires complete schema-2 results on the same set (including the
retained `legacy_eval_set_id` when appropriate), matching task, duration, robot
start mode, seed and action horizon. Paired starts must have matching policy
seeds. Rates include Wilson intervals; paired McNemar uses one selected repeat
and reports unadjusted p-values. Repeats are not independent extra benchmark
starts. A one-episode wiring run is not a meaningful policy-quality comparison.

For a folder of sweep results, `./s2r sweep /workspace/s2r/outputs/my-sweep` runs the
existing sweep summarizer and writes `summary.json` and `summary.md` there. Use
it for the supported sweep layout; it does not make mismatched experiments paired.

## Smoke checks and geometry errors

`--smoke-test` uses the configured smoke selection/timeouts for any compatible
policy. It is a quick diagnostic, not a completed full benchmark. Do not compare
an early-stopped smoke subset as if it covered the full fixed schedule.

If evaluation says the live scene differs from the set, restore the intended
scene or capture a new snapshot and build a new set with a new identity. Do not
edit the frozen geometry, replace its digest, disable the guard, or reuse old
results as paired evidence. The reset order remains robot, object starts, then
tracking state; custom task variants must preserve it.

Schema-1 object sidecars remain readable. `s2r migrate` converts dataset sidecars;
it does not rewrite frozen eval sets. Schema conversion preserves old set IDs as
`legacy_eval_set_id`, but does not make different physical scenes equivalent.
