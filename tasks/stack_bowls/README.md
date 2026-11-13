# Stack three bowls

Stack the yellow, red and blue bowls upright inside one another. **Any color
order is accepted.** Begin with three separate bowls. Move two bowls, release
them, and keep the complete nested stack stable for one second. Each bowl must
remain upright; overturned bowls and a gripper-supported stack do not count.

The supplied `~/pavan/bowl.usdc` is retained as `assets/bowl.source.usdc`.
`build_asset.py` writes the active `assets/bowl.usd` with the complete 412,122
triangle visual and its authored face-varying normals, without decimation.
Although the supplied USD declares metres, its coordinates span about 120 by
120 by 50 units; they are normalized to 120 mm diameter and 50 mm height.
Separate convex wall sectors and the measured interior floor preserve the
hollow cavity. `assets/provenance.json` records the source hash, dimensions,
normal count, and conversion details.
The mass is an explicit 55 g simulation assumption, not a measured value.

```bash
./s2r datagen --source scripted --config_path configs/datagen/stack_bowls.yaml
```

This creates one successful SO101 simulation demonstration, using rim grasps;
it does not train a policy. See the generation report for accepted episodes and
failures. The task's progress stages also appear in the paired evaluation GUI.

The supplied one-episode recipe uses a separated SO101 start fixture and records
red → yellow → blue from bottom to top. The success rules accept all six orders.
The planner pinches the thin rim, uses a fixed raised ready posture, and carries
above a support position within the SO101 wrist range. Contact capacity is increased for the compound bowl meshes;
the shared SO101 contact material and physical solver settings are retained.

Render the saved episode from its joint observations and all three object pose
tracks, without advancing physics or modifying the dataset:

```bash
docker exec -w /workspace/s2r s2r-sim s2r-entrypoint python \
  tools/diagnostics/render_task_episode.py \
  --dataset /workspace/datasets/stack_bowls_one_episode \
  --arm so101 --task stack_bowls --headless \
  --output /workspace/s2r/outputs/stack_bowls/render
```

The output is `episode.mp4`, with external and wrist views, plus a replay
validation report. Choose a new destination/output directory when repeating a
run; existing datasets and renders are preserved.

Success is measured from the actual simulated poses and velocities:

- Adjacent bowl centres must align within 9 mm, with a 12–35 mm nesting pitch
  and at most 15 degrees between their upward axes.
- The bottom bowl must remain within 12 degrees of vertical. Every opening must
  face upward within 30 degrees, allowing cumulative lean in a nested stack.
- Two distinct bowls must have been carried. All three must be released and
  stable for 30 control steps (one second), with the gripper clear of the stack.

The recipe uses bounded pose IK with Newton FK/Jacobians and SciPy, smooth jaw
commands, and fresh support-pose feedback before the second pickup. It is a
scripted demonstration, not learned-policy or broad-start-coverage validation.
