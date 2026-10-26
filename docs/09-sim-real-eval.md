# 09 — Paired sim/real evaluation

[Guide hub](../README.md) · Previous: [Real-arm rollout](08-real-arm-rollout.md)

`sim_real_eval` runs one cube start in Isaac and then on the physical SO-101 follower, one after the other, with the
same policy. The cube start is spawned with the leader arm, as in recording. Both halves query one policy server, so
only one copy of the model is in GPU memory. After each trial it reports the per-stage gap between sim and real, which
shows where the simulator is optimistic. It runs in the teleop container (`./so101 sim-real-eval`) and serves a page at
http://localhost:8766.

> **Not yet run on hardware.** The trial flow, stage labelling and report are covered by tests with a fake robot,
> a fake simulator and a fake policy. The real arm has not been run through this page. Do the shadow pass below first.

## Before you start

1. The follower is calibrated and the teleop container has the arm and cameras attached, as in
   [Real-arm rollout](08-real-arm-rollout.md#before-the-first-run). `/root/env` has `ROBOT_PORT`, `ROBOT_ID`,
   `CAMERA_GRIPPER`, `CAMERA_EXTERNAL` and `CONTROL_FPS`.
2. The leader arm is connected (`TELEOP_PORT`/`TELEOP_ID` in `/root/env`, or `--leader_port`/`--leader_id`). It drives
   the sim arm between trials, as in recording.
3. A policy server is up for the checkpoint, either on its own or from the host-managed lifecycle:

```bash
./so101 serve --policy groot_n17 --checkpoint <ckpt> --models-dir /external_storage/models
# or
./docker/eval.sh <config>.yaml --server_only
```

## Start

```bash
./so101 sim-real-eval --policy groot_n17 --lang_instruction "Pick up the blue cube and place it in the white box"
```

Useful flags:

- `--enable-motion` lets the arm move. Without it the arm is read and the policy is queried, but nothing is sent.
- `--no_real` opens the page and runs the sim half without an arm connected.
- `--settle_steps` sets how many sim steps the spawned cube settles before its pose is recorded (default 30).
- `--resume outputs/sim_real_eval/<time>` continues an earlier session and appends to its trials.
- `--gui_port` changes the page port (default 8766).

Open http://localhost:8766 on the machine running the container. Over SSH, forward the port from your own machine
with `./so101 tunnel --host <gpu-host> --port 8766`, then open the same address locally.

## One trial

1. **Position with the leader.** Between trials the leader arm teleoperates the sim arm, as in recording. Move it so the
   sim gripper is where the cube should start.
2. **Spawn cube.** The cube appears beside the sim gripper's fixed jaw, at the same placement as the recorder's spawn.
   It settles for a moment and its pose is recorded. The sim arm then returns to its home (start) pose. The page shows
   that pose in raw LeRobot units.
3. **Match the real scene.** Put the real arm in that start pose and place the real cube where the sim cube is, using
   the overlay. Use the opacity slider; Difference mode turns matching areas black. Then press **Placed**.
4. **Run sim, then Run real.** Run sim restarts from exactly the recorded scene: the arm at home and the recorded cube
   pose. The policy runs in Isaac and the stage is detected automatically. Then press **Run real**, which is enabled
   only after a sim run and **Placed**. The arm moves only if `--enable-motion` is set. **Stop** works at any time.
5. **Save or Discard.** Pick the furthest stage the real arm reached and press **Save**. **Discard** drops the trial and
   writes nothing. The leader drives the sim arm again only after Save or Discard.

**Run sim** and **Run real** can be pressed again to re-run that half; re-running clears the result and label that come
after it.

## Stages

Both halves use the same stage names, so a sim trial and a real trial compare directly.

| Stage | Meaning |
|---|---|
| `none` | The gripper tip never came within 3 cm of the cube |
| `reached` | The gripper tip came within 3 cm of the cube |
| `grasped` | The cube is held in the gripper |
| `lifted` | The held cube was raised off the table |
| `over_box` | The held cube is above the box footprint |
| `placed` | The cube was released inside the box |
| `success` | The task's success check passes |

Label the real arm with the same meaning: the furthest stage it reached during its run.

## Safety

- **Shadow mode is the default.** The policy is queried and its targets are logged, but nothing is sent to the arm.
  Check that the logged targets are sensible poses for the scene before enabling motion.
- **With `--enable-motion`** the arm follows the same rules as `real_rollout`: a start is refused when the first target
  is more than `--max_start_error` (default 20 raw units) from the current pose; each command moves at most
  `--max_relative_target` (default 10); every target is clamped to its joint range; the arm holds its last goal on
  stop, policy timeout or robot error.
- **Sim and real never run at the same time.** The page runs one half at a time, and Stop works for both.
- **The arm keeps torque after the page exits** unless `--release-on-exit` is set, as in
  [Real-arm rollout](08-real-arm-rollout.md#what-happens-to-the-arm-when-it-stops). Lower it to a rest pose by hand first.

## Outputs

Each session writes to `outputs/sim_real_eval/<time>/` (or the folder given with `--output`):

- `trials.jsonl` holds one line per saved trial, appended on each save. Each trial stores the cube's start position
  (x, y, yaw in the robot base frame), its world pose (`cube_pose_w`) and the arm start pose, alongside its sim result,
  real result and label.
- `report.json` (schema `so101_sim_real_eval_v1`) is rewritten atomically on each save. It holds:
  - `header`: policy, checkpoint, instruction, motion flag and safety limits;
  - `trials`: every saved trial with its cube start, sim result, real result and label;
  - `summary`: per-stage rates for sim, real and the gap (sim minus real); a confusion matrix of sim stage by real
    stage; paired success counts (both, sim only, real only, neither) with an exact McNemar test.
- `real_trial_NNN.jsonl` holds the per-step real trace for each real run.

Neither side uses grasp assistance or action smoothing, so both halves see the same raw policy output. A difference
between sim and real comes from the scene and the arm, not from the client.

## Limitations

- The Isaac and hardware paths are tested only with fakes. Expect the first real run to surface problems.
- Matching the scene is manual. The real cube is placed by eye using the overlay, and the real arm is put in the start
  pose by hand, so small offsets count against the gap. The page does not move the real arm to the recorded start pose.
- The sim side runs one environment, so `--num_envs` must be 1 or unset.
