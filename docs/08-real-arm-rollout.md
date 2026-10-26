# 08 — Running a policy on the real arm

[Guide hub](../README.md) · Previous: [Policy plugins](07-policy-plugins.md)

`real_rollout` runs a served policy on the physical SO-101 follower with the same client the simulator uses
(`SimRemotePolicy`): one action queue, smoothing, grasp controller and plugin client, with the real arm
supplying raw LeRobot units (-100..100 for the body joints, 0..100 for the gripper) instead of radians.
It runs in the teleop container and starts no simulator.

> **Not yet run on hardware.** The control loop, the safety rules and the script flow are covered by tests with
> a fake robot and a fake policy. No arm has been moved with this code. Do the shadow run below first.

## Before the first run

1. The follower is calibrated: `lerobot-calibrate --robot.type=so101_follower --robot.port=$ROBOT_PORT --robot.id=$ROBOT_ID`.
   `real_rollout` never starts calibration; it refuses to run without a calibration file for `ROBOT_ID`, or when
   the arm's stored calibration does not match that file.
2. The teleop container is running with the arm and cameras attached (README, step 2), and `/root/env` has
   `ROBOT_PORT`, `ROBOT_ID`, `CAMERA_GRIPPER`, `CAMERA_EXTERNAL` and `CONTROL_FPS`.
3. A policy server is up for the checkpoint, for example on the GPU machine:

```bash
./so101 serve --policy molmoact2 --checkpoint molmoact2_so101 --models-dir ~/models
```

## 1. Shadow run (no motion)

```bash
./so101 rollout --policy molmoact2 --lang_instruction "Pick up the blue cube and place it in the white box"
```

The arm is read and the policy is queried, but nothing is sent. Per-step state and target are written to
`outputs/real_rollout/<time>/episode_000.jsonl`. Vary the scene (move the cube and the box) and check that the
logged targets are sensible poses for what the cameras see and move the way the scene says they should. Do not
force the arm by hand while the script is connected.

## 2. Enable motion

```bash
./so101 rollout --policy molmoact2 --lang_instruction "..." --enable-motion --episodes 5
```

Keep a hand on the power switch. Each episode:

- waits for you to reset the scene and press Enter;
- refuses to start if the first target is more than `--max_start_error` (default 20 raw units) from the current pose;
- sends at most `--max_relative_target` (default 10 raw units) of change per command, enforced by the follower driver,
  and clamps every target to its joint's range;
- stops sending and holds the last goal if the policy server times out or fails, or the robot reports an error;
- ends after `--episode_seconds` (default 30), then asks you to label it: `y` success, `n` failure, `s` skip.

Ctrl-C stops sending and writes the report.

## What happens to the arm when it stops

The arm **keeps torque on and holds its last goal**, including after the script exits. It does not drop. Cut power,
or pass `--release-on-exit` (the arm will fall), to release it. Lower it to a rest pose by hand first.

## Results

`report.json` holds the run settings, one entry per episode (status, steps, seconds, outcome, inference calls) and a
summary where only labelled episodes count towards the success rate. It is a separate format from the simulator's
fixed-start benchmark report, so `compare_eval_results.py` does not read it.

Next: [Paired sim/real evaluation](09-sim-real-eval.md)
