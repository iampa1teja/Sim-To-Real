# Compare MolmoAct2 on real and simulated starts

Run from the v2 checkout with the sim container running and the s2r Molmo
server listening on port 5005:

```bash
./s2r eval-sim-real
```

Open http://localhost:8765. This reuses the recording GUI. The follower uses its stable USB serial path, so ACM port renumbering does not select the leader. The default config is
`configs/eval/sim_real.yaml`; it selects the calibrated `rpd_follower`, room and
wrist cameras, and the s2r Molmo server. It does not restart that server.

1. Move the leader to position the physical follower before Start. The sim mirrors
   the follower’s measured joints. Positioning uses the same per-command limits as
   rollout; Start stops leader control and hands control to the policy.
   Both serial ports use stable USB identities. `--no-leader` disables this setup mode.
2. Keep the gripper open. Place the physical cube beside the fixed jaw using the
   same convention as paired recording, and click **Spawn object**. Compare the
   room and wrist views. Real cube placement is manual: this does not measure its
   pose or prove millimetre alignment.
3. Click **Start real → sim**. The real policy trial runs for 150 seconds. The
   physical follower then holds its final target. The simulator restores the saved
   starting joints and object poses and runs its own 150-second trial, using its
   own images and joint observations. Both use the same loaded policy server.
4. **Stop evaluation** cancels the comparison, including the pending second trial.
   Exit closes devices; the follower keeps torque enabled. Restart the command
   for another comparison; a completed output is never overwritten by Start. Stop
   also works during leader positioning and leaves positioning disabled until relaunch.

Outputs are in `outputs/eval-sim-real/<timestamp>/`: `real.mp4` and `sim.mp4`
(each with room and wrist side by side), their JSONL measured/commanded joint logs,
and `report.json` with starts, calibration, timing, and termination reason.
These are visual comparisons; real success is not automatically labelled.
Inference runs in a worker so camera capture and Stop remain responsive. Both
sides apply the same raw joint limits, maximum target offset, and initial-action
check. Video uses the configured capture FPS and repeats the previous frame if capture falls behind; logs retain measured wall times.

A nonmoving camera preview:

```bash
./s2r eval-sim-real --shadow --preview-seconds 10
```

For shorter manual trials, pass `--episode_seconds 30`. `--shadow` prevents real
motor commands even when Start is pressed. To use a standard s2r policy server,
pass `--backend s2r --policy_port 5556` (the checkpoint is chosen on the server).
The optional `--backend local-lerobot` adapter is only for the separate custom
LeRobot TCP server; it must never be pointed at an s2r server. The default uses
the standard s2r transport and policy plugin.

The prepared remote server on `rpuser1@192.168.29.76` serves the same source
checkpoint through local SSH-forwarded port **5556**. Its model files are under
`/media/rpuser1/New Volume/s2r/models/molmoact2-so101`; keep that volume mounted.
The local server remains on **5005**. To choose remote inference for a new run:

```bash
./s2r eval-sim-real --policy_port 5556
```

If the tunnel is not already running, keep this in another terminal:

```bash
./s2r tunnel --host rpuser1@192.168.29.76 --port 5556
```

Check the remote server with:

```bash
ssh rpuser1@192.168.29.76 'docker logs --tail 20 s2r-serve-molmoact2-5556'
```
