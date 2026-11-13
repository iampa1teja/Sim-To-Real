# Compare a policy on real and simulated starts

```bash
./s2r eval-sim-real
```

Open **http://localhost:8765**. The evaluation console uses the same layout as the real-arm testing
GUI: a policy dropdown, task prompt, server address/port, duration, Run, Stop and recent-run reports.
**Spawn object** is its additional control. The simulator starts with the command; physical devices
open only when you click **Connect real arm**. Disconnect stops a comparison and releases the device
connections while retaining follower motor torque.

1. Connect the real arm. With the default config, the calibrated leader can position the follower
   before a trial. `--no-leader` keeps positioning read-only. Stop disables leader positioning until
   you reconnect.
2. Keep the gripper open, position the physical cube beside the fixed jaw, and click **Spawn object**
   to place its simulation counterpart. This does not measure or move the real cube. Spawn is disabled
   during a comparison. `--no-object` explicitly skips this setup requirement.
3. Select the policy, enter the task prompt and set the duration **per side**. The selected model server
   must already be running; the default endpoint is `localhost:5556`. The dropdown selects the client
   protocol, while the server selects the loaded checkpoint.
4. Click **Run real → sim**. The real trial runs first. The simulation then restores the saved real
   joint pose and spawned object states, refreshes its cameras and runs the second trial. Both sides use
   the same frozen prompt, policy, endpoint, client options and duration. Editing these fields is disabled
   until the comparison ends.
5. To end a trial early, select the furthest stage you observed in **Real trial ended at** and click
   **End real trial → run sim**. The arm holds its measured pose, its gripper keeps its last goal,
   and simulation starts from the saved comparison start. Select the simulation's stage and click
   **End simulation trial** when it finishes. The choices come from the task manifest, including
   **Not reached** and its final **Success** stage. A stage selection is required before ending either side.
6. **Stop** or **Esc** cancels both trials. Pending inference replies are discarded after cancellation.
   Closing the controlling browser also stops the run after its heartbeat expires. You can reposition,
   spawn again and run another comparison without overwriting the earlier one.

There are exactly **two camera panels**. Each dropdown can show the real external, real gripper,
simulation external or simulation wrist camera. The defaults compare the two external views. Changing
these dropdowns affects the display only: each policy continues receiving its configured room/wrist inputs.
Simulation previews remain available when the physical arm is disconnected.

Each comparison is saved under `outputs/eval-sim-real/<session>/<comparison>/` with `real.mp4`,
`sim.mp4`, their measured/commanded joint logs and `report.json`. Reports retain the prompt, selected
policy, endpoint, calibration provenance, client transforms, starting state, termination reasons and
server descriptions. Recent runs link directly to these reports. A finished execution does not
automatically label the physical task as successful.
The history also compares real/simulation stages and their elapsed times. These labels are explicitly
operator supplied; a timer ending without a selection remains unlabelled. Reports retain the selected
milestone, cumulative stage flags, confirmation time and whether both sides reached the same stage.

The config is `configs/eval/sim_real.yaml`; CLI flags override it. Camera defaults come from the current
local `.env`. Both serial ports use stable USB identities. Per-policy client options are selected from
`policy_options`, so MolmoAct2 coordinate settings are not applied when selecting a different policy.
Both sides retain the calibrated joint bounds, first-target distance check and relative-target limit.
The action horizon is unchanged.

To preview/infer without physical motor commands:

```bash
./s2r eval-sim-real --shadow --no-leader
```

`--preview-seconds 10` closes the session after a short preview. To use a remote model, set its address
and port in **Connection settings**, or forward its standard s2r port first:

```bash
./s2r tunnel --host user@server --port 5556
```

The optional `--backend local-lerobot` adapter is for the separate trusted loopback LeRobot TCP server.
It supports MolmoAct2 only and must not be pointed at a standard s2r server.
