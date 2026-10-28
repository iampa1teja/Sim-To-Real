# 09 — Sim-vs-real calibration check

[Guide hub](../README.md) · [Recording](04-recording.md)

The passive logger compares the previous completed teleop tick before the next
command or sim reset. It records follower-mapped leader commands, unclipped
encoder-derived real angles, actual PhysX read-back, and the clamped sim target.
The logger adds no commands to either arm. **Ordinary teleoperation moves both
arms.** Only the operator starts a live session.

Encoder-based checks cannot detect a zero-pose error made during calibration:
real angles use the same encoder calibration as the sim. Matching a rendered
pose, an inclinometer, or an independent camera reference is needed for that.

## Operator run sheet

Use a teleop container started with the writable repository, calibration, GPU,
display and device mounts in [Setup](01-setup.md). A read-only analysis container
with no device/display mounts cannot run this procedure. From the host:

```bash
docker exec -it teleop bash
```

All following commands run **inside that interactive teleop shell**, from the
repository root. Confirm the physical port roles; these are the operator's
specified ports, not automatic USB identification:

```bash
cd /workspace/Sim-to-Real-SO-101-Workshop
setenv TELEOP_PORT /dev/ttyACM0
setenv ROBOT_PORT /dev/ttyACM1
setenv TELEOP_ID rpd_teleop
setenv ROBOT_ID rpd_robot
setenv HF_LEROBOT_CALIBRATION /root/.cache/huggingface/lerobot/calibration
```

1. **Operator calibration, only if a new calibration is intended.** These
   commands modify calibration files; the implementation and verification work
   never executes them. Existing calibrated arms can proceed to step 2.
   Calibration must finish and exit before teleoperation starts.

   ```bash
   lerobot-calibrate --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id="$TELEOP_ID"
   lerobot-calibrate --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID"
   ```

2. Select the **files visible inside the container**. `setenv` persists these
   variables to `docker/env`; the `export` equivalents work for one shell.

   ```bash
   setenv SO101_FOLLOWER_CALIBRATION "$HF_LEROBOT_CALIBRATION/robots/so101_follower/$ROBOT_ID.json"
   setenv SO101_LEADER_CALIBRATION "$HF_LEROBOT_CALIBRATION/teleoperators/so101_leader/$TELEOP_ID.json"
   ```

   At startup, check both printed absolute paths and SHA-256 values. The sim,
   LeRobot and log header must identify the same files. A changed file or a
   different loaded `calibration_fpath` causes a failure before connection.

3. Start **one** of these sessions. Choose a new log filename for every run;
   existing JSONL files are never overwritten. The examples create no dataset.

   ```bash
   pick_place_agent --enable_real_follower --no_calibration \
     --control_input terminal \
     --calibration_check_log outputs/calibration-check/paired-01.jsonl
   ```

   Or, to inspect the generic teleop's existing wrist-start re-zero behavior:

   ```bash
   lerobot_agent --enable_real_follower --no_calibration --spawn_port 0 \
     --calibration_check_log outputs/calibration-check/generic-01.jsonl
   ```

   Both logging flags automatically disable calibration and reject an
   uncalibrated device. `--joint_angle_log` remains an alias-compatible way to
   obtain the telemetry, including its original JSONL frame fields.

   Sweep each joint slowly for about two minutes; hold still at about five
   distinct poses per joint, away from limits. Hold for at least one second
   so the default 0.25-second stillness window has time to settle. Keep hands
   clear during ordinary teleop. **Stop with Ctrl-C in the teleop terminal**;
   Ctrl-C, SIGTERM, normal exit and Python exceptions run the summary finalizer.
   Physical emergency stopping uses the arm's power switch. Do not send SIGKILL
   if you need the automatic finalizer; offline summarization remains available.

4. Read the reports. A second shell can watch while teleop runs:

   ```bash
   python3 scripts/calibration_check.py watch --log outputs/calibration-check/paired-01.jsonl
   ```

   Teleop and both log watchers show one fixed terminal table with Real, Sim,
   Sim-real and flag columns. Only changed values update, once per second;
   ordinary messages and terminal commands remain above the table. The gripper
   has its own separated row. Flags are `S` (still), `M` (moving), `L` (at a
   limit), and `B` (a suspected gap above 180°). Redirected output uses plain
   text tables. Stopping restores normal terminal scrolling.

   Ctrl-C there stops only the file watcher. After stopping teleop:

   ```bash
   cat outputs/calibration-check/paired-01.jsonl.summary.md
   python3 scripts/calibration_check.py summarize \
     --log outputs/calibration-check/paired-01.jsonl \
     --static-velocity-deg-s 1 --static-window-s 0.25 --pass-deg 1 --warn-deg 3
   ```

   This reproduces `<log>.summary.json` and `<log>.summary.md`. `--output PATH`
   additionally writes the JSON report there and cannot replace the raw log.

5. Stop teleop, support the follower, and perform the independent pose check:

   ```bash
   python3 scripts/calibration_check.py poses --supported \
     --follower-port "$ROBOT_PORT" --follower-id "$ROBOT_ID" \
     --log outputs/calibration-check/poses-01.jsonl
   ```

   The sim displays all-zero and five named poses spanning the interior of
   each joint range. Match the supported, torque-off real arm to the rendered
   pose, then press Enter. A closed sim window, Ctrl-C, or `q` + Enter stops
   sampling and disconnects with torque still off. The guarded session permits
   only torque-off register writes (including `Lock=0` used by the driver);
   it forbids configuration, motion, torque-enable and calibration writes.

   Each capture contains 30 samples. Any joint std above 0.3 normalized units
   rejects that pose. The CLI prints candidate offsets and saves
   `<log>.candidates.json`, including the raw samples and connection evidence.
   The measured gap is **real − target**; the additive mapping candidate is
   **target − real**. CONSISTENT requires at least five accepted poses per
   joint and a max-minus-min candidate spread ≤1°. Saturated joints are
   excluded. Gripper offsets are outside this pose check. **Nothing is applied.**

## Interpreting the report

The header records calibration paths/hashes, the measured setup hash, the exact
mapping, runtime limits, Git commit, thresholds and limitations. Each frame has
the three signed gaps, limit/static flags and a suspected branch-error flag.
Differences are never wrapped: a gap above 180° stays above 180° and is flagged.

For each body joint, both all-frame and static, not-at-limit tables report mean
signed gap, mean absolute gap, median, population std, p95 absolute gap, maximum
absolute gap, regression slope and correlation against real angle. The same
statistics are reported for real tracking (`real − commanded`) and sim tracking
(`sim − sim_target`). The gripper has its own tables because its closed anchor
is unresolved.

The stillness gate requires velocities of **both real and sim** below the
configured threshold over the entire short window. Startup frames cannot yet
prove stillness. A joint is at-limit if a command, observation or target reaches,
exceeds, or is within 0.05° of a runtime bound. Exclusion counts overlap.
The first observation has unknown previous command/target, recorded as null.

PASS means mean |sim−real| ≤1° on still, unsaturated frames; WARN means ≤3°;
larger means FAIL. NO_DATA means no qualifying frames, not a pass. These verdicts
do not certify physical zero calibration or sufficient range coverage. A slope
near 0 is consistent with a constant offset; near −2 suggests sign inversion;
other slopes suggest a scale error. Constant real angles provide no slope.

## Calibration selection and recording compatibility

`SO101_FOLLOWER_CALIBRATION` and `SO101_LEADER_CALIBRATION` win over ID-based
paths. Otherwise, `ROBOT_ID` / `TELEOP_ID` select
`$HF_LEROBOT_CALIBRATION/robots/so101_follower/<id>.json` and
`.../teleoperators/so101_leader/<id>.json`. The default calibration root is
`~/.cache/huggingface/lerobot/calibration`. Paths expand `~` and resolve symlinks.
Arbitrary filenames are supported: LeRobot's file lookup uses the selected
filename stem while the interface retains its logical arm ID.

If neither paths nor IDs are set in the environment, simulation retains the
existing `real_setup.json` mapping and embedded-calibration stale check. The
setup file, USD assets and physical calibration files are never regenerated by
these tools. The mapping selection also reaches real_rollout; its explicit
`--calibration_file` has highest priority.

When files are selected, recording contracts hash both selected calibration
files as well as the mapping and existing setup provenance. Resuming a dataset
under different calibration bytes is refused; choose a new dataset root.
The no-selection contract remains unchanged.

## Earlier +192° elbow and +88° roll gaps

See the [original report's trace decomposition](reports/2026-10-27-joint-angle-check/report.md#follow-up--decomposing-the-large-gaps).
The old sim tracked its targets closely; those targets already disagreed with
the observed follower. Heavy elbow saturation and disputed port-role provenance
prevent physical offset claims. The generic agent's wrist-start re-zero can
create a constant roll target gap. The new header identifies that behavior;
`pick_place_agent` uses its existing absolute wrist mapping.

No new hardware capture or physical zero measurement was performed while
building this tool. Return the new JSONL and summary for the per-joint average
gap/verdict assessment after the operator run.

## Verification — 2026-10-28

| Check | Result |
| --- | --- |
| Complete CPU suite | 852 passed, 27 skipped; 281 subtests passed |
| Focused host suites | 120 passed, 1 skipped; 30 subtests passed |
| Flake8, all 18 changed Python files | Passed with the repository configuration |
| Protected setup JSON, calibration fixture, real follower/leader files | Original hashes retained |

The in-place table follow-up has 59 passing focused checks plus virtual-terminal
checks for cell updates, terminal input, resizing and scroll restoration.

The full suite used the existing teleop image's pinned dependencies in a
disposable CPU container: no GPU requests, serial devices, or network access.
Datasets, assets and policy sources were mounted read-only. Its bootstrap made
SDK dependencies available to child Python processes and created an empty
container-local `/external_storage/models` parent for dry-run command tests;
it did not mount model weights or start training, simulation or a hardware
session. The focused tests ran on the host. The host's complete suite initially
failed on missing optional dependencies; the complete green result above is
from the CPU container.

Existing test fixtures now represent the selected-calibration initializer and
the training wrapper's statistics-module import. No upstream GR00T source was
changed. Synthetic tests also exercise full and interrupted torque-off pose
sessions with fake motors, PhysX getter selection, unclipped angles, and
exclusion of repeated reference targets.

Local raw evidence is under
`outputs/calibration-evidence/2026-10-28-calibration-check/`: `container_pytest.log`,
`targeted_pytest.log`, `flake8.log`, `protected_inputs.json`, and
`worker_isolation.txt`. `synthetic-demo.jsonl` and its summaries are explicitly
marked synthetic; they are not physical measurements.
