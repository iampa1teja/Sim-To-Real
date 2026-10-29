# 09 — Sim-vs-real calibration check

[Guide hub](../README.md) · [Recording](04-recording.md) · [Real rollout](08-real-arm-rollout.md)

One operator session produces an independent pose calibration verdict and a
separate tracking diagnosis. Only the operator starts hardware sessions. The
preview, summaries and combined report need no hardware. The lock command only
updates simulation calibration and writes an audit record; it never edits
LeRobot calibration files, motor settings or datasets.

## Session command sheet

Start the writable, GPU/display/device-mounted teleop container using
[Setup](01-setup.md), then enter it from the host:

```bash
docker exec -it teleop bash
```

All commands below run inside that shell. These port roles are the operator's
existing assignments; confirm them against the physical connections. Use Isaac's
Python wrapper for these commands (plain host Python cannot render Isaac).

```bash
cd /workspace/Sim-to-Real-SO-101-Workshop
export TELEOP_PORT=/dev/ttyACM0 ROBOT_PORT=/dev/ttyACM1
export TELEOP_ID=rpd_teleop ROBOT_ID=rpd_robot
export HF_LEROBOT_CALIBRATION=/root/.cache/huggingface/lerobot/calibration
export CALIB_RUN=outputs/calibration-check/$(date +%Y%m%d-%H%M%S)
mkdir -p "$CALIB_RUN"
```

1. **Replaced servo only:** run motor setup before calibration, following the
   driver's prompts to connect individual motors. This command configures motor
   IDs/baud rates; do not run it for an unchanged calibrated arm.

   ```bash
   lerobot-setup-motors --robot.type=so101_follower --robot.port="$ROBOT_PORT"
   # Only if a leader servo was replaced:
   lerobot-setup-motors --teleop.type=so101_leader --teleop.port="$TELEOP_PORT"
   ```

2. **If recalibration is intended:** calibrate follower, then leader, and let
   each command finish. Support the arm and follow the driver's zero/range
   prompts. These are operator commands that write motor/calibration settings.
   Unchanged arms can retain their existing files.

   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port="$ROBOT_PORT" --robot.id="$ROBOT_ID"
   lerobot-calibrate --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" --teleop.id="$TELEOP_ID"
   ```

3. Explicitly select the follower mapping being checked for this shell. This
   is required after recalibration until the new mapping is locked. IDs still
   determine which files hardware clients load; their mappings must agree.

   ```bash
   export SO101_FOLLOWER_CALIBRATION="$HF_LEROBOT_CALIBRATION/robots/so101_follower/$ROBOT_ID.json"
   export SO101_LEADER_CALIBRATION="$HF_LEROBOT_CALIBRATION/teleoperators/so101_leader/$TELEOP_ID.json"
   /workspace/isaaclab/_isaac_sim/python.sh scripts/calibration_check.py poses \
     --preview --output "$CALIB_RUN/preview"
   ```

   Inspect `preview/contact-sheet.png` from the room camera and `preview.json`.
   The reference view preserves the room camera position/orientation and widens
   its field of view by scaling focal lengths to 35%, so the full arm is visible.
   Every pose must say OK. `--headless` renders without a display. Preview never
   imports LeRobot or opens serial devices. It checks the versioned TOML's joint
   coverage and margins, table clearance and nonadjacent-link clearance from
   collision hulls carried by PhysX link transforms. Mechanically connected
   adjacent links and the operator-approved wrist/jaw mechanical interface
   are exempt from self-clearance (recorded explicitly in the TOML/evidence); the base/shoulder mounting assembly is exempt from
   the table-plane check, as approved by the operator. Its fixed shoulder
   clearance of 18.6 mm is still recorded. The plane comes from the upper
   surface of the actual room tabletop mesh, using the measured table normal. The required clearance is at least 2 cm. The hand
   support check bounds moving weight, gripper reach and height; it is a
   geometric plausibility check, not an ergonomic or physical clearance guarantee.

4. Stop all other arm clients. Support the follower before releasing torque,
   then match each displayed pose and press Enter to capture 30 encoder samples:

   ```bash
   /workspace/isaaclab/_isaac_sim/python.sh scripts/calibration_check.py poses \
     --supported --follower-port "$ROBOT_PORT" --follower-id "$ROBOT_ID" \
     --log "$CALIB_RUN/poses.jsonl"
   ```

   Torque remains off. The guard permits only `Torque_Enable=0` and `Lock=0`;
   it blocks calibration, configuration, motion and torque-enable writes.
   Ctrl-C, closing the window, or `q` + Enter stops and finalizes the evidence.
   The log header contains the pose TOML hash, mapping, limits, calibration
   identities, Git commit/dirty state and thresholds. Existing logs are refused.
   Motion with sample std >0.3 normalized units rejects the capture; saturated
   joints and repeated joint targets do not count toward the verdict.

5. Exit the pose session. Start ordinary teleoperation for about **two minutes**,
   sweeping slowly and holding at least five distinct angles per joint for at
   least one second. Teleoperation moves both arms; keep hands clear.

   ```bash
   pick_place_agent --enable_real_follower --no_calibration \
     --follower_id "$ROBOT_ID" --robot_id "$TELEOP_ID" --control_input terminal \
     --calibration_check_log "$CALIB_RUN/gap.jsonl"
   ```

   Stop with Ctrl-C in the teleop terminal. The passive logger adds no commands.
   It observes the previous completed tick before a new command or reset, and
   retains the original gap JSONL schema. A separate terminal may watch with
   `python scripts/calibration_check.py watch --log <gap.jsonl>`.

6. Produce the combined report:

   ```bash
   /workspace/isaaclab/_isaac_sim/python.sh scripts/calibration_check.py report \
     --poses "$CALIB_RUN/poses.jsonl" --gap "$CALIB_RUN/gap.jsonl" \
     --output "$CALIB_RUN/report"
   cat "$CALIB_RUN/report/report.md"
   ```

7. If the independent pose result is **LOCKABLE**, lock it as the default:

   ```bash
   /workspace/isaaclab/_isaac_sim/python.sh scripts/calibration_check.py lock \
     --report "$CALIB_RUN/report/report.json"
   unset SO101_FOLLOWER_CALIBRATION SO101_LEADER_CALIBRATION
   ```

   Lock refuses a non-LOCKABLE report, changed follower bytes, or a changed
   `real_setup.json`. It calls `scripts/update_sim_calibration.py` with the
   report's follower file and ID, then writes
   `docs/reports/calibration-locks/<YYYYMMDD>-<follower_id>.json`. That audit records
   calibration/setup hashes, report hash, pose-set hash and current Git identity.
   It prints the exact `git add`, `git commit` and `git tag calib-<YYYYMMDD>`
   commands for you to review and run. It does not execute them. An existing
   same-day lock record is never overwritten.

## Decision rules

For each body joint, the pose summary reports median **real − target**, spread
(max minus min), accepted distinct joint targets, verdict and reason:

| Verdict | Absolute median | Spread | Accepted poses |
| --- | --- | --- | --- |
| PASS | ≤1° | ≤1° | ≥5 |
| WARN | ≤2° | ≤1° | ≥5 |
| FAIL | Any other result | Any other result | Fewer than 5 is labelled INSUFFICIENT |

Overall LOCKABLE requires every body joint to be PASS or WARN. Gripper results
are separate and never block. Both `poses` and `report` accept
`--pose-pass-deg`, `--pose-warn-deg`, `--pose-spread-deg`, and `--pose-min-poses`.
Report inherits recorded thresholds unless explicitly overridden, and records
all values used. Visual matching accuracy depends on the operator/reference;
encoder agreement alone cannot establish the physical zero pose.

Tracking is labelled **NOT a calibration verdict**. On still frames, the report
shows mean ± population std and correlation with real joint angle for sim −
real, real − commanded, and sim − sim target. It includes still frames at limits
in those statistics and distributions, and reports the fraction at limits.
Real-tracking and sim-tracking PASS/WARN/FAIL verdicts use mean absolute gap on
still, unsaturated frames (default 1°/3°). NO_DATA never means PASS. Stillness
requires both real and sim velocities below 1°/s throughout a 0.25-second window.
Differences are linear and never wrapped; >180° gaps remain visible.

The recommended actions are deterministic:

- Pose FAIL: recalibrate that joint; INSUFFICIENT also calls out missing evidence.
- Mean absolute real − commanded >3°: real servo problem; inspect gains, load or blockage.
- |Mean sim − sim target| >1° and |angle correlation| >0.8: sim stiffness too low;
  inspect the named actuator group in `assets/so101.py`.
- More than 5% of frames at a runtime limit: limit mismatch.

The report includes input identities, Git state, embedded-calibration match,
rule thresholds and two PNG plots (pose offsets with ±1°/±2° bands, and tracking
gap distributions). If matplotlib is unavailable it explicitly records that
plots were skipped. Different follower/setup/mapping identities across the two
sessions are rejected rather than combined.

## Selection and recording compatibility

Simulation defaults to the locked `assets/real_setup.json` mapping.
`ROBOT_ID` and `TELEOP_ID` never select a simulation mapping. Only an explicit
`SO101_FOLLOWER_CALIBRATION` or a client's explicit calibration path changes it.
The embedded-calibration stale check still applies to the default.

Hardware CLI IDs take precedence over their environment defaults. They select
LeRobot files under `$HF_LEROBOT_CALIBRATION` (default
`~/.cache/huggingface/lerobot/calibration`), in `robots/so101_follower` and
`teleoperators/so101_leader`. No filename stem replaces a CLI ID. The explicit
`real_rollout --calibration_file` input is loaded in memory while retaining the
requested ID. Before connecting a follower, the client derives the mapping
from the file LeRobot loaded and compares it with the sim mapping. A mismatch
prints both source hashes and a per-joint min/max difference table and refuses
connection. Missing leader files only affect leader clients.

With no explicit calibration selection, `recording_contract()` retains the
exact pre-`0a32271` contract/hash, even with environment IDs set. Explicit
selection includes calibration hashes; changing them requires a new dataset
root. Neither reporting nor locking rewrites recorded datasets.

## Preview evidence — 2026-10-29

The committed [contact sheet](reports/2026-10-29-calibration-preview/contact-sheet.png)
and [PhysX clearance evidence](reports/2026-10-29-calibration-preview/preview.json)
contain eight OK poses. Minimum nonexempt table clearance is 57.9 mm and minimum
nonexempt self-clearance is 25.7 mm. Body-joint range coverage is 62.5–67.3%.
The wrist/jaw interface and base/shoulder mounting exemptions were explicitly
approved by the operator; the measured shoulder-to-table gap is 18.6 mm.

## Development verification

The implementation is tested with synthetic logs and fake motors, plus a
hardware-free Isaac preview. The work report at
`outputs/calibration_report_work_20261029.md` records commands, commits, baseline
and final checks. No physical calibration verdict is claimed until the operator
captures the independent poses and gap session.

Verified on 2026-10-29 in the existing teleop image's CPU environment:

| Check | origin/main baseline | Final |
| --- | --- | --- |
| Full CPU suite | 856 passed, 27 skipped | 881 passed, 27 skipped |
| Subtests | 281 passed | 281 passed |
| Flake8 (repository config, 17 changed Python files) | — | Passed |
| Isaac reference preview | — | 8/8 OK, approved exemptions recorded |

The exact suite invocation is `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests -q`.
The image bootstrap supplies its existing SDK dependencies to child processes;
CPU test containers have no GPU, serial devices, or network. A writable temporary
config location is required by the existing eval-config test. The host default
Python lacks pytest, and system Python lacks optional test dependencies.
A synthetic end-to-end lock test ran the actual updater in a temporary repository;
the real setup and follower/leader files retained their original hashes.
