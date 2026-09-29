# Troubleshooting and open questions

[Guide hub](../README.md)

This guide combines this fork's error paths with the
[NVIDIA troubleshooting guide](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/troubleshooting.html).
Course advice applies to its own commands; use this fork's checked flag spelling
and dataset rules below. Preserve recordings when investigating failures.

## Serial ports and calibration

“Physical leader serial port … does not exist” is a preflight failure, before
motion. Rediscover the port inside teleop, verify device mounts and use the
correct leader/follower IDs. No automatic port swapping is performed. “Requires
read/write permission” needs host device permissions; reconnects may change them.
Use the [Setup](01-setup.md) discovery/group commands. Two recorder processes
must not share the same serial device.

Missing all motors suggests missing motor power, even if USB enumerates. A single
missing motor suggests cabling/communication; check the chain and power. Follow
calibration prompts with unobstructed mechanical travel, not cable-limited stops.
Wrong ranges or IDs can make commanded angles wrong. The baseline checker is a
warning aid, not proof that a calibration is appropriate for your arm.

## Camera problems

Rediscover OpenCV camera indices after reconnecting hardware. Verify the saved
test images before assigning wrist/external. Check lens cover/focus, USB access,
resolution and a competing process holding the device. The physical eval adapter
requires `front` and `wrist`; the recorder's physical camera names are `external`
and `gripper`; MyRoom sim names are `realsense_rgb` and `wrist_cam`. Use the
appropriate config/mapping, not a guessed universal rename.

No real GUI panes in a sim-only run is expected. The real follower must be enabled
to open its configured cameras. Re-aiming a camera needs a new scene/dataset
version even when schema validation passes.

## X11, GUI and render lag

Check host `DISPLAY`, X authorization, GPU access and the launch mounts in
[Setup](01-setup.md). `list_envs` opens a simulator window and has no headless CLI.
The copied workshop teleop command uses host networking and Xauthority; unlike
the real-robot command, it does not mount `/tmp/.X11-unix`. Whether that recipe
works with your X server/socket configuration needs local validation.

The recorder browser is separate from Isaac's viewport. If it is blank, inspect
browser network/console errors: React and Babel are fetched from a CDN. Keep
`--gui_host` on localhost. If its port is occupied, use the existing `--gui_port`
option. Lowering `--gui_fps` reduces preview frequency, not dataset timing.

Keep Fabric enabled. Prior local alignment checks found stale wrist-camera poses
with `--disable_fabric`. Lower visual load before changing control FPS; saved FPS
and image timing must remain consistent. The recorder requires FPS to divide the
physics rate; heavy rendering may make wall-clock collection slower than its
nominal simulation timeline. Validate the recorded videos, not only the viewport.

## Memory and model loading

Training OOM: reduce `--global_batch_size`, keep it divisible by GPU count and
avoid concurrent Isaac/training jobs on the same GPU. Accumulation is applied
separately in the pinned implementation; it does not automatically shrink the
per-device batch. See [Training](05-training.md). There is no measured promise
that the example fits every 24 GB GPU. Host RAM/data-loader pressure is distinct
from GPU memory exhaustion.

A server that exits before becoming ready needs its model/GPU/library logs checked.
Ensure the model directory contains config and weights, and that the matching
Ada/Blackwell image was built. The Blackwell image's nightly dependencies are
not a completely reproducible software pin even though GR00T itself is pinned.

## Hugging Face authentication and rate limits

Authenticate in the environment doing the download or upload; do not paste a
token into a tracked file. Example placeholder is `<your_hf_token>`, never a real
credential. Authentication and permissions are separate from network availability.

**Training machine**, activated training environment:

```bash
hf auth login
```

The declared `lerobot_push_dataset` console command has a missing callable;
[Training](05-training.md#optional-hub-upload) provides the existing module
invocation. Its caught initialization errors do not produce a reliable nonzero
exit status, so read the output and verify upload completion.

## Resume refusal and empty datasets

Recorder resume checks FPS, feature names/shapes, frame and episode counts,
sim/real alignment, and sidecar timelines. Use the original recording
configuration; do not edit metadata to bypass those checks. A changed scene or
camera schema needs a new dataset root. A nonempty directory without LeRobot
metadata is refused. A dataset created but exited before saving any episode may
fail to reopen because metadata exists without frame/episode parquet; preserve
it and use a fresh root rather than deleting a potentially valuable dataset.

Preparation resume is different: `--resume` retries `<output>.building`. Source
hashes must match and exclusions must be identical (order/duplicates normalize).
An existing completed output is never overwritten. Use a different output name
for a different exclusion set. Preparation checks only the pinned GR00T HEAD;
it does not attest that the checkout has no local code modifications.

## Unencoded episodes and encoding failures

Saved parquet/sidecars do not imply videos are ready. In the PickPlace GUI, use
Encode videos or Encode & Exit and wait for pending count zero. Exit without
encoding deliberately leaves recoverable PNGs. Relaunch the same dataset to
recover pending saved episodes; recovery requires a contiguous suffix and exactly
the expected PNG count for every camera. “Published video is missing”, “Cannot
recover episode … expected … PNGs” and “Unencoded episodes must be a contiguous
suffix” indicate integrity problems, not a request to silently regenerate data.

Encoding runs in a child process with staging and an encoder log. Check disk
space and the reported encoder error. Pending images are retained after failure;
retry Encode videos after resolving the cause. Do not delete pending PNGs, move
individual episode files, or train on a partly encoded root. Unsaved interrupted
frames are not reconstructed into a valid demonstration automatically.

## Policy server port and evaluation starts

“Port … is unavailable” comes from host TCP preflight. Stop the intended existing
listener or pass another `--port` to the host runner; it forwards that port to
server/client. A manually launched client uses `--policy_port`, not the server's
spelling. Do not start both an independent server and the host runner on 5555.

“Recorded starts directory missing” requires the sim dataset's `pick_place_meta`
directory. “Cube starts and colour instructions require a pick-place Eval task”
means the direct client was launched with a different/default task. Missing/invalid
sidecars, overlap with the current box or impossible random sampling are explicit
failures. Recheck scene alignment rather than moving a recorded start silently.

## Open questions and code findings

These were found while documenting; no code fixes are bundled with these docs.

| Finding | Consequence / unresolved work |
| --- | --- |
| `lerobot_push_dataset:main` is declared, but no `main` exists | Broken console entry point; use the existing module path pending a code fix |
| `list_envs` launches at import and only closes the app in its module `__main__` block | Console invocation may leave Isaac running until manually closed; no headless parser |
| `so101_eval.py` defines `timeout` but never uses it | Do not promise a 60-second bounded run; stop with Ctrl+C |
| Real eval plot docstring says JSON + PNG, but only PNG is written | No real result/trajectory JSON or success-rate output exists there |
| `SO101Control` hard-codes initial/home poses; sim eval has a ten-step hard-coded initial action | Must be checked/adapted for other setups, not described as calibrated universal poses |
| Sidecars omit contact, box transform and success labels | Exact offline success scoring cannot be reconstructed from them alone |
| Newly created zero-episode datasets enter resume loader path | Empty-dataset relaunch is not guaranteed; needs a dedicated implementation/test |
| Generic task registration is insufficient for recorded-start eval | Recorder, direct evaluator and host script have fixed PickPlace names/IDs |
| `env setup/` and `visual_matching/` mentioned by older docs are ignored/untracked | Fresh clones lack the room builder and calibration workflow; tracked reproducible replacements are needed |
| GR00T training env and converter env differ | A single documented, tested dependency recipe covering preparation and training remains to be validated per platform |
| FinetuneConfig describes effective batch, but Trainer computes per-device batch before accumulation | Use implementation semantics described in the training guide |
| No fork image registry tag, no 24 GB pick-place timing benchmark | Build locally; hardware capacity/driver/throughput claims need measurement |
| No tracked paired-data merger or colour replay writer | Sim+real training/colour-regenerated datasets require additional work |
| Host evaluator scrapes `xhost +` from the hub README | Its launch block must remain present until the script is decoupled from docs |
| Historical Docker notes used obsolete paths and unsupported visualization flags | New commands use the tracked `docker/real/scripts` files and actual config fields |
| Browser GUI needs CDN assets; teleop X11 recipe lacks socket mount | Offline browser and X-server compatibility need environment-specific validation |

Use the [evaluation validation record](../docker/eval_pick_place_validation.md)
for CPU checks versus outstanding GPU checks. Documentation validation confirms
links/CLI spelling; it does not calibrate hardware, test GPU rendering, establish
training convergence, or certify a real-robot rollout.
