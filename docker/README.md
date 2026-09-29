# Docker entry point

[Guide hub](../README.md)

The maintained container guide is [01 — Setup](../docs/01-setup.md). It includes
both image builds, exact launch mounts, additional shells, device discovery,
calibration, persistent `docker/env` settings and editable extension installation.

- [Simulation and real-robot launch commands](../docs/01-setup.md#build-and-start)
- [Calibration and manual control](../docs/01-setup.md#calibration-diagnostics-and-manual-control)
- [Host evaluation runner and every flag](../docs/06-evaluation.md#host-runner)
- [Real policy server and rollout](../docs/06-evaluation.md#real-robot-rollout)
- [Validation history and GPU checks](eval_pick_place_validation.md)
- [Troubleshooting](../docs/troubleshooting.md)

The teleop launch block also remains in the hub because `eval_pick_place.sh`
extracts it for its missing-container diagnostic. Use the checked paths in the
new guides rather than the former `real_robot/` or `lerobot_so101_teleop/` paths.
