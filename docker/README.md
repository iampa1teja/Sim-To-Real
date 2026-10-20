# Docker entry point

[Guide hub](../README.md)

The maintained container guide is [01 — Setup](../docs/01-setup.md). It includes
both image builds, exact launch mounts, additional shells, device discovery,
calibration, persistent `docker/env` settings and editable extension installation.

- [Simulation launch and policy-server image builds](../docs/01-setup.md#build-and-start)
- [Docker training and data preparation](../docs/05-training.md)
- [Calibration and manual control](../docs/01-setup.md#calibration-diagnostics-and-manual-control)
- [Host evaluation runner and every flag](../docs/06-evaluation.md#host-runner)
- [GR00T N1.7 milestone policy server](../docs/06-evaluation.md#gr00t-n17-200k-simulation-server)
- [Troubleshooting](../docs/troubleshooting.md)

The teleop launch block also remains in the hub because `eval.sh`
extracts it for its missing-container diagnostic. Use the checked paths in the
new guides rather than the former `real_robot/` or `lerobot_so101_teleop/` paths.
