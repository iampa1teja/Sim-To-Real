# Getting started

Run commands from the repository root. `./s2r` is the supported program entry point;
it prints the command it runs. Put `--dry-run` before the command to inspect a plan
without starting Docker or changing a dataset.

## Prepare the host and containers

Use a Linux host with Python 3.11 or newer, `uv`, Docker Compose, and an NVIDIA GPU
that the configured Docker runtime can expose. The simulator image pins its Isaac
Lab/Newton and LeRobot dependencies; do not install another Isaac package into the
host environment. Image builds and policy dependency/model installation need their
respective upstream downloads and licenses. The initial build is much larger than
a normal application image.

```bash
uv sync --group host
cp -n .env.example .env
./s2r --help
./s2r check --static
./s2r build sim
docker compose up -d sim
./s2r envs
./s2r check --arm so101 --scene blank_table --task pick_place_cube --unit rpd_robot
```

The static check validates manifests and referenced files without Isaac. The full
check additionally starts a headless environment, resets and steps it, resolves
joint/body names, and renders its RGB cameras. It does not certify grasping,
learned policy quality, or physical robot calibration. `blank_table` has no SO-101
mount file: it uses the scene anchor and the arm's default joint positions.

For only the host launcher and static checks, a smaller environment is sufficient:
`uv venv .venv` followed by `uv pip install --python .venv/bin/python numpy`.
Use `uv sync --group host` when you also need the host analysis/test libraries.

`./s2r build serve` and `./s2r build train` build the policy images when needed.
`./s2r pull sim --registry <namespace> --tag <tag>` is available when your team has
already published an appropriately licensed image. Match `S2R_TAG` in Compose to
the tag you built or pulled.

## Paths and selectors

| Host path / setting | Simulator container |
| --- | --- |
| Repository root | `/workspace/s2r` |
| `./datasets` (default `S2R_DATASETS`) | `/workspace/datasets` |
| `./outputs` | `/workspace/s2r/outputs` |
| `configs/...yaml` | Relative to `/workspace/s2r` |

Arguments to record, eval, datagen, eval-set and calibration-replay are paths
**inside the simulator container**. Arguments such as `s2r train --dataset` and
`s2r serve --models-dir` name **host** paths that the CLI mounts in a separate
policy container. See [training](05-training.md) for those mount names.

Every run chooses `arm`, `scene`, `task`, and `variant`. Explicit flags override
YAML values; YAML values override `S2R_ARM`, `S2R_SCENE`, and `S2R_TASK` defaults.
Changing `.env` does not rewrite explicit selectors in a YAML. A per-unit arm also
needs its mapping unit: `unit: rpd_robot` for generation/evaluation, or
`follower.unit: rpd_robot` for recording. These checked-in units describe the
measured project robots, not an arbitrary newly connected SO-101.

## First dataset without hardware

The scripted expert needs no leader, follower, policy server, or trained model.
This command uses the measured meeting-room composition and a demonstrated seed:

```bash
./s2r datagen --config_path configs/datagen/scripted.yaml \
  --seed 101 --attempts 1 --episodes 1 \
  --destination /workspace/datasets/first_pick_place \
  --report /workspace/s2r/outputs/first_pick_place.json
```

Use a new destination each time. A successful run writes a LeRobot dataset with
parquet actions/states, both RGB videos, named-object `task_meta/` trajectories,
and `datagen_meta/` annotations. The report must have `complete: true`, the
requested accepted count, and `final_success: true` for every saved episode.
An exhausted attempt budget can return failure with no saved episodes; inspect
its recorded failure reasons. Passing one seed is not a distribution-wide success
claim. [Data generation](08-data-generation.md) covers stacking and Mimic.

## Teleoperation and paired recording

`configs/record/sim_only.yaml` disables the real follower but still uses a physical
leader. It is not the hardware-free path above. Set the leader port/id in `.env`,
choose a mapping matching your device, and enable the hardware/display overlay:

```bash
docker compose -f compose.yaml -f compose.hardware.yaml up -d sim
./s2r --dry-run record --config_path configs/record/sim_only.yaml
./s2r record --config_path configs/record/sim_only.yaml
```

Before recording, copy the YAML to your run config and set `dataset.repo_id`,
`dataset.root`, and `dataset.task_name`; their template values are null. Configure
the terminal controls and display/GUI mode for your session. The browser GUI is
not covered by these guides. For paired recording use the paired-real config and
configure the follower, camera devices, raw LeRobot device calibration, and unit
mapping together. Do not assume the example USB paths or camera indices identify
your devices. [Adding an arm](02-add-an-arm.md) explains calibration boundaries.

## Continue

- [Add an arm](02-add-an-arm.md), [scene](03-add-a-scene.md), or [task](04-add-a-task.md).
- [Prepare and train](05-training.md), then [evaluate on a fixed benchmark](06-evaluation.md).
- [Add a policy plugin](07-policy-plugins.md) or [generate more data](08-data-generation.md).
- [Verification status](restructure-status.md) distinguishes completed checks from limitations.

For the full CPU suite, use the simulator image's installed SDK dependencies:

```bash
docker exec -w /workspace/s2r -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=src \
  s2r-sim s2r-entrypoint python -m pytest tests -q
```

The host environment supports static tools and CPU libraries; it does not contain
all SDK modules used by the complete test suite. Stop the idle container with
`docker compose stop sim` when finished. Dataset/output bind mounts persist.
