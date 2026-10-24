# so101: one command for every policy

`./so101 <command>` is the front door for serving, training and evaluating any policy under `policies/`.
It needs Python 3.11 or newer on the machine where you type it, and Docker. Add `--dry-run` to any
command to see exactly what it would run.

| Command | What it does |
| --- | --- |
| `./so101 policies` | List policies, their default port, and the train commands they declare. |
| `./so101 prepare --policy P --dataset SRC --output OUT` | Convert a recorded dataset for the policy (`[train.commands].prepare`). |
| `./so101 train --policy P --dataset D --output OUT [--base-model M]` | Fine-tune (`[train.commands].train`). GR00T also takes `--recipe train\|sim\|failures\|dit`. |
| `./so101 verify --policy P --dataset D` | Check a prepared dataset before spending GPU hours. |
| `./so101 serve --policy P --checkpoint C [--models-dir DIR] [--port N] [--option k=v]` | Start the policy server container. |
| `./so101 eval <config>.yaml [flags]` | Run `docker/eval.sh` with everything after `eval` passed through. |
| `./so101 record [pick_place_agent flags]` | Record demonstrations: runs `pick_place_agent` in the local `teleop` container with `/root/env` loaded. |
| `./so101 rollout --policy P --lang_instruction "..." [--enable-motion]` | Run a served policy on the real arm with `real_rollout` in the `teleop` container. Shadow mode (no motion) unless `--enable-motion`. See [08 — Running a policy on the real arm](../docs/08-real-arm-rollout.md). |
| `./so101 tunnel --host user@box --port N` | Forward a remote server's port to this machine. |
| `./so101 sync --host user@box --remote-root /abs/path [--path datasets/x]` | Copy `policies/`, `configs/` and `eval_sets/` (plus any `--path`) to the remote repo copy. Never deletes. |

Arguments after `--` on prepare, train and verify go to the policy's script unchanged. For `eval`, `record` and
`rollout` every argument after the command goes to the wrapped program, so put `--dry-run` first to only print the
command: `./so101 --dry-run rollout --policy molmoact2 ...`. These three always run on the machine with the display
and the arm, never on a remote Docker host.

## How a policy plugs in

A policy declares its commands in `policy.toml` (see [policies/README.md](../policies/README.md)). Today the
GR00T and MolmoAct2 policies run through the wrappers that already own their containers
(`docker/train.py`, `docker/molmo_train.sh`); `cli/so101_cli/plan.py` maps the flags onto what those read.
Every other policy takes the generic path: a train container runs the declared script with `SO101_POLICY`,
`SO101_POLICY_DIR`, `SO101_DATASET`, `SO101_OUTPUT` and `SO101_BASE_MODEL` set. That path needs the
shared train image, which arrives with the image phase; until then pass `--image` to point it at one.

## Training or serving on another machine

Docker resolves bind mounts where the daemon runs, so a remote run uses paths on the remote machine.

```bash
./so101 sync --host gpu@box --remote-root /srv/so101 --path datasets/pick_place_v1
./so101 serve --policy molmoact2 --checkpoint molmoact2_so101 --models-dir /srv/models \
    --host gpu@box --remote-root /srv/so101
./so101 tunnel --host gpu@box --port 5556        # in another terminal
./so101 eval pick_place_molmo_so101.yaml --external_server
```

The simulator stays on the machine that has the display and the Isaac container; only the policy server
moves. `eval` itself cannot run on a remote Docker host, because it execs into the local `teleop` container.
GR00T and MolmoAct2 training need the repo and data at the same absolute paths, so run those on the GPU
machine itself.

## Layout

```
so101                     shell shim: puts cli/ on PYTHONPATH and runs python3 -m so101_cli
cli/so101_cli/plan.py     pure planners: each returns the exact command, never runs anything
cli/so101_cli/main.py     argument parsing, printing and running a plan
tests/test_so101_cli.py   exact-command tests for every subcommand, local and remote
```
