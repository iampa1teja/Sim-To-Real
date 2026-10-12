# Policy server

One container serves any policy found in `../policies/`. The image holds only uv, CUDA, system
libraries and the shared server core. Each policy's Python environment is built on its first start
and kept on a named volume.

## Build

```bash
./policy_servers/build.sh
```

## Run

```bash
docker run --rm --gpus all --network host \
  -v "$PWD/policies:/policies" \
  -v policy-cache:/opt/cache \
  -v "$HOME/models:/workspace/models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  policy-server:latest \
  --policy groot_n17 --checkpoint /workspace/models/<checkpoint> --port 5560 \
  --option embodiment_tag=NEW_EMBODIMENT
```

- `--list` (the default command) prints the policy folders the container can see.
- `--policy <name>` selects `policies/<name>/`. Everything after it goes to `server_core.launcher`:
  `--checkpoint`, `--port`, `--host`, `--device`, and repeatable `--option key=value` for
  policy-specific settings.
- The first start of a policy runs `uv sync` and can take minutes. Later starts reuse
  `/opt/cache/venvs/<name>`. Keep the `policy-cache` volume to keep them.
- `/policies` is mounted read-write so uv can write a missing `uv.lock`. Commit that file.
- During development also mount `policy_servers/core` at `/opt/so101/server_core` and
  `source/sim_to_real_so101/policy_client` at `/opt/so101/policy_client` to skip rebuilds.

## Cleaning up

```bash
docker volume rm policy-cache     # drops every policy venv, interpreter and checkout
```
