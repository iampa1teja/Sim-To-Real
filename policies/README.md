# Policies

Each folder here is one servable policy. Adding a policy means adding a folder; nothing else is edited.

```
policies/<name>/
├── policy.toml       manifest: name, python version, cameras, horizon, runtime env
├── pyproject.toml    model dependencies (omit if policy.toml sets environment.git + rev)
├── uv.lock           committed lockfile
├── client.py         runs where the robot / sim runs. No torch. Builds requests, parses responses.
├── server.py         runs inside this policy's venv in the container. Loads and runs the model.
└── train/            optional: the scripts named in policy.toml [train.commands]
```

## Training a policy

A policy that can be trained declares it in `policy.toml`; nothing outside its folder is edited.

```toml
[train]
dataset_format = "lerobot_v2.1"     # lerobot_v2.1 or lerobot_v3: what `prepare` hands to `train`
extras = []                         # optional dependency groups from pyproject.toml

[train.commands]
prepare = "train/prepare.sh"        # optional
train = "train/train.sh"            # required once [train] exists
verify = "train/verify.sh"          # optional
```

Each command is a script path inside the policy folder. The train container runs it with the repo
mounted and these variables set: `SO101_POLICY_DIR`, `SO101_DATASET`, `SO101_OUTPUT`,
`SO101_BASE_MODEL`. Arguments after the command name are passed to the script unchanged. A policy
with no `[train]` table can only be served. The template ships stub scripts that exit with an error
until you fill them in.

## Add a policy

1. `new_policy <name>` copies `_template/` to `policies/<name>/`.
2. Fill `policy.toml`, list dependencies in `pyproject.toml`, run `uv lock` there, commit `uv.lock`.
3. Implement the functions in `client.py` and `server.py` (see `docs/07-policy-plugins.md`), and fill the
   `train/` scripts or delete the `[train]` table.
4. Check it: `pytest tests/policies/test_conformance.py -k <name>`.
5. Serve and evaluate with `--policy <name>`.

Folders starting with `_` and manifests with `enabled = false` are skipped.
