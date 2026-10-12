# Policies

Each folder here is one servable policy. Adding a policy means adding a folder; nothing else is edited.

```
policies/<name>/
├── policy.toml       manifest: name, python version, cameras, horizon, runtime env
├── pyproject.toml    model dependencies (omit if policy.toml sets environment.git + rev)
├── uv.lock           committed lockfile
├── client.py         runs where the robot / sim runs. No torch. Builds requests, parses responses.
└── server.py         runs inside this policy's venv in the container. Loads and runs the model.
```

## Add a policy

1. `new_policy <name>` copies `_template/` to `policies/<name>/`.
2. Fill `policy.toml`, list dependencies in `pyproject.toml`, run `uv lock` there, commit `uv.lock`.
3. Implement the functions in `client.py` and `server.py` (see `docs/07-policy-plugins.md`).
4. Check it: `pytest tests/policies/test_conformance.py -k <name>`.
5. Serve and evaluate with `--policy <name>`.

Folders starting with `_` and manifests with `enabled = false` are skipped.
