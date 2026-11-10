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

## Arm and action units

The platform's typed `ArmSpec` defines joint order, USD names, roles, and physical units. Current client plugins
receive raw device-state vectors and return raw device-action vectors in that order. Normalized body joints use
`-100..100` and gripper joints `0..100`. With `use_degrees = true`, body values are the driver's calibrated
encoder-degree coordinate; its midpoint/scale can differ from the arm's physical frame. Grippers stay normalized.
Selected mappings retain explicit raw endpoints when needed, including reversed motor directions. Policy-specific
embodiment conventions remain the plugin's concern; clients must not treat raw degree codes as USD angles.

`JointMapping` owns the simulator boundary: calibrated spans map device units to physical degrees/metres, runtime
limits clamp targets, and revolute angles become radians for the articulation. Calibration units are selected by
recording `follower.unit` or evaluation/rollout/replay `--unit`; they live under
`arms/<arm>/calibration/units/`. Clients should keep camera and joint ordering explicit and preserve the action units
used to train their checkpoint. The standard client shares `DefaultClientPolicy` request/response handling.

Camera names come from the scene's fixed-camera declarations and the arm's mounted-camera declarations.
The meeting room retains `realsense_rgb` and `wrist_cam`; the plain table uses `room` and `wrist_cam`.
Configure the policy camera mapping for the scene used to record or evaluate its checkpoint.

Task stages and benchmark reporting are driven by the selected task plugin. Schema-2 datasets retain every
tracked object's poses in `task_meta/`; legacy `pick_place_meta/` datasets remain readable. Fixed eval sets carry
task/arm identity, joint order and their saved raw-unit mapping. Their final declared stage determines success.
The three approved frozen-set conversions retain `legacy_eval_set_id` for paired comparisons; geometry checks
still apply. See the phase status for hold-state smoke evidence, which does not establish learned-policy success.

Clients receive `configure(ArmInfo)` with the selected joint names, roles, types and raw-unit mapping.
The default/example client derives its action dimension from this metadata. A fixed-embodiment plugin declares
`[action].dim = 6` (as GR00T and Molmo do); incompatible arms are rejected before inference.
`s2r check --static --arm <arm> --policy <name>` checks that selected pairing without rejecting unrelated policies.

The Phase 4 serving image `s2r-serve:v2-phase4` is built, and the renamed evaluation entrypoint completed the
example hold-state policy's nine-episode schedule. Actual GR00T and Molmo serving/evaluation still await GPU
coordination; an existing host Molmo server is using GPU memory. These gates and the complete conformance
command remain pending. A completed hold-state schedule does not establish learned-policy task success.
The current GR00T and Molmo training recipes retain their SO101/pick-place assumptions;
a generic arm loader alone does not make those model recipes embodiment-independent. See
[phase status](../docs/restructure-status.md) for the acceptance gates still being verified.

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

Each command is a script path inside the policy folder (`.sh` runs with bash, `.py` with the
policy's Python). The train container runs it with the repo
mounted and these variables set: `S2R_POLICY_DIR`, `S2R_DATASET`, `S2R_OUTPUT`,
`S2R_BASE_MODEL`, `S2R_PYTHON` (the interpreter of the policy's separate training venv) and
`S2R_PROJECT_DIR`. Packages the policy needs only for training go in `train/requirements.txt`. Arguments after the command name are passed to the script unchanged. A policy
with no `[train]` table can only be served. The template ships stub scripts that exit with an error
until you fill them in.

## Add a policy

1. `s2r new policy <name>` copies `_template/` to `policies/<name>/`.
2. Fill `policy.toml`, list dependencies in `pyproject.toml`, run `uv lock` there, commit `uv.lock`.
3. Implement the functions in `client.py` and `server.py` (see `docs/07-policy-plugins.md`), and fill the
   `train/` scripts or delete the `[train]` table.
4. Check it: `pytest tests/policies/test_conformance.py -k <name>`.
5. Serve and evaluate with `--policy <name>`.

The template, example and GR00T clients share `core.policy_client.default.DefaultClientPolicy`;
subclass it when a policy uses the standard joint-state/image request and absolute-joint action response.
Model-specific request formats can still implement `ClientPolicy` directly.

Folders starting with `_` and manifests with `enabled = false` are skipped.
