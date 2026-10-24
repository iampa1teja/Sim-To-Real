# 07 - Policy plugins: what to learn and what to write

You write every `.py` file listed here. The non-Python files already exist: `policies/**/policy.toml`,
`policies/**/pyproject.toml`, `policy_servers/{Dockerfile,entrypoint.sh,build.sh,README.md}`.

## 1. What to learn first

| Topic | Why you need it | Where |
|---|---|---|
| uv: `uv sync --frozen`, `uv lock`, `uv run --project`, `UV_PROJECT_ENVIRONMENT`, `--python` | The container builds one venv per policy | docs.astral.sh/uv (Projects, Locking, Python versions) |
| `tomllib` | Reading `policy.toml` (stdlib, Python 3.11+) | Python docs |
| ZeroMQ REQ/REP, `RCVTIMEO`, `LINGER` | The wire transport | zguide.zeromq.org ch.1-2; pyzmq docs |
| msgpack and custom encode/decode hooks | Numpy arrays over the wire | msgpack-python docs; read `MsgSerializer` in `gr00t_client/server_client.py` |
| `importlib.util.spec_from_file_location` | Loading `client.py` / `server.py` from a folder path | Python docs, importlib |
| `abc.ABC` / `typing.Protocol` | The adapter contract | Python docs |
| `dataclasses` | Manifest object | Python docs |
| `argparse` (repeatable options via `action="append"`) | Launcher and CLI | Python docs |
| Docker: layers, `ENTRYPOINT` vs `CMD`, bind mounts vs named volumes, `--gpus all` | The container | docs.docker.com (Build, Storage) |
| NVIDIA Container Toolkit | GPU inside the container | NVIDIA docs |
| Your own code | You are moving it, not rewriting it | `gr00t_client/{server_client,smoothing,grasp}.py`, `GR00TRemotePolicy` in `utils/lerobot_interface.py`, `policies/groot_n17/train/launch_gr00t_server_n17.py`, `policies/groot_n17/train/gr00t_model_profiles.py` |
| GR00T N1.7 server and `Gr00tPolicy` API | `groot_n17/server.py` | Isaac-GR00T repo at the pinned rev: `gr00t/eval/run_gr00t_server.py` |
| MolmoAct2 | `molmoact2/{client,server}.py` | Its model card and repo: input format, image preprocessing, action space, normalization, chunk length, language handling |

## 2. Rules the design depends on

- `client.py` and everything under `policy_client/` must import without torch or any model library.
  They run in the Isaac container and the host env.
- `policy_client/transport.py` must use only relative imports. The server container imports the same
  file as top-level package `policy_client`; the teleop container imports it as
  `sim_to_real_so101.policy_client`.
- `server_core/discovery.py` and `policy_client/discovery.py` read manifests with stdlib only.
- The request format between a policy's `client.py` and `server.py` is private to that folder. The
  shared code only sees "an observation goes in, an `(H, J)` array of absolute joint targets comes out".
- Normalization happens inside the policy, never in shared code.

## 3. Python files to write

### `policy_servers/core/` (runs inside the container)

**`__init__.py`**: empty.

**`discovery.py`**
- Imports: `dataclasses`, `pathlib`, `tomllib`, `logging`, `re`, `os`.
- `@dataclass PolicyManifest`: `name, description, enabled, default_port, python, git, rev, cameras, language, default_horizon, representation, env, path, train`. `train` is `None` or a `TrainSpec` (`dataset_format`, `commands`, `extras`) read from the optional `[train]` table; see `policies/README.md`.
- `load_manifest(policy_dir) -> PolicyManifest`: parse and validate (name equals folder name, regex, required keys, SHA length). Raise `ValueError` naming the missing key.
- `discover(policies_dir=None) -> dict[str, PolicyManifest]`: scan `*/policy.toml`, skip `_*` and disabled, log and skip invalid ones. Default dir from `POLICIES_DIR`.

**`contract.py`**
- Imports: `abc`, `typing`, `numpy`.
- `class ServerPolicy(ABC)`: `load(self, checkpoint, device, options: dict)`, `reset(self, seed=None) -> dict`, `infer(self, request: dict) -> dict`, `info(self) -> dict` (manifest-like facts the policy knows at runtime, e.g. native chunk length).
- `load_server_policy(policy_dir) -> ServerPolicy`: import `server.py` by path (importlib) and call its `create()` factory; check the return is a `ServerPolicy`.

**`server.py`**
- Imports: `zmq`, `traceback`, `logging`, `dataclasses`, `typing`, and `MsgSerializer` from `policy_client.transport`.
- `class PolicyServer`: port of the existing server. `__init__(policy, host, port, api_token=None)`, `register_endpoint`, `run`, `_validate_token`, `_handle_ping`, `_kill_server`. Endpoints: `ping`, `kill`, `get_action` (calls `policy.infer`), `reset` (calls `policy.reset`), `describe` (calls `policy.info`), `seed_policy` (calls `policy.reset(seed)` and returns `{"seed": seed}`; keep this so benchmark seeding still works).

**`launcher.py`**
- Imports: `argparse`, `sys`, `logging`, `pathlib`, then `discovery.load_manifest`, `contract.load_server_policy`, `server.PolicyServer`.
- `build_parser()`: `--policy-dir` (required, set by entrypoint), `--checkpoint`, `--host` default `*`, `--port` default from manifest, `--device` default `cuda`, `--option KEY=VALUE` (`action="append"`), `--api-token`.
- `parse_options(pairs) -> dict[str, str]`.
- `main(argv=None)`: build parser, load manifest, create policy, `policy.load(...)`, wrap in `PolicyServer`, `run()`. Must be runnable as `python -m server_core.launcher`.

### `source/sim_to_real_so101/policy_client/` (replaces `gr00t_client/`)

**`__init__.py`**: keep light; no heavy imports.

**`transport.py`**
- Move `MsgSerializer`, `PolicyClient` and `BasePolicy`-free parts from `gr00t_client/server_client.py`. Drop the `PolicyServer` class (it now lives in `server_core`) and every `sim_to_real_so101.*` import.
- Imports: `io`, `msgpack`, `numpy`, `zmq`, `typing`.
- `class MsgSerializer`: `to_bytes`, `from_bytes`, `encode_custom_classes`, `decode_custom_classes`.
- `class PolicyClient`: `__init__(host, port, timeout_ms, api_token)`, `ping`, `kill_server`, `call_endpoint`, `get_action(observation, options)`, `reset(options)`, `describe()`, `seed(seed)`, `_init_socket`, `_reset_socket`.

**`types.py`**: `ActionChunk` dataclass (`actions: np.ndarray` of shape `(H, J)`, `info: dict`), nothing model-specific.

**`contract.py`**
- Imports: `abc`, `numpy`, `typing`.
- `class ClientPolicy(ABC)`: `from_manifest(cls, manifest, options)`, `build_request(self, joint_positions, images: dict[str, np.ndarray], instruction: str) -> dict`, `parse_response(self, response: dict, joint_order: list[str]) -> np.ndarray` (shape `(H, J)`, absolute joints), `reset(self)`.
- `load_client_policy(name, manifest, options) -> ClientPolicy`.

**`discovery.py`**: same API as the server one (`PolicyManifest`, `load_manifest`, `discover`), stdlib only. Find `policies/` via `POLICIES_DIR`, else walk up from this file to the repo root.

**`remote_policy.py`**
- Imports: `deque`, `numpy`, `torch` (guarded: the host env has it, Isaac has it), `datetime`, `smoothing`, `grasp`, `transport.PolicyClient`, `contract`.
- `class RemotePolicy`: the body of `GR00TRemotePolicy` with the GR00T parts removed. Same constructor arguments plus `policy_name`. Methods: `connect`, `reset`, `steps_until_query`, `close_timing`, `get_action`.
- It calls `client_policy.build_request` and `parse_response`. Everything else (queue, smoothing, grasp, seeding, timing) stays as it is.
- `_add_batch_time_dims`, `_sim_obs_to_groot_inputs` and `_decode_action_chunk` move into `policies/groot_n17/client.py`.

**`smoothing.py`, `grasp.py`**: move unchanged. Only fix the import lines (`sim_to_real_so101.gr00t_client.*` becomes relative or `sim_to_real_so101.policy_client.*`).

**`cli.py`**
- Imports: `argparse`, `discovery`.
- `add_policy_arguments(parser)`: `--policy` (choices from `discover()`), `--host`, `--port`, `--policy_option KEY=VALUE` (repeatable).
- `policy_from_args(args) -> tuple[PolicyManifest, dict]`.

### `policies/<name>/`

**`groot_n17/client.py`**
- Imports: `numpy`, `ClientPolicy` from `sim_to_real_so101.policy_client.contract`.
- `class Groot17Client(ClientPolicy)`: move `_add_batch_time_dims`, the GR00T key mapping from `_sim_obs_to_groot_inputs`, and `_decode_action_chunk` here. `build_request` returns `{"observation": ..., "options": None}`; `parse_response` returns the `(H, J)` array. Expose `create()`.

**`groot_n17/server.py`**
- Imports: `runpy`/`sys` or direct `gr00t` imports, plus `ServerPolicy`.
- `class Groot17Server(ServerPolicy)`: `load` reproduces what `policies/groot_n17/train/launch_gr00t_server_n17.py` does (revision check, offline Cosmos processor, then build `Gr00tPolicy` from `--checkpoint` and `options["embodiment_tag"]`). `infer` calls the policy's `get_action`. `reset` supports seeds.
- Copy `offline_cosmos_processor` from `policies/groot_n17/train/gr00t_offline_cosmos.py` and the pin from `policies/groot_n17/train/gr00t_model_profiles.py` into helper files in this folder. Do not import from `scripts/`: it is not mounted.
- `create()` factory.

**`molmoact2/client.py`, `server.py`**: same shape as GR00T. Everything specific (preprocessing, prompt format, de-normalization, chunk length, joint order, units) comes from the model card. If the model's action space is not absolute joints, convert in `parse_response`.

**`_template/client.py`, `_template/server.py`**: skeletons with every required function raising `NotImplementedError` and a docstring saying what it must return.

### Other Python files

**`source/sim_to_real_so101/scripts/new_policy.py`**: `main()` takes a name, validates it, copies `policies/_template/` to `policies/<name>/`, rewrites the name in `policy.toml` and `pyproject.toml`. Register it in `pyproject.toml` `[project.scripts]` as `new_policy`.

**Edits**: `utils/lerobot_interface.py` (use `RemotePolicy`), `scripts/lerobot_eval.py` (`--policy` via `cli.py`), `scripts/sweep_action_horizon.py`, `scripts/plot_grasp.py` (imports only).

**Tests**
- `tests/policy_client/test_transport.py`, `test_smoothing.py`, `test_grasp.py`: moved with new import paths.
- `tests/policies/test_conformance.py`: parametrized over `discover()`. For each policy check the manifest, import `client.py` with torch blocked, check `build_request` accepts a fake observation. Server side is skipped on the host when the model library is missing.
- `tests/policy_servers/test_launcher.py`: option parsing, bad manifest errors.

## 4. Evaluation: `docker/eval.sh <config>.yaml`

`docker/eval.sh` replaced `eval_pick_place.sh`. All settings come from a YAML file
(`configs/eval/pick_place.yaml`; `docker/eval_config.py` validates it). It starts `policy-server:latest`
with `--policy <name> --checkpoint ... --port ... --option k=v`, mounts `./policies` and a `policy-cache`
volume, and runs `lerobot_eval --policy <name> --policy_option k=v` in teleop. `seed_policy` is built
into `server_core`, so no separate benchmark server is needed. See [Evaluation](06-evaluation.md).

## 5. Order to build it in

1. `policy_client/{transport,smoothing,grasp}.py` moved; old tests green.
2. `server_core` with a dummy policy that returns zeros; ping and `get_action` round trip.
3. `groot_n17` client and server; compare actions against the old container on one recorded observation.
4. `RemotePolicy` and the `--policy` flag.
5. `molmoact2`.
6. `new_policy` and the conformance test.
7. The `eval.sh` change (done); then remove `gr00t_client/`, `docker/real/Dockerfile.n17`, `entrypoint.n17.sh`.
