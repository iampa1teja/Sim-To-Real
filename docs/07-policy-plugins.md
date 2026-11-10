# 07 — Policy plugins

A policy lives entirely in `policies/<name>/`. Its client runs beside the robot/simulator and its server runs in
an isolated policy environment. Add the folder, declare its contract, then run `s2r check`; no `src/` edit is needed.
This guide builds a complete echo policy that repeats the current raw joint state. It tests wiring and preserves
the selected arm's order/units; it does not perform a task.

## Create the folder

From the repository root, with the [host environment](01-getting-started.md) available:

```bash
./s2r new policy echo_policy
```

Replace the following files. The scaffold's unused `train/` stubs can remain: omitting `[train]` below makes this
a serving-only policy. The name must match the folder and use lowercase letters, digits and underscores.

`policies/echo_policy/policy.toml`:

```toml
[policy]
name = "echo_policy"
description = "Repeats raw joint state for client/server wiring checks."
enabled = true
default_port = 5610

[environment]
python = "3.12"

[observation]
cameras = ["room", "wrist"]
language = true

[action]
dim = "arm"
default_horizon = 4
representation = "absolute_joint"
```

`dim = "arm"` accepts the selected arm's joint count. Declare a positive integer such as `dim = 6` for a fixed
embodiment; incompatible arms are rejected before inference. This count check does not establish semantic joint
or unit compatibility with a model. Folder names starting with `_` and `enabled = false` plugins are skipped by
normal discovery.

`policies/echo_policy/pyproject.toml`:

```toml
[project]
name = "policy-echo-policy"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy", "pyzmq", "msgpack"]
```

The shared wire/server needs NumPy, pyzmq and msgpack. Pin the resolved environment with `uv.lock`:

```bash
(cd policies/echo_policy && uv lock)
```

Commit that lockfile with the plugin. The container uses `uv sync --frozen`. Alternatively, set
`[environment].git` and `.rev` to an upstream project and a full 40-character commit SHA; that checkout's
`pyproject.toml`/lockfile then own the environment. The GR00T plugin demonstrates this alternative.

## Implement the client

`policies/echo_policy/client.py`:

```python
import numpy as np

from core.policy_client.contract import ArmInfo, ClientPolicy
from core.policy_client.default import DefaultClientPolicy
from core.policy_client.discovery import PolicyManifest


class EchoClient(DefaultClientPolicy):
    def configure(self, arm: ArmInfo) -> None:
        self.arm = arm

    def build_request(self, joint_positions, images, instruction):
        state = np.asarray(joint_positions, dtype=np.float32)
        if state.shape != (len(self.arm.joint_names),) or not np.isfinite(state).all():
            raise ValueError("Expected one finite value per configured joint")
        prefix = self.options.get("instruction_prefix", "")
        request = super().build_request(state, images, prefix + instruction)
        request["joint_names"] = list(self.arm.joint_names)
        return request

    def parse_response(self, response, joint_order):
        if tuple(joint_order) != self.arm.feature_names:
            raise ValueError("Joint order changed after configure")
        actions = super().parse_response(response, joint_order)
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned nonfinite actions")
        return actions


def create(manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy:
    return EchoClient.from_manifest(manifest, options)
```

`create(manifest, options)` must return a `ClientPolicy`. The inherited `from_manifest` classmethod stores
manifest camera names/language behavior and the string-valued client options. `DefaultClientPolicy.reset()`
has no history to clear; override it if the client adds episode state.

`configure(ArmInfo)` is called before inference. It supplies `joint_names`, `roles` (`arm`/`gripper`),
`types` (`revolute`/`prismatic`) and complete `joint_mapping`. `feature_names` appends `.pos` in the same order;
`gripper_indices`/`arm_indices` select roles without assuming a motor name or position. A custom model client
can implement `ClientPolicy` directly with these signatures:

```python
@classmethod
def from_manifest(cls, manifest: PolicyManifest, options: dict[str, str]) -> ClientPolicy: ...
def configure(self, arm: ArmInfo) -> None: ...  # optional no-op in the base class
# joint_positions: (J,), images: camera -> uint8 (H,W,3)
def build_request(self, joint_positions, images, instruction) -> dict: ...
def parse_response(self, response: dict, joint_order: list[str]) -> np.ndarray: ...
def reset(self) -> None: ...
```

Inputs and returned absolute actions are **raw device/dataset units**, not simulator radians. Normalized body
joints conventionally use −100…100 and grippers 0…100. Degree-mode bodies use the driver's calibrated encoder
coordinate, which can differ from physical degree zero; prismatic physical spans use metres. Mapping entries
carry `joint_min`/`joint_max` and optional `raw_min`/`raw_max`, including reversed endpoints. Keep model-specific
normalization and embodiment conventions in this plugin. The simulator's `JointMapping` handles the physical
boundary; the echo policy leaves all values untouched.

The standard request contains `state: float32 (J,)`, `images: {camera: uint8 (H,W,3)}` and `instruction: str`;
this example adds `joint_names`. Standard replies contain `actions: float32 (H,J)`. The client and server jointly
own any additions. Transport encodes NumPy arrays as embedded `.npy` bytes inside msgpack; avoid object arrays
and Python model objects. Clients must import without Torch, Isaac, Kit or USD.

## Implement the server

`policies/echo_policy/server.py`:

```python
import numpy as np

from policy_server.contract import ServerPolicy


class EchoServer(ServerPolicy):
    def __init__(self):
        self.horizon = 4
        self.calls = 0

    def load(self, checkpoint: str, device: str, options: dict[str, str]) -> None:
        # This example needs no weights. Real plugins load heavy libraries here.
        self.horizon = int(options.get("horizon", "4"))
        if self.horizon < 1:
            raise ValueError("horizon must be positive")

    def reset(self, seed: int | None = None) -> dict:
        self.calls = 0
        return {"seed": seed}

    def infer(self, request: dict) -> dict:
        state = np.asarray(request["state"], dtype=np.float32)
        if state.ndim != 1 or len(request["joint_names"]) != len(state):
            raise ValueError("State and joint_names disagree")
        if not np.isfinite(state).all():
            raise ValueError("State must be finite")
        self.calls += 1
        return {"actions": np.repeat(state[None], self.horizon, axis=0)}

    def info(self) -> dict:
        return {"name": "echo_policy", "horizon": self.horizon, "calls": self.calls}


def create() -> ServerPolicy:
    return EchoServer()
```

The server loader calls `create()` without arguments and then
`load(checkpoint: str, device: str, options: dict[str, str])`. `reset(seed=None)` clears per-episode state;
`infer(request)` returns the plugin's reply; `info()` returns a dict whose `name` matches the manifest. Keep
`info()` usable before loading a model so static/conformance checks can inspect it. The server container exposes
the shared package as `policy_server`, hence its import above; client imports use `core.policy_client`.

Server `--option KEY=VALUE` and evaluator/rollout `--policy_option KEY=VALUE` configure different processes.
Only pass a setting to both when your plugin explicitly needs both. Do not assume a client option is forwarded
to the server or that `ArmInfo` automatically crosses the wire.

## Check the folder on CPU

```bash
./s2r check --static --arm so101 --policy echo_policy
PYTHONPATH=src uv run --group host python -m pytest \
  tests/policies/test_conformance.py -k echo_policy -q
```

Static checks parse manifests and Python files without importing model code. The conformance test loads the
client/server factories without executing `server.load()`. Neither check establishes inference correctness. The following complete CPU check exercises both real plugin
factories, the selected-arm contract, cameras, options and the actual wire encoding. It uses a synthetic arm with
mixed joint types and two gripper axes, so it needs no hardware/calibration file:

```bash
PYTHONPATH=src:src/core uv run --group host python - <<'PY'
from pathlib import Path
import numpy as np
from core.policy_client.contract import ArmInfo, load_client_policy
from core.policy_client.discovery import load_manifest
from core.policy_client.transport import MsgSerializer
from policy_server.contract import load_server_policy

folder = Path("policies/echo_policy")
manifest = load_manifest(folder)
names = ("slide", "yaw", "left_finger", "right_finger")
arm = ArmInfo("four_axis", names, ("arm", "arm", "gripper", "gripper"),
              ("prismatic", "revolute", "prismatic", "prismatic"),
              {name: {"joint_min": 0.0, "joint_max": 1.0} for name in names})
client = load_client_policy("echo_policy", manifest, {"instruction_prefix": "Check: "})
client.configure(arm)
server = load_server_policy(folder)
server.load("", "cpu", {"horizon": "4"})
server.reset(seed=1984)
state = np.array([25, -12, 40, 40], dtype=np.float32)
images = {name: np.zeros((2, 3, 3), dtype=np.uint8) for name in manifest.cameras}
request = MsgSerializer.from_bytes(MsgSerializer.to_bytes(client.build_request(state, images, "hold")))
assert request["instruction"] == "Check: hold"
reply = MsgSerializer.from_bytes(MsgSerializer.to_bytes(server.infer(request)))
actions = client.parse_response(reply, list(arm.feature_names))
np.testing.assert_array_equal(actions, np.repeat(state[None], 4, axis=0))
assert server.info()["name"] == "echo_policy"
client.reset()
print("PASS: configured four-joint client/server wire round trip")
PY
```

No model or simulator is launched by these CPU checks. The generic conformance test checks every enabled plugin
with a compatible installed arm. A fixed embodiment needs that arm installed and a calibration unit when the arm
requires per-unit calibration. Add plugin-specific tests for semantic joint/normalization conventions.

## Serve and connect

Build the serving image if needed. The current `s2r serve` Docker planner requests NVIDIA GPU access even though
the echo server itself performs only NumPy work. The command below is therefore a runtime/GPU step, separate
from the CPU checks:

```bash
./s2r build serve --tag local
mkdir -p models
./s2r serve --policy echo_policy --models-dir "$PWD/models" --checkpoint unused \
  --image s2r-serve:local --port 5610 --option horizon=4 --foreground
```

`--checkpoint` is required by the launcher; this example ignores its value and needs no file at `models/unused`.
Ctrl-C stops this foreground server. For a short wiring evaluation, create
`policies/echo_policy/eval.yaml` (replace `YOUR_UNIT` with the arm's calibrated unit ID):

```yaml
arm: so101
scene: blank_table
task: pick_place_cube
variant: eval
unit: YOUR_UNIT
policy:
  name: echo_policy
  port: 5610
  options:
    instruction_prefix: "Check: "
eval:
  episodes: 1
  action_horizon: 4
  episode_length_s: 1
  rename_map:
    wrist_cam: wrist
run:
  gui: false
  rerun: false
```

In another terminal with `s2r-sim` running:

```bash
./s2r eval --config_path policies/echo_policy/eval.yaml
```

For `rpd_meeting_room`, change `scene` and add `realsense_rgb: room` under `eval.rename_map`.
The manifest names policy inputs, while the scene
and arm name physical cameras. An unmapped scene camera keeps its name; every required policy camera must
arrive. The echo policy holds state, so a task failure is expected in this wiring check. Use
[evaluation](06-evaluation.md) for a fixed benchmark and [training](05-training.md) for learned checkpoints.

## Training commands

To make a policy trainable, add actual scripts in its folder and declare them:

```toml
[train]
dataset_format = "lerobot_v2.1"
extras = []

[train.commands]
prepare = "train/prepare.sh"
train = "train/train.sh"
verify = "train/verify.sh"
```

`train` is required once `[train]` exists; `prepare`/`verify` are optional. A `.sh` script runs with Bash and a
`.py` script with the policy Python. `s2r prepare|train|verify --policy <name> ... -- <script flags>` forwards the
trailing flags unchanged. The container exports `S2R_POLICY_DIR`, `S2R_DATASET`, `S2R_OUTPUT`, `S2R_BASE_MODEL`,
`S2R_PYTHON` and `S2R_PROJECT_DIR`; use these paths rather than host paths inside the script. Training-only
dependencies belong in `train/requirements.txt` or declared extras. The manifest supports `lerobot_v2.1` and
`lerobot_v3`; your preparer must actually produce its declared format. See [training](05-training.md) for the
current tested SO-101 pick-place recipe and read-only dataset/base-model mounts.
