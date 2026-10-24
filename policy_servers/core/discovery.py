"""Read and validate policies/<name>/policy.toml inside the policy server container.

Stdlib only: this module is imported by the launcher before any policy venv exists.
Keep the API identical to source/sim_to_real_so101/policy_client/discovery.py.
"""
from dataclasses import dataclass
import logging
import os
import re
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

TRAIN_COMMANDS = ("prepare", "train", "verify")
DATASET_FORMATS = ("lerobot_v2.1", "lerobot_v3")


@dataclass(frozen=True)
class TrainSpec:
    """The optional [train] table: what the train container runs for this policy.

    commands maps a command name (prepare, train, verify) to a script path relative to the policy
    folder. extras names optional dependency groups the train container installs on top of the
    policy environment.
    """

    dataset_format: str
    commands: dict[str, str]
    extras: tuple[str, ...]


@dataclass(frozen=True)
class PolicyManifest:
    """Validated contents of one policies/<name>/policy.toml.

    Fields map to the manifest tables:
      [policy]       name, description, enabled, default_port
      [environment]  python, git, rev
      [observation]  cameras (tuple of str), language
      [action]       default_horizon, representation
      [runtime.env]  env (dict[str, str])
      [train]        train (TrainSpec, or None when the policy cannot be trained here)
    path is the policy folder the manifest was read from.
    """

    name: str
    description: str
    enabled: bool
    default_port: int
    python: str
    git: str
    rev: str
    cameras: tuple[str, ...]
    language: bool
    default_horizon: int
    representation: str
    env: dict[str, str]
    path: Path
    train: TrainSpec | None = None


def default_policies_dir() -> Path:
    configured_dir = os.environ.get("POLICIES_DIR")
    if configured_dir is not None:
        return Path(configured_dir)

    module_dir = Path(__file__).resolve().parent
    for parent in (module_dir, *module_dir.parents):
        policies_dir = parent / "policies"
        if policies_dir.is_dir():
            return policies_dir

    raise FileNotFoundError(
        f"Could not find a policies directory above {module_dir}; "
        "set POLICIES_DIR to its location"
    )


def _parse_train(table) -> TrainSpec | None:
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError("train must be a table")

    dataset_format = table.get("dataset_format", "lerobot_v2.1")
    if dataset_format not in DATASET_FORMATS:
        raise ValueError(f"train.dataset_format must be one of {', '.join(DATASET_FORMATS)}")

    extras = table.get("extras", [])
    if not isinstance(extras, list) or any(not isinstance(extra, str) or not extra for extra in extras):
        raise ValueError("train.extras must be a list of non-empty strings")

    commands = table.get("commands")
    if not isinstance(commands, dict) or "train" not in commands:
        raise ValueError("train.commands must be a table with at least a 'train' command")
    for command, script in commands.items():
        if command not in TRAIN_COMMANDS:
            raise ValueError(f"Unknown train command {command!r}; use {', '.join(TRAIN_COMMANDS)}")
        parts = script.replace("\\", "/").split("/") if isinstance(script, str) else None
        if not parts or script.startswith("/") or ".." in parts or "" in parts:
            raise ValueError(f"train.commands.{command} must be a relative path inside the policy folder")

    return TrainSpec(dataset_format=dataset_format, commands=dict(commands), extras=tuple(extras))


def load_manifest(policy_dir: str | Path) -> PolicyManifest:
    policy_dir = Path(policy_dir)
    manifest_path = policy_dir / "policy.toml"

    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found")

    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        name = data["policy"]["name"]
        enabled = data.get("policy", {}).get("enabled", True)
        description = data.get("policy", {}).get("description", "")
        default_port = data["policy"]["default_port"]
        python_version = data["environment"]["python"]
        git = data.get("environment", {}).get("git", "")
        rev = data.get("environment", {}).get("rev", "")
        cameras = data["observation"]["cameras"]
        language = data.get("observation", {}).get("language", False)
        default_horizon = data["action"]["default_horizon"]
        representation = data.get("action", {}).get("representation", "absolute_joint")
        runtime_env = data.get("runtime", {}).get("env", {})

    except KeyError as err:
        raise ValueError(f"Missing required key: {err.args[0]}") from err

    except tomllib.TOMLDecodeError as err:
        raise ValueError(f"Invalid TOML: {err}") from err

    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("policy.name must match [a-z][a-z0-9_]*")

    if name != policy_dir.name:
        raise ValueError(f"policy.name {name!r} must match folder {policy_dir.name!r}")

    if not isinstance(description, str):
        raise ValueError("policy.description must be a string")

    if not isinstance(enabled, bool):
        raise ValueError("policy.enabled must be a boolean")

    if type(default_port) is not int or not 1 <= default_port <= 65535:
        raise ValueError("policy.default_port must be an integer between 1 and 65535")

    if not isinstance(python_version, str) or not re.fullmatch(r"3\.\d+", python_version):
        raise ValueError("environment.python must match '3.<digits>'")

    if git:
        if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
            raise ValueError("environment.rev must be a 40-character lowercase hex SHA")
    elif rev:
        raise ValueError("environment.rev requires environment.git")

    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(camera, str) or not camera for camera in cameras)
            or len(cameras) != len(set(cameras))):
        raise ValueError("observation.cameras must be a non-empty list of unique non-empty strings")

    cameras = tuple(cameras)

    if not isinstance(language, bool):
        raise ValueError("observation.language must be a boolean")

    if type(default_horizon) is not int or default_horizon < 1:
        raise ValueError("action.default_horizon must be an integer >= 1")

    if representation != "absolute_joint":
        raise ValueError("action.representation must be 'absolute_joint'")

    if not isinstance(runtime_env, dict):
        raise ValueError("runtime.env must be a table")

    for key, value in runtime_env.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"Invalid runtime.env key: {key}")
    runtime_env = {key: str(value) for key, value in runtime_env.items()}

    train = _parse_train(data.get("train"))

    manifest = PolicyManifest(
        name=name,
        description=description,
        enabled=enabled,
        default_port=default_port,
        python=python_version,
        cameras=cameras,
        language=language,
        default_horizon=default_horizon,
        representation=representation,
        git=git,
        rev=rev,
        env=runtime_env,
        path=policy_dir,
        train=train,
    )
    return manifest


def discover(policies_dir: str | Path | None = None) -> dict[str, PolicyManifest]:
    policies_dir = default_policies_dir() if policies_dir is None else Path(policies_dir)
    manifests = {}
    if not policies_dir.is_dir():
        return manifests

    for policy_dir in sorted(policies_dir.iterdir()):
        if not policy_dir.is_dir():
            continue
        if policy_dir.name.startswith("_"):
            continue
        try:
            manifest = load_manifest(policy_dir)
        except (FileNotFoundError, ValueError) as err:
            log.warning("Skipping invalid policy %s: %s", policy_dir, err)
            continue
        if manifest.enabled:
            manifests[manifest.name] = manifest

    return manifests
