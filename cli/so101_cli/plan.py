"""Pure command planners for the so101 CLI.

Every function returns a Plan and never runs Docker, ssh or rsync, so tests can assert exact commands.
Paths given with a remote host are paths on that host: Docker resolves bind mounts where the daemon runs.
"""
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re

TRAIN_COMMANDS = ("prepare", "train", "verify")
GROOT_RECIPES = ("train", "sim", "failures", "dit")
IMAGE_ROLES = ("teleop", "serve", "train")
DEFAULT_TRAIN_IMAGE = "so101-train:latest"
DEFAULT_SERVE_IMAGE = "policy-server:latest"
# Names that the README, docker/eval.sh and configs/eval/*.yaml still use. build and pull keep them pointing at
# the new images so nothing has to be edited after pulling from a registry.
LEGACY_ALIASES = {"teleop": "teleop-docker:latest", "serve": DEFAULT_SERVE_IMAGE}
# role -> (Dockerfile, build target). serve and train are two targets of one file and share every layer.
DOCKERFILES = {"teleop": ("docker/sim/Dockerfile", ""), "serve": ("policy_servers/Dockerfile", "serve"),
               "train": ("policy_servers/Dockerfile", "train")}
SYNC_PATHS = ("policies", "configs", "eval_sets")
CONTAINER_REPO = "/workspace/repo"
CONTAINER_POLICIES = "/policies"  # POLICIES_DIR in policy_servers/Dockerfile; the entrypoint reads policies from here
TELEOP_CONTAINER = "teleop"
TELEOP_REPO = "/workspace/Sim-to-Real-SO-101-Workshop"


class CliError(ValueError):
    """A request that cannot be turned into a command; the message is shown to the user."""


@dataclass(frozen=True)
class Plan:
    argv: tuple
    env: dict = field(default_factory=dict)  # overlay on the caller's environment
    docker_host: str = ""                    # ssh://user@box when Docker must run on a remote machine
    cwd: str = ""
    make_dirs: tuple = ()                    # local folders to create before running (a bind mount needs its source)
    runs_on_dry_run: bool = False            # the wrapped tool has its own dry run: run it, do not just print


def host_path(value, remote, what):
    """A path as the Docker daemon sees it: absolute on the remote host, resolved here otherwise."""
    if remote:
        if not PurePosixPath(value).is_absolute():
            raise CliError(f"{what} must be an absolute path on the remote host, got {value!r}")
        return value
    return str(Path(value).expanduser().resolve())


def relative_path(value, what):
    parts = PurePosixPath(value.replace("\\", "/")).parts
    if not parts or PurePosixPath(value).is_absolute() or ".." in parts:
        raise CliError(f"{what} must be a relative path without '..', got {value!r}")
    return "/".join(parts)


def remote_settings(host, remote_root):
    if host and not remote_root:
        raise CliError("--host needs --remote-root, the absolute repo path on that machine (see 'so101 sync')")
    return (f"ssh://{host}" if host else ""), bool(host)


def bind(source, target, readonly=False):
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def plan_serve(manifest, repo, *, checkpoint, models_dir="~/models", port=None, options=(), image=DEFAULT_SERVE_IMAGE,
               hf_home="", host="", remote_root="", foreground=False):
    """Mirror docker/eval.sh's policy-server command so serving and evaluating start the same container."""
    docker_host, remote = remote_settings(host, remote_root)
    root = remote_root.rstrip("/") if remote else str(Path(repo).resolve())
    checkpoint = relative_path(checkpoint, "--checkpoint")
    port = port or manifest.default_port
    for option in options:
        if "=" not in option:
            raise CliError(f"--option must look like key=value, got {option!r}")
    if not remote and not hf_home:
        hf_home = str(Path.home() / ".cache" / "huggingface")
    argv = ["docker", "run", *([] if foreground else ["-d"]), "--rm", "--name", f"so101-serve-{manifest.name}-{port}",
            "--network", "host", "--gpus", "all",
            "-v", f"{root}/policies:/policies", "-v", "policy-cache:/opt/cache"]
    if hf_home:
        argv += ["-v", f"{host_path(hf_home, remote, '--hf-home')}:/root/.cache/huggingface"]
    argv += ["-v", f"{host_path(models_dir, remote, '--models-dir')}:/workspace/models", image,
             "--policy", manifest.name, "--checkpoint", f"/workspace/models/{checkpoint}", "--port", str(port)]
    for option in options:
        argv += ["--option", option]
    return Plan(tuple(argv), docker_host=docker_host)


def _train_inputs(command, dataset, output):
    if command in ("prepare", "train") and not dataset:
        raise CliError(f"{command} needs --dataset")
    if command in ("prepare", "train") and not output:
        raise CliError(f"{command} needs --output")
    if command == "verify" and not dataset:
        raise CliError("verify needs --dataset")


def _groot(command, repo, *, dataset, output, base_model, recipe, extra, dry_run):
    # shortcut: wraps docker/train.py, which owns the pinned GR00T container; remove when the phase 6 train
    # image runs every policy's [train.commands] directly.
    if recipe and command != "train":
        raise CliError("--recipe only applies to the train command")
    forwarded, env = list(extra), {}
    if command == "prepare":
        forwarded += ["--source", dataset, "--output", output]
    elif command == "verify":
        forwarded += ["--dataset", dataset]
    else:
        env = {name: value for name, value in (("DATASET", dataset), ("OUT", output), ("BASE_MODEL", base_model)) if value}
    workflow = recipe or command
    argv = ["python3", str(Path(repo) / "docker" / "train.py"), workflow, *(["--dry_run"] if dry_run else []), *forwarded]
    return Plan(tuple(argv), env, runs_on_dry_run=True)


def _molmo(command, repo, *, dataset, output, base_model, recipe, extra, dry_run):
    # shortcut: wraps docker/molmo_train.sh, which owns the validated LoRA runtime; remove with _groot.
    if recipe:
        raise CliError("--recipe is not used by molmoact2")
    script, env = str(Path(repo) / "docker" / "molmo_train.sh"), {}
    if command == "prepare":
        argv = [script, "python", "policies/molmoact2/train/prepare_molmo_lora.py", "--source", dataset,
                "--output", output, *extra]
    else:
        argv = [script, "train" if command == "train" else "probe", *extra]
        env["MOLMO_DATASET"] = dataset
        if command == "train":
            env["MOLMO_OUT"] = output
        if base_model:
            env["MOLMO_BASE"] = base_model
    return Plan(tuple(argv), env)


ADAPTERS = {"groot_n17": _groot, "molmoact2": _molmo}


def plan_train(command, manifest, repo, *, dataset="", output="", base_model="", recipe="", extra=(),
               image=DEFAULT_TRAIN_IMAGE, host="", remote_root="", dry_run=False, generic=False, hf_home="",
               host_uid=None, host_gid=None):
    """Plan prepare / train / verify for one policy from its [train] table."""
    if manifest.train is None:
        raise CliError(f"policy {manifest.name!r} has no [train] table in its policy.toml")
    script = manifest.train.commands.get(command)
    if script is None:
        raise CliError(f"policy {manifest.name!r} declares no {command!r} command")
    _train_inputs(command, dataset, output)
    docker_host, remote = remote_settings(host, remote_root)
    if manifest.name in ADAPTERS and not generic:
        if remote:
            raise CliError(f"{manifest.name} trains through a wrapper that needs the repo and data at the same "
                           "absolute paths; run it on the GPU machine itself (ssh there, then ./so101 train ...)")
        return ADAPTERS[manifest.name](command, repo, dataset=dataset, output=output, base_model=base_model,
                                       recipe=recipe, extra=extra, dry_run=dry_run)
    if recipe:
        raise CliError("--recipe is only used by the GR00T wrapper")
    root = remote_root.rstrip("/") if remote else str(Path(repo).resolve())
    env = {"SO101_POLICY": manifest.name, "SO101_POLICY_DIR": f"{CONTAINER_POLICIES}/{manifest.name}"}
    if not remote and host_uid is not None and host_gid is not None:
        env.update(SO101_HOST_UID=str(host_uid), SO101_HOST_GID=str(host_gid))
    argv = ["docker", "run", "--rm", "--init", "--ipc", "host", "--shm-size", "8g", "--workdir", CONTAINER_REPO]
    if command == "train":
        # NVML reports host PIDs; the busy-GPU guard needs their names to recognize Xorg.
        argv += ["--gpus", "all", "--pid", "host"]
    argv += bind(root, CONTAINER_REPO, readonly=True)
    # Writable like serve's mount: uv writes a missing uv.lock into the policy folder.
    argv += bind(f"{root}/policies", CONTAINER_POLICIES)
    make_dirs = ()
    for flag, value, target, writable, variable in (
            ("--dataset", dataset, "/data/dataset", False, "SO101_DATASET"),
            ("--output", output, "/data/output", True, "SO101_OUTPUT"),
            ("--base-model", base_model, "/data/base_model", False, "SO101_BASE_MODEL")):
        if value:
            source = host_path(value, remote, flag)
            destination = target
            if flag == "--output":
                if command == "prepare":
                    # Preparation atomically renames a sibling staging directory into a new destination.
                    output_path = PurePosixPath(source) if remote else Path(source)
                    if not output_path.name:
                        raise CliError("prepare --output must name a new directory")
                    source = str(output_path.parent)
                    destination = f"{target}/{output_path.name}"
                if not remote:
                    make_dirs = (source,)
            argv += bind(source, target, readonly=not writable)
            env[variable] = destination
    argv += ["-v", "policy-cache:/opt/cache"]
    if command == "train" and not remote and not hf_home:
        hf_home = str(Path.home() / ".cache" / "huggingface")
    if hf_home:
        argv += bind(host_path(hf_home, remote, "--hf-home"), "/root/.cache/huggingface", readonly=True)
    for name, value in sorted(env.items()):
        argv += ["--env", f"{name}={value}"]
    path = f"{CONTAINER_POLICIES}/{manifest.name}/{script}"
    runner = ["bash", path] if path.endswith(".sh") else ["sh", "-c", 'exec "${SO101_PYTHON:-python3}" "$@"', "so101", path]
    return Plan(tuple([*argv, image, "--env", manifest.name, "--", *runner, *extra]), docker_host=docker_host,
                make_dirs=make_dirs)


def image_name(role, registry="", tag="latest"):
    if role not in IMAGE_ROLES:
        raise CliError(f"unknown image {role!r}; use {', '.join(IMAGE_ROLES)} or all")
    if registry and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", registry.rstrip("/")):
        raise CliError(f"invalid registry or namespace {registry!r}")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}", tag):
        raise CliError(f"invalid image tag {tag!r}")
    return f"{registry.rstrip('/') + '/' if registry else ''}so101-{role}:{tag}"


def default_image(role, environ):
    """The image serve/train use when --image is not given: the registry copy if SO101_REGISTRY is set."""
    if environ.get("SO101_REGISTRY"):
        return image_name(role, environ["SO101_REGISTRY"], environ.get("SO101_TAG", "latest"))
    return DEFAULT_SERVE_IMAGE if role == "serve" else DEFAULT_TRAIN_IMAGE


def plan_build(roles, repo, *, registry="", tag="latest", push=False, allow_isaac_push=False):
    """One `docker build` per role (plus a `docker push` when asked); never builds anything itself."""
    plans = []
    for role in roles:
        name = image_name(role, registry, tag)
        if push and not registry:
            raise CliError("--push needs --registry (or SO101_REGISTRY), for example your Docker Hub namespace")
        if push and role == "teleop" and not allow_isaac_push:
            raise CliError("the teleop image is built on NVIDIA's Isaac Lab image; check its licence allows "
                           "redistributing derived images, then pass --allow-isaac-push")
        dockerfile, target = DOCKERFILES[role]
        tags = [name, *([LEGACY_ALIASES[role]] if role in LEGACY_ALIASES else [])]
        argv = ["docker", "build", *(["--target", target] if target else []),
                *[part for item in tags for part in ("-t", item)], "-f", dockerfile, "."]
        plans.append(Plan(tuple(argv), cwd=str(Path(repo).resolve())))
        if push:
            plans.append(Plan(("docker", "push", name)))
    return plans


def plan_pull(roles, *, registry, tag="latest"):
    """Pull the registry images and point the legacy local names at them."""
    if not registry:
        raise CliError("pull needs --registry (or SO101_REGISTRY)")
    plans = []
    for role in roles:
        name = image_name(role, registry, tag)
        plans.append(Plan(("docker", "pull", name)))
        if role in LEGACY_ALIASES:
            plans.append(Plan(("docker", "tag", name, LEGACY_ALIASES[role])))
    return plans


def plan_eval(repo, args):
    """docker/eval.sh already orchestrates server and rollout; the CLI only gives it one front door."""
    return Plan(("bash", str(Path(repo) / "docker" / "eval.sh"), *args), runs_on_dry_run=True)


def plan_teleop(program, args, container=TELEOP_CONTAINER):
    """Run a console script inside the running teleop container, with the settings from /root/env loaded.

    Always local: the simulator display and a real arm's USB device live on the machine running the container.
    """
    return Plan(("docker", "exec", "-it", "-w", TELEOP_REPO, container, "bash", "-c",
                 'source /root/env 2>/dev/null; console_script=$(command -v "$1") || exit; shift; '
                 'exec /workspace/isaaclab/_isaac_sim/python.sh "$console_script" "$@"', "so101", program, *args))


def plan_tunnel(host, port):
    return Plan(("ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"{port}:localhost:{port}", host))


def plan_sync(repo, host, remote_root, paths=()):
    """Copy what the remote containers bind-mount. Never deletes anything on the remote side."""
    if not host or not remote_root:
        raise CliError("sync needs --host and --remote-root")
    if not PurePosixPath(remote_root).is_absolute():
        raise CliError("--remote-root must be an absolute path on the remote host")
    chosen = [relative_path(path, "--path") for path in (paths or SYNC_PATHS)]
    argv = ["rsync", "-azR", "--info=progress2", "--exclude=__pycache__", "--exclude=.git",
            *[f"./{path}" for path in chosen], f"{host}:{remote_root.rstrip('/')}/"]
    return Plan(tuple(argv), cwd=str(Path(repo).resolve()))
