"""Pure command planners for the s2r CLI (v1 cli/so101_cli/plan.py, renamed).

Every function returns a Plan and never runs Docker, ssh or rsync, so tests can assert exact commands.
Paths given with a remote host are paths on that host: Docker resolves bind mounts where the daemon runs.

Images: `s2r-sim` (docker/sim/Dockerfile: Isaac Lab 3.0 + LeRobot, teleop/record/eval/rollout client)
and `s2r-serve` / `s2r-train` (two targets of docker/policy/Dockerfile).
"""
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
import sys

from core.policy_client.discovery import TRAIN_COMMANDS

IMAGE_ROLES = ("sim", "serve", "train")
# role -> (Dockerfile, build target). serve and train share every layer of docker/policy/Dockerfile.
DOCKERFILES = {"sim": ("docker/sim/Dockerfile", ""), "serve": ("docker/policy/Dockerfile", "serve"),
               "train": ("docker/policy/Dockerfile", "train")}
SYNC_PATHS = ("arms", "scenes", "tasks", "policies", "configs")
CONTAINER_REPO = "/workspace/repo"
CONTAINER_POLICIES = "/policies"  # POLICIES_DIR in docker/policy/Dockerfile
SIM_CONTAINER = "s2r-sim"          # running sim container (override with S2R_SIM_CONTAINER)
SIM_REPO = "/workspace/s2r"        # repo path inside the sim image (docker/sim/Dockerfile WORKDIR)
SIM_PROGRAMS = {
    "record": "sim.teleop.agent", "eval": "sim.eval.run", "rollout": "core.rollout.real_rollout",
    "spawn": "cli.spawn", "envs": "cli.list_envs", "calibration-replay": "sim.datagen.calibration_replay",
    "calibration-report": "core.calibration.report", "eval-set": "core.benchmark.eval_set",
    "calibrate": "core.calibration.calibrate", "calibration-check": "core.calibration.check",
    "calibration-stats": "core.calibration.stats",
}
DATAGEN_MODULES = {name: "sim.datagen." + name for name in ("scripted", "mimic", "dr_replay")}



class CliError(ValueError):
    """A request that cannot be turned into a command; the message is shown to the user."""


@dataclass(frozen=True)
class Plan:
    argv: tuple
    env: dict = field(default_factory=dict)  # overlay on the caller's environment
    docker_host: str = ""                    # ssh://user@box when Docker must run on a remote machine
    cwd: str = ""
    make_dirs: tuple = ()                    # local folders to create before running (a bind mount needs its source)


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
        raise CliError("--host needs --remote-root, the absolute repo path on that machine (see 's2r sync')")
    return (f"ssh://{host}" if host else ""), bool(host)


def bind(source, target, readonly=False):
    return ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]


def image_name(role, registry="", tag="latest"):
    if role not in IMAGE_ROLES:
        raise CliError(f"unknown image {role!r}; use {', '.join(IMAGE_ROLES)} or all")
    if registry and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", registry.rstrip("/")):
        raise CliError(f"invalid registry or namespace {registry!r}")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,127}", tag):
        raise CliError(f"invalid image tag {tag!r}")
    return f"{registry.rstrip('/') + '/' if registry else ''}s2r-{role}:{tag}"


def default_image(role, environ):
    """The image serve/train use when --image is not given: the registry copy if S2R_REGISTRY is set."""
    return image_name(role, environ.get("S2R_REGISTRY", ""), environ.get("S2R_TAG", "latest"))


def plan_serve(manifest, repo, *, checkpoint, models_dir="~/models", port=None, options=(), image="s2r-serve:latest",
               hf_home="", host="", remote_root="", foreground=False):
    """Start the policy server container for one policy (entrypoint.sh --policy <name> <launcher args>)."""
    docker_host, remote = remote_settings(host, remote_root)
    root = remote_root.rstrip("/") if remote else str(Path(repo).resolve())
    checkpoint = relative_path(checkpoint, "--checkpoint")
    port = port or manifest.default_port
    for option in options:
        if "=" not in option:
            raise CliError(f"--option must look like key=value, got {option!r}")
    if not remote and not hf_home:
        hf_home = str(Path.home() / ".cache" / "huggingface")
    argv = ["docker", "run", *([] if foreground else ["-d"]), "--rm", "--name", f"s2r-serve-{manifest.name}-{port}",
            "--network", "host", "--gpus", "all",
            "-v", f"{root}/policies:{CONTAINER_POLICIES}", "-v", "policy-cache:/opt/cache"]
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


def plan_train(command, manifest, repo, *, dataset="", output="", base_model="", extra=(), image="s2r-train:latest",
               host="", remote_root="", hf_home="", host_uid=None, host_gid=None):
    """Plan prepare / train / verify for one policy from its [train] table (train image, --env mode).

    The S2R_* variables are the contract the policies' train scripts and docker/policy/entrypoint.sh share.
    """
    if manifest.train is None:
        raise CliError(f"policy {manifest.name!r} has no [train] table in its policy.toml")
    script = manifest.train.commands.get(command)
    if script is None:
        raise CliError(f"policy {manifest.name!r} declares no {command!r} command")
    _train_inputs(command, dataset, output)
    docker_host, remote = remote_settings(host, remote_root)
    root = remote_root.rstrip("/") if remote else str(Path(repo).resolve())
    env = {"S2R_POLICY": manifest.name, "S2R_POLICY_DIR": f"{CONTAINER_POLICIES}/{manifest.name}"}
    if not remote and host_uid is not None and host_gid is not None:
        env.update(S2R_HOST_UID=str(host_uid), S2R_HOST_GID=str(host_gid))
    argv = ["docker", "run", "--rm", "--init", "--ipc", "host", "--shm-size", "8g", "--workdir", CONTAINER_REPO]
    if command == "train":
        # NVML reports host PIDs; a busy-GPU guard needs their names to recognize Xorg.
        argv += ["--gpus", "all", "--pid", "host"]
    argv += bind(root, CONTAINER_REPO, readonly=True)
    # Writable: uv writes a missing uv.lock into the policy folder.
    argv += bind(f"{root}/policies", CONTAINER_POLICIES)
    make_dirs = ()
    for flag, value, target, writable, variable in (
            ("--dataset", dataset, "/data/dataset", False, "S2R_DATASET"),
            ("--output", output, "/data/output", True, "S2R_OUTPUT"),
            ("--base-model", base_model, "/data/base_model", False, "S2R_BASE_MODEL")):
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
    runner = ["bash", path] if path.endswith(".sh") else ["sh", "-c", 'exec "${S2R_PYTHON:-python3}" "$@"', "s2r", path]
    return Plan(tuple([*argv, image, "--env", manifest.name, "--", *runner, *extra]), docker_host=docker_host,
                make_dirs=make_dirs)


def plan_build(roles, repo, *, registry="", tag="latest", push=False, allow_isaac_push=False):
    """One `docker build` per role (plus a `docker push` when asked)."""
    plans = []
    for role in roles:
        name = image_name(role, registry, tag)
        if push and not registry:
            raise CliError("--push needs --registry (or S2R_REGISTRY), for example your Docker Hub namespace")
        if push and role == "sim" and not allow_isaac_push:
            raise CliError("the sim image is built on NVIDIA's Isaac Lab image; check its licence allows "
                           "redistributing derived images, then pass --allow-isaac-push")
        dockerfile, target = DOCKERFILES[role]
        argv = ["docker", "build", *(["--target", target] if target else []), "-t", name, "-f", dockerfile, "."]
        plans.append(Plan(tuple(argv), cwd=str(Path(repo).resolve())))
        if push:
            plans.append(Plan(("docker", "push", name)))
    return plans


def plan_pull(roles, *, registry, tag="latest"):
    """Pull the registry images and tag them with the local names."""
    if not registry:
        raise CliError("pull needs --registry (or S2R_REGISTRY)")
    plans = []
    for role in roles:
        name = image_name(role, registry, tag)
        plans.append(Plan(("docker", "pull", name)))
        plans.append(Plan(("docker", "tag", name, image_name(role, "", tag))))
    return plans


def plan_sim(program, args, container=SIM_CONTAINER, *, git_commit=""):
    """Run a console script inside the running sim container through its entrypoint (Isaac env + PYTHONPATH).

    Always local: the simulator display and a real arm's USB devices live on the machine running the container.
    """
    return Plan(("docker", "exec", "-it", *(["-e", f"S2R_GIT_COMMIT={git_commit}"] if git_commit else []),
                 "-w", SIM_REPO, container, "s2r-entrypoint", "python", "-m", program, *args))


def plan_host(module, args, repo):
    return Plan((sys.executable, "-m", module, *args), cwd=str(Path(repo).resolve()),
                env={"PYTHONPATH": str(Path(repo).resolve() / "src")})


def plan_new(kind, name, repo):
    return plan_host("cli.scaffold", (kind, name, "--repo", str(Path(repo).resolve())), repo)


def plan_check(repo, *, arm=None, scene=None, task=None, policy=None, unit=None, variant="teleop",
               static_only=False, container=SIM_CONTAINER, git_commit="", compositions=None):
    selected = [item for key, value in (("arm", arm), ("scene", scene), ("task", task), ("policy", policy), ("unit", unit))
                if value for item in ("--" + key, value)]
    plans = [plan_host("cli.check", ("--repo", str(Path(repo).resolve()), "--static-only", *selected), repo)]
    if not static_only:
        for selected_arm, selected_scene, selected_task in compositions or [(arm, scene, task)]:
            runtime = [item for key, value in (("arm", selected_arm), ("scene", selected_scene), ("task", selected_task))
                       if value for item in ("--" + key, value)]
            plans.append(plan_sim("sim.check", (*runtime, "--variant", variant), container, git_commit=git_commit))
    return plans


def plan_datagen(source, args=(), container=SIM_CONTAINER, *, git_commit=""):
    if source not in DATAGEN_MODULES:
        raise CliError(f"Unknown data source {source!r}; choose {', '.join(DATAGEN_MODULES)}")
    return plan_sim(DATAGEN_MODULES[source], args, container, git_commit=git_commit)


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
