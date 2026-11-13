"""Pure command planners for the s2r CLI.

Every function returns a Plan and never runs Docker, ssh or rsync, so tests can assert exact commands.
Paths given with a remote host are paths on that host: Docker resolves bind mounts where the daemon runs.

Images: `s2r-sim` (docker/sim/Dockerfile: Isaac Lab 3.0 + LeRobot, teleop/record/eval/rollout client)
and `s2r-serve` / `s2r-train` (two targets of docker/policy/Dockerfile).
"""
from dataclasses import dataclass, field, replace
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
SIM_SELECTION_ENV = ("S2R_ARM", "S2R_SCENE", "S2R_TASK", "S2R_UNIT")
SIM_RUNTIME_ENV = (*SIM_SELECTION_ENV, "S2R_ROOM_USD", "S2R_EVAL_STARTS",
                   "S2R_FOLLOWER_CALIBRATION", "S2R_LEADER_CALIBRATION", "S2R_EVAL_CHECKPOINT")
SIM_PROGRAMS = {
    "test-gui": "core.rollout.testing_gui",
    "eval-sim-real": "sim.eval.sim_real",
    "record": "sim.teleop.agent", "eval": "sim.eval.run", "rollout": "core.rollout.real_rollout",
    "spawn": "cli.spawn", "envs": "cli.list_envs", "calibration-replay": "sim.datagen.calibration_replay",
    "calibration-report": "core.calibration.report", "eval-set": "core.benchmark.eval_set",
    "calibrate": "core.calibration.calibrate", "calibration-check": "core.calibration.check",
    "calibration-stats": "core.calibration.stats", "compare": "core.benchmark.compare",
    "sweep": "core.benchmark.sweep",
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
               host="", remote_root="", hf_home="", host_uid=None, host_gid=None, verify_gpu=False):
    """Plan prepare / train / verify for one policy from its [train] table (train image, --env mode).

    The S2R_* variables are the contract the policies' train scripts and docker/policy/entrypoint.sh share.
    """
    if verify_gpu and command != "verify":
        raise CliError("--gpu is only supported for verify")
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
    if command == "train" or (command == "verify" and verify_gpu):
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


def plan_sim(program, args, container=SIM_CONTAINER, *, git_commit="", selection_env=None):
    """Run a console script inside the running sim container through its entrypoint (Isaac env + PYTHONPATH).

    Always local: the simulator display and a real arm's USB devices live on the machine running the container.
    Selectors are explicit planner inputs; only the CLI reads the host environment.
    """
    selection_env = selection_env or {}
    if set(selection_env) - set(SIM_RUNTIME_ENV):
        raise CliError("Unknown simulation runtime environment key")
    if any(not isinstance(value, str) for value in selection_env.values()):
        raise CliError("Simulation selector values must be strings")
    selectors = [value for name in SIM_RUNTIME_ENV if name in selection_env
                 for value in ("-e", f"{name}={selection_env[name]}")]
    return Plan(("docker", "exec", "-it", *(["-e", f"S2R_GIT_COMMIT={git_commit}"] if git_commit else []), *selectors,
                 "-w", SIM_REPO, container, "s2r-entrypoint", "python", "-m", program, *args))


def plan_host(module, args, repo):
    return Plan((sys.executable, "-m", module, *args), cwd=str(Path(repo).resolve()),
                env={"PYTHONPATH": str(Path(repo).resolve() / "src")})


def plan_new(kind, name, repo, *, urdf=None, ee_link=None, gripper=(), base_link=None,
             camera=(), package_path=(), ros2_control=None, rename=(), grasp_depth_m=.01,
             continuous_limit_deg=180., force=False, container=SIM_CONTAINER,
             git_commit="", selection_env=None):
    if urdf is None:
        if ee_link or gripper or base_link or camera or package_path or ros2_control or rename or force:
            raise CliError("URDF import flags require --urdf")
        return plan_host("cli.scaffold", (kind, name, "--repo", str(Path(repo).resolve())), repo)
    if kind != "arm":
        raise CliError("--urdf is only supported for new arm")
    if not ee_link or not gripper:
        raise CliError("URDF import requires --ee_link and at least one --gripper")
    from core.arms.urdf import find_packages
    root = Path(repo).resolve()
    source = Path(urdf).expanduser().resolve()
    packages = find_packages(source)
    for pair in package_path:
        package, sep, path = pair.partition('=')
        if not sep or not package or not path:
            raise CliError(f"--package_path expects pkg=dir: {pair!r}")
        packages[package] = Path(path).expanduser().resolve()
    external = set()
    def mounted(path):
        path = Path(path).expanduser().resolve()
        if path.is_relative_to(root):
            return str(Path(SIM_REPO) / path.relative_to(root))
        external.add(path if path.is_dir() else path.parent)
        return str(path)
    command = [name, '--urdf', mounted(source), '--ee_link', ee_link]
    for joint in gripper:
        command += ['--gripper', joint]
    if base_link:
        command += ['--base_link', base_link]
    for flag, values in (('camera', camera), ('rename', rename)):
        for value in values:
            command += ['--' + flag, value]
    for package, directory in sorted(packages.items()):
        command += ['--package_path', package + '=' + mounted(directory)]
    if ros2_control:
        command += ['--ros2_control', mounted(ros2_control)]
    command += ['--grasp_depth_m', str(grasp_depth_m), '--continuous_limit_deg', str(continuous_limit_deg)]
    if force:
        command.append('--force')
    if not external:
        # Repository paths already have the running sim container's bind mount.
        return plan_sim('sim.arms.urdf_import', command, container, git_commit=git_commit,
                        selection_env=selection_env)
    mounts = bind(str(root), SIM_REPO)
    if (root / 'datasets').is_dir():
        mounts += bind(str(root / 'datasets'), '/workspace/datasets')
    for directory in sorted(external):
        mounts += bind(str(directory), str(directory), readonly=True)
    env = ['-e', 'S2R_GIT_COMMIT=' + git_commit] if git_commit else []
    for key, value in (selection_env or {}).items():
        env += ['-e', key + '=' + value]
    return Plan(('docker', 'run', '--rm', '--gpus', 'all', '--network', 'host', *env, *mounts,
                 '-w', SIM_REPO, 's2r-sim:latest', 's2r-entrypoint', 'python', '-m',
                 'sim.arms.urdf_import', *command))


def plan_check(repo, *, arm=None, scene=None, task=None, policy=None, unit=None, variant="teleop",
               static_only=False, container=SIM_CONTAINER, git_commit="", compositions=None, scene_json=None,
               selection_env=None):
    compositions = list(compositions or [(arm, scene, task)])
    if scene_json is not None:
        if static_only:
            raise CliError("--scene-json needs a runtime check; omit --static")
        if len(compositions) != 1 or not all(compositions[0]):
            raise CliError("--scene-json needs one composition; select --arm, --scene and --task")
    selected = [item for key, value in (("arm", arm), ("scene", scene), ("task", task), ("policy", policy), ("unit", unit))
                if value for item in ("--" + key, value)]
    host = plan_host("cli.check", ("--repo", str(Path(repo).resolve()), "--static-only", *selected), repo)
    if selection_env and "S2R_ROOM_USD" in selection_env:
        # Host validation checks manifest assets; runtime overrides can name
        # container-only paths and are validated in the headless composition.
        host = replace(host, env={**host.env, "S2R_ROOM_USD": ""})
    plans = [host]
    if not static_only:
        for selected_arm, selected_scene, selected_task in compositions:
            runtime = [item for key, value in (("arm", selected_arm), ("scene", selected_scene), ("task", selected_task))
                       if value for item in ("--" + key, value)]
            output = ("--scene-json", str(scene_json)) if scene_json is not None else ()
            plans.append(plan_sim("sim.check", (*runtime, "--variant", variant, *output), container,
                                  git_commit=git_commit, selection_env=selection_env))
    return plans


def plan_datagen(source, args=(), container=SIM_CONTAINER, *, git_commit="", selection_env=None):
    if source is None:
        # YAML is read where the simulator/config files live. Host dry runs stay
        # dependency-free and faithfully show this dispatcher invocation.
        return plan_sim("cli.datagen", args, container, git_commit=git_commit, selection_env=selection_env)
    if source not in DATAGEN_MODULES:
        raise CliError(f"Unknown data source {source!r}; choose {', '.join(DATAGEN_MODULES)}")
    return plan_sim(DATAGEN_MODULES[source], args, container, git_commit=git_commit, selection_env=selection_env)


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
