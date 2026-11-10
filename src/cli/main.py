"""s2r command line: parse a request, turn it into a Plan, print it and (unless --dry-run) run it."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
USAGE_NOTE = (
    "Sim commands (record, eval, rollout, spawn, envs, calibration-replay, calibration-report, eval-set, "
    "calibrate, calibration-check, calibration-stats) "
    "pass every argument after the command to the program inside the running sim container; to only print the "
    "command put --dry-run first: 's2r --dry-run record --config_path configs/record/sim_only.yaml'. "
    "Remote GPU: run 's2r sync', then serve/train with --host user@box --remote-root /abs/path. To evaluate a "
    "remote server from a local simulator: 's2r tunnel --host user@box --port N', then 's2r eval ...'."
)


def build_parser():
    from .plan import IMAGE_ROLES, TRAIN_COMMANDS, default_image

    parser = argparse.ArgumentParser(prog="s2r", description=__doc__, epilog=USAGE_NOTE)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    commands.add_parser("policies", help="list policies under policies/ and what each can do")

    def remote(sub):
        sub.add_argument("--host", default="", help="user@box: run Docker on that machine over ssh")
        sub.add_argument("--remote-root", default="", help="absolute path of this repo on --host")
        sub.add_argument("--dry-run", action="store_true", help="print the command without running it")

    for name in TRAIN_COMMANDS:
        sub = commands.add_parser(name, help=f"run a policy's {name} command in the train container")
        sub.add_argument("--policy", required=True)
        sub.add_argument("--dataset", default="", help="dataset folder (source for prepare)")
        sub.add_argument("--output", default="", help="output folder (prepared dataset or checkpoints)")
        sub.add_argument("--base-model", default="", help="checkpoint to start from")
        sub.add_argument("--hf-home", default="", help="Hugging Face cache (mounted read-only)")
        sub.add_argument("--image", default=default_image("train", os.environ))
        remote(sub)
        sub.add_argument("extra", nargs=argparse.REMAINDER, help="after --: passed to the policy's script unchanged")

    serve = commands.add_parser("serve", help="start the policy server container")
    serve.add_argument("--policy", required=True)
    serve.add_argument("--checkpoint", required=True, help="path relative to --models-dir")
    serve.add_argument("--models-dir", default="~/models")
    serve.add_argument("--hf-home", default="")
    serve.add_argument("--port", type=int)
    serve.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    serve.add_argument("--image", default=default_image("serve", os.environ))
    serve.add_argument("--foreground", action="store_true")
    remote(serve)

    for name in ("build", "pull"):
        sub = commands.add_parser(name, help=("build the images from this checkout" if name == "build" else
                                              "pull the images from a registry"))
        sub.add_argument("images", nargs="?", default="all", choices=(*IMAGE_ROLES, "all"))
        sub.add_argument("--registry", default=os.environ.get("S2R_REGISTRY", ""),
                         help="registry namespace, e.g. your Docker Hub user (default: $S2R_REGISTRY)")
        sub.add_argument("--tag", default=os.environ.get("S2R_TAG", "latest"))
        if name == "build":
            sub.add_argument("--push", action="store_true", help="docker push each built image to --registry")
            sub.add_argument("--allow-isaac-push", action="store_true",
                             help="confirm that NVIDIA's licence allows publishing the sim image")
        sub.add_argument("--dry-run", action="store_true", help="print the commands without running them")

    tunnel = commands.add_parser("tunnel", help="forward a remote policy server port to this machine")
    tunnel.add_argument("--host", required=True)
    tunnel.add_argument("--port", type=int, required=True)

    sync = commands.add_parser("sync", help="rsync arms, scenes, tasks, policies and configs to the remote repo copy")
    sync.add_argument("--host", required=True)
    sync.add_argument("--remote-root", required=True)
    sync.add_argument("--path", action="append", default=[], help="extra repo-relative path (e.g. datasets/x)")
    sync.add_argument("--dry-run", action="store_true")
    new = commands.add_parser("new", help="create an arm, scene, task or policy from its template")
    new.add_argument("kind", choices=("arm", "scene", "task", "policy"))
    new.add_argument("name")
    new.add_argument("--dry-run", action="store_true")
    check = commands.add_parser("check", help="validate all extensions, then spawn the selected composition headlessly")
    check.add_argument("--arm")
    check.add_argument("--scene")
    check.add_argument("--task")
    check.add_argument("--policy")
    check.add_argument("--unit")
    check.add_argument("--variant", default="teleop")
    check.add_argument("--static", "--static-only", dest="static_only", action="store_true",
                       help="validate every extension on the host without launching Isaac")
    check.add_argument("--dry-run", action="store_true")
    for name in ("record", "eval", "rollout", "spawn", "envs", "calibrate", "calibration-check",
                 "calibration-stats", "calibration-report", "calibration-replay", "eval-set", "datagen", "migrate"):
        commands.add_parser(name, help="run the corresponding Python module (arguments passed through)")
    return parser


def load_policies(repo):
    sys.path.insert(0, str(repo / "src"))
    from core.policy_client.discovery import discover

    return discover(repo / "policies")


def describe(policies):
    lines = [f"{'policy':<14} {'port':<5} {'train commands':<22} dataset"]
    for name, manifest in sorted(policies.items()):
        train = manifest.train
        commands = ",".join(train.commands) if train else "-"
        lines.append(f"{name:<14} {manifest.default_port:<5} {commands:<22} {train.dataset_format if train else '-'}")
    return "\n".join(lines)


def make_plan(args, repo, policies):
    from . import plan as paths

    if args.command in paths.SIM_PROGRAMS:
        container = os.environ.get("S2R_SIM_CONTAINER", paths.SIM_CONTAINER)
        return paths.plan_sim(paths.SIM_PROGRAMS[args.command], args.args, container, git_commit=source_revision(repo))
    if args.command == "new":
        return paths.plan_new(args.kind, args.name, repo)
    if args.command == "check":
        compositions = check_compositions(repo, args.arm, args.scene, args.task)
        arm, scene, task = compositions[0]
        return paths.plan_check(repo, arm=arm, scene=scene, task=task, policy=args.policy, unit=args.unit,
                                variant=args.variant, static_only=args.static_only,
                                container=os.environ.get("S2R_SIM_CONTAINER", paths.SIM_CONTAINER), git_commit=source_revision(repo),
                                compositions=compositions)
    if args.command == "datagen":
        return paths.plan_datagen(args.source, args.args, os.environ.get("S2R_SIM_CONTAINER", paths.SIM_CONTAINER),
                                  git_commit=source_revision(repo))
    if args.command == "migrate":
        return paths.plan_host("core.datasets.task_meta", args.args, repo)
    if args.command in ("build", "pull"):
        roles = paths.IMAGE_ROLES if args.images == "all" else (args.images,)
        if args.command == "pull":
            return paths.plan_pull(roles, registry=args.registry, tag=args.tag)
        return paths.plan_build(roles, repo, registry=args.registry, tag=args.tag, push=args.push,
                                allow_isaac_push=args.allow_isaac_push)
    if args.command == "tunnel":
        return paths.plan_tunnel(args.host, args.port)
    if args.command == "sync":
        return paths.plan_sync(repo, args.host, args.remote_root, args.path)
    if args.policy not in policies:
        raise paths.CliError(f"unknown policy {args.policy!r}; available: {', '.join(sorted(policies)) or 'none'}")
    manifest = policies[args.policy]
    if args.command == "serve":
        return paths.plan_serve(manifest, repo, checkpoint=args.checkpoint, models_dir=args.models_dir, port=args.port,
                                options=args.option, image=args.image, hf_home=args.hf_home, host=args.host,
                                remote_root=args.remote_root, foreground=args.foreground)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    return paths.plan_train(args.command, manifest, repo, dataset=args.dataset, output=args.output,
                            base_model=args.base_model, extra=extra, image=args.image, host=args.host,
                            remote_root=args.remote_root, hf_home=args.hf_home,
                            host_uid=os.getuid() if hasattr(os, "getuid") else None,
                            host_gid=os.getgid() if hasattr(os, "getgid") else None)


def source_revision(repo):
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def check_compositions(repo, arm=None, scene=None, task=None):
    base = (arm or os.environ.get("S2R_ARM", "so101"), scene or os.environ.get("S2R_SCENE", "rpd_meeting_room"),
            task or os.environ.get("S2R_TASK", "pick_place_cube"))
    rows = [base]
    if not any((arm, scene, task)):
        for index, (folder, marker) in enumerate((("arms", "arm.toml"), ("scenes", "scene.toml"), ("tasks", "task.toml"))):
            for path in sorted((Path(repo) / folder).iterdir()):
                if not path.name.startswith("_") and (path / marker).is_file():
                    row = list(base)
                    row[index] = path.name
                    if tuple(row) not in rows:
                        rows.append(tuple(row))
    return rows


def execute(plan, dry_run, environ=None):
    if isinstance(plan, list):  # build and pull are several commands; stop at the first failure
        for step in plan:
            code = execute(step, dry_run, environ)
            if code:
                return code
        return 0
    environ = dict(os.environ if environ is None else environ)
    environ.update(plan.env)
    if plan.docker_host:
        environ["DOCKER_HOST"] = plan.docker_host
    overlay = [*plan.env.items(), *([("DOCKER_HOST", plan.docker_host)] if plan.docker_host else [])]
    prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in overlay)
    print(f"+ {prefix}{shlex.join(plan.argv)}", flush=True)
    if dry_run:
        return 0
    for folder in plan.make_dirs:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return subprocess.run(plan.argv, env=environ, cwd=plan.cwd or None).returncode


def main(argv=None, repo=REPO):
    if sys.version_info < (3, 11):
        print(f"s2r needs Python 3.11 or newer (found {sys.version.split()[0]}).", file=sys.stderr)
        return 2
    from .plan import SIM_PROGRAMS, CliError

    argv = list(sys.argv[1:] if argv is None else argv)
    leading_dry_run = argv[:1] == ["--dry-run"]
    if leading_dry_run:
        argv = argv[1:]
    # Sim commands hand every argument, flags included, to the wrapped program; argparse's REMAINDER
    # rejects a first argument that starts with a dash, so they bypass the parser.
    if argv and argv[0] == "datagen":
        parser = argparse.ArgumentParser(prog="s2r datagen")
        parser.add_argument("--source", choices=("scripted", "mimic", "dr_replay"))
        parsed, extra = parser.parse_known_args(argv[1:])
        if parsed.source is None and not any(arg == "--config_path" or arg.startswith("--config_path=") for arg in extra):
            parser.error("provide --source or --config_path with a source field")
        args = argparse.Namespace(command="datagen", source=parsed.source, args=extra, dry_run=leading_dry_run)
        policies = {}
    elif argv and argv[0] == "migrate":
        args = argparse.Namespace(command="migrate", args=argv[1:], dry_run=leading_dry_run)
        policies = {}
    elif argv and argv[0] in SIM_PROGRAMS:
        args = argparse.Namespace(command=argv[0], args=argv[1:], dry_run=leading_dry_run)
        policies = {}
    else:
        args = build_parser().parse_args(argv)
        policies = load_policies(repo)
    if args.command == "policies":
        print(describe(policies))
        return 0
    try:
        plan = make_plan(args, repo, policies)
    except CliError as error:
        print(f"s2r: error: {error}", file=sys.stderr)
        return 2
    if args.command == "datagen" and args.source is not None and not leading_dry_run:
        module = Path(repo) / "src" / "sim" / "datagen" / (args.source + ".py")
        if not module.is_file():
            print(f"s2r: data source {args.source!r} is not implemented yet ({module.name}).", file=sys.stderr)
            return 2
    return execute(plan, getattr(args, "dry_run", False) or leading_dry_run)
