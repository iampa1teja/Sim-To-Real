"""so101 command line: parse a request, turn it into a Plan, print it and (unless --dry-run) run it."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
# Commands that hand every argument, flags included, to another program. argparse cannot do this reliably:
# a REMAINDER positional rejects a first argument that starts with a dash.
PASSTHROUGH = {"eval": "", "record": "pick_place_agent", "rollout": "real_rollout"}
USAGE_NOTE = (
    "For eval, record and rollout, flags after the command belong to the wrapped program; to only print its command "
    "put --dry-run first: 'so101 --dry-run rollout --policy ...'. "
    "Remote GPU: run 'so101 sync', then serve/train with --host user@box --remote-root /abs/path. To evaluate a "
    "remote server from a local simulator: 'so101 tunnel --host user@box --port N', then "
    "'so101 eval <config>.yaml --external_server'."
)


def build_parser():
    from .plan import GROOT_RECIPES, IMAGE_ROLES, TRAIN_COMMANDS, default_image

    parser = argparse.ArgumentParser(prog="so101", description=__doc__, epilog=USAGE_NOTE)
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
        sub.add_argument("--hf-home", default="", help="Hugging Face cache for the generic container (mounted read-only)")
        sub.add_argument("--recipe", default="", choices=("",) + GROOT_RECIPES, help="GR00T training recipe (train only)")
        sub.add_argument("--image", default=default_image("train", os.environ),
                         help="train image for policies without a wrapper (default so101-train:latest, or the "
                              "SO101_REGISTRY copy)")
        sub.add_argument("--generic", action="store_true", help="skip the policy's wrapper and use the generic path")
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

    evaluate = commands.add_parser("eval", help="run docker/eval.sh; everything after 'eval' is passed to it")
    evaluate.add_argument("args", nargs=argparse.REMAINDER)

    for name in ("build", "pull"):
        sub = commands.add_parser(name, help=("build the images from this checkout" if name == "build" else
                                              "pull the images from a registry and keep the old local names working"))
        sub.add_argument("images", nargs="?", default="all", choices=(*IMAGE_ROLES, "all"))
        sub.add_argument("--registry", default=os.environ.get("SO101_REGISTRY", ""),
                         help="registry namespace, e.g. your Docker Hub user (default: $SO101_REGISTRY)")
        sub.add_argument("--tag", default=os.environ.get("SO101_TAG", "latest"))
        if name == "build":
            sub.add_argument("--push", action="store_true", help="docker push each built image to --registry")
            sub.add_argument("--allow-isaac-push", action="store_true",
                             help="confirm that NVIDIA's licence allows publishing the teleop image")
        sub.add_argument("--dry-run", action="store_true", help="print the commands without running them")

    for name, program, summary in (
            ("record", "pick_place_agent", "record demonstrations: runs pick_place_agent in the teleop container"),
            ("rollout", "real_rollout", "run a served policy on the real arm: runs real_rollout in the teleop container")):
        sub = commands.add_parser(name, help=summary + "; everything after the command is passed to it")
        sub.set_defaults(program=program)
        sub.add_argument("args", nargs=argparse.REMAINDER)

    tunnel = commands.add_parser("tunnel", help="forward a remote policy server port to this machine")
    tunnel.add_argument("--host", required=True)
    tunnel.add_argument("--port", type=int, required=True)

    sync = commands.add_parser("sync", help="rsync policies, configs and eval sets to the remote repo copy")
    sync.add_argument("--host", required=True)
    sync.add_argument("--remote-root", required=True)
    sync.add_argument("--path", action="append", default=[], help="extra repo-relative path (e.g. datasets/pick_place_v1)")
    sync.add_argument("--dry-run", action="store_true")
    return parser


def load_policies(repo):
    sys.path.insert(0, str(repo / "source"))
    from sim_to_real_so101.policy_client.discovery import discover

    return discover(repo / "policies")


def describe(policies):
    lines = [f"{'policy':<14} {'port':<5} {'train commands':<22} dataset"]
    for name, manifest in sorted(policies.items()):
        train = manifest.train
        commands = ",".join(train.commands) if train else "-"
        lines.append(f"{name:<14} {manifest.default_port:<5} {commands:<22} {train.dataset_format if train else '-'}")
    return "\n".join(lines)


def make_plan(args, repo, policies):
    from . import plan

    if args.command == "eval":
        return plan.plan_eval(repo, args.args)
    if args.command in ("build", "pull"):
        roles = plan.IMAGE_ROLES if args.images == "all" else (args.images,)
        if args.command == "pull":
            return plan.plan_pull(roles, registry=args.registry, tag=args.tag)
        return plan.plan_build(roles, repo, registry=args.registry, tag=args.tag, push=args.push,
                               allow_isaac_push=args.allow_isaac_push)
    if args.command in ("record", "rollout"):
        return plan.plan_teleop(args.program, args.args)
    if args.command == "tunnel":
        return plan.plan_tunnel(args.host, args.port)
    if args.command == "sync":
        return plan.plan_sync(repo, args.host, args.remote_root, args.path)
    if args.policy not in policies:
        raise plan.CliError(f"unknown policy {args.policy!r}; available: {', '.join(sorted(policies)) or 'none'}")
    manifest = policies[args.policy]
    if args.command == "serve":
        return plan.plan_serve(manifest, repo, checkpoint=args.checkpoint, models_dir=args.models_dir, port=args.port,
                               options=args.option, image=args.image, hf_home=args.hf_home, host=args.host,
                               remote_root=args.remote_root, foreground=args.foreground)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    return plan.plan_train(args.command, manifest, repo, dataset=args.dataset, output=args.output,
                           base_model=args.base_model, recipe=args.recipe, extra=extra, image=args.image,
                           host=args.host, remote_root=args.remote_root, dry_run=args.dry_run, generic=args.generic,
                           hf_home=args.hf_home, host_uid=os.getuid() if hasattr(os, "getuid") else None,
                           host_gid=os.getgid() if hasattr(os, "getgid") else None)


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
    prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in [*plan.env.items(), *([("DOCKER_HOST", plan.docker_host)] if plan.docker_host else [])])
    print(f"+ {prefix}{shlex.join(plan.argv)}", flush=True)
    if dry_run and not plan.runs_on_dry_run:
        return 0
    for folder in plan.make_dirs:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return subprocess.run(plan.argv, env=environ, cwd=plan.cwd or None).returncode


def main(argv=None, repo=REPO):
    if sys.version_info < (3, 11):
        print(f"so101 needs Python 3.11 or newer (found {sys.version.split()[0]}).", file=sys.stderr)
        return 2
    from .plan import CliError

    argv = list(sys.argv[1:] if argv is None else argv)
    leading_dry_run = argv[:1] == ["--dry-run"]
    if leading_dry_run:
        argv = argv[1:]
    if argv and argv[0] in PASSTHROUGH:
        args = argparse.Namespace(command=argv[0], args=argv[1:], program=PASSTHROUGH[argv[0]], dry_run=leading_dry_run)
    else:
        args = build_parser().parse_args(argv)
    policies = load_policies(repo)
    if args.command == "policies":
        print(describe(policies))
        return 0
    try:
        plan = make_plan(args, repo, policies)
    except CliError as error:
        print(f"so101: error: {error}", file=sys.stderr)
        return 2
    return execute(plan, getattr(args, "dry_run", False) or leading_dry_run)
