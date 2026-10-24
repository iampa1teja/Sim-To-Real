"""so101 command line: parse a request, turn it into a Plan, print it and (unless --dry-run) run it."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
USAGE_NOTE = (
    "Remote GPU: run 'so101 sync', then serve/train with --host user@box --remote-root /abs/path. To evaluate a "
    "remote server from a local simulator: 'so101 tunnel --host user@box --port N', then "
    "'so101 eval <config>.yaml --external_server'."
)


def build_parser():
    from .plan import DEFAULT_SERVE_IMAGE, DEFAULT_TRAIN_IMAGE, GROOT_RECIPES, TRAIN_COMMANDS

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
        sub.add_argument("--recipe", default="", choices=("",) + GROOT_RECIPES, help="GR00T training recipe (train only)")
        sub.add_argument("--image", default=DEFAULT_TRAIN_IMAGE, help="train image for policies without a wrapper")
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
    serve.add_argument("--image", default=DEFAULT_SERVE_IMAGE)
    serve.add_argument("--foreground", action="store_true")
    remote(serve)

    evaluate = commands.add_parser("eval", help="run docker/eval.sh; everything after 'eval' is passed to it")
    evaluate.add_argument("args", nargs=argparse.REMAINDER)

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
                           host=args.host, remote_root=args.remote_root, dry_run=args.dry_run, generic=args.generic)


def execute(plan, dry_run, environ=None):
    environ = dict(os.environ if environ is None else environ)
    environ.update(plan.env)
    if plan.docker_host:
        environ["DOCKER_HOST"] = plan.docker_host
    prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in [*plan.env.items(), *([("DOCKER_HOST", plan.docker_host)] if plan.docker_host else [])])
    print(f"+ {prefix}{shlex.join(plan.argv)}", flush=True)
    if dry_run and not plan.runs_on_dry_run:
        return 0
    return subprocess.run(plan.argv, env=environ, cwd=plan.cwd or None).returncode


def main(argv=None, repo=REPO):
    if sys.version_info < (3, 11):
        print(f"so101 needs Python 3.11 or newer (found {sys.version.split()[0]}).", file=sys.stderr)
        return 2
    from .plan import CliError

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
    return execute(plan, getattr(args, "dry_run", False))
