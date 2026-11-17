"""Merge audited generation shards into a fresh dataset root."""

from pathlib import Path as _Path
import argparse as _argparse
import shutil as _shutil
import sys as _sys

from core.datasets.consolidate import consolidate as _consolidate
from core.datasets.merge import (ShardSummary as _ShardSummary, apply_merge_meta as _apply_merge_meta,
                                merge_frames as _merge_frames, plan_merge as _plan_merge)
from core.generate.errors import GenerationError as _GenerationError


def main(argv=None):
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=_Path)
    parser.add_argument("--dst_root", required=True, type=_Path)
    parser.add_argument("--repo_id", required=True)
    args = parser.parse_args(argv)
    writing = False
    try:
        if args.dst_root.exists():
            raise _GenerationError(f"Destination already exists: {args.dst_root}")
        shards = [_ShardSummary.from_root(root) for root in args.shards]
        plan = _plan_merge(shards)
        writing = True
        _merge_frames(plan, shards, args.dst_root, args.repo_id)
        _apply_merge_meta(plan, shards, args.dst_root)
        _consolidate(args.dst_root)
    except _GenerationError as exc:
        if writing and args.dst_root.exists():
            _shutil.rmtree(args.dst_root)
        print(f"{type(exc).__name__}: {exc}", file=_sys.stderr)
        return 1
    print(f"Merged {len(shards)} shards, {plan.total} episodes into {args.dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
