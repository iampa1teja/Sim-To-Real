"""Upload a consolidated dataset after checking its current audit."""

from pathlib import Path as _Path
import argparse as _argparse
import sys as _sys

from core.datasets.upload import (BucketStorage as _BucketStorage, HubStorage as _HubStorage,
                                 plan_upload as _plan_upload, run_upload as _run_upload)
from core.generate.errors import GenerationError as _GenerationError


def _storage(args):
    if args.repo_id:
        from huggingface_hub import HfApi
        return _HubStorage(HfApi(), args.repo_id, revision=args.revision)
    import fsspec
    fs, _ = fsspec.core.url_to_fs(args.bucket)
    return _BucketStorage(fs, args.bucket)


def main(argv=None, *, storage=None):
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=_Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo_id")
    group.add_argument("--bucket")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--message", default="Upload dataset")
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    try:
        if storage is None:
            storage = _storage(args)
        plan = _plan_upload(args.root, storage, allow_overwrite=args.allow_overwrite)
        if args.dry_run:
            for item in plan.items:
                print(item.remote)
        else:
            _run_upload(plan, storage, message=args.message)
        print(f"{'Planned' if args.dry_run else 'Uploaded'} {len(plan.items)} files; skipped {len(plan.skipped)}")
        return 0
    except _GenerationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
