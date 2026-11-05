#!/usr/bin/env python3
"""Run the pinned official N1.7 server with its local offline Cosmos processor."""

import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

from gr00t_model_profiles import N17_PIN, transformers_hub_cache
from gr00t_offline_cosmos import offline_cosmos_processor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--groot", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    groot = args.groot.expanduser().resolve()
    server = groot / "gr00t/eval/run_gr00t_server.py"
    if not server.is_file():
        parser.error(f"Official N1.7 server missing: {server}")
    revision = subprocess.check_output(
        ["git", "-C", str(groot), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != N17_PIN:
        parser.error(f"GR00T N1.7 revision differs from the required pin {N17_PIN}")
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_hub_cache())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("GROOT_SKIP_HF_MODEL_WEIGHTS", "GROOT_HF_LOCAL_FIRST", "PYTEST_CURRENT_TEST"):
        os.environ.pop(name, None)
    sys.dont_write_bytecode = True
    old_path, old_argv, old_cwd = sys.path[:], sys.argv[:], Path.cwd()
    try:
        sys.path.insert(0, str(groot))
        import gr00t
        imported = Path(gr00t.__file__).resolve()
        expected = groot / "gr00t/__init__.py"
        if imported != expected:
            raise RuntimeError(f"GR00T imported from {imported}; expected {expected}")
        with offline_cosmos_processor():
            os.chdir(groot)
            sys.argv = [str(server), *forwarded]
            runpy.run_path(str(server), run_name="__main__")
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
