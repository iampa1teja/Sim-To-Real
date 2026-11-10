"""Dispatch data generation to a source-specific module."""
import argparse
import importlib.util
import runpy

from .plan import DATAGEN_MODULES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=DATAGEN_MODULES)
    args, extra = parser.parse_known_args(argv)
    module = DATAGEN_MODULES[args.source]
    if importlib.util.find_spec(module) is None:
        parser.exit(2, f"Data source {args.source!r} is not implemented yet ({module}).\n")
    import sys
    sys.argv = [module, *extra]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
