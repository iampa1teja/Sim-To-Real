"""Dispatch data generation to a source-specific module."""
import argparse
import importlib.util
from pathlib import Path
import runpy

from .plan import DATAGEN_MODULES


def resolve_source(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=DATAGEN_MODULES)
    args, extra = parser.parse_known_args(argv)
    if args.source is None:
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument("--config_path", type=Path)
        config = config_parser.parse_known_args(extra)[0].config_path
        if config is None:
            parser.error("provide --source or --config_path with a source field")
        import yaml

        try:
            values = yaml.safe_load(config.read_text())
        except (OSError, yaml.YAMLError) as error:
            parser.error(f"cannot read {config}: {error}")
        if not isinstance(values, dict):
            parser.error(f"{config}: configuration must be a mapping")
        args.source = values.get("source")
        if not isinstance(args.source, str) or args.source not in DATAGEN_MODULES:
            parser.error(f"{config}: source must be one of {', '.join(DATAGEN_MODULES)}")
    return args.source, extra


def main(argv=None):
    source, extra = resolve_source(argv)
    module = DATAGEN_MODULES[source]
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Data source {source!r} is not implemented yet ({module}).")
    import sys
    previous_argv = sys.argv
    try:
        sys.argv = [module, *extra]
        # Match python -m: lazy module attributes and class lookups must resolve
        # against the executing source module, not this dispatcher's __main__.
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
