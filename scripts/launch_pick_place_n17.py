#!/usr/bin/env python3
"""Scope N1.7 fine-tuning modalities, then run the pinned official launcher.

Usage: python launch_pick_place_n17.py --groot PATH <official FinetuneConfig flags>
The official launcher still creates the model, processor, datasets and trainer.
"""

import argparse
import importlib
import os
from pathlib import Path
import runpy
import subprocess
import sys


PIN = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def scope_dataset_modalities(config):
    """Exclude unused code defaults while retaining each selected dataset config.

    AutoProcessor.from_pretrained subsequently merges these selected overrides
    into its saved checkpoint modalities, preserving pretrained embodiment tags.
    """
    if getattr(config.training, "skip_weight_loading", False):
        raise ValueError("Real N1.7 fine-tuning requires pretrained weight loading")
    selected = []
    for dataset in config.data.datasets:
        tag = getattr(dataset.embodiment_tag, "value", dataset.embodiment_tag)
        if not isinstance(tag, str) or not tag:
            raise ValueError("Every selected dataset needs an embodiment tag")
        if tag not in selected:
            selected.append(tag)
    if not selected:
        raise ValueError("N1.7 fine-tuning requires a selected dataset")
    missing = [tag for tag in selected if tag not in config.data.modality_configs]
    if missing:
        raise ValueError(f"Selected dataset modality configuration missing: {missing}")
    config.data.modality_configs = {
        tag: config.data.modality_configs[tag] for tag in selected
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--groot", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    groot = args.groot.expanduser().resolve()
    launcher = groot / "gr00t/experiment/launch_finetune.py"
    if not launcher.is_file():
        parser.error(f"Official N1.7 launcher missing: {launcher}")
    revision = subprocess.check_output(
        ["git", "-C", str(groot), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        parser.error(f"GR00T N1.7 revision differs from the required pin {PIN}")
    for name in ("GROOT_SKIP_HF_MODEL_WEIGHTS", "GROOT_HF_LOCAL_FIRST", "PYTEST_CURRENT_TEST"):
        os.environ.pop(name, None)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    old_path, old_argv, old_cwd = sys.path[:], sys.argv[:], Path.cwd()
    experiment, original_run = None, None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        sys.path.insert(0, str(groot))
        import gr00t
        imported = Path(gr00t.__file__).resolve()
        expected = groot / "gr00t/__init__.py"
        if imported != expected:
            raise RuntimeError(f"GR00T imported from {imported}; expected {expected}")
        experiment = importlib.import_module("gr00t.experiment.experiment")
        original_run = experiment.run

        def run_selected(config):
            scope_dataset_modalities(config)
            from gr00t_offline_cosmos import offline_cosmos_processor
            with offline_cosmos_processor():
                return original_run(config)

        experiment.run = run_selected
        os.chdir(groot)
        sys.argv = [str(launcher), *forwarded]
        runpy.run_path(str(launcher), run_name="__main__")
    finally:
        if experiment is not None and original_run is not None:
            experiment.run = original_run
        sys.path[:] = old_path
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
