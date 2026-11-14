"""Translate the container mount contract without importing a model or CUDA."""
import argparse
import os
from pathlib import Path


def training_args(argv, environ=None):
    environ = os.environ if environ is None else environ
    result = list(argv)
    keys = {value.split('=', 1)[0] for value in result if value.startswith('--')}
    # A complete explicit LeRobot config owns its defaults, including resume.
    if '--config_path' in keys or '--help' in keys or '-h' in result:
        return result

    def default(key, value):
        if value is not None and key not in keys:
            result.append(f'{key}={value}')

    dataset = environ.get('S2R_DATASET')
    for index, value in enumerate(argv):
        if value.startswith('--dataset.root='):
            dataset = value.split('=', 1)[1]
        elif value == '--dataset.root' and index + 1 < len(argv):
            dataset = argv[index + 1]
    default('--dataset.root', dataset)
    if dataset:
        default('--dataset.repo_id', 'local/molmo_prepared')
        default('--policy.norm_stats_path', str(Path(dataset) / 'meta/molmo_norm_stats.json'))
        default('--policy.norm_tag', 'workshop_pick_place_v1_as_real')
    default('--output_dir', environ.get('S2R_OUTPUT'))
    if '--policy.path' not in keys:
        default('--policy.type', 'molmoact2')
        default('--policy.checkpoint_path', environ.get('S2R_BASE_MODEL'))
    return result


def probe_args(argv=None, environ=None):
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description='Run actual Molmo LoRA GPU optimizer probes')
    parser.add_argument('--dataset', type=Path, default=environ.get('S2R_DATASET'))
    parser.add_argument('--checkpoint', type=Path, default=environ.get('S2R_BASE_MODEL'))
    output = environ.get('S2R_OUTPUT')
    parser.add_argument('--report', type=Path, default=str(Path(output) / 'verification.json') if output else None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=16)
    parser.add_argument('--updates', type=int, default=2)
    args = parser.parse_args(argv)
    if any(getattr(args, key) is None for key in ('dataset', 'checkpoint', 'report')):
        parser.error('dataset, checkpoint and report need explicit flags or S2R_DATASET/BASE_MODEL/OUTPUT')
    if min(args.batch_size, args.accumulation, args.updates) < 1:
        parser.error('batch size, accumulation and updates must be positive')
    return args
