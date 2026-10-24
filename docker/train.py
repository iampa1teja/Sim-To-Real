#!/usr/bin/env python3
"""Run the pinned GR00T training tools in Docker, retaining host output paths."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
GROOT = '/Isaac-GR00T'
PYTHON = f'{GROOT}/.venv/bin/python'
WORKFLOWS = ('train', 'sim', 'failures', 'dit', 'prepare', 'verify', 'check', 'python')
SETTINGS = (
    'BATCH', 'MAX_STEPS', 'SAVE_STEPS', 'SAVE_TOTAL_LIMIT', 'MILESTONE_STEPS',
    'MILESTONE_KEEP', 'RESUME', 'DATALOADER_NUM_WORKERS', 'MEASURED_S_PER_STEP',
    'LEARNING_RATE', 'FT_STEPS', 'BAD_EPISODES', 'PREFLIGHT_ONLY',
    'PYTORCH_CUDA_ALLOC_CONF',
)


def absolute(value):
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def build_command(workflow, forwarded, environ):
    """Pure command construction: dry runs neither create paths nor call Docker."""
    env = dict(environ)
    external = Path('/external_storage/models')
    names = {'train': 'so101_pick_place_v2_nvinit', 'sim': 'so101_pick_place_sim_n17_100k',
             'failures': 'so101_pick_place_sim_n17_200k_failures_ft'}
    defaults = {
        'BASE_MODEL': external / ('so101_pick_place_sim_n17_200k/milestones/checkpoint-200000'
                                 if workflow == 'failures' else 'nv_so_arm_n17'),
        'DATASET': REPO / 'datasets' / ('pick_place_sim' if workflow == 'sim' else 'pick_place_v2_gr00t'),
        'OUT': external / names.get(workflow, 'so101_pick_place_v2_nvinit'),
    }
    paths = ({name: absolute(env.get(name, default)) for name, default in defaults.items()}
             if workflow in ('train', 'sim', 'failures') else {})
    if workflow == 'failures':
        paths.pop('DATASET')  # The wrapper prepares and selects CLEAN/FAILED itself.
    if workflow == 'python':
        paths.update({name: absolute(env[name]) for name in defaults if env.get(name)})
    for name in ('MILESTONE_DIR', 'SOURCE', 'CLEAN', 'FAILED', 'RESULTS'):
        if env.get(name):
            paths[name] = absolute(env[name])
    extras = [absolute(p) for p in env.get('EXTRA_DATASETS', '').split(':') if p]
    spec = importlib.util.spec_from_file_location('train_model_profiles', REPO / 'policies/groot_n17/train/gr00t_model_profiles.py')
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    hub = profiles.transformers_hub_cache(env)
    hf_home = absolute(env.get('HF_HOME', Path.home() / '.cache/huggingface'))
    cache = absolute(env.get('TRAIN_CACHE', Path.home() / '.cache/so101-training'))
    mounts = {}

    def mount(path, writable=False):
        # Mount the closest existing parent for a not-yet-created output. Docker
        # must not silently create missing model/dataset paths as root.
        path = absolute(path)
        while not path.exists():
            path = path.parent
        if path == Path('/'):
            raise ValueError('No usable parent for a training mount; create its storage directory first')
        if ',' in str(path):
            raise ValueError('Docker training mount paths cannot contain commas')
        mounts[path] = mounts.get(path, False) or writable

    mount(REPO, workflow in ('prepare', 'failures', 'python'))
    for name, path in paths.items():
        mount(path, name in ('OUT', 'MILESTONE_DIR', 'CLEAN', 'FAILED'))
    for path in extras:
        mount(path)
    mount(hf_home, True)
    mount(hub, True)
    # Created by main only after validating the image, never by a dry run.
    mounts[cache] = True

    forwarded = list(forwarded)
    writable_flags = {'--output', '--output_dir'}
    path_flags = {'--source', '--output', '--dataset', '--init_ckpt', '--output_dir'}
    for index, argument in enumerate(forwarded):
        flag, separator, value = argument.partition('=')
        if flag == '--groot':
            raise ValueError('GR00T is pinned inside the image; omit --groot')
        if flag not in path_flags:
            continue
        value = value if separator else forwarded[index + 1] if index + 1 < len(forwarded) else ''
        if not value or value.startswith('--'):
            raise ValueError(f'{flag} requires a path')
        path = absolute(value)
        mount(path, flag in writable_flags)
        if separator:
            forwarded[index] = f'{flag}={path}'
        else:
            forwarded[index + 1] = str(path)
    if workflow == 'dit':
        supplied = {argument.partition('=')[0] for argument in forwarded}
        if '--init_ckpt' not in supplied:
            mount(external / 'so101_pick_place_sim_n17_200k/milestones/checkpoint-200000')
        if '--output_dir' not in supplied:
            mount(external / 'so101_pick_place_sim_n17_200k_dit_only', True)

    container_env = {name: env[name] for name in SETTINGS if name in env}
    container_env.update({name: str(path) for name, path in paths.items()})
    container_env.update(
        SO101_TRAIN_IN_CONTAINER='1', GROOT=GROOT, CUDA_HOME='/usr/local/cuda',
        GROOT_N17_CHECKOUT=GROOT, GROOT_N17_PYTHON=PYTHON,
        PYTHONPATH=GROOT, HF_HOME=str(hf_home), HF_HUB_CACHE=str(hub),
        XDG_CACHE_HOME=str(cache),
        TORCH_HOME=str(cache / 'torch'), TRITON_CACHE_DIR=str(cache / 'triton'),
        EXTRA_DATASETS=':'.join(map(str, extras)),
    )
    command = ['docker', 'run', '--rm', '--init', '--network', 'none', '--shm-size', '8g',
               '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', str(REPO)]
    if workflow not in ('prepare', 'verify'):
        # NVML reports host PIDs. Let the existing busy-GPU guard resolve their
        # names so the idle Xorg baseline is distinguished from compute jobs.
        command.extend(['--gpus', 'all', '--pid', 'host'])
    for group in sorted(set(os.getgroups()) - {os.getgid()}):
        command.extend(['--group-add', str(group)])
    for path, writable in sorted(mounts.items(), key=lambda item: (len(item[0].parts), str(item[0]))):
        command.extend(['--mount', f'type=bind,source={path},target={path}' + ('' if writable else ',readonly')])
    for name, value in sorted(container_env.items()):
        command.extend(['--env', f'{name}={value}'])
    command.append(env.get('TRAIN_IMAGE', 'so101-groot-train:n1.7'))
    shell_tools = {'train': 'policies/groot_n17/train/train_pick_place.sh', 'sim': 'policies/groot_n17/train/train_pick_place_sim_100k.sh',
                   'failures': 'policies/groot_n17/train/train_pick_place_failures_ft.sh', 'dit': 'policies/groot_n17/train/train_dit.sh'}
    if workflow in shell_tools:
        command.extend(['bash', str(REPO / shell_tools[workflow]), *forwarded])
    elif workflow == 'prepare':
        command.extend([PYTHON, str(REPO / 'policies/groot_n17/train/prepare_pick_place_gr00t.py'),
                        '--groot', GROOT, '--model_profile', 'so_arm_n17', *forwarded])
    elif workflow == 'verify':
        command.extend([PYTHON, str(REPO / 'policies/groot_n17/train/verify_pick_place_training.py'), *forwarded])
    elif workflow == 'python':
        command.extend([PYTHON, *forwarded])
    else:
        command.extend([PYTHON, str(REPO / 'policies/groot_n17/train/check_training_runtime.py'), *forwarded])
    return command, cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Examples: ./docker/train.sh --dry_run; ./docker/train.sh --preflight_only; '
        './docker/train.sh prepare --source datasets/pick_place_v1 --output datasets/pick_place_sim. '
        'Training settings retain their environment-variable interface; output defaults to /external_storage/models.'))
    parser.add_argument('workflow', nargs='?', choices=WORKFLOWS, default='train')
    parser.add_argument('--dry_run', action='store_true', help='Print the Docker command without running or creating anything')
    args, forwarded = parser.parse_known_args(argv)
    try:
        command, cache = build_command(args.workflow, forwarded, os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    print('Docker command: ' + shlex.join(command), flush=True)
    if args.dry_run or os.environ.get('DRY_RUN') == '1':
        return 0
    image = os.environ.get('TRAIN_IMAGE', 'so101-groot-train:n1.7')
    result = subprocess.run(['docker', 'image', 'inspect', image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode:
        parser.exit(1, f'Missing training image {image}; run ./docker/train/build.sh first.\n')
    cache.mkdir(parents=True, exist_ok=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    sys.exit(main())
