"""GR00T preparation retains current and historical S2R object metadata."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np

from core.datasets.task_meta import read_task_meta, save_task_trajectory

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'workspace/policies/groot_n17/train'
spec = importlib.util.spec_from_file_location('groot_sidecar_copy', SCRIPTS / 'dataset_sidecars.py')
sidecars = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sidecars)


def test_schema2_preparation_renumbers_every_object_without_changing_source(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'prepared'
    poses = {'blue_cube': np.array([[.1, .2, .3, 1, 0, 0, 0], [.2, .3, .4, 1, 0, 0, 0]]),
             'target': np.array([[.4, .5, .6, 1, 0, 0, 0], [.4, .5, .6, 1, 0, 0, 0]])}
    save_task_trajectory(source, 7, poses, {name: (.02, .02, .02) for name in poses}, 30,
                         task='pick_place_cube', arm='so101')
    original = source / 'task_meta/episode_000007.json'
    before = original.read_bytes()
    [copied] = sidecars.copy_task_sidecars(source, target, {0: 7}, {7: 2})
    data = read_task_meta(copied, episode_index=0, num_frames=2)
    assert data['task'] == 'pick_place_cube' and data['arm'] == 'so101'
    assert set(data['objects']) == set(poses)
    for name, values in poses.items():
        np.testing.assert_array_equal(data['objects'][name]['position'], values[:, :3])
        np.testing.assert_array_equal(data['objects'][name]['orientation_wxyz'], values[:, 3:])
    assert original.read_bytes() == before


def test_schema1_preparation_keeps_legacy_layout(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'prepared'
    path = source / 'pick_place_meta/episode_000003.json'
    path.parent.mkdir(parents=True)
    data = dict(schema_version=1, episode_index=3, num_frames=1, fps=30, units='m',
                reference_frame='robot base link (base)', cube_size_m=[.02, .02, .02],
                position=[[.1, .2, .3]], orientation_wxyz=[[1, 0, 0, 0]])
    path.write_text(json.dumps(data))
    [copied] = sidecars.copy_task_sidecars(source, target, {0: 3}, {3: 1})
    assert copied.parent.name == 'pick_place_meta'
    assert json.loads(copied.read_text()) == {**data, 'episode_index': 0}


def test_small_training_flags_reach_the_existing_recipe_without_starting_cuda(tmp_path):
    environment = dict(os.environ, GROOT=str(tmp_path),
                       DATASET=str(tmp_path / 'data'), OUT=str(tmp_path / 'out'),
                       BASE_MODEL=str(tmp_path / 'base'))
    result = subprocess.run(['bash', str(SCRIPTS / 'train_pick_place.sh'), '--batch', '1',
        '--max-steps', '1', '--save-steps', '1', '--save-total-limit', '1', '--dry-run'],
        env=environment, capture_output=True, text=True, check=True)
    assert '--global-batch-size 1' in result.stdout
    assert '--max-steps 1' in result.stdout and '--save-steps 1' in result.stdout
    assert 'Dry run only' in result.stdout
    assert not (tmp_path / 'out').exists()


def test_download_command_preserves_every_hf_include_pattern(tmp_path):
    spec = importlib.util.spec_from_file_location('groot_model_profiles', SCRIPTS / 'gr00t_model_profiles.py')
    profiles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiles)
    destination = tmp_path / 'model folder $literal'
    argv = shlex.split(profiles.download_command(destination))
    assert argv[:2] == ['hf', 'download']
    # Installed HF download uses nargs='*' with the default store action.
    # Repeating --include overwrites earlier patterns rather than extending them.
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_id')
    parser.add_argument('--revision')
    parser.add_argument('--include', nargs='*')
    parser.add_argument('--local-dir')
    args = parser.parse_args(argv[2:])
    assert args.repo_id == profiles.SO_ARM_REPO and args.revision == profiles.SO_ARM_REVISION
    assert args.local_dir == str(destination)
    assert args.include == ['config.json', 'processor_config.json', 'embodiment_id.json',
                            'statistics.json', 'model-*.safetensors', 'model.safetensors.index.json']
