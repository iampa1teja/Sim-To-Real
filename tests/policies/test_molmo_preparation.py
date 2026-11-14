"""Retained SO-101 recipe: portable payloads, exact codec and container arguments."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.arms.calibration import resolve_calibration
from utils.arms.mapping import units_to_physical
from utils.arms.arms import load_arm_spec
from utils.datasets.sources import recorded_joint_mapping
from utils.hashing import tree_sha256

SCRIPTS = Path(__file__).resolve().parents[2] / 'workspace/policies/molmoact2/train'


def module(name):
    spec = importlib.util.spec_from_file_location('test_molmo_' + name, SCRIPTS / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    arm, _ = load_arm_spec('so101')
    mapping = resolve_calibration(arm, unit='rpd_robot').mapping
    for folder in ('meta', 'data/chunk-000', 'videos/camera', 'task_meta', 'datagen_meta'):
        (root / folder).mkdir(parents=True)
    features = {key: {'names': list(arm.feature_names)} for key in ('observation.state', 'action')}
    (root / 'meta/info.json').write_text(json.dumps(dict(features=features, total_frames=3, total_episodes=1)))
    (root / 'meta/stats.json').write_text('{}')
    marker = dict(arm=arm.name, joint_names=list(arm.joint_names), use_degrees=False, joint_mapping=mapping)
    (root / 'meta/recording_alignment.json').write_text(json.dumps(marker))
    (root / 'meta/datagen.json').write_text(json.dumps(dict(joint_mapping=mapping)))
    (root / 'videos/camera/episode.mp4').write_bytes(b'unchanged-video-payload')
    (root / 'task_meta/episode_000000.json').write_text('{"objects":{"a":{},"b":{}}}')
    (root / 'datagen_meta/episode_000000.json').write_text('{"frames":[]}')
    values = np.array([[-80, -40, 0, 20, 60, 0], [0, 10, 20, 30, 40, 50], [80, 60, 40, 20, 0, 100]],
                      dtype=np.float32)
    pq.write_table(pa.table({'episode_index': [0, 0, 0], 'observation.state': values.tolist(),
                             'action': values.tolist()}), root / 'data/chunk-000/file.parquet')
    return root, arm, mapping, values


def test_prepare_from_container_environment_preserves_physical_mapping_and_payloads(source, tmp_path, monkeypatch):
    root, arm, mapping, values = source
    before = tree_sha256(root)
    output = tmp_path / 'prepared'
    monkeypatch.setenv('S2R_DATASET', str(root))
    monkeypatch.setenv('S2R_OUTPUT', str(output))
    monkeypatch.delenv('S2R_UNIT', raising=False)
    module('prepare_molmo_lora').main([])
    converted = np.array(pq.read_table(output / 'data/chunk-000/file.parquet')['action'].to_pylist())
    prepared, _ = recorded_joint_mapping(output, arm)
    kwargs = dict(joint_order=arm.joint_names, gripper_joints=arm.gripper_joint_names)
    np.testing.assert_allclose(units_to_physical(converted, prepared, **kwargs),
                               units_to_physical(values, mapping, **kwargs), atol=2e-5, rtol=0)
    np.testing.assert_array_equal(converted[:, -1], values[:, -1])
    assert tree_sha256(root) == before
    assert not (output / 'videos').is_symlink()
    assert tree_sha256(root / 'videos') == tree_sha256(output / 'videos')
    for folder in ('task_meta', 'datagen_meta'):
        assert tree_sha256(root / folder) == tree_sha256(output / folder)
    assert (output / 'meta/source_recording_alignment.json').read_bytes() == (root / 'meta/recording_alignment.json').read_bytes()
    assert not (output / 'meta/datagen.json').exists()
    stats = json.loads((output / 'meta/molmo_norm_stats.json').read_text())['metadata_by_tag']
    expected = converted * [1, -1, 1, 1, 1, 1] + [0, 90, 90, 0, 0, 0]
    np.testing.assert_allclose(stats['workshop_pick_place_v1_as_real']['action_stats']['q01'],
                               np.quantile(expected, .01, axis=0), atol=1e-5, rtol=0)


def test_prepare_rejects_wrong_features_and_mapping_without_fallback(source, tmp_path):
    root, arm, mapping, values = source
    prep = module('prepare_molmo_lora')
    with pytest.raises(ValueError, match='differs from the source mapping'):
        prep.prepare(root, tmp_path / 'wrong_unit', unit='rpd_follower')
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['action']['names'].reverse()
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='feature order'):
        prep.prepare(root, tmp_path / 'wrong_order')
    assert not (tmp_path / 'wrong_order').exists()
    (root / 'meta/recording_alignment.json').unlink()
    with pytest.raises(ValueError, match='saved joint mapping'):
        prep.prepare(root, tmp_path / 'missing_mapping')


def test_training_and_gpu_probe_respect_explicit_arguments():
    inputs = module('command_inputs')
    env = {'S2R_DATASET': '/data/dataset', 'S2R_OUTPUT': '/data/output', 'S2R_BASE_MODEL': '/data/base_model'}
    argv = inputs.training_args(['--dataset.root', '/other', '--output_dir=/chosen', '--steps=2'], env)
    assert '--dataset.root=/data/dataset' not in argv and '--output_dir=/data/output' not in argv
    assert '--policy.norm_stats_path=/other/meta/molmo_norm_stats.json' in argv
    assert '--policy.checkpoint_path=/data/base_model' in argv and '--steps=2' in argv
    assert inputs.training_args(['--config_path=resume.json'], env) == ['--config_path=resume.json']
    args = inputs.probe_args(['--report', '/chosen.json', '--updates', '1'], env)
    assert args.dataset == Path('/data/dataset') and args.checkpoint == Path('/data/base_model')
    assert args.report == Path('/chosen.json') and args.updates == 1
    with pytest.raises(SystemExit):
        inputs.probe_args([], {})
