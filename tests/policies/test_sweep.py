"""CPU checks for calibration bounds, timed playback and target-aligned measurements."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def sweep():
    return module(ROOT / 'workspace/policies/sweep/make_sweep.py').make_sweep()


def test_sweep_bounds_speed_and_exact_holds(sweep):
    ticks = np.asarray(sweep['ticks'])
    assert len(sweep['holds']) == 54
    assert sweep['joint_names'][-2:] == ['wrist_roll', 'gripper']
    assert np.all(ticks >= sweep['raw_min']) and np.all(ticks <= sweep['raw_max'])
    assert np.all(ticks[:, :-1] >= -100) and np.all(ticks <= 100) and np.all(ticks[:, -1] >= 0)
    speeds = np.abs(np.diff(ticks, axis=0)) * sweep['deg_per_raw'] * sweep['fps']
    assert speeds.max() <= 15 + 1e-9
    for hold in sweep['holds']:
        start, end = hold['start_tick'], hold['end_tick']
        assert end - start == sweep['hold_s'] * sweep['fps']
        np.testing.assert_allclose(ticks[start:end], np.tile(hold['raw'], (end - start, 1)))
        # Only the swept joint leaves home, including every transition.
        j = sweep['joint_names'].index(hold['joint'])
        others = [k for k in range(len(sweep['joint_names'])) if k != j]
        np.testing.assert_allclose(ticks[start, others], np.asarray(sweep['home_raw'])[others])
        assert not np.allclose(ticks[start - 1], hold['raw'], atol=1e-10, rtol=0)
        assert not np.allclose(ticks[end], hold['raw'], atol=1e-10, rtol=0)
        raw = np.asarray(hold['raw'])
        source = json.loads((ROOT / 'workspace/arms/so101/calibration/units/rpd_follower.json').read_text())['source']['joints']
        for i, name in enumerate(sweep['joint_names']):
            fraction = (raw[i] - sweep['raw_min'][i]) / (sweep['raw_max'][i] - sweep['raw_min'][i])
            assert hold['expected_encoder_counts'][i] == pytest.approx(
                source[name]['range_min'] + fraction * (source[name]['range_max'] - source[name]['range_min']))
    np.testing.assert_allclose(ticks[-1], sweep['home_raw'])


def test_home_uses_scene_mount_and_exclusion_preserves_indices(sweep):
    from utils.arms.calibration import load_unit
    from utils.arms.spec import load_arm_spec
    from utils.arms.mapping import units_to_physical
    from utils.geometry import mount_joint_positions
    from utils.scene import load_scene_spec, load_mount_spec
    arm = load_arm_spec('so101')[0]
    mount = mount_joint_positions(arm, load_mount_spec(load_scene_spec('rpd_meeting_room'), arm))
    physical = units_to_physical(sweep['home_raw'], load_unit(arm, 'rpd_follower').mapping,
                                 arm.joint_names, gripper_joints=arm.gripper_joint_names)
    np.testing.assert_allclose(physical, np.rad2deg([mount[j.usd] for j in arm.joints]))
    excluded = module(ROOT / 'workspace/policies/sweep/make_sweep.py').make_sweep(exclude='wrist_roll:0,gripper:8')
    assert len(excluded['holds']) == 52
    assert not any((h['joint'], h['index']) in {('wrist_roll', 0), ('gripper', 8)} for h in excluded['holds'])
    assert np.max(np.abs(np.diff(excluded['ticks'], axis=0)) * excluded['deg_per_raw'] * excluded['fps']) <= 15


def test_server_ramp_reset_padding_and_independent_clients(sweep, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'src/core'))
    server = module(ROOT / 'workspace/policies/sweep/server.py').create()
    path = tmp_path / 'sweep.json'
    # Shorten playback while keeping a non-home endpoint to test end padding.
    sweep['ticks'] = [sweep['home_raw'], sweep['holds'][0]['raw']]
    path.write_text(json.dumps(sweep))
    server.load('none', 'cpu', {'trajectory': str(path), 'horizon': '30'})
    home = np.asarray(sweep['home_raw'])
    start = np.clip(home + 20, sweep['raw_min'], sweep['raw_max'])
    request = dict(state=start, sweep_stream='real')
    chunks = [server.infer(request)['actions']]
    assert chunks[0].dtype == np.float32 and chunks[0].shape == (30, 6)
    np.testing.assert_allclose(chunks[0][0], start, atol=1e-5)
    np.testing.assert_allclose(server.observed_start, start)
    # A second client begins from its own start even while real is in flight.
    other = server.infer(dict(state=home, sweep_stream='sim'))['actions']
    np.testing.assert_allclose(other[0], home, atol=1e-5)
    while sum(len(c) for c in chunks) < 400:
        chunks.append(server.infer(request)['actions'])
    rows = np.concatenate(chunks)
    ramp_count = int(np.ceil(1.5 * np.max(np.abs(home - start) * sweep['deg_per_raw']) / 15 * sweep['fps']))
    assert np.max(np.abs(np.diff(rows[:ramp_count + 1], axis=0)) * sweep['deg_per_raw'] * sweep['fps']) < 15.001
    np.testing.assert_allclose(rows[ramp_count], home, atol=1e-5)
    np.testing.assert_allclose(rows[-1], home, atol=1e-5)
    assert server.info()['tick'] >= server.info()['total_ticks']
    server.reset()
    assert server.info()['tick'] == 0
    np.testing.assert_equal(server.infer(request)['actions'], chunks[0])


def test_sweep_client_supports_room_camera_only():
    from core.policy_client.discovery import get_manifest
    client = module(ROOT / 'workspace/policies/sweep/client.py')
    manifest = get_manifest('sweep')
    policy = client.create(manifest, {'cameras': 'room'})
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    request = policy.build_request(np.zeros(6), {'room': image}, '')
    assert list(request['images']) == ['room']
    assert policy.cameras == ('room',)
    assert client.create(manifest, {}).cameras == ('room', 'wrist')
    with pytest.raises(ValueError, match='subset'):
        client.create(manifest, {'cameras': 'unknown'})
