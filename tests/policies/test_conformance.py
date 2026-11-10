"""Every enabled policy declares a loadable client/server and NumPy wire format."""
from pathlib import Path

import numpy as np
import pytest

from core.arms.calibration import resolve_calibration
from core.arms.spec import load_arm_spec
from core.policy_client.contract import ArmInfo, ClientPolicy, load_client_policy
from core.policy_client.discovery import discover
from core.policy_client.transport import MsgSerializer

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = discover(ROOT / 'policies')


@pytest.mark.parametrize('name', sorted(MANIFESTS))
def test_enabled_plugin_factories_and_wire_round_trip(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'src/core'))
    from policy_server.contract import ServerPolicy, load_server_policy
    from policy_client.transport import MsgSerializer as ServerSerializer

    manifest = MANIFESTS[name]
    assert manifest.name == manifest.path.name
    arm = load_arm_spec('so101')[0]
    info = ArmInfo.from_arm(arm, resolve_calibration(arm, unit='rpd_robot').mapping)
    assert manifest.action_dim in ('arm', len(info.joint_names))
    client = load_client_policy(name, manifest, {})
    assert isinstance(client, ClientPolicy)
    client.configure(info)
    server = load_server_policy(manifest.path)
    assert isinstance(server, ServerPolicy)
    assert server.info()['name'] == name
    state = np.array([-17., -45., 20., 35., 7., 60.], dtype=np.float32)
    images = {camera: np.arange(18, dtype=np.uint8).reshape(2, 3, 3) for camera in manifest.cameras}
    request = ServerSerializer.from_bytes(MsgSerializer.to_bytes(client.build_request(state, images, 'move')))
    assert request['state'].shape == (len(info.joint_names),)
    assert list(request['images']) == list(manifest.cameras)
    response = MsgSerializer.from_bytes(ServerSerializer.to_bytes({'actions': np.stack([request['state']] * 2)}))
    result = client.parse_response(response, list(info.feature_names))
    assert result.shape == (2, len(info.joint_names))
    np.testing.assert_allclose(result, np.stack([state] * 2), atol=1e-4, rtol=0)
    client.reset()
