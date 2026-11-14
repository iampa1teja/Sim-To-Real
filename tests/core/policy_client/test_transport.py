"""The real request/reply transport runs against a CPU policy server."""
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np

from core.policy_client.contract import ArmInfo
from core.policy_client.remote_policy import RemotePolicy

ROOT = Path(__file__).resolve().parents[3]


def test_remote_policy_configure_seed_and_actions_over_real_zmq(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'src/core'))
    from policy_server.contract import load_server_policy
    from policy_server.server import PolicyServer

    ready = Queue()

    def serve():
        policy = load_server_policy(ROOT / 'workspace/policies/example')
        policy.load('', 'cpu', {})
        server = PolicyServer(policy, host='127.0.0.1', port=0, timeout_ms=1000, api_token='local-test')
        ready.put(server.port)
        try:
            server.run()
        finally:
            server.close()

    thread = Thread(target=serve, daemon=True)
    thread.start()
    port = ready.get(timeout=5)
    names = ('left', 'yaw', 'reach', 'right')
    info = ArmInfo('four_axes', names, ('gripper', 'arm', 'arm', 'gripper'),
                   ('prismatic', 'revolute', 'prismatic', 'prismatic'),
                   {name: dict(joint_min=0., joint_max=1.) for name in names})
    policy = RemotePolicy(info.feature_names, policy_name='example', host='127.0.0.1', port=port,
                          api_token='local-test', timeout_ms=1000, arm_info=info)
    try:
        policy.connect()
        policy.reset(seed=1984)
        state = np.array([25., -12., 50., 75.], dtype=np.float32)
        images = {name: np.arange(18, dtype=np.uint8).reshape(2, 3, 3) for name in ('room', 'wrist')}
        for _ in range(9):
            assert policy.get_raw_action(state, images) == dict(zip(info.feature_names, state))
        assert policy.inference_calls == 2
        assert policy.client.describe()['calls'] == 2
        policy.client.call_endpoint('kill', requires_input=False)
    finally:
        policy.close()
        policy.close()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_transport_close_is_explicit_and_idempotent():
    from types import SimpleNamespace
    from core.policy_client.transport import PolicyClient
    calls = []
    client = PolicyClient.__new__(PolicyClient)
    client.socket = SimpleNamespace(close=lambda **kwargs: calls.append(('socket', kwargs)))
    client.context = SimpleNamespace(term=lambda: calls.append(('context', {})))
    client.close(); client.close()
    assert calls == [('socket', {'linger': 0}), ('context', {})]
