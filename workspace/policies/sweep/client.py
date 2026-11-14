"""Standard absolute-joint client for the sweep plugin."""
from core.policy_client.default import DefaultClientPolicy
from uuid import uuid4


class SweepClientPolicy(DefaultClientPolicy):
    """Keep the paired tool's two clients independent on one policy server."""

    def reset(self):
        self.stream = uuid4().hex

    def build_request(self, joint_positions, images, instruction):
        if not hasattr(self, 'stream'):
            self.reset()
        request = super().build_request(joint_positions, images, instruction)
        request['sweep_stream'] = self.stream
        return request


def create(manifest, options):
    policy = SweepClientPolicy.from_manifest(manifest, options)
    if 'cameras' in options:
        cameras = tuple(options['cameras'].split(','))
        if not cameras or not set(cameras) <= set(manifest.cameras):
            raise ValueError('cameras must be a nonempty subset of the manifest cameras')
        policy.cameras = cameras
    return policy
