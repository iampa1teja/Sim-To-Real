"""Passive gripper coordinates at the articulation boundary, never in actions."""
import torch

from isaaclab.envs.mdp import JointPositionAction, reset_joints_by_offset

from ._common import _arm_spec, as_torch, env_indices


def mimic_control_mode(robot, arm):
    """Inspect the live Newton constraints, never import-provenance metadata."""
    if hasattr(robot, '_s2r_mimic_runtime'):
        return robot._s2r_mimic_runtime
    from isaaclab_newton.physics import NewtonManager
    model = NewtonManager.get_model()
    mode = 'manual'
    if model is not None and model.constraint_mimic_count:
        rows = zip(model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy(),
                   model.constraint_mimic_coef0.numpy(), model.constraint_mimic_coef1.numpy(),
                   model.constraint_mimic_enabled.numpy())
        constraints = [(model.joint_label[a], model.joint_label[b], float(offset), float(scale))
                       for a,b,offset,scale,enabled in rows if enabled]
        sources = {joint.name:joint.usd for joint in arm.action_joints}
        followers = [joint for joint in arm.joints if joint.mimic is not None]
        if followers and all(any(a.rsplit('/',1)[-1] == joint.usd
                and b == a.rsplit('/',1)[0] + '/' + sources[joint.mimic.joint]
                and abs(offset-joint.mimic.offset)<1e-7 and abs(scale-joint.mimic.multiplier)<1e-7
                for a,b,offset,scale in constraints) for joint in followers):
            mode = 'constraint'
    if mode == 'constraint':
        follower_ids = robot.find_joints([joint.usd for joint in arm.joints if joint.mimic], preserve_order=True)[0]
        # A passive follower must not fight the converted equality constraint
        # with an independent position drive aimed at its reset coordinate.
        robot.write_joint_stiffness_to_sim(0., joint_ids=follower_ids)
        robot.write_joint_damping_to_sim(0., joint_ids=follower_ids)
    robot._s2r_mimic_runtime = mode
    print(f'[mimic] {mode}: {arm.name}', flush=True)
    return mode


def mimic_targets(arm, action_positions):
    """Return follower targets in declaration order from source action coordinates."""
    indices = {joint.name: index for index, joint in enumerate(arm.action_joints)}
    return torch.stack([action_positions[..., indices[joint.mimic.joint]] * joint.mimic.multiplier
                        + joint.mimic.offset for joint in arm.joints if joint.mimic is not None], dim=-1)


def write_mimic_state(env, env_ids=None):
    """Synchronize passive positions and velocities after any direct state reset."""
    arm = _arm_spec(env)
    if not arm.mimic_joints:
        return
    robot = env.scene['robot']
    ids = env_indices(env, env_ids)
    source_ids = robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)[0]
    followers = [joint for joint in arm.joints if joint.mimic is not None]
    follower_ids = robot.find_joints([joint.usd for joint in followers], preserve_order=True)[0]
    position = mimic_targets(arm, as_torch(robot.data.joint_pos)[ids][:, source_ids])
    source_indices = {joint.name: index for index, joint in enumerate(arm.action_joints)}
    velocity = torch.stack([as_torch(robot.data.joint_vel)[ids, source_ids[source_indices[joint.mimic.joint]]]
                            * joint.mimic.multiplier for joint in followers], dim=-1)
    robot.write_joint_state_to_sim_index(position=position, velocity=velocity,
                                         joint_ids=follower_ids, env_ids=ids)
    if mimic_control_mode(robot, arm) != 'constraint':
        robot.set_joint_position_target_index(target=position, joint_ids=follower_ids, env_ids=ids)


def reset_joints_with_mimic(env, env_ids, position_range, velocity_range, asset_cfg):
    reset_joints_by_offset(env, env_ids, position_range=position_range, velocity_range=velocity_range, asset_cfg=asset_cfg)
    write_mimic_state(env, env_ids)


class MimicJointPositionAction(JointPositionAction):
    """Dispatch source actions and, unless a validated constraint owns them, followers."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._arm = _arm_spec(env)
        self._followers = [joint for joint in self._arm.joints if joint.mimic is not None]
        self._follower_ids = self._asset.find_joints([joint.usd for joint in self._followers],
                                                  preserve_order=True)[0]

    def apply_actions(self):
        super().apply_actions()
        if self._followers and mimic_control_mode(self._asset, self._arm) != 'constraint':
            targets = mimic_targets(self._arm, as_torch(self.processed_actions))
            self._asset.set_joint_position_target_index(target=targets, joint_ids=self._follower_ids)
