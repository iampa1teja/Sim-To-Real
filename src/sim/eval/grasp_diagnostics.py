# sim/eval/grasp_diagnostics.py  (ported from v1 utils/grasp_diagnostics.py)
"""Read-only, in-memory measurements of a pick-place grasp attempt."""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Live body quaternions are XYZW; converted to WXYZ for the numpy rotation helper.
# - Arm-specific fingertip geometry and measurement are optional arm-folder hooks.
# - Gripper percent comes from the arm's JointMapping instead of v1's robot_iface bounds.
from __future__ import annotations

import math
import logging

import numpy as np

log = logging.getLogger(__name__)


class GraspTrace:
    """Aggregate closest approach and representative closing events; no I/O."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.command = None
        self.step = 0
        self.closest = None
        self.closest_closing = None
        self.first_closing = None
        self.closing_events = 0
        self._previous_gripper = None
        self._closing = False

    def observe(self, measurement, command_gripper_pct):
        measurement = dict(measurement, command_gripper_pct=float(command_gripper_pct))
        if self.closest is None or measurement['tool_error_mm'] < self.closest['tool_error_mm']:
            self.closest = measurement
        closing = self._previous_gripper is not None and command_gripper_pct < self._previous_gripper - .1
        if closing:
            if not self._closing:
                self.closing_events += 1
                if self.first_closing is None:
                    self.first_closing = measurement
            if self.closest_closing is None or measurement['tool_error_mm'] < self.closest_closing['tool_error_mm']:
                self.closest_closing = measurement
        self._closing = closing
        self._previous_gripper = command_gripper_pct

    def summary(self):
        return dict(closest=self.closest, closest_closing=self.closest_closing,
                    first_closing=self.first_closing, closing_events=self.closing_events,
                    coordinate_frame='fixed gripper; tool point is cube centre against fixed fingertip',
                    jaw_gap_source='opposing CAD fingertip centres projected on gripper X')


def _read_frame(torch, robot, body_ids, joint_ids, cube_position, command):
    """Transfer one consistent telemetry frame with a single device sync (quaternions returned WXYZ)."""
    from sim.envs.mdp.resets import _wxyz, as_torch
    packed = torch.cat((as_torch(robot.data.body_link_pos_w)[0, body_ids].reshape(-1),
                        as_torch(robot.data.body_link_quat_w)[0, body_ids].reshape(-1),
                        cube_position, command, as_torch(robot.data.joint_pos)[0, joint_ids]))
    frame = packed.detach().cpu().numpy()
    joints = len(command)
    return (frame[:6].reshape(2, 3), _wxyz(frame[6:14].reshape(2, 4)),
            frame[14:17], frame[17:17 + joints], frame[17 + joints:17 + 2 * joints])


def _tracking_errors(command, measured, revolute):
    """Report angular and linear tracking errors separately."""
    error = np.abs(np.asarray(command) - np.asarray(measured))
    revolute = np.asarray(revolute, dtype=bool)
    result = {}
    if revolute.any():
        result["max_joint_tracking_error_deg"] = float(np.max(error[revolute]) * 180 / math.pi)
    if (~revolute).any():
        result["max_joint_tracking_error_m"] = float(np.max(error[~revolute]))
    return result


def install_grasp_diagnostics(env, mapping, step_dt):
    """Observe the success term before Isaac's automatic reset, without changing it.

    mapping: core.arms.mapping.JointMapping used to express the gripper in raw percent.
    """
    import torch
    from isaaclab.sim import get_current_stage

    from core.arms.spec import load_arm_spec, load_arm_utils
    from sim.envs.mdp.resets import as_torch

    arm, _ = load_arm_spec()
    utilities = load_arm_utils(arm)
    offsets = getattr(utilities, "grasp_tip_offsets", None)
    measure = getattr(utilities, "measure_grasp", None)
    gripper_indices = mapping.gripper_mask.nonzero().flatten()
    if (arm.gripper is None or len(arm.gripper.contact_bodies) != 1 or len(gripper_indices) != 1
            or offsets is None or measure is None):
        log.warning("Grasp geometry diagnostics unavailable for arm %s", arm.name)
        return None
    robot = env.scene['robot']
    body_ids = [robot.find_bodies(name)[0][0] for name in (arm.links.ee, arm.gripper.contact_bodies[0])]
    joint_ids, _ = robot.find_joints(list(arm.usd_joint_names), preserve_order=True)
    revolute = np.array([joint.type == "revolute" for joint in arm.joints])
    cube_size = float(env.scene['blue_cube'].cfg.spawn.size[0])
    fixed_tip, moving_tip = offsets(get_current_stage(), env.scene.env_prim_paths[0] + '/Robot', arm,
                                   cube_size / 2)
    gripper_index = int(gripper_indices[0].item())
    trace = GraspTrace()
    cfg = env.termination_manager.get_term_cfg('success')
    original = cfg.func

    def gripper_pct(radians):
        return float(mapping.to_raw(torch.as_tensor(radians))[gripper_index].item())

    def observe(raw, **params):
        result = original(raw, **params)
        if trace.command is not None:
            positions, quaternions, cube, command, measured = _read_frame(
                torch, robot, body_ids, joint_ids, as_torch(raw.scene['blue_cube'].data.root_pos_w)[0], trace.command)
            value = measure(positions[0], quaternions[0], positions[1], quaternions[1],
                                  cube, fixed_tip, moving_tip, cube_size)
            value.update(sim_time_s=trace.step * step_dt,
                         **_tracking_errors(command, measured, revolute),
                         measured_gripper_pct=gripper_pct(measured))
            trace.observe(value, gripper_pct(command))
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg('success', cfg)
    return trace
