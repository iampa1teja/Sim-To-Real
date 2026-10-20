"""Read-only, in-memory measurements of a pick-place grasp attempt."""
from __future__ import annotations

import math
import numpy as np

from .pick_place_eval import rotation_matrix


def measure_grasp(fixed_position, fixed_quaternion, moving_position, moving_quaternion,
                  cube_position, fixed_tip, moving_tip, cube_size_m):
    """Measure the opposing CAD fingertip centres using live rigid-body poses.

    Gap is projected on the fixed jaw's closing axis, not the distance between
    link origins. Tool error is expressed in the fixed gripper frame.
    """
    rotation = rotation_matrix(fixed_quaternion)
    fixed = np.asarray(fixed_position) + rotation @ fixed_tip
    moving = np.asarray(moving_position) + rotation_matrix(moving_quaternion) @ moving_tip
    gap = float(((moving - fixed) @ rotation)[0])
    tool = fixed + rotation[:, 0] * (cube_size_m / 2)
    error = (tool - cube_position) @ rotation
    return dict(tool_error_xyz_mm=(error * 1000).tolist(),
                tool_error_mm=float(np.linalg.norm(error) * 1000),
                jaw_gap_mm=gap * 1000, cube_width_mm=cube_size_m * 1000)


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


def _tip_offsets(stage, root, half_size):
    """Read mesh vertices once, then retain only two body-local contact points."""
    from pxr import Gf, Usd, UsdGeom
    cache = UsdGeom.XformCache()
    gripper = stage.GetPrimAtPath(root + '/gripper')
    jaw = stage.GetPrimAtPath(root + '/jaw')
    gripper_matrix = cache.GetLocalToWorldTransform(gripper)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    result = []
    for body, fragment in ((gripper, '/visuals/wrist_roll_follower_so101_v1/'),
                           (jaw, '/visuals/moving_jaw_so101_v1/')):
        mesh = next((p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.IsA(UsdGeom.Mesh) and fragment in str(p.GetPath())), None)
        if mesh is None:
            raise RuntimeError(f'Cannot measure grasp: missing fingertip mesh under {body.GetPath()}')
        transform = cache.GetLocalToWorldTransform(mesh) * gripper_matrix.GetInverse()
        points = np.asarray([transform.Transform(Gf.Vec3d(*map(float, p)))
                             for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]) * scale
        tip_z = points[:, 2].min()
        band = points[points[:, 2] <= tip_z + 2 * half_size]
        # The fixed jaw is left of the moving jaw in the authored zero pose.
        x = band[:, 0].max() if body == gripper else band[:, 0].min()
        point = np.array([x, (band[:, 1].min() + band[:, 1].max()) / 2, tip_z + half_size])
        body_from_gripper = gripper_matrix * cache.GetLocalToWorldTransform(body).GetInverse()
        local = body_from_gripper.Transform(Gf.Vec3d(*(point / scale)))
        result.append(np.asarray(local) * scale)
    return result


def _read_frame(torch, robot, body_ids, cube_position, command):
    """Transfer one consistent telemetry frame with a single device sync."""
    packed = torch.cat((robot.data.body_pos_w[0, body_ids].reshape(-1),
                        robot.data.body_quat_w[0, body_ids].reshape(-1),
                        cube_position, command, robot.data.joint_pos[0]))
    frame = packed.detach().cpu().numpy()
    return (frame[:6].reshape(2, 3), frame[6:14].reshape(2, 4),
            frame[14:17], frame[17:23], frame[23:29])


def install_grasp_diagnostics(env, robot_iface, step_dt):
    """Observe the success term before Isaac's automatic reset, without changing it."""
    import torch
    from isaacsim.core.utils.stage import get_current_stage
    robot = env.scene['robot']
    body_ids = [robot.find_bodies(name)[0][0] for name in ('gripper', 'jaw')]
    cube_size = float(env.scene['blue_cube'].cfg.spawn.size[0])
    fixed_tip, moving_tip = _tip_offsets(get_current_stage(), env.scene.env_prim_paths[0] + '/Robot', cube_size / 2)
    trace = GraspTrace()
    mapping_min = robot_iface.mapping_joint_mins.detach().cpu().numpy()
    mapping_max = robot_iface.mapping_joint_maxs.detach().cpu().numpy()
    cfg = env.termination_manager.get_term_cfg('success')
    original = cfg.func

    def observe(raw, **params):
        result = original(raw, **params)
        if trace.command is not None:
            positions, quaternions, cube, command, measured = _read_frame(
                torch, robot, body_ids, raw.scene['blue_cube'].data.root_pos_w[0], trace.command)
            value = measure_grasp(positions[0], quaternions[0], positions[1], quaternions[1],
                                  cube, fixed_tip, moving_tip, cube_size)
            value.update(sim_time_s=trace.step * step_dt,
                         max_joint_tracking_error_deg=float(np.max(np.abs(command - measured)) * 180 / math.pi),
                         measured_gripper_pct=float((measured[-1] * 180 / math.pi - mapping_min[-1])
                                                    / (mapping_max[-1] - mapping_min[-1]) * 100))
            gripper = (command[-1] * 180 / math.pi - mapping_min[-1]) / (mapping_max[-1] - mapping_min[-1]) * 100
            trace.observe(value, gripper)
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg('success', cfg)
    return trace
