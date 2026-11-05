# sim/eval/grasp_diagnostics.py  (ported from v1 utils/grasp_diagnostics.py)
"""Read-only, in-memory measurements of a pick-place grasp attempt."""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Live body quaternions are XYZW; converted to WXYZ for the numpy rotation helper.
# - Fixed-jaw body and mesh come from arm.toml (links.ee, links.fixed_jaw_mesh). The moving-jaw body and mesh
#   are still the SO-101 names below (arm.toml has no fields for them yet).
# - Gripper percent comes from the arm's JointMapping instead of v1's robot_iface bounds.
from __future__ import annotations

import math

import numpy as np

from core.tasks.loader import load_task_module

MOVING_JAW_BODY = "jaw"
MOVING_JAW_MESH = "moving_jaw_so101_v1"


def _rotation_matrix(quaternion_wxyz):
    return load_task_module("eval", "pick_place_cube").rotation_matrix(quaternion_wxyz)


def measure_grasp(fixed_position, fixed_quaternion, moving_position, moving_quaternion,
                  cube_position, fixed_tip, moving_tip, cube_size_m):
    """Measure the opposing CAD fingertip centres using live rigid-body poses (quaternions WXYZ).

    Gap is projected on the fixed jaw's closing axis, not the distance between
    link origins. Tool error is expressed in the fixed gripper frame.
    """
    rotation = _rotation_matrix(fixed_quaternion)
    fixed = np.asarray(fixed_position) + rotation @ fixed_tip
    moving = np.asarray(moving_position) + _rotation_matrix(moving_quaternion) @ moving_tip
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


def _tip_offsets(stage, root, fixed_body, half_size):
    """Read mesh vertices once, then retain only two body-local contact points."""
    from pxr import Gf, Usd, UsdGeom

    from sim.scene.gripper_geometry import fixed_jaw_mesh_name
    cache = UsdGeom.XformCache()
    gripper = stage.GetPrimAtPath(f"{root}/{fixed_body}")
    jaw = stage.GetPrimAtPath(f"{root}/{MOVING_JAW_BODY}")
    gripper_matrix = cache.GetLocalToWorldTransform(gripper)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    result = []
    for body, fragment in ((gripper, f'/visuals/{fixed_jaw_mesh_name()}/'),
                           (jaw, f'/visuals/{MOVING_JAW_MESH}/')):
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
    """Transfer one consistent telemetry frame with a single device sync (quaternions returned WXYZ)."""
    from sim.envs.mdp.resets import _wxyz, as_torch
    packed = torch.cat((as_torch(robot.data.body_link_pos_w)[0, body_ids].reshape(-1),
                        as_torch(robot.data.body_link_quat_w)[0, body_ids].reshape(-1),
                        cube_position, command, as_torch(robot.data.joint_pos)[0]))
    frame = packed.detach().cpu().numpy()
    joints = len(command)
    return (frame[:6].reshape(2, 3), _wxyz(frame[6:14].reshape(2, 4)),
            frame[14:17], frame[17:17 + joints], frame[17 + joints:17 + 2 * joints])


def install_grasp_diagnostics(env, mapping, step_dt):
    """Observe the success term before Isaac's automatic reset, without changing it.

    mapping: core.arms.mapping.JointMapping used to express the gripper in raw percent.
    """
    import torch
    from isaaclab.sim import get_current_stage

    from sim.envs.base_env_cfg import EE_LINK
    from sim.envs.mdp.resets import as_torch
    robot = env.scene['robot']
    body_ids = [robot.find_bodies(name)[0][0] for name in (EE_LINK, MOVING_JAW_BODY)]
    cube_size = float(env.scene['blue_cube'].cfg.spawn.size[0])
    fixed_tip, moving_tip = _tip_offsets(get_current_stage(), env.scene.env_prim_paths[0] + '/Robot', EE_LINK,
                                         cube_size / 2)
    gripper_index = int(mapping.gripper_mask.nonzero()[0].item())
    trace = GraspTrace()
    cfg = env.termination_manager.get_term_cfg('success')
    original = cfg.func

    def gripper_pct(radians):
        return float(mapping.to_raw(torch.as_tensor(radians))[gripper_index].item())

    def observe(raw, **params):
        result = original(raw, **params)
        if trace.command is not None:
            positions, quaternions, cube, command, measured = _read_frame(
                torch, robot, body_ids, as_torch(raw.scene['blue_cube'].data.root_pos_w)[0], trace.command)
            value = measure_grasp(positions[0], quaternions[0], positions[1], quaternions[1],
                                  cube, fixed_tip, moving_tip, cube_size)
            value.update(sim_time_s=trace.step * step_dt,
                         max_joint_tracking_error_deg=float(np.max(np.abs(command - measured)) * 180 / math.pi),
                         measured_gripper_pct=gripper_pct(measured))
            trace.observe(value, gripper_pct(command))
        return result

    cfg.func = observe
    env.termination_manager.set_term_cfg('success', cfg)
    return trace
