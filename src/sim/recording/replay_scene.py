# sim/recording/replay_scene.py  (ported from v1 utils/replay_scene.py)
"""Pose recorded pick-place frames in the live scene and render them with a chosen appearance.

Kinematic replay: joint states and the object pose are written directly and the scene is
rendered without stepping physics, so the motion is exactly the recorded one.
"""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Joints: raw dataset units -> radians through core.arms.mapping.JointMapping (v1 LeRobotSO101Interface is gone),
#   written to the arm.toml joints in order with the *_index write APIs.
# - Sidecar poses are WXYZ (v1 format); converted to XYZW and composed with Isaac Lab's combine_frame_transforms
#   (v1 replay_math.base_to_world is gone). Base body comes from arm.toml.
# - Cube colour and room light setters come from tasks/pick_place_cube/mdp.py.

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from core.tasks.loader import load_task_module
from sim.envs.mdp.resets import (
    as_torch,
    set_camera_focal_length,
    set_camera_pose_offset,
    set_robot_color,
    sync_pick_place_camera_intrinsics,
)
from sim.generate.dr_variants import DRDraw
from sim.recording.lerobot_recorder import split_visual_obs


class SceneReplayer:
    """Applies per-episode appearance and per-frame poses to one (unwrapped) Isaac Lab env."""

    def __init__(self, env, mapping, dr_ranges, cameras,
                 external_camera="camera_realsense_rgb", wrist_camera="camera_wrist_cam",
                 render_passes=2, object_name="blue_cube", task="pick_place_cube"):
        """mapping: core.arms.mapping.JointMapping; cameras: recorded sim camera names (without camera_)."""
        from sim.envs.base_env_cfg import BASE_LINK, JOINT_NAMES

        self.env = env
        self.mapping = mapping
        self.dr_ranges = dr_ranges
        self.cameras = list(cameras)
        self.external_camera = external_camera
        self.wrist_camera = wrist_camera
        self.render_passes = render_passes
        self.robot = env.scene["robot"]
        self.cube = env.scene[object_name]
        self.task_mdp = load_task_module("mdp", task)
        self.base_body_id = self.robot.find_bodies(BASE_LINK)[0][0]
        self.joint_ids, _ = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        self.env_ids = torch.arange(env.num_envs, device=env.device)

    def apply(self, color_rgb, color_name, draw: DRDraw | None) -> None:
        """Set the cube colour and every DR term present in `draw`.

        No reset step is needed: each term is written absolutely (camera offsets are relative
        to the camera's cached base pose), and terms that are None are never touched.
        """
        self.task_mdp.set_cube_color(self.env, self.env_ids, color_rgb, color_name)
        if draw is None:
            return
        if draw.exposure is not None:
            self.task_mdp.set_pick_place_room_light(self.env, self.env_ids, draw.exposure)
        if draw.robot_color is not None:
            set_robot_color(self.env, self.dr_ranges.robot_colors[draw.robot_color])
        focal_changed = False
        if draw.camera_external is not None:
            params = draw.camera_external
            prim_path = self.env.scene[self.external_camera].cfg.prim_path
            set_camera_pose_offset(self.env, prim_path, params["pos"], params["rot"])
            set_camera_focal_length(self.env, SceneEntityCfg(self.external_camera), params["focal"])
            focal_changed = True
        if draw.camera_wrist_focal is not None:
            set_camera_focal_length(self.env, SceneEntityCfg(self.wrist_camera), draw.camera_wrist_focal)
            focal_changed = True
        if focal_changed:
            # The room cameras use a calibrated OpenCV lens model; keep it in step with focalLength.
            sync_pick_place_camera_intrinsics(self.env, self.env_ids)

    def pose_frame(self, state_raw, pose_base) -> None:
        """Write one recorded frame: joints from observation.state, object from its base-link pose (WXYZ)."""
        q = self.mapping.to_radians(np.asarray(state_raw)).reshape(1, -1)
        q = q.to(as_torch(self.robot.data.joint_pos).dtype)
        self.robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q),
                                                  joint_ids=self.joint_ids, env_ids=self.env_ids)

        base_pos = as_torch(self.robot.data.body_link_pos_w)[:, self.base_body_id]
        base_quat = as_torch(self.robot.data.body_link_quat_w)[:, self.base_body_id]
        pose = torch.as_tensor(np.asarray(pose_base), dtype=base_pos.dtype, device=base_pos.device).reshape(1, 7)
        quat_xyzw = torch.cat((pose[:, 4:7], pose[:, 3:4]), dim=-1)
        pos_w, quat_w = math_utils.combine_frame_transforms(base_pos, base_quat, pose[:, :3], quat_xyzw)
        self.cube.write_root_pose_to_sim_index(root_pose=torch.cat((pos_w, quat_w), dim=-1), env_ids=self.env_ids)
        self.cube.write_root_velocity_to_sim_index(root_velocity=pos_w.new_zeros((1, 6)), env_ids=self.env_ids)

    def _camera_sensors(self):
        """The two recorded cameras first, then any other camera sensor in the scene (once each)."""
        seen, cameras = set(), []
        candidates = [self.env.scene[self.external_camera], self.env.scene[self.wrist_camera]]
        candidates += [s for s in self.env.scene.sensors.values() if hasattr(s.cfg, "data_types")]
        for camera in candidates:
            if id(camera) not in seen:
                seen.add(id(camera))
                cameras.append(camera)
        return cameras

    def render_images(self) -> dict:
        """Render the posed frame and return {camera: rgb} exactly as the recorder takes it."""
        # render() also flushes the written poses to the renderer (forward()); a second pass
        # guards against the RTX renderer returning the previous frame.
        for _ in range(self.render_passes):
            self.env.sim.render()
        # No physics step means sensor timestamps do not advance: force a fresh read.
        for camera in self._camera_sensors():
            camera.update(0.0, force_recompute=True)
        observation = self.env.observation_manager.compute()
        images, _, _ = split_visual_obs(observation["visual"], self.cameras)
        return images
