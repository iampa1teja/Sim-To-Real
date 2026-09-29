"""Pose recorded pick-place frames in the live scene and render them with a chosen appearance.

Kinematic replay: joint states and the cube pose are written directly and the scene is
rendered without stepping physics, so the motion is exactly the recorded one.
"""

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from sim_to_real_so101.mdp import (
    set_camera_focal_length,
    set_camera_pose_offset,
    set_cube_color,
    set_pick_place_room_light,
    set_robot_color,
    sync_pick_place_camera_intrinsics,
)
from sim_to_real_so101.utils.dr_variants import DRDraw
from sim_to_real_so101.utils.replay_math import base_to_world


class SceneReplayer:
    """Applies per-episode appearance and per-frame poses to one (unwrapped) Isaac Lab env."""

    def __init__(self, env, interface, dr_ranges,
                 external_camera="camera_realsense_rgb", wrist_camera="camera_wrist_cam",
                 render_passes=2):
        self.env = env
        self.interface = interface
        self.dr_ranges = dr_ranges
        self.external_camera = external_camera
        self.wrist_camera = wrist_camera
        self.render_passes = render_passes
        self.robot = env.scene["robot"]
        self.cube = env.scene["blue_cube"]
        self.base_body_id = self.robot.find_bodies("base")[0][0]
        self.env_ids = torch.arange(env.num_envs, device=env.device)

    def apply(self, color_rgb, color_name, draw: DRDraw | None) -> None:
        """Set the cube colour and every DR term present in `draw`.

        No reset step is needed: each term is written absolutely (camera offsets are relative
        to the camera's cached base pose), and terms that are None are never touched.
        """
        set_cube_color(self.env, self.env_ids, color_rgb, color_name)
        if draw is None:
            return
        if draw.exposure is not None:
            set_pick_place_room_light(self.env, self.env_ids, draw.exposure)
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
            # MyRoom cameras use a calibrated OpenCV lens model; keep it in step with focalLength.
            sync_pick_place_camera_intrinsics(self.env, self.env_ids)

    def pose_frame(self, state_raw, pose_base) -> None:
        """Write one recorded frame: joints from observation.state, cube from its base-link pose."""
        state = torch.as_tensor(np.asarray(state_raw), dtype=torch.float32, device=self.env.device)
        q = self.interface.get_mapped_actions_vectorized(state).reshape(1, -1)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))

        base_pos = self.robot.data.body_pos_w[0, self.base_body_id]
        base_quat = self.robot.data.body_quat_w[0, self.base_body_id]
        pose = torch.as_tensor(np.asarray(pose_base), dtype=base_pos.dtype, device=base_pos.device)
        pos_w, quat_w = base_to_world(base_pos, base_quat, pose[:3], pose[3:7])
        self.cube.write_root_pose_to_sim(torch.cat((pos_w, quat_w)).reshape(1, 7))
        self.cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=base_pos.dtype, device=base_pos.device))

    def _cameras(self):
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
        """Render the posed frame and return images formatted exactly like the recorder's."""
        # render() also flushes the written poses to the renderer (forward()); a second pass
        # guards against the RTX renderer returning the previous frame.
        for _ in range(self.render_passes):
            self.env.sim.render()
        # No physics step means sensor timestamps do not advance: force a fresh read.
        for camera in self._cameras():
            camera.update(0.0, force_recompute=True)
        observation = self.env.observation_manager.compute()
        _, images, _, _ = self.interface.sim_to_real_dataset_processor(
            observation["policy"]["joint_pos_obs"][0], observation["visual"])
        return images
