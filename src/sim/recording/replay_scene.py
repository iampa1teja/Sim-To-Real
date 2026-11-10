# sim/recording/replay_scene.py  (ported from v1 utils/replay_scene.py)
"""Pose recorded task frames in the live scene and render them with a chosen appearance.

Kinematic replay: joint states and the object pose are written directly and the scene is
rendered without stepping physics, so the motion is exactly the recorded one.
"""
# Changes from v1 for Isaac Lab 3.0 / Newton:
# - Joints: raw dataset units -> radians through core.arms.mapping.JointMapping (v1 LeRobotSO101Interface is gone),
#   written to the arm.toml joints in order with the *_index write APIs.
# - Sidecar poses are WXYZ (v1 format); converted to XYZW and composed with Isaac Lab's combine_frame_transforms
#   (v1 replay_math.base_to_world is gone). Base body comes from arm.toml.
# - Object colour comes from the task; declared scene lighting is independent of the task.

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from core.tasks.loader import load_task, load_task_module
from sim.envs.mdp._common import as_torch
from sim.envs.mdp.appearance import set_robot_color
from sim.envs.mdp.cameras import (
    set_camera_focal_length,
    set_camera_pose_offset,
)
from sim.datagen.dr_variants import DRDraw
from sim.scene.cameras import sync_camera_intrinsics
from sim.scene.appearance import set_scene_light_exposure
from sim.recording.lerobot_recorder import split_visual_obs


class SceneReplayer:
    """Applies per-episode appearance and per-frame poses to one (unwrapped) Isaac Lab env."""

    def __init__(self, env, mapping, dr_ranges, cameras,
                 external_camera=None, wrist_camera=None,
                 render_passes=2, object_name=None, task=None, arm=None):
        """mapping: core.arms.mapping.JointMapping; cameras: recorded sim camera names (without camera_)."""
        from core.arms.spec import load_arm_spec
        from core.scenes import load_scene_spec

        self.env = env
        self.arm = arm or getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]
        self.mapping = mapping
        self.dr_ranges = dr_ranges
        self.cameras = list(cameras)
        scene = getattr(env.cfg, "scene_spec", None) or load_scene_spec()
        fixed = [f"camera_{name}" for name in scene.cameras]
        attached = [f"camera_{name}" for name in self.arm.cameras]
        self.external_camera = external_camera or next(iter(fixed), None)
        self.wrist_camera = wrist_camera or next(iter(attached), None)
        self.render_passes = render_passes
        self.robot = env.scene["robot"]
        self.task = load_task(task)
        names = (object_name,) if object_name is not None else self.task.spec.tracked_objects
        self.objects = {name: env.scene[name] for name in names}
        self.task_mdp = load_task_module("mdp", self.task.name)
        self.base_body_id = self.robot.find_bodies(self.arm.links.base)[0][0]
        self.joint_ids, _ = self.robot.find_joints(list(self.arm.usd_joint_names), preserve_order=True)
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
            set_scene_light_exposure(self.env, self.env_ids, draw.exposure)
        if draw.robot_color is not None:
            set_robot_color(self.env, self.dr_ranges.robot_colors[draw.robot_color])
        focal_changed = False
        if draw.camera_external is not None:
            if self.external_camera is None:
                raise ValueError("Camera pose DR requires a fixed scene camera")
            params = draw.camera_external
            prim_path = self.env.scene[self.external_camera].cfg.prim_path
            set_camera_pose_offset(self.env, prim_path, params["pos"], params["rot"])
            set_camera_focal_length(self.env, SceneEntityCfg(self.external_camera), params["focal"])
            focal_changed = True
        if draw.camera_wrist_focal is not None:
            if self.wrist_camera is None:
                raise ValueError("Wrist focal DR requires an arm-mounted camera")
            set_camera_focal_length(self.env, SceneEntityCfg(self.wrist_camera), draw.camera_wrist_focal)
            focal_changed = True
        if focal_changed:
            # The room cameras use a calibrated OpenCV lens model; keep it in step with focalLength.
            sync_camera_intrinsics(self.env, self.env_ids)

    def pose_frame(self, state_raw, pose_base) -> None:
        """Write one recorded frame: joints from observation.state, object from its base-link pose (WXYZ)."""
        q = self.mapping.to_sim(np.asarray(state_raw)).reshape(1, -1)
        q = q.to(as_torch(self.robot.data.joint_pos).dtype)
        self.robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q),
                                                  joint_ids=self.joint_ids, env_ids=self.env_ids)

        base_pos = as_torch(self.robot.data.body_link_pos_w)[:, self.base_body_id]
        base_quat = as_torch(self.robot.data.body_link_quat_w)[:, self.base_body_id]
        if not isinstance(pose_base, dict):
            if len(self.objects) != 1:
                raise ValueError("Multi-object replay requires a named pose map")
            pose_base = {next(iter(self.objects)): pose_base}
        if set(pose_base) != set(self.objects):
            raise ValueError(f"Replay objects {sorted(pose_base)} != task objects {sorted(self.objects)}")
        for name, values in pose_base.items():
            pose = torch.as_tensor(np.asarray(values), dtype=base_pos.dtype, device=base_pos.device).reshape(1, 7)
            quat_xyzw = torch.cat((pose[:, 4:7], pose[:, 3:4]), dim=-1)
            pos_w, quat_w = math_utils.combine_frame_transforms(base_pos, base_quat, pose[:, :3], quat_xyzw)
            obj = self.objects[name]
            obj.write_root_pose_to_sim_index(root_pose=torch.cat((pos_w, quat_w), dim=-1), env_ids=self.env_ids)
            obj.write_root_velocity_to_sim_index(root_velocity=pos_w.new_zeros((1, 6)), env_ids=self.env_ids)

    def _camera_sensors(self):
        """Recorded cameras first, then any other camera sensor in the scene, once each."""
        seen, cameras = set(), []
        candidates = [self.env.scene[f"camera_{name}"] for name in self.cameras]
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
