"""Keep the registered physics actor out of the work area between episodes."""

import torch
from pxr import UsdGeom

from isaaclab.managers import SceneEntityCfg
from sim_to_real_so101.mdp import reset_object_pose, reset_pick_place_state


class EpisodeCube:
    def __init__(self, env):
        self.env = env.unwrapped
        self.asset = self.env.scene["blue_cube"]
        self.visible = False
        self.images = [
            UsdGeom.Imageable(self.env.sim.stage.GetPrimAtPath(path + "/CUBE"))
            for path in self.env.scene.env_prim_paths
        ]

    @torch.inference_mode()
    def hide(self):
        self.visible = False
        for image in self.images:
            image.MakeInvisible()
        self.park()
        reset_pick_place_state(self.env, None)
        self.env.scene["contact_grasp"].reset()

    @torch.inference_mode()
    def park(self):
        if self.visible:
            return
        # Deleting/deactivating a registered prim invalidates Isaac Lab's
        # rigid-body/contact views. Hide and park it below the room instead.
        pose = self.asset.data.default_root_state[:, :7].clone()
        pose[:, :3] = self.env.scene.env_origins
        pose[:, 2] -= 10.0
        self.asset.write_root_pose_to_sim(pose)
        self.asset.write_root_velocity_to_sim(torch.zeros_like(self.asset.data.root_vel_w))

    @torch.inference_mode()
    def show(self):
        ids = torch.arange(self.env.num_envs, device=self.env.device)
        reset_object_pose(self.env, ids, SceneEntityCfg("blue_cube"), {})
        for image in self.images:
            image.MakeVisible()
        self.visible = True
        # Refresh cameras before a zero-countdown episode records its first frame.
        self.env.sim.render()
        for sensor in self.env.scene.sensors.values():
            if hasattr(sensor.cfg, "data_types"):
                sensor.update(0.0, force_recompute=True)
