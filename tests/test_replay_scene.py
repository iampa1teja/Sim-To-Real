"""Step 6 - utils/replay_scene.py: SceneReplayer, tested against a fake scene.

CPU only: python -m unittest tests.test_replay_scene -v
(Real rendering is checked separately by the GPU fidelity run: replay blue with no DR
and compare against the original frames, PSNR > 30 dB.)

Contract under test
- imports its setters from sim_to_real_so101.mdp:
      set_cube_color, set_robot_color, set_pick_place_room_light,
      set_camera_pose_offset, set_camera_focal_length
  and base_to_world from sim_to_real_so101.utils.replay_math
- SceneReplayer(env, interface, dr_ranges,
                external_camera="camera_realsense_rgb", wrist_camera="camera_wrist_cam")
    env is the UNWRAPPED Isaac Lab env (env.scene, env.sim, env.observation_manager, env.num_envs)
- apply(color_rgb, color_name, draw):
      always   set_cube_color(env, <ids or None>, color_rgb, color_name)
      draw.exposure          -> set_pick_place_room_light(env, ids, exposure)
      draw.robot_color       -> set_robot_color(env, dr_ranges.robot_colors[name])
      draw.camera_external   -> set_camera_pose_offset(env, <external camera cfg.prim_path>, pos, rot)
                                set_camera_focal_length(env, <cfg named external_camera>, focal)
      draw.camera_wrist_focal-> set_camera_focal_length(env, <cfg named wrist_camera>, focal)
      any focal change       -> sync_pick_place_camera_intrinsics(env, ids)
      draw None / field None -> that setter is NOT called
- pose_frame(state_raw (6,), pose_base (7,) = x y z qw qx qy qz in the base-link frame):
      state_raw may be numpy: convert to a float32 torch tensor on the env device first, then
      q = interface.get_mapped_actions_vectorized(state)  -> robot.write_joint_state_to_sim(q (1,6), zeros (1,6))
      cube world pose = base_to_world(live base-link pose, pose_base)
          -> cube.write_root_pose_to_sim((1,7)); cube.write_root_velocity_to_sim(zeros (1,6))
- render_images() -> dict camera_name -> image:
      calls env.sim.render() (at least once; more if needed for fresh frames),
      refreshes each camera with camera.update(dt, force_recompute=True), then
      obs = env.observation_manager.compute() and returns the 2nd item of
      interface.sim_to_real_dataset_processor(obs["policy"]["joint_pos_obs"][0], obs["visual"])
      (same image formatting as the recorder)
"""
import math
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _fakes import install_stubs, load_module, yaw_quat  # noqa: E402

PATCHER = None
MDP = None
SC = None
DV = None
SETTERS = ("set_cube_color", "set_robot_color", "set_pick_place_room_light",
           "set_camera_pose_offset", "set_camera_focal_length", "sync_pick_place_camera_intrinsics")


def setUpModule():
    global PATCHER, MDP, SC, DV
    MDP = types.ModuleType("sim_to_real_so101.mdp")
    for name in SETTERS:
        setattr(MDP, name, MagicMock(name=name))
    PATCHER = install_stubs({"sim_to_real_so101.mdp": MDP})
    load_module("sim_to_real_so101.utils.replay_math", "source/sim_to_real_so101/utils/replay_math.py")
    DV = load_module("sim_to_real_so101.utils.dr_variants", "source/sim_to_real_so101/utils/dr_variants.py")
    SC = load_module("sim_to_real_so101.utils.replay_scene", "source/sim_to_real_so101/utils/replay_scene.py")


def tearDownModule():
    PATCHER.stop()


class FakeScene(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensors = {}


def make_env(base_pos=(0.0, 0.0, 0.0), base_quat=(1.0, 0.0, 0.0, 0.0)):
    robot = MagicMock(name="robot")
    robot.find_bodies.return_value = ([1], ["base"])
    robot.data.body_pos_w = torch.tensor([[[9.0, 9.0, 9.0], list(base_pos)]])     # body 1 is "base"
    robot.data.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0], list(base_quat)]])
    cube = MagicMock(name="cube")
    scene = FakeScene(
        robot=robot, blue_cube=cube,
        camera_realsense_rgb=SimpleNamespace(
            cfg=SimpleNamespace(prim_path="/World/envs/env_.*/realsense_camera_rgb"), update=MagicMock()),
        camera_wrist_cam=SimpleNamespace(
            cfg=SimpleNamespace(prim_path="/World/envs/env_.*/Robot/gripper/wrist_cam"), update=MagicMock()),
    )
    visual = {"rgb_realsense_rgb": torch.zeros(1, 4, 4, 3)}
    env = SimpleNamespace(
        num_envs=1, device="cpu", scene=scene, sim=MagicMock(name="sim"),
        observation_manager=MagicMock(name="obs"),
    )
    env.observation_manager.compute.return_value = {
        "policy": {"joint_pos_obs": torch.arange(6.0).unsqueeze(0)}, "visual": visual}
    return env, robot, cube, visual


def _mapped(raw):
    # Like the real LeRobotSO101Interface.get_mapped_actions_vectorized: torch-only math
    # (torch.zeros_like + tensor arithmetic), so a numpy array breaks it.
    if not isinstance(raw, torch.Tensor):
        raise TypeError("get_mapped_actions_vectorized needs a torch.Tensor (convert the numpy state first)")
    return raw.float() * 0.01


def make_interface():
    iface = MagicMock(name="interface")
    iface.get_mapped_actions_vectorized.side_effect = _mapped
    iface.sim_to_real_dataset_processor.return_value = (
        torch.zeros(6), {"realsense_rgb": "IMG_A", "wrist_cam": "IMG_B"}, {}, {})
    return iface


def ranges():
    return DV.DRRanges(exposure=(-1, 1), robot_colors={"teal": (0.0, 0.8, 0.5)},
                       camera_pos={"x": (-0.01, 0.01)}, camera_rot={"yaw": (-0.05, 0.05)},
                       focal_length=(12.0, 15.0))


def arg(call, index, name):
    return call.args[index] if len(call.args) > index else call.kwargs[name]


class Base(unittest.TestCase):
    def setUp(self):
        for name in SETTERS:
            getattr(MDP, name).reset_mock()
        self.env, self.robot, self.cube, self.visual = make_env()
        self.iface = make_interface()
        self.replayer = SC.SceneReplayer(self.env, self.iface, ranges())


class ApplyTests(Base):
    def test_no_draw_only_sets_cube_colour(self):
        self.replayer.apply((0.8, 0.05, 0.05), "red", None)
        MDP.set_cube_color.assert_called_once()
        call = MDP.set_cube_color.call_args
        self.assertIs(call.args[0], self.env)
        self.assertEqual(tuple(call.args[2]), (0.8, 0.05, 0.05))
        self.assertEqual(call.args[3], "red")
        for name in SETTERS[1:]:
            getattr(MDP, name).assert_not_called()

    def test_full_draw(self):
        draw = DV.DRDraw(exposure=0.25, robot_color="teal",
                         camera_external={"pos": {"x": 0.004, "y": 0.0, "z": 0.0},
                                          "rot": {"roll": 0.0, "pitch": 0.0, "yaw": 0.01}, "focal": 13.0},
                         camera_wrist_focal=14.0)
        self.replayer.apply((0, 0, 1), "blue", draw)
        self.assertEqual(MDP.set_pick_place_room_light.call_args.args[2], 0.25)
        MDP.set_robot_color.assert_called_once_with(self.env, (0.0, 0.8, 0.5))
        pose_call = MDP.set_camera_pose_offset.call_args
        self.assertEqual(pose_call.args[1], "/World/envs/env_.*/realsense_camera_rgb")
        self.assertEqual(pose_call.args[2], draw.camera_external["pos"])
        self.assertEqual(pose_call.args[3], draw.camera_external["rot"])
        focal = {c.args[1].name: c.args[2] for c in MDP.set_camera_focal_length.call_args_list}
        self.assertEqual(focal, {"camera_realsense_rgb": 13.0, "camera_wrist_cam": 14.0})
        # MyRoom's calibrated OpenCV lens must follow focal-length changes (as in the DR eval)
        MDP.sync_pick_place_camera_intrinsics.assert_called()

    def test_partial_draw_leaves_other_terms_alone(self):
        self.replayer.apply((0, 0, 1), "blue", DV.DRDraw(exposure=-0.3, robot_color=None,
                                                         camera_external=None, camera_wrist_focal=None))
        MDP.set_pick_place_room_light.assert_called_once()
        for name in ("set_robot_color", "set_camera_pose_offset", "set_camera_focal_length"):
            getattr(MDP, name).assert_not_called()


class PoseFrameTests(Base):
    def test_joint_state_written_from_mapped_state(self):
        state = np.array([10, -20, 30, -40, 50, 60], dtype=np.float32)
        self.replayer.pose_frame(state, np.array([0.2, 0.0, 0.01, 1, 0, 0, 0]))
        call = self.robot.write_joint_state_to_sim.call_args
        position, velocity = arg(call, 0, "position"), arg(call, 1, "velocity")
        torch.testing.assert_close(torch.as_tensor(position).reshape(1, 6).float(),
                                   torch.tensor(state).unsqueeze(0) * 0.01)
        self.assertEqual(float(torch.as_tensor(velocity).abs().sum()), 0.0)
        self.assertEqual(tuple(torch.as_tensor(velocity).shape), (1, 6))

    def test_cube_pose_uses_live_base_link(self):
        self.env, self.robot, self.cube, _ = make_env(base_pos=(1.0, 2.0, 3.0), base_quat=yaw_quat(math.pi / 2))
        self.replayer = SC.SceneReplayer(self.env, self.iface, ranges())
        self.replayer.pose_frame(np.zeros(6, dtype=np.float32), np.array([0.1, 0.0, 0.01, 1, 0, 0, 0]))
        pose = torch.as_tensor(arg(self.cube.write_root_pose_to_sim.call_args, 0, "root_pose")).double()
        self.assertEqual(tuple(pose.shape), (1, 7))
        torch.testing.assert_close(pose[0, :3], torch.tensor([1.0, 2.1, 3.01], dtype=torch.float64),
                                   atol=1e-6, rtol=0)
        q = pose[0, 3:]
        expected = torch.tensor(yaw_quat(math.pi / 2), dtype=torch.float64)
        self.assertLess(min((q - expected).abs().max().item(), (q + expected).abs().max().item()), 1e-6)
        velocity = torch.as_tensor(arg(self.cube.write_root_velocity_to_sim.call_args, 0, "root_velocity"))
        self.assertEqual(tuple(velocity.shape), (1, 6))
        self.assertEqual(float(velocity.abs().sum()), 0.0)


class RenderTests(Base):
    def test_renders_then_formats_like_the_recorder(self):
        images = self.replayer.render_images()
        self.assertGreaterEqual(self.env.sim.render.call_count, 1)
        self.assertEqual(images, {"realsense_rgb": "IMG_A", "wrist_cam": "IMG_B"})
        call = self.iface.sim_to_real_dataset_processor.call_args
        torch.testing.assert_close(torch.as_tensor(call.args[0]).float(), torch.arange(6.0))
        self.assertIs(call.args[1], self.visual)

    def test_cameras_are_force_refreshed(self):
        # Without a physics step, sensor timestamps do not advance, so a plain update()
        # can return the PREVIOUS frame. The recorder's EpisodeCube uses force_recompute=True.
        self.replayer.render_images()
        for name in ("camera_realsense_rgb", "camera_wrist_cam"):
            update = self.env.scene[name].update
            self.assertTrue(update.called, f"{name} not updated")
            self.assertTrue(update.call_args.kwargs.get("force_recompute"),
                            f"{name}.update(...) must pass force_recompute=True")


if __name__ == "__main__":
    unittest.main()
