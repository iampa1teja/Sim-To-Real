"""Step 1 - mdp/resets.py: deterministic DR setters + randomizers that use them.

CPU only (fake USD/Isaac): python -m unittest tests.test_resets_dr -v

Contract under test
- set_robot_color(env, rgb)
- set_cube_color(env, env_ids, rgb, name)
- set_light_exposure(env, env_ids, exposure, asset_cfg)
- set_pick_place_room_light(env, env_ids, exposure)
- get_camera_base_pose(env, prim_path_pattern) -> {"pos", "quat"} (cached first read)
- set_camera_pose_offset(env, prim_path_pattern, pos_offset, rot_offset)  (relative to base; never accumulates)
- set_camera_focal_length(env, asset_cfg, focal_length)
- every randomize_* draws a value, calls its set_*, and RETURNS what it applied:
    randomize_robot_color -> colour name
    randomize_camera_focal_length -> float
    randomize_camera_pose -> (pos_offset, rot_offset)
    randomize_light_exposure / randomize_pick_place_room_light -> float
    randomize_cube_color -> list of names, one per env id (in env_ids order)
"""
import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__))
from _fakes import (  # noqa: E402
    CURRENT, FakeMaterial, FakePrim, FakeShader, FakeStage, Quatd, SceneEntityCfg,
    install_stubs, load_module, quat_from_euler_xyz, quat_mul,
)

ENV_NS = "/World/envs/env_.*"
PATCHER = None
R = None


def setUpModule():
    global PATCHER, R
    PATCHER = install_stubs()
    R = load_module("sim_to_real_so101.mdp.resets", "source/sim_to_real_so101/mdp/resets.py")


def tearDownModule():
    PATCHER.stop()


def make_env(num_envs=2):
    return SimpleNamespace(num_envs=num_envs, device="cpu", scene={})


def quat_of(prim):
    value = prim.GetAttribute("xformOp:orient").Get()
    return torch.tensor(value.as_tuple() if isinstance(value, Quatd) else value, dtype=torch.float64)


class Base(unittest.TestCase):
    def setUp(self):
        CURRENT.stage = FakeStage()
        R._default_poses_cache.clear()
        torch.manual_seed(0)


# ── setters ──────────────────────────────────────────────────────────────────

class RobotColorTests(Base):
    def test_sets_diffuse_on_robot_shader(self):
        shader = CURRENT.stage.add(FakePrim(
            "/World/envs/env_0/Robot/Looks/material_a_3d_printed/Shader",
            attrs={"inputs:diffuse_color_constant": (1, 1, 1)}))
        env = make_env()
        env.scene["robot"] = SimpleNamespace(cfg=SimpleNamespace(prim_path=f"{ENV_NS}/Robot"))
        R.set_robot_color(env, (0.1, 0.2, 0.3))
        self.assertEqual(tuple(shader.GetAttribute("inputs:diffuse_color_constant").Get()), (0.1, 0.2, 0.3))


class CubeColorTests(Base):
    def cube_env(self, with_shader=True, with_input=True):
        inputs = {"diffuseColor": (0, 0, 1)} if with_input else {}
        shader = FakeShader("/World/envs/env_0/CUBE/Looks/Shader", inputs)
        mesh = FakePrim("/World/envs/env_0/CUBE/geometry",
                        material=FakeMaterial(shader) if with_shader else None)
        CURRENT.stage.add(FakePrim("/World/envs/env_0/CUBE", children=[mesh]))
        env = make_env(num_envs=1)
        env.scene["blue_cube"] = SimpleNamespace(
            root_physx_view=SimpleNamespace(prim_paths=["/World/envs/env_0/CUBE"]))
        return env, shader

    def test_sets_colour_and_records_name(self):
        env, shader = self.cube_env()
        R.set_cube_color(env, torch.tensor([0]), (0.8, 0.05, 0.05), "red")
        self.assertEqual(tuple(shader.GetInput("diffuseColor").Get()), (0.8, 0.05, 0.05))
        self.assertEqual(env._pick_place_cube_color, ["red"])

    def test_none_env_ids_means_all(self):
        env, shader = self.cube_env()
        R.set_cube_color(env, None, (0.8, 0.05, 0.05), "red")
        self.assertEqual(env._pick_place_cube_color, ["red"])

    def test_missing_shader_or_input_raises(self):
        env, _ = self.cube_env(with_shader=False)
        with self.assertRaisesRegex(RuntimeError, "No bound surface shader"):
            R.set_cube_color(env, torch.tensor([0]), (1, 0, 0), "red")
        CURRENT.stage = FakeStage()
        env, _ = self.cube_env(with_input=False)
        with self.assertRaisesRegex(RuntimeError, "diffuseColor"):
            R.set_cube_color(env, torch.tensor([0]), (1, 0, 0), "red")


class LightTests(Base):
    def test_set_light_exposure_sets_every_prim(self):
        lights = [CURRENT.stage.add(FakePrim(f"/World/envs/env_{i}/Light", attrs={"inputs:exposure": 0.0}))
                  for i in range(2)]
        env = make_env()
        env.scene["light"] = SimpleNamespace(prim_paths=[p.path for p in lights])
        R.set_light_exposure(env, torch.tensor([0, 1]), 1.25, SceneEntityCfg("light"))
        self.assertEqual([p.GetAttribute("inputs:exposure").Get() for p in lights], [1.25, 1.25])

    def test_room_light_sets_existing_tubelights_and_skips_missing(self):
        tube = CURRENT.stage.add(FakePrim("/World/envs/env_0/Room/Room/Lighting/Tubelight",
                                          attrs={"inputs:exposure": 0.0}))
        env = make_env()
        env.scene["room"] = SimpleNamespace(prim_paths=["/World/envs/env_0/Room", "/World/envs/env_1/Room"])
        R.set_pick_place_room_light(env, torch.tensor([0, 1]), -0.5)  # env_1 has no tube light: no error
        self.assertEqual(tube.GetAttribute("inputs:exposure").Get(), -0.5)


class CameraPoseTests(Base):
    PATTERN = "{ENV_REGEX_NS}/realsense_camera_rgb"

    def setUp(self):
        super().setUp()
        self.base_pos = (0.6, 0.03, 1.26)
        self.base_quat = Quatd(math.cos(0.3), 0.0, math.sin(0.3), 0.0)
        self.cams = [CURRENT.stage.add(FakePrim(
            f"/World/envs/env_{i}/realsense_camera_rgb",
            attrs={"xformOp:translate": self.base_pos, "xformOp:orient": self.base_quat})) for i in range(2)]
        self.env = make_env()

    def test_base_pose_is_cached_from_first_read(self):
        first = R.get_camera_base_pose(self.env, self.PATTERN)
        self.assertEqual(tuple(first["pos"]), self.base_pos)
        self.cams[0].GetAttribute("xformOp:translate").Set((9, 9, 9))
        self.assertEqual(tuple(R.get_camera_base_pose(self.env, self.PATTERN)["pos"]), self.base_pos)

    def test_missing_camera_returns_none(self):
        self.assertIsNone(R.get_camera_base_pose(self.env, "{ENV_REGEX_NS}/nope"))
        R.set_camera_pose_offset(self.env, "{ENV_REGEX_NS}/nope", {"x": 1}, {})  # no error

    def test_offset_is_relative_to_base_and_never_accumulates(self):
        pos, rot = {"x": 0.01, "y": -0.02, "z": 0.005}, {"roll": 0.0, "pitch": 0.02, "yaw": -0.03}
        for _ in range(3):  # applying the same offset repeatedly must give the same pose
            R.set_camera_pose_offset(self.env, self.PATTERN, pos, rot)
        expected_pos = tuple(b + pos[a] for b, a in zip(self.base_pos, "xyz"))
        delta = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.02]), torch.tensor([-0.03]))[0]
        expected_quat = quat_mul(torch.tensor([self.base_quat.as_tuple()]), delta.unsqueeze(0))[0].double()
        for cam in self.cams:  # every env's camera gets the same pose
            for got, want in zip(cam.GetAttribute("xformOp:translate").Get(), expected_pos):
                self.assertAlmostEqual(got, want, places=9)
            torch.testing.assert_close(quat_of(cam), expected_quat, atol=1e-6, rtol=0)

    def test_zero_offset_restores_base(self):
        R.set_camera_pose_offset(self.env, self.PATTERN, {"x": 0.05}, {"yaw": 0.2})
        R.set_camera_pose_offset(self.env, self.PATTERN, {}, {})
        for got, want in zip(self.cams[0].GetAttribute("xformOp:translate").Get(), self.base_pos):
            self.assertAlmostEqual(got, want, places=9)
        torch.testing.assert_close(quat_of(self.cams[0]),
                                   torch.tensor(self.base_quat.as_tuple(), dtype=torch.float64), atol=1e-6, rtol=0)


class FocalLengthTests(Base):
    def test_sets_every_matching_camera(self):
        cams = [CURRENT.stage.add(FakePrim(f"/World/envs/env_{i}/wrist_cam", attrs={"focalLength": 20.0}))
                for i in range(2)]
        env = make_env()
        env.scene["camera_wrist_cam"] = SimpleNamespace(cfg=SimpleNamespace(prim_path="{ENV_REGEX_NS}/wrist_cam"))
        R.set_camera_focal_length(env, SceneEntityCfg("camera_wrist_cam"), 13.5)
        self.assertEqual([c.GetAttribute("focalLength").Get() for c in cams], [13.5, 13.5])


# ── randomizers: draw in range -> call setter -> return the applied value ────

class RandomizerTests(Base):
    def test_robot_color(self):
        env = make_env()
        with patch.object(R, "set_robot_color") as setter:
            for _ in range(20):
                name = R.randomize_robot_color(env, None, color_names=["teal", "black"])
                self.assertIn(name, ("teal", "black"), "randomize_robot_color must return the colour name")
                setter.assert_called_with(env, R.ROBOT_COLORS[name])

    def test_focal_length(self):
        env, cfg = make_env(), SceneEntityCfg("camera_wrist_cam")
        with patch.object(R, "set_camera_focal_length") as setter:
            for _ in range(20):
                value = R.randomize_camera_focal_length(env, None, (12.0, 15.0), cfg)
                self.assertIsInstance(value, float)
                self.assertTrue(12.0 <= value <= 15.0)
                setter.assert_called_with(env, cfg, value)

    def test_camera_pose(self):
        CURRENT.stage.add(FakePrim("/World/envs/env_0/cam", attrs={"xformOp:translate": (0, 0, 0)}))
        env = make_env()
        pos_range = {"x": (-0.01, 0.01), "z": (0.0, 0.02)}
        rot_range = {"yaw": (-0.05, 0.05)}
        with patch.object(R, "set_camera_pose_offset") as setter:
            for _ in range(20):
                result = R.randomize_camera_pose(env, None, "{ENV_REGEX_NS}/cam", pos_range, rot_range)
                self.assertIsNotNone(result, "randomize_camera_pose must return (pos_offset, rot_offset)")
                pos, rot = result
                self.assertEqual(set(pos), {"x", "y", "z"})
                self.assertEqual(set(rot), {"roll", "pitch", "yaw"})
                self.assertTrue(-0.01 <= pos["x"] <= 0.01 and 0.0 <= pos["z"] <= 0.02)
                self.assertEqual((pos["y"], rot["roll"], rot["pitch"]), (0.0, 0.0, 0.0))
                self.assertTrue(-0.05 <= rot["yaw"] <= 0.05)
                setter.assert_called_with(env, "{ENV_REGEX_NS}/cam", pos, rot)

    def test_camera_pose_without_camera_is_noop(self):
        with patch.object(R, "set_camera_pose_offset") as setter:
            R.randomize_camera_pose(make_env(), None, "{ENV_REGEX_NS}/missing", {"x": (0, 1)}, {})
            setter.assert_not_called()

    def test_light_exposure(self):
        env, cfg = make_env(), SceneEntityCfg("light")
        with patch.object(R, "set_light_exposure") as setter:
            value = R.randomize_light_exposure(env, None, (-1.0, 1.0), cfg)
            self.assertTrue(-1.0 <= value <= 1.0)
            setter.assert_called_with(env, None, value, cfg)

    def test_room_light(self):
        env, ids = make_env(), torch.tensor([0, 1])
        with patch.object(R, "set_pick_place_room_light") as setter:
            value = R.randomize_pick_place_room_light(env, ids, (0.5, 0.75))
            self.assertTrue(0.5 <= value <= 0.75)
            setter.assert_called_once()
            self.assertEqual(setter.call_args.args[2], value)

    def test_cube_color_per_env(self):
        env = make_env(num_envs=3)
        env.scene["blue_cube"] = SimpleNamespace(root_physx_view=SimpleNamespace(prim_paths=[]))
        colors = {"blue": (0, 0, 1), "red": (0.8, 0.05, 0.05)}
        with patch.object(R, "set_cube_color") as setter:
            names = R.randomize_cube_color(env, torch.tensor([0, 2]), colors)
            self.assertEqual(len(names), 2, "randomize_cube_color must return one name per env id")
            self.assertEqual(setter.call_count, 2)
            for call, name, index in zip(setter.call_args_list, names, (0, 2)):
                self.assertEqual(call.args[2], colors[name])
                self.assertEqual(call.args[3], name)
                self.assertEqual(call.args[1].tolist(), [index])

    def test_cube_color_empty_palette(self):
        env = make_env()
        env.scene["blue_cube"] = SimpleNamespace(root_physx_view=SimpleNamespace(prim_paths=[]))
        with self.assertRaisesRegex(ValueError, "palette"):
            R.randomize_cube_color(env, None, {})


if __name__ == "__main__":
    unittest.main()
