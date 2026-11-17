"""Isaac environment and recorder adapters for the CPU generation protocols."""

from pathlib import Path as _Path

import numpy as _np

from core.arms.spec import load_arm_spec as _load_arm_spec
from core.generate.errors import EnvError as _EnvError
from core.generate.labels import label_columns as _label_columns
from core.generate.protocols import ExpertObs as _ExpertObs, ObjectState as _ObjectState
from core.tasks.base import Pose as _Pose


class RecorderEpisodeWriter:
    def __init__(self, recorder):
        self._recorder = recorder
        self.root = _Path(recorder.dataset_root)

    def episode_count(self):
        return len(self.episode_lengths())

    def episode_lengths(self):
        return self._recorder.episode_lengths()

    def add_frame(self, action_raw, state_raw, images, task, labels):
        frame = self._recorder.make_sim_frame(action_raw, state_raw, images, extra=_label_columns(labels))
        frame["task"] = task
        self._recorder.add_frame(frame)

    def discard_episode(self):
        self._recorder.clear_episode_buffer()

    def save_episode(self):
        return self._recorder.save_episode()

    def finalize(self):
        self._recorder.finalize()


class IsaacGenEnv:
    def __init__(self, env, task, *, arm, mapping, chain, gripper, cameras, palette_rgb):
        from sim.envs.mdp.resets import as_torch
        self._env, self._raw = env, env.unwrapped
        self._task, self._arm, self._mapping = task, arm, mapping
        self._chain, self._grip = chain, gripper
        self._cameras, self._palette = dict(cameras), dict(palette_rgb)
        self.fps = round(1/self._raw.step_dt)
        spec = _load_arm_spec(arm)[0]
        self.joint_names = tuple(spec["joints"]["names"])
        self.camera_names = tuple(cameras)
        self._robot = self._raw.scene["robot"]
        self._raw._generation_arm = arm
        self._joint_ids, found = self._robot.find_joints(list(self.joint_names), preserve_order=True)
        if tuple(found) != self.joint_names:
            raise _EnvError("Articulation joint order does not match arm.toml")
        term = self._raw.action_manager.get_term("joint_positions")
        def array(value):
            if hasattr(value, "torch") or hasattr(value, "detach"):
                return as_torch(value).detach().cpu().numpy()
            return _np.asarray(value)
        if not _np.allclose(array(term._scale), 1.) or not _np.allclose(array(term._offset), 0.):
            raise _EnvError("Generation requires absolute joint targets with scale 1 and offset 0")
        term_ids = getattr(term, "_joint_ids", slice(None))
        if isinstance(term_ids, slice):
            term_ids = list(range(len(self._robot.joint_names)))[term_ids]
        self._action_columns = [list(term_ids).index(i) for i in self._joint_ids]
        self._action_dim = len(term_ids)
        self._frame = 0

    def _target_tensor(self, q):
        import torch
        from sim.envs.mdp.resets import as_torch
        dtype = as_torch(self._robot.data.joint_pos).dtype
        target = torch.zeros((1, self._action_dim), device=self._raw.device, dtype=dtype)
        target[:, self._action_columns] = torch.as_tensor(_np.array(q, copy=True), device=self._raw.device, dtype=dtype)
        return target

    def reset(self, spec):
        import torch
        from sim.envs.mdp.objects import park_all, set_object_pose_base, set_object_color
        from sim.envs.mdp.resets import as_torch
        if set(spec.objects)-self._task.spec.objects.keys() or len(spec.initial_q) != len(self.joint_names):
            raise _EnvError("Spec objects or joint count do not match the environment")
        if self._raw.num_envs != 1:
            raise _EnvError("Generation requires one environment")
        self._spec, self._frame, self._ever_grasped = spec, 0, False
        self._placed_run, self._previous_run, self._signal_frame = 0, 0, None
        self._env.reset()
        park_all(self._raw, self._task.spec.objects)
        for name, state in spec.objects.items():
            if state.active:
                set_object_pose_base(self._raw, name, state.pose)
                if state.color is not None:
                    set_object_color(self._raw, name, self._palette[state.color])
        q = torch.as_tensor(spec.initial_q, device=self._raw.device, dtype=as_torch(self._robot.data.joint_pos).dtype).reshape(1, -1)
        ids = torch.zeros(1, device=self._raw.device, dtype=torch.long)
        self._robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), joint_ids=self._joint_ids, env_ids=ids)
        self._robot.set_joint_position_target_index(target=q, joint_ids=self._joint_ids, env_ids=ids)
        self._raw.scene.write_data_to_sim()
        self._raw.sim.forward()
        self._raw.scene.update(self._raw.physics_dt)
        # Contact trackers must take their rest height from the sampled poses,
        # after reset events have parked objects and generation has placed them.
        if "reset_tracking" in self._raw.event_manager.active_terms.get("reset", ()):
            tracking = self._raw.event_manager.get_term_cfg("reset_tracking")
            tracking.func(self._raw, ids, **tracking.params)
        if spec.held_object is not None:
            name = spec.held_object
            obj = self._task.spec.objects[name]
            dims = obj.size if obj.size is not None else obj.params["size_outer"]
            obs = self.observe()
            point = list(obs.tcp.position)
            if obj.size is None and obj.params.get("origin") == "bottom":
                point[2] -= dims[2]/2
            set_object_pose_base(self._raw, name, _Pose(tuple(point), obs.tcp.quat_xyzw))
            close = _np.array(spec.initial_q)
            close[self._grip.index] = self._grip.closed_rad
            for _ in range(8):
                self._env.step(self._target_tensor(close))
            self._frame = 0
        return self.observe()

    def step(self, q_target):
        self._env.step(self._target_tensor(q_target))
        self._frame += 1
        return self.observe()

    def observe(self):
        from sim.envs.mdp.objects import object_poses_in_base
        from sim.envs.mdp.resets import as_torch
        observations = self._raw.observation_manager.compute()
        ee = as_torch(observations["policy"]["ee_frame_state"])[0].detach().cpu().numpy()
        ee = _Pose(tuple(float(v) for v in ee[:3]), tuple(float(v) for v in ee[3:]))
        q = as_torch(self._robot.data.joint_pos)[0, self._joint_ids].detach().cpu().numpy().astype(_np.float64)
        poses = object_poses_in_base(self._raw, self._task.spec.objects)
        objects = {name: _ObjectState(pose, name in self._spec.objects and self._spec.objects[name].active,
                                      self._spec.objects[name].color if name in self._spec.objects else None)
                   for name, pose in poses.items()}
        group = observations.get("subtask_terms")
        signals = {} if group is None else {name: bool(as_torch(value).reshape(-1)[0].item()) for name, value in group.items()}
        if group is not None and "cube_grasped" in signals:
            holding = signals["cube_grasped"]
            placed = signals.get("cube_placed", False)
            self._ever_grasped |= holding
            if self._signal_frame != self._frame:
                self._previous_run, self._signal_frame = self._placed_run, self._frame
            self._placed_run = self._previous_run+1 if placed else 0
            signals.update(holding=holding, grasped=self._ever_grasped, lifted=self._ever_grasped,
                           placed=placed, success=self._placed_run >= self._task.spec.raw.get("success", {}).get("confirm_steps", 25))
        return _ExpertObs(self._frame, q, ee, self._chain.tcp_from_ee(ee), self._grip.gap_from_angle(float(q[self._grip.index])), objects, signals)

    def set_object_pose(self, name, pose):
        from sim.envs.mdp.objects import set_object_pose_base
        if name not in self._task.spec.objects:
            raise _EnvError(f"Unknown object: {name}")
        set_object_pose_base(self._raw, name, pose)

    def render(self):
        from sim.envs.mdp.resets import as_torch
        self._raw.sim.render()
        images = {}
        for name in self.camera_names:
            camera = self._raw.scene[f"camera_{name}"]
            camera.update(0., force_recompute=True)
            pixels = as_torch(camera.data.output["rgb"])[0, ..., :3]
            response = getattr(camera, "apply_response", None)
            if response is not None:
                pixels = response(pixels.unsqueeze(0))[0]
            images[name] = pixels.detach().cpu().numpy().astype(_np.uint8, copy=True)
        return images

    def close(self):
        self._env.close()
