"""One physical-step/recording path for generated and calibrated replay episodes.

Frames describe the state before their recorded action. Success is observed after
that action through the task's real signals and StageTrace; failed buffers are discarded.
"""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np

from core.eval.geometry import rotation_matrix
from utils.geometry import quaternion_multiply
from core.tasks.base import Pose, StepState
from core.tasks.stages import StageTrace
from utils.datasets.task_meta import save_task_trajectory, validate_task_meta


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def relative_pose(base, pose):
    bx, by, bz, bw = base.quat_xyzw
    position = rotation_matrix((bw, bx, by, bz)).T @ (np.asarray(pose.position) - base.position)
    quaternion = np.asarray(quaternion_multiply((-bx, -by, -bz, bw), pose.quat_xyzw))
    quaternion /= np.linalg.norm(quaternion)
    return Pose.from_lists(position, quaternion)


def pose_matrix(pose):
    x, y, z, w = pose.quat_xyzw
    result = np.eye(4)
    result[:3, :3] = rotation_matrix((w, x, y, z))
    result[:3, 3] = pose.position
    return result.tolist()


def _json_state(state):
    return None if state is None else asdict(state)


class SuccessRecorder:
    """Commit a LeRobot buffer and its aligned sidecars only after task success."""
    def __init__(self, writer, task, arm, object_sizes, fps, *, control_offset_m=None):
        self.writer, self.task, self.arm = writer, task, arm
        self.fps = fps
        self.control_offset_m = None if control_offset_m is None else np.asarray(control_offset_m, dtype=float).tolist()
        self.object_sizes = {name: tuple(size) for name, size in object_sizes.items()}
        if set(self.object_sizes) != set(task.spec.tracked_objects):
            raise ValueError("Generated recordings must include every tracked task object")
        self.poses = {name: [] for name in self.object_sizes}
        self.annotations = []
        self.trace = StageTrace(task)
        self.steps = 0
        self.first_success_step = None
        self.closed = False
        if writer.frame_count:
            raise ValueError("Start an attempt with an empty recorder buffer")

    def append(self, action, state, images, poses_base, *, annotation=None):
        if self.closed:
            raise RuntimeError("Attempt has already finished")
        if set(poses_base) != set(self.poses):
            raise ValueError("Frame objects differ from the declared tracked objects")
        poses = {name: np.asarray(values, dtype=float).copy() for name, values in poses_base.items()}
        if any(value.shape != (7,) or not np.isfinite(value).all() for value in poses.values()):
            raise ValueError("Every recorded object needs one finite base-frame WXYZ pose")
        frame = self.writer.make_sim_frame(action, state, images)
        self.writer.add_frame(frame)
        for name, value in poses.items():
            self.poses[name].append(value.tolist())
        self.annotations.append(annotation)

    def observe(self, state):
        if self.steps >= self.writer.frame_count:
            raise ValueError("Success observation has no corresponding recorded action")
        flags = self.trace.observe(state)
        self.steps += 1
        if self.task.is_success(self.trace.stages) and self.first_success_step is None:
            self.first_success_step = self.steps
        if self.annotations[-1] is not None:
            self.annotations[-1]["subtask_term_signals"] = dict(self.trace.stages)
        return flags

    def finish(self, *, attempt=0, seed=None, reason=None):
        if self.closed:
            raise RuntimeError("Attempt has already finished")
        frames = self.writer.frame_count
        if frames != self.steps or any(len(values) != frames for values in self.poses.values()):
            raise ValueError("Action, observation and task-pose timelines are not aligned")
        result = self.trace.finish()
        latched = self.task.is_success(result["stages"])
        final_success = bool(self.trace.last_state is not None and
                             self.task.is_success(self.task.observe(self.trace.last_state)))
        accepted = latched and final_success and frames > 0
        if latched and not final_success:
            reason = "success_lost_before_episode_end"
            result["failure_mode"] = reason
        index, metadata_path = None, None
        if accepted:
            payload = {"schema_version": 2, "task": self.task.name, "arm": self.arm.name,
                       "units": "m", "reference_frame": f"robot base link ({self.arm.links.base})",
                       "episode_index": self.writer.episode_index, "num_frames": frames, "fps": self.fps,
                       "objects": {name: {"size": list(self.object_sizes[name]),
                                          "position": [row[:3] for row in values],
                                          "orientation_wxyz": [row[3:] for row in values]}
                                   for name, values in self.poses.items()}}
            # Validate before publishing a dataset episode; a malformed/misaligned
            # sidecar must never leave an accepted episode without its trajectory.
            validate_task_meta(payload, tracked_objects=self.task.spec.tracked_objects)
            index = self.writer.save_episode()
            save_task_trajectory(self.writer.dataset_root, index, self.poses, self.object_sizes, self.fps,
                                 task=self.task.name, arm=self.arm.name,
                                 reference_frame=f"robot base link ({self.arm.links.base})")
            metadata_path = Path(self.writer.dataset_root) / "datagen_meta" / f"episode_{index:06d}.json"
            write_json(metadata_path, {"schema_version": 1, "episode_index": index, "attempt": attempt,
                       "seed": seed, "num_frames": frames, "fps": self.fps, "stages": result["stages"],
                       "control_offset_m": self.control_offset_m,
                       "first_success_step": self.first_success_step, "last_state": _json_state(self.trace.last_state),
                       "frames": self.annotations})
        else:
            self.writer.clear_episode_buffer()
        self.closed = True
        return {"attempt": attempt, "index": index, "accepted": accepted, "success": accepted,
                "latched_success": latched, "final_success": final_success,
                **result, "frames": frames, "steps": self.steps, "seed": seed,
                "first_success_step": self.first_success_step, "last_state": _json_state(self.trace.last_state),
                "camera_frames": {name: frames for name in self.writer.cameras},
                "object_frames": {name: len(values) for name, values in self.poses.items()},
                "metadata_path": str(metadata_path) if metadata_path is not None else None, "reason": reason}

    def abort(self):
        self.writer.clear_episode_buffer()
        self.closed = True


class LiveRollout:
    """Synchronous physics executor shared with scripted, Mimic and audit replay."""
    def __init__(self, env, task, arm, mapping, recorder, *, control_offset=None, observer=None, target_pose_fn=None):
        self.env = env
        self.raw = getattr(env, "unwrapped", env)
        if self.raw.num_envs != 1:
            raise ValueError("Generated dataset rollouts currently require num_envs=1")
        self.task, self.arm, self.mapping, self.recorder = task, arm, mapping, recorder
        self.observer, self.target_pose_fn = observer, target_pose_fn
        self.control_offset = np.asarray(control_offset if control_offset is not None else (0., 0., 0.), dtype=float)
        self.robot = self.raw.scene["robot"]
        self.joint_ids = self.robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)[0]
        self.ee_id = self.robot.find_bodies(arm.links.ee)[0][0]
        self.base_id = self.robot.find_bodies(arm.links.base)[0][0]
        self.gripper_ids = [index for index, joint in enumerate(arm.action_joints) if joint.role == "gripper"]
        self.last_state = None

    @staticmethod
    def _numpy(value):
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "torch"):
            value = value.torch
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def snapshot(self, signals=None):
        positions = self._numpy(self.robot.data.body_link_pos_w)[0]
        rotations = self._numpy(self.robot.data.body_link_quat_w)[0]
        joints = self._numpy(self.robot.data.joint_pos)[0, self.joint_ids]
        objects = {name: Pose.from_lists(self._numpy(self.raw.scene[name].data.root_pos_w)[0],
                                        self._numpy(self.raw.scene[name].data.root_quat_w)[0])
                   for name in self.task.spec.objects}
        values = {}
        for name, value in (signals or {}).items():
            if isinstance(value, (bool, int, float, str)):
                values[name] = value
            else:
                array = self._numpy(value)
                selected = array[0] if array.ndim else array
                values[name] = selected.item() if np.size(selected) == 1 else selected.tolist()
        return StepState(objects, Pose.from_lists(positions[self.ee_id], rotations[self.ee_id]),
                         float(np.mean(joints[self.gripper_ids])) if self.gripper_ids else 0., values)

    def control_pose(self, state):
        x, y, z, w = state.ee.quat_xyzw
        position = np.asarray(state.ee.position) + rotation_matrix((w, x, y, z)) @ self.control_offset
        return Pose.from_lists(position, state.ee.quat_xyzw)

    def step(self, target_sim, *, action_raw=None, target_eef_pose=None, gripper_action=None, phase=None):
        import torch
        from sim.envs.mdp._common import as_torch
        from sim.recording.lerobot_recorder import split_visual_obs

        target = torch.as_tensor(target_sim, device=self.raw.device, dtype=torch.float32).reshape(1, -1)
        if target.shape[1] != len(self.arm.action_joints) or not torch.isfinite(target).all():
            raise ValueError("Generated action must contain finite values in manifest joint order")
        before = self.snapshot()
        base = Pose.from_lists(self._numpy(self.robot.data.body_link_pos_w)[0, self.base_id],
                               self._numpy(self.robot.data.body_link_quat_w)[0, self.base_id])
        poses = {}
        for name in self.task.spec.tracked_objects:
            pose = relative_pose(base, before.objects[name])
            x, y, z, w = pose.quat_xyzw
            poses[name] = (*pose.position, w, x, y, z)
        visual = self.raw.observation_manager.compute()["visual"]
        images, _, _ = split_visual_obs(visual, self.recorder.writer.cameras)
        state_raw = self.mapping.to_raw(as_torch(self.robot.data.joint_pos)[0, self.joint_ids])
        if action_raw is None:
            action_raw = self.mapping.to_raw(target[0])
        elif not torch.allclose(self.mapping.to_sim(action_raw).reshape(1, -1), target, atol=1e-6, rtol=0):
            raise ValueError("Recorded raw action does not decode to the executed simulator target")
        requested_pose = target_eef_pose
        if self.target_pose_fn is not None:
            target_eef_pose = self.target_pose_fn(target)
        target_matrix = None if target_eef_pose is None else pose_matrix(relative_pose(base, target_eef_pose))
        annotation = {"eef_pose": pose_matrix(relative_pose(base, self.control_pose(before))),
                      "target_eef_pose": target_matrix,
                      "requested_eef_pose": None if requested_pose is None else pose_matrix(relative_pose(base, requested_pose)),
                      "object_pose": {name: pose_matrix(relative_pose(base, pose)) for name, pose in before.objects.items()},
                      "gripper_action": self._numpy(action_raw)[self.gripper_ids].tolist(), "phase": phase}
        self.recorder.append(action_raw, state_raw, images, poses, annotation=annotation)
        result = self.env.step(target)
        if len(result) >= 4 and any(bool(torch.as_tensor(value).any()) for value in result[2:4]):
            raise RuntimeError("Generated rollouts require termination/autoreset disabled")
        self.last_state = self.snapshot(self.task.signals(self.raw))
        self.recorder.observe(self.last_state)
        if self.observer is not None:
            self.observer(annotation)
        return result
