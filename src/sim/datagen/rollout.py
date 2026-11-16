"""One physical-step/recording path for generated and calibrated replay episodes.

Frames describe the state before their recorded action. Success is observed after
that action through the task's real signals and StageTrace; failed buffers are discarded.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import os
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np

from utils.geometry import quaternion_multiply, rotation_matrix_wxyz, xyzw_to_wxyz
from core.tasks.task import Pose, StepState
from core.tasks.stages import StageTrace
from utils.datasets.task_meta import save_task_trajectory, validate_task_meta
from utils.io import write_json
from utils.geometry import relative_pose, pose_matrix, wxyz_to_xyzw
from sim.actions.action import ExpertContext


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
            result["failure_mode"] = "failed_at_" + self.task.stages[-1]
            result["timeout_holding"] = self.trace.holding
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
        position = np.asarray(state.ee.position) + rotation_matrix_wxyz(xyzw_to_wxyz(state.ee.quat_xyzw)) @ self.control_offset
        return Pose.from_lists(position, state.ee.quat_xyzw)

    def step(self, target_sim, *, action_raw=None, target_eef_pose=None, gripper_action=None, phase=None):
        import torch
        from sim.workspace.envs.mdp._common import as_torch
        from sim.recording.recorder import split_visual_obs

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


def task_with_start_options(task, options):
    """Keep per-run sampling settings separate from the registered task plugin."""
    if not options:
        return task
    start = {**task.spec.start, **options}
    if np.any(np.asarray(start['xy_min']) >= start['xy_max']):
        raise ValueError('Start xy_min must be below xy_max on both axes')
    if start['yaw_deg'][0] > start['yaw_deg'][1]:
        raise ValueError('Start yaw_deg must be in ascending order')
    if not isinstance(start.get('frame'), str) or not start['frame']:
        raise ValueError('Start frame must be a nonempty scene frame name')
    return type(task)(replace(task.spec, start=start, raw={**task.spec.raw, 'start': start}))


def jitter_start_positions(arm, positions, radians, seed):
    """Vary only revolute arm joints; fingers and the fixed mount stay unchanged."""
    result = np.array(positions, copy=True)
    rng = np.random.default_rng(seed)
    for index, joint in enumerate(arm.action_joints):
        if joint.role == 'arm' and joint.type == 'revolute':
            low, high = np.deg2rad(joint.limits)
            result[:, index] = np.clip(result[:, index] + rng.uniform(-radians, radians, len(result)),
                                       low+1e-5, high-1e-5)
    return result


def reset_task_tracking(raw):
    term = getattr(raw.cfg.events, "reset_tracking", None)
    if term is not None:
        prepared = raw.event_manager.get_term_cfg("reset_tracking")
        prepared.func(raw, None, **prepared.params)


def _matrix_pose(matrix):
    import torch
    from isaaclab.utils.math import quat_from_matrix
    quaternion = quat_from_matrix(torch.as_tensor(matrix[:3, :3], dtype=torch.float64, device="cpu"))
    return Pose.from_lists(matrix[:3, 3], quaternion.numpy())


@dataclass
class GenerationContext:
    args: object
    env: object
    task: object
    arm: object
    mapping: object
    calibration: object
    cameras: dict
    writer: object
    ik: object
    kinematics: object
    source: str = "scripted"

    @property
    def raw(self):
        return getattr(self.env, "unwrapped", self.env)

    def target_pose(self, target):
        from utils.geometry import quaternion_multiply, rotation_matrix_wxyz, xyzw_to_wxyz

        matrix = self.kinematics.fk(LiveRollout._numpy(target).reshape(-1))
        local = _matrix_pose(matrix)
        robot = self.raw.scene["robot"]
        base_id = robot.find_bodies(self.arm.links.base)[0][0]
        p = LiveRollout._numpy(robot.data.body_link_pos_w)[0, base_id]
        q = LiveRollout._numpy(robot.data.body_link_quat_w)[0, base_id]
        world = p + rotation_matrix_wxyz(xyzw_to_wxyz(q)) @ local.position
        return Pose.from_lists(world, quaternion_multiply(q, local.quat_xyzw))

    def gripper_target(self, target, fraction=None, gap_m=None):
        """Interpolate each gripper's declared physical closed/open codec endpoints."""
        result = target.clone()
        if gap_m is not None:
            if fraction is not None:
                raise ValueError("Specify a gripper fraction or a physical gap, not both")
            grip = self.arm.gripper
            if grip is None:
                raise ValueError("A physical gripper command requires an arm gripper")
            if grip.kind == "parallel":
                q = grip.q_for_gap(gap_m)
                ids = [index for index, joint in enumerate(self.arm.action_joints) if joint.role == "gripper"]
                primary=self.arm.action_joints[ids[0]]
                low,high=primary.limits
                fraction=(q-low)/(high-low)
                for index in ids:
                    joint=self.arm.action_joints[index]
                    result[:,index]=joint.limits[0]+fraction*(joint.limits[1]-joint.limits[0])
                return result
            from utils.arms.arms import load_arm_utils
            hook = getattr(load_arm_utils(self.arm), 'gripper_q_for_gap', None)
            if hook is not None:
                ids = [index for index, joint in enumerate(self.arm.action_joints) if joint.role == "gripper"]
                result[:, ids] = hook(gap_m)
                return result
            if grip.closed_gap_m is None or grip.open_gap_m is None:
                fraction = 0. if gap_m <= 0 else float(self.args.expert_options.get('gripper_open_fraction', .45))
            else:
                fraction = max(0., min(1., (gap_m - grip.closed_gap_m) / (grip.open_gap_m - grip.closed_gap_m)))
                if grip.open_position is not None and grip.closed_position is not None and grip.open_position < grip.closed_position:
                    fraction = 1. - fraction
        if fraction is None:
            raise ValueError("A gripper opening is required")
        raw = self.mapping.to_raw(result)
        ids = [index for index, joint in enumerate(self.arm.action_joints) if joint.role == "gripper"]
        raw[:, ids] = self.mapping.raw_mins[ids] + float(fraction) * (
            self.mapping.raw_maxs[ids] - self.mapping.raw_mins[ids])
        decoded = self.mapping.to_sim(raw)
        result[:, ids] = decoded[:, ids]
        return result

    def begin_attempt(self, seed, *, reset=True, warmup=True, observer=None):
        import torch
        from sim.workspace.envs.mdp._common import as_torch

        self.raw.cfg.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if reset:
            self.env.reset()
        jitter = getattr(self.args, 'start_joint_jitter_rad', 0.)
        if jitter:
            robot = self.raw.scene['robot']
            ids = robot.find_joints(list(self.arm.action_usd_joint_names), preserve_order=True)[0]
            positions = as_torch(robot.data.joint_pos)[:, ids]
            varied = positions.new_tensor(jitter_start_positions(self.arm, positions.detach().cpu().numpy(), jitter, seed))
            robot.write_joint_state_to_sim_index(position=varied, velocity=torch.zeros_like(varied), joint_ids=ids)
            robot.set_joint_position_target_index(target=varied, joint_ids=ids)
            from sim.workspace.envs.mdp.mimic import write_mimic_state
            write_mimic_state(self.raw)
        if warmup:
            ids = self.raw.scene["robot"].find_joints(list(self.arm.action_usd_joint_names), preserve_order=True)[0]
            hold = as_torch(self.raw.scene["robot"].data.joint_pos)[:, ids].clone()
            from sim.actions.action import gripper_commands
            widths = [max(obj.size[:2]) for name, obj in self.task.spec.objects.items()
                      if name in self.task.spec.tracked_objects and obj.size is not None]
            opened, _ = gripper_commands(SimpleNamespace(
                arm=self.arm, options=self.args.expert_options), max(widths, default=.02))
            hold = self.gripper_target(hold, opened.get('gripper_open_fraction'), opened.get('gripper_open_gap_m'))
            for _ in range(self.args.warmup_steps):
                self.env.step(hold)
            reset_task_tracking(self.raw)
        self.ik.reset()
        sizes = {name: self.task.spec.objects[name].size or self.task.spec.objects[name].params.get("size_outer")
                 for name in self.task.spec.tracked_objects}
        recorder = SuccessRecorder(self.writer, self.task, self.arm, sizes, self.args.fps,
                                   control_offset_m=self.ik.position_offset_m)
        return LiveRollout(self.env, self.task, self.arm, self.mapping, recorder,
                           control_offset=self.ik.position_offset_m, observer=observer, target_pose_fn=self.target_pose)

    def close(self):
        try:
            self.writer.finalize()
        finally:
            self.env.close()


def generation_ik_options(task, args, *, use_expert=True):
    """Shared scripted/Mimic options: task-owned tolerances, weight and unchanged servo tuning."""
    from sim.actions.ik import IKOptions

    expert = task.spec.raw.get("expert", {}) if use_expert else {}
    return IKOptions(integral_gain=getattr(args, "ik_integral_gain", 1.0),
                     orientation_weight=expert.get("orientation_weight", .05),
                     orientation_tolerance_rad=expert.get("orientation_tolerance_rad", .03),
                     damping=getattr(args, "ik_damping", .0125),
                     max_integral_bias_rad=getattr(args, "ik_max_integral_bias_rad", .08),
                     max_integral_bias_m=getattr(args, "ik_max_integral_bias_m", .005))


def setup_context(args, *, env_factory=None, sample_starts=True, source=None, use_expert_ik=True):
    """Build a task environment after Kit launch; Mimic can supply its real env class."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.managers import EventTermCfg
    import sim.workspace.envs
    from utils.arms.arms import load_arm_spec
    from utils.arms.calibration import resolve_sim_calibration
    from utils.arms.mapping import JointMapping
    from utils.datasets.alignment import mark_recording_root, recording_contract
    from sim.workspace.envs.mdp.starts import reset_from_starts
    from sim.workspace.envs.scene_builder import drop_unused_camera_outputs
    from sim.datagen.dr_replay import discover_sim_cameras
    from sim.actions.ik import make_ik, make_kinematics
    from sim.recording.recorder import LeRobotRecorder

    if Path(args.destination).exists():
        raise FileExistsError(args.destination)
    task = sim.workspace.envs.register(scene=args.scene, task=args.task)
    task = task_with_start_options(task, getattr(args, 'start_options', {}))
    arm, _ = load_arm_spec(args.arm)
    calibration = resolve_sim_calibration(arm, unit=args.unit)
    cfg = parse_env_cfg(task.gym_id(args.variant), device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    cfg.seed = args.seed
    initial_joints = getattr(args, 'initial_joints_sim', None)
    if initial_joints is not None:
        if np.shape(initial_joints) != (len(arm.action_joints),) or not np.isfinite(initial_joints).all():
            raise ValueError('initial_joints_sim must contain one finite value per action joint')
        for value, joint in zip(initial_joints,arm.action_joints):
            limits = np.deg2rad(joint.limits) if joint.type=='revolute' else joint.limits
            if not limits[0] <= value <= limits[1]:
                raise ValueError(f'Initial joint {joint.name} is outside its physical limits')
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(arm.action_usd_joint_names,initial_joints)))
    cfg.sim_joint_mapping = calibration.mapping
    cfg.terminations = None
    cfg.episode_length_s = (args.max_steps + getattr(args, "warmup_steps", 0) + 1) / args.fps
    if not math.isclose(1 / (cfg.decimation * cfg.sim.dt), args.fps, abs_tol=1e-6):
        raise ValueError("Generation FPS must match the task's physical control timestep")
    cfg.sim.render_interval = cfg.decimation
    for name in ('njmax', 'nconmax'):
        value = getattr(args, 'solver_' + name, None)
        if value is not None:
            setattr(cfg.sim.physics.solver_cfg, name, value)
    cfg.events.reset_from_starts = (EventTermCfg(func=reset_from_starts, mode="reset", params={
        "task": task, "starts_dir": str(args.starts_dir) if args.starts_dir else None,
        "mode": "random", "seed": None, "reset_robot": False,
    }) if sample_starts else None)
    if (sample_starts and (source or getattr(args, "source", "scripted")) == "scripted" and
            getattr(args, "start_mode", "random") == "grid" and not args.starts_dir):
        from sim.datagen.starts import GridStarts

        cfg.events.reset_from_starts.params["start_sampler"] = GridStarts(
            task, cells_per_axis=args.cells_per_axis, yaw_steps=args.yaw_steps,
            max_starts=args.max_starts, seed=args.seed)
    drop_unused_camera_outputs(cfg)
    env = env_factory(cfg) if env_factory is not None else gym.make(task.gym_id(args.variant), cfg=cfg)
    raw = getattr(env, "unwrapped", env)
    writer = None
    try:
        cameras = discover_sim_cameras(env)
        mapping = JointMapping.from_arm(arm, calibration.mapping, device=raw.device)
        writer = LeRobotRecorder(task_name=task.instruction, repo_id=args.repo_id, dataset_root=str(args.destination),
                                 fps=args.fps, device=raw.device, cameras=cameras, arm=arm)
        writer.init_dataset()
        producer = source or getattr(args, "source", "scripted")
        contract = recording_contract(follower_id=None, leader_id=f"datagen:{producer}", calibration_selection=calibration)
        contract.update(start_sync="sim_state", wrist_rezero=False, sim_wrist_relative=False,
                        recorded_action="absolute_sim_target_in_device_units",
                        note="Generated actions and observations use the selected device codec; no relative teleop wrist shift.")
        mark_recording_root(args.destination, contract)
        expert = task.spec.raw.get("expert", {}) if use_expert_ik else {}
        options = generation_ik_options(task, args, use_expert=use_expert_ik)
        ik = make_ik(raw.scene["robot"], arm, position_offset_m=expert.get("control_offset"),
                     solver=expert.get("ik_solver"), control_dt=cfg.decimation * cfg.sim.dt, options=options)
        kinematics = make_kinematics(arm, position_offset_m=ik.position_offset_m)
        return GenerationContext(args, env, task, arm, mapping, calibration, cameras, writer, ik, kinematics, producer)
    except BaseException:
        try:
            if writer is not None:
                writer.finalize()
        finally:
            env.close()
        raise


def make_report(context, *, source_provenance=None):
    from dataclasses import asdict
    from utils.provenance import git_commit
    arm, args = context.arm, context.args
    return {"schema_version": 1, "source": context.source, "dataset": str(Path(args.destination).resolve()),
            "repo_id": args.repo_id, "arm": arm.name, "task": context.task.name, "scene": args.scene,
            "variant": args.variant, "fps": args.fps, "joint_names": list(arm.joint_names),
            "joint_units": [joint.unit for joint in arm.action_joints], "joint_roles": [joint.role for joint in arm.action_joints],
            "joint_mapping": context.calibration.mapping, "calibration": context.calibration.provenance,
            "control_offset_m": np.asarray(context.ik.position_offset_m).tolist(),
            "ik_controller": {
                "library": getattr(context.ik, "library_name", "isaaclab.controllers.DifferentialIKController"),
                "method": getattr(context.ik, "method_name", arm.ik.method),
                "joint_limit_avoidance_gain": arm.ik.joint_limit_avoidance_gain,
                "mode": getattr(context.ik, "mode_name", {
                    "scripted": "native_jacobian_feedback", "mimic": "bounded_command_pose_inverse",
                    "calibration_replay": "recorded_joint_command_replay"}.get(context.source, context.source)),
                "integral_feedback_applied": context.source == "scripted" and context.ik.options.integral_gain > 0,
                "control_dt_s": 1 / args.fps, "options": asdict(context.ik.options)},
            "seed": args.seed, "git_commit": git_commit(Path(__file__).resolve().parents[3]),
            "start_sampling": {"region": dict(context.task.spec.start),
                               "joint_jitter_rad": getattr(args, 'start_joint_jitter_rad', 0.),
                               "initial_joints_sim": getattr(args, 'initial_joints_sim', None)},
            "source_provenance": source_provenance or {}, "episodes": [], "accepted_episodes": 0,
            "requested_episodes": getattr(args, "episodes", None),
            "attempt_limit": getattr(args, "attempts", None), "attempts_completed": 0, "complete": False}


def publish_report(context, report):
    from sim.datagen.starts import GridStarts

    sampler = getattr(context.raw, "_task_starts", None)
    if isinstance(sampler, GridStarts):
        report["start_sampling"].update(mode="grid", coverage=sampler.coverage())
    report["accepted_episodes"] = sum(row["accepted"] for row in report["episodes"])
    report["attempts_completed"] = len(report["episodes"])
    destination = context.args.report or Path(str(context.args.destination) + ".report.json")
    write_json(destination, report)
    write_json(Path(context.args.destination) / "meta/datagen.json", report)


def carried_object_offset(state, control_pose, name, previous=None):
    """Track in-hand slip while held; freeze the attachment after release."""
    if previous is None or state.signals.get('holding', False):
        return np.asarray(control_pose.position) - state.objects[name].position
    return np.asarray(previous)


def execute_expert(context, rollout, expert):
    from sim.eval.scene import capture_scene

    initial = rollout.snapshot()
    geometry = capture_scene(context.raw, context.task)
    a, b, _ = geometry["table_plane_world"]
    normal = np.array([-a, -b, 1.])
    normal /= np.linalg.norm(normal)
    arm_base = Pose.from_lists(geometry["base_position_world"], wxyz_to_xyzw(geometry["base_quaternion_wxyz"]))
    planning_context = ExpertContext(initial, rollout.control_pose(initial), tuple(normal),
                                     context.args.expert_options, arm_base=arm_base, arm=context.arm)
    waypoints = expert.plan(planning_context)
    dt = 1 / context.args.fps
    steps = 0
    rollout.waypoint_trace = []

    def boundary(event, waypoint, elapsed):
        if not getattr(context.args, "debug", False):
            return
        state = rollout.last_state or initial
        actual = rollout.control_pose(state)
        record = {
            "event": event, "waypoint": waypoint.name, "step": steps, "elapsed_s": elapsed,
            "actual_tcp_pose": {"position": list(actual.position), "quat_xyzw": list(actual.quat_xyzw)},
            "desired_tcp_pose": {"position": list(waypoint.pose.position), "quat_xyzw": list(waypoint.pose.quat_xyzw)},
            "objects": {name: {"position": list(pose.position), "quat_xyzw": list(pose.quat_xyzw)}
                        for name, pose in state.objects.items()},
            "signals": dict(state.signals), "stages": dict(rollout.recorder.trace.stages),
        }
        if getattr(context.args, "debug", False):
            import re
            from PIL import Image
            from sim.recording.recorder import split_visual_obs

            report_path = context.args.report or Path(str(context.args.destination) + ".report.json")
            directory = report_path.parent / (report_path.stem + "-diagnostics") / f"attempt_{getattr(rollout, 'attempt', 0):04d}"
            if directory.resolve().is_relative_to(Path(context.args.destination).resolve()):
                directory = Path(context.args.destination).parent / (Path(context.args.destination).name + "-diagnostics") / directory.name
            directory.mkdir(parents=True, exist_ok=True)
            # Rendering adds no physical steps or task-signal reads. Force the
            # sensors current so the picture describes this exact boundary.
            for _ in range(2):
                context.raw.sim.render()
            for sensor in context.raw.scene.sensors.values():
                if "rgb" in getattr(sensor.cfg, "data_types", ()):
                    sensor.update(0., force_recompute=True)
            visual = context.raw.observation_manager.compute()["visual"]
            images, _, _ = split_visual_obs(visual, context.cameras)
            record["images"] = {}
            for name, pixels in images.items():
                label = re.sub(r"[^a-zA-Z0-9_-]", "_", waypoint.name + "_" + name)
                path = directory / f"{len(rollout.waypoint_trace):02d}_{event}_{label}.png"
                Image.fromarray(LiveRollout._numpy(pixels)[..., :3].astype(np.uint8)).save(path)
                record["images"][name] = os.path.relpath(path, report_path.parent)
        rollout.waypoint_trace.append(record)

    def failure_diagnostics(waypoint, requested_position, requested_quaternion, command, elapsed, phase_steps):
        if not getattr(context.args, "debug", False):
            return
        from sim.workspace.envs.mdp._common import as_torch

        actual = rollout.control_pose(rollout.last_state)
        final = context.ik.solve(waypoint.pose.position, requested_quaternion, advance=False)
        robot = context.raw.scene["robot"]
        ids = robot.find_joints(list(context.arm.action_usd_joint_names), preserve_order=True)[0]
        measured = LiveRollout._numpy(as_torch(robot.data.joint_pos)[:, ids])[0]
        commanded = LiveRollout._numpy(command).reshape(-1)
        rollout.diagnostics = {
            "waypoint": waypoint.name, "elapsed_s": elapsed, "phase_steps": phase_steps, "total_steps": steps,
            "actual_tcp_pose": {"position": list(actual.position), "quat_xyzw": list(actual.quat_xyzw)},
            "desired_tcp_pose": {"position": list(waypoint.pose.position), "quat_xyzw": list(waypoint.pose.quat_xyzw)},
            "last_requested_tcp_pose": {"position": np.asarray(requested_position).tolist(),
                                        "quat_xyzw": np.asarray(requested_quaternion).tolist()},
            "position_error_m": float(final.position_error[0]),
            "position_error_vector_world_m": (np.asarray(waypoint.pose.position) - actual.position).tolist(),
            "orientation_error_rad": float(final.orientation_error[0]),
            "orientation_axes": list(context.arm.ik.orientation_axes),
            "position_tolerance_m": waypoint.tolerance_m,
            "orientation_tolerance_rad": context.ik.options.orientation_tolerance_rad,
            "joint_names": list(context.arm.joint_names),
            "sim_joint_units": ["rad" if joint.type == "revolute" else "m" for joint in context.arm.action_joints],
            "measured_joint_positions_sim": measured.tolist(),
            "last_commanded_joint_positions_sim": commanded.tolist(),
            "joint_command_error_sim": (commanded - measured).tolist(),
            "integral_bias_sim": LiveRollout._numpy(context.ik.integral_bias).reshape(-1).tolist(),
            "final_ik_limited": bool(final.limited[0]),
        }

    attachment_offsets={}
    previous_grip_command = None
    for waypoint in waypoints:
        state=rollout.last_state or initial
        start = rollout.control_pose(state)
        adapt = getattr(expert, "adapt_waypoint", None)
        if adapt is not None:
            waypoint = adapt(waypoint, replace(planning_context, state=state, control_pose=start))
        object_goal = None
        if waypoint.object_ref is not None:
            object_goal = np.asarray(waypoint.pose.position)
            name=waypoint.object_ref
            if name not in state.objects:
                raise ValueError(f"Waypoint object reference {name!r} is not in the task state")
            if name not in attachment_offsets:
                attachment_offsets[name]=np.asarray(start.position)-state.objects[name].position
            waypoint=replace(waypoint,pose=Pose.from_lists(
                np.asarray(waypoint.pose.position)+attachment_offsets[name],waypoint.pose.quat_xyzw))
        boundary("start", waypoint, 0.)
        gripper_ramp = float(context.args.expert_options.get('gripper_ramp_seconds', 0.))
        if gripper_ramp > 0:
            from sim.workspace.envs.mdp._common import as_torch
            robot = context.raw.scene['robot']
            action_ids = robot.find_joints(list(context.arm.action_usd_joint_names), preserve_order=True)[0]
            grip_ids = [i for i,j in enumerate(context.arm.action_joints) if j.role=='gripper']
            start_grip = (previous_grip_command.clone() if previous_grip_command is not None else
                          as_torch(robot.data.joint_pos)[:,action_ids][:,grip_ids].clone())
        for tick in range(1, math.ceil(waypoint.timeout_s / dt) + 1):
            elapsed = tick * dt
            if object_goal is not None and context.args.expert_options.get('object_feedback', False):
                current = rollout.last_state or initial
                attachment_offsets[name] = carried_object_offset(
                    current, rollout.control_pose(current), name, attachment_offsets[name])
                waypoint = replace(waypoint, pose=Pose.from_lists(
                    object_goal + attachment_offsets[name], waypoint.pose.quat_xyzw))
            alpha = min(1., elapsed / waypoint.duration_s)
            alpha = alpha * alpha * (3 - 2 * alpha)
            position = np.asarray(start.position) * (1 - alpha) + np.asarray(waypoint.pose.position) * alpha
            first, last = np.asarray(start.quat_xyzw), np.asarray(waypoint.pose.quat_xyzw)
            last = last if np.dot(first, last) >= 0 else -last
            quaternion = first * (1 - alpha) + last * alpha
            quaternion /= np.linalg.norm(quaternion)
            orient = getattr(expert, "control_orientation", None)
            if orient is not None:
                current = rollout.last_state or initial
                quaternion = orient(waypoint, current, rollout.control_pose(current),
                                    quaternion, planning_context, alpha)
            solved = context.ik.solve(position, quaternion)
            target = context.gripper_target(solved.joint_positions, waypoint.gripper_open_fraction, waypoint.gripper_open_gap_m)
            if gripper_ramp > 0:
                ramp = min(1., elapsed/gripper_ramp)
                ramp = ramp*ramp*(3-2*ramp)
                target[:,grip_ids] = start_grip*(1-ramp)+target[:,grip_ids]*ramp
            rollout.step(target, target_eef_pose=Pose.from_lists(position, quaternion), phase=waypoint.name)
            if gripper_ramp > 0:
                previous_grip_command = target[:,grip_ids].clone()
            steps += 1
            if context.task.is_success(rollout.recorder.trace.stages):
                boundary("task_success", waypoint, elapsed)
                return "task_success"
            if steps >= context.args.max_steps:
                failure_diagnostics(waypoint, position, quaternion, target, elapsed, tick)
                boundary("max_steps", waypoint, elapsed)
                return "max_steps"
            if elapsed >= waypoint.duration_s:
                final = context.ik.solve(waypoint.pose.position, quaternion, advance=False)
                if (float(final.position_error[0]) <= waypoint.tolerance_m and
                        float(final.orientation_error[0]) <= context.ik.options.orientation_tolerance_rad):
                    boundary("complete", waypoint, elapsed)
                    break
        else:
            failure_diagnostics(waypoint, position, quaternion, target, elapsed, tick)
            boundary("timeout", waypoint, elapsed)
            return "waypoint_timeout:" + waypoint.name
    return "expert_finished_without_success"


class DiagnosticTrace:
    """Copy buffers already observed by the shared executor; never step or evaluate a task."""

    def __init__(self, context, attempt, seed):
        report = context.args.report or Path(str(context.args.destination) + ".report.json")
        self.path = report.with_name(report.stem + '-diagnostics') / f'attempt_{attempt:04d}.json'
        if self.path.resolve().is_relative_to(Path(context.args.destination).resolve()):
            raise ValueError('Mimic diagnostic traces must be outside the dataset destination')
        self.data = {'schema_version': 1, 'source': 'isaaclab_mimic', 'attempt': attempt, 'seed': seed,
                     'joint_names': list(context.arm.joint_names),
                     'sim_joint_units': ['rad' if joint.type == 'revolute' else 'm' for joint in context.arm.action_joints],
                     'alignment': 'annotations are pre-action; states, signals and measured joints are post-step',
                     'initial_state': None, 'initial_joint_positions_sim': None, 'steps': []}

    @staticmethod
    def _joints(rollout):
        return LiveRollout._numpy(rollout.robot.data.joint_pos)[0, rollout.joint_ids].copy().tolist()

    def begin(self, rollout):
        self.data['initial_state'] = _json_state(rollout.snapshot())
        self.data['initial_joint_positions_sim'] = self._joints(rollout)

    def append(self, rollout, target):
        state = _json_state(rollout.last_state)
        self.data['steps'].append({
            'frame_index': rollout.recorder.steps - 1,
            'commanded_joint_positions_sim': LiveRollout._numpy(target).reshape(-1).copy().tolist(),
            'measured_joint_positions_sim': self._joints(rollout),
            'task_signals': deepcopy(state['signals']), 'state': state,
            'latched_stages': dict(rollout.recorder.trace.stages),
        })

    def write(self, rollout, result, error):
        self.data['annotations'] = [] if rollout is None else deepcopy(rollout.recorder.annotations)
        self.data['result'], self.data['error'] = result, error
        write_json(self.path, self.data)
