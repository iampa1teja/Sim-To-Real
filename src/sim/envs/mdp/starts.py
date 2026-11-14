"""Place declared objects from recorded, fixed-benchmark or task-sampled starts."""
from pathlib import Path
import os

import numpy as np
import torch

from utils.arms.spec import load_arm_spec
from core.tasks.base import Pose
from core.tasks.loader import load_task
from sim.envs.mdp._common import _wxyz, _xyzw, as_torch, env_indices


def _poses(objects):
    return {name: Pose.from_lists(row["position"], _xyzw(row["orientation_wxyz"]))
            for name, row in objects.items()}


class _TaskStarts:
    def __init__(self, task, seed):
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.scene = None

    def sample(self, accepts_random=None):
        from core.tasks.base import sample_start_in_base

        for _ in range(10000):
            poses = sample_start_in_base(self.task, self.rng, self.scene)
            objects = {name: dict(position=np.asarray(pose.position),
                                  orientation_wxyz=_wxyz(pose.quat_xyzw)) for name, pose in poses.items()}
            if accepts_random is None or accepts_random(objects):
                return -1, objects, "random"
        raise ValueError(f"Cannot sample a valid start for task {self.task.name!r} after 10000 attempts")


def _dataset_root(path):
    path = Path(path)
    return path.parent if path.name in ("task_meta", "pick_place_meta") else path


def reset_from_starts(
    env, env_ids, *, task=None, starts_dir=None, mode="cycle", random_fraction=0.0, seed=None,
    reset_robot=True, eval_set=None, splits=None, repeats=1, start_offset=0, benchmark_selection="all",
):
    from isaaclab.utils.math import quat_mul
    from utils.arms.mapping import dataset_state_to_radians
    from core.eval.eval_set import EvalSetStarts
    from core.eval.geometry import resting_pose, upright_orientation
    from core.eval.starts import RecordedStarts
    from sim.eval.scene import capture_scene

    task = load_task(task or getattr(env.cfg, "task_name", None)) if isinstance(task, (str, type(None))) else task
    arm = getattr(env.cfg, "arm_spec", None) or load_arm_spec()[0]
    ids = env_indices(env, env_ids)
    if not len(ids):
        return
    starts_dir = starts_dir or os.environ.get("S2R_EVAL_STARTS")
    seed = seed if seed is not None else env.cfg.seed
    key = (task.name, arm.name, str(starts_dir), mode, random_fraction, seed, bool(reset_robot),
           str(eval_set), str(splits), repeats, start_offset, benchmark_selection)
    if getattr(env, "_task_starts_key", None) != key:
        if eval_set:
            sampler = EvalSetStarts(eval_set, splits, repeats, seed, start_offset=start_offset,
                                   selection=benchmark_selection, arm=arm, task=task)
        elif starts_dir:
            sampler = RecordedStarts(starts_dir, mode, random_fraction, seed,
                                     robot_states_root=_dataset_root(starts_dir) if reset_robot else None,
                                     arm=arm, task=task)
        else:
            sampler = _TaskStarts(task, seed)
        env._task_starts, env._task_starts_key = sampler, key
    sampler = env._task_starts
    robot = env.scene["robot"]
    env.sim.forward()
    base_ids, _ = robot.find_bodies(arm.links.base)
    if len(base_ids) != 1:
        raise ValueError(f"Starts require one robot base body named {arm.links.base!r}")
    base_id = base_ids[0]
    body_pos, body_quat = as_torch(robot.data.body_link_pos_w), as_torch(robot.data.body_link_quat_w)
    if not hasattr(env, "_task_start_index"):
        env._task_start_index = [-1] * env.num_envs
        env._task_start_zone = [None] * env.num_envs
    object_rows = {name: [] for name in task.spec.tracked_objects}
    joint_starts = []
    joint_ids, _ = robot.find_joints(list(arm.action_usd_joint_names), preserve_order=True)
    for index in ids.tolist():
        live = capture_scene(env, task, index)
        base_p = body_pos[index, base_id].detach().cpu().numpy()
        base_q = _wxyz(body_quat[index, base_id].detach().cpu().numpy())
        plane = live.get("table_plane_world")
        if hasattr(sampler, "scene"):
            sampler.scene = live

        def accepts_random(objects):
            return task.start_is_valid(_poses(objects), live)

        start, objects, zone = sampler.sample(accepts_random)
        if set(objects) != set(task.spec.tracked_objects):
            raise ValueError(f"Start objects {sorted(objects)} differ from tracked objects {list(task.spec.tracked_objects)}")
        if not task.start_is_valid(_poses(objects), live):
            raise ValueError(f"Start {start} is invalid in the live scene; verify the dataset and object poses")
        for name, row in objects.items():
            position, orientation = np.asarray(row["position"]), np.asarray(row["orientation_wxyz"])
            spec = task.spec.objects[name]
            asset = env.scene[name]
            size = spec.size or spec.params.get("size_outer") or getattr(asset.cfg.spawn, "size", None)
            if plane is not None and size is not None and spec.params.get("rest_on_table", True):
                if start == -1 or (eval_set and sampler.current["split"] != "train"):
                    orientation = upright_orientation(base_q, plane, orientation)
                world, _ = resting_pose(position, orientation, base_p, base_q, plane, size)
            else:
                from core.eval.geometry import rotation_matrix
                world = base_p + rotation_matrix(base_q) @ position
            world_orientation = quat_mul(body_quat[index, base_id],
                torch.as_tensor(_xyzw(orientation), device=env.device, dtype=body_quat.dtype))
            object_rows[name].append((world, world_orientation))
        env._task_start_index[index], env._task_start_zone[index] = start, zone
        if eval_set:
            env._benchmark_start = dict(sampler.current)
        if reset_robot:
            if isinstance(sampler, _TaskStarts):
                joint_starts.append(as_torch(robot.data.joint_pos)[index, joint_ids].detach().cpu().numpy())
            else:
                mapping = getattr(sampler, "joint_mapping", None) or getattr(env.cfg, "sim_joint_mapping", None)
                if eval_set:
                    mapping = sampler.data.get("joint_mapping", sampler.data.get("scene", {}).get("joint_mapping", mapping))
                joint_starts.append(dataset_state_to_radians(sampler.robot_state(start), mapping, arm=arm))
    if reset_robot:
        joints = torch.as_tensor(np.asarray(joint_starts), device=env.device, dtype=as_torch(robot.data.joint_pos).dtype)
        robot.write_joint_state_to_sim_index(position=joints, velocity=torch.zeros_like(joints),
                                            joint_ids=joint_ids, env_ids=ids)
        robot.set_joint_position_target_index(target=joints, joint_ids=joint_ids, env_ids=ids)
        from sim.envs.mdp.mimic import write_mimic_state
        write_mimic_state(env, ids)
    for name, rows in object_rows.items():
        asset = env.scene[name]
        position = torch.as_tensor(np.asarray([row[0] for row in rows]), device=env.device,
                                   dtype=as_torch(asset.data.root_pos_w).dtype)
        pose = torch.cat((position, torch.stack([row[1] for row in rows])), dim=-1)
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        asset.write_root_velocity_to_sim_index(root_velocity=position.new_zeros((len(ids), 6)), env_ids=ids)


def assert_reset_event_order(env, env_ids):
    names = env.event_manager.active_terms["reset"]
    required = ("reset_robot_position", "reset_from_starts", "reset_tracking")
    if any(name not in names for name in required) or not all(
            names.index(a) < names.index(b) for a, b in zip(required, required[1:])):
        raise RuntimeError(f"Invalid task reset order: {names}; require {' < '.join(required)}")
    if "eval_robot_color" in names and not names.index("reset_set_robot_visual_material") < names.index("eval_robot_color"):
        raise RuntimeError(f"Robot colour randomization would be overwritten: {names}")
    print(f"[Task eval] Reset event order: {names}")
