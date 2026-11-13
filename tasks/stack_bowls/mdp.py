"""Task-owned grasp, upright nesting, release and stable confirmation signals."""
import torch
import isaaclab.utils.math as math_utils
from sim.envs.mdp import as_torch, env_indices
from sim.envs.mdp._common import _ee_body
from .geometry import nesting_metrics, any_nested_pair


def reset_tracking(env, env_ids):
    from . import TASK
    ids = env_indices(env, env_ids)
    positions = torch.stack([as_torch(env.scene[n].data.root_pos_w) for n in TASK.spec.tracked_objects], 1)
    if not hasattr(env, '_bowls'):
        env._bowls = dict(rest_z=positions[:,:,2].clone(),
            ever_lifted=torch.zeros(positions.shape[:2], device=env.device, dtype=torch.bool),
            ever_grasped=torch.zeros(positions.shape[:2], device=env.device, dtype=torch.bool),
            confirm=torch.zeros(env.num_envs, device=env.device, dtype=torch.long), step=-1)
    data = env._bowls
    data['rest_z'][ids] = positions[ids,:,2]
    data['ever_lifted'][ids] = False
    data['ever_grasped'][ids] = False
    data['confirm'][ids] = 0
    data['step'] = -1
    env.scene['contact_grasp'].reset(ids)


def stage_signals(env):
    from . import TASK
    if not hasattr(env, '_bowls'):
        reset_tracking(env, None)
    data = env._bowls
    if data['step'] == env.common_step_counter:
        return data['signals']
    s = TASK.spec.raw['success']
    objects = [env.scene[n] for n in TASK.spec.tracked_objects]
    p = torch.stack([as_torch(o.data.root_pos_w) for o in objects], 1)
    q = torch.stack([as_torch(o.data.root_quat_w) for o in objects], 1)
    v = torch.stack([as_torch(o.data.root_vel_w) for o in objects], 1)
    data['rest_z'] = torch.minimum(data['rest_z'], p[:,:,2])
    lifted = p[:,:,2]-data['rest_z'] >= s['min_lift']
    force = as_torch(env.scene['contact_grasp'].data.force_matrix_w)
    contact = torch.linalg.vector_norm(force, dim=-1).sum(1)
    held = (contact > s['force_threshold']) & lifted
    data['ever_grasped'] |= held
    data['ever_lifted'] |= lifted & data['ever_grasped']
    paired, upright, stable = nesting_metrics(p, q, v, s)
    released = contact.max(1).values < .02
    warmed = env.episode_length_buf >= s['warmup_steps']
    stacked = paired.all(1) & upright.all(1) & stable.all(1) & released & warmed
    # Two distinct objects must have been physically carried in this episode.
    carried = data['ever_lifted'].sum(1)
    robot = env.scene['robot']
    body, _ = _ee_body(env, robot)
    ee = as_torch(robot.data.body_link_pos_w)[:,body]
    eq = as_torch(robot.data.body_link_quat_w)[:,body]
    offset = ee.new_tensor(env.cfg.arm_spec.ik.position_offset_m).expand_as(ee)
    tcp = ee + math_utils.quat_apply(eq, offset)
    distance = torch.linalg.vector_norm(tcp[:,None]-p, dim=-1).min(1).values
    eligible = stacked & (carried >= 2) & (distance >= s.get('gripper_clearance_m', .085))
    data['confirm'] = torch.where(eligible, data['confirm']+1, 0)
    data['signals'] = dict(reached=distance<.075, grasped=data['ever_grasped'].any(1),
        lifted=carried>=1, first_bowl_stacked=any_nested_pair(p,s,q)&upright.all(1)&stable.all(1)&released&(carried>=1),
        second_bowl_lifted=carried>=2, all_bowls_stacked=eligible,
        success=data['confirm']>=s['confirm_steps'], holding=held.any(1),
        reach_distance=distance, carried_bowls=carried, nested_pair_count=paired.sum(1))
    data['step'] = env.common_step_counter
    return data['signals']


def success(env):
    return stage_signals(env)['success']
