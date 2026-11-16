"""Three original hollow bowls: order-free nesting, stages and composed rim moves."""
import math
import numpy as np

from sim.actions import approach, hold, pick, place
from sim.actions.action import Check
from sim.actions.checks.grasped import Carried
from sim.actions.moves.rim import Rim
from utils.geometry import Pose

OBJECTS = ("yellow_bowl", "red_bowl", "blue_bowl")
SETTINGS = {'color_order': 'any',
 'min_lift': 0.012,
 'force_threshold': 0.15,
 'warmup_steps': 20,
 'confirm_steps': 30,
 'xy_tolerance': 0.009,
 'nested_height_min': 0.012,
 'nested_height_max': 0.035,
 'max_tilt_degrees': 30.0,
 'max_support_tilt_degrees': 15.0,
 'max_bottom_tilt_degrees': 12.0,
 'max_speed': 0.025,
 'max_angular_speed': 0.35,
 'gripper_clearance_m': 0.085}


def nesting_metrics(positions, quaternions, velocities, settings):
    import torch
    order = torch.argsort(positions[:,:,2], dim=1)
    sorted_positions = torch.gather(positions, 1, order[:,:,None].expand(-1,-1,3))
    difference = sorted_positions[:,1:]-sorted_positions[:,:-1]
    paired = (torch.linalg.vector_norm(difference[:,:,:2], dim=-1) <= settings['xy_tolerance']) & (
        difference[:,:,2] >= settings['nested_height_min']) & (
        difference[:,:,2] <= settings['nested_height_max'])
    # Compare neighboring bowl axes, allowing cumulative lean in a nested
    # stack. The bottom bowl must still stand upright relative to gravity.
    x,y,z,w = quaternions.unbind(-1)
    up = torch.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),dim=-1)
    sorted_up = torch.gather(up,1,order[:,:,None].expand(-1,-1,3))
    support_aligned = (sorted_up[:,1:]*sorted_up[:,:-1]).sum(-1) >= math.cos(
        math.radians(settings.get('max_support_tilt_degrees',15.)))
    paired &= support_aligned
    upright = up[:,:,2] >= math.cos(math.radians(settings['max_tilt_degrees']))
    bottom_valid = sorted_up[:,0,2] >= math.cos(math.radians(settings.get('max_bottom_tilt_degrees',12.)))
    upright &= bottom_valid[:,None]
    stable = (torch.linalg.vector_norm(velocities[:,:,:3], dim=-1) <= settings['max_speed']) & (
        torch.linalg.vector_norm(velocities[:,:,3:], dim=-1) <= settings['max_angular_speed'])
    return paired, upright, stable


def any_nested_pair(positions, settings, quaternions=None):
    import torch
    delta = positions[:,:,None,:]-positions[:,None,:,:]
    mask = (torch.linalg.vector_norm(delta[:,:,:,:2],dim=-1) <= settings['xy_tolerance']) & (
        delta[:,:,:,2] >= settings['nested_height_min']) & (delta[:,:,:,2] <= settings['nested_height_max'])
    if quaternions is not None:
        x,y,z,w = quaternions.unbind(-1)
        up = torch.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),dim=-1)
        aligned = (up[:,:,None,:]*up[:,None,:,:]).sum(-1) >= math.cos(
            math.radians(settings.get('max_support_tilt_degrees',15.)))
        mask &= aligned
    return mask.any(2).any(1)


carried = Carried(OBJECTS, min_lift=SETTINGS['min_lift'], force=SETTINGS['force_threshold'], reset_all=True)


class Nesting(Check):
    """The task-specific order-free nesting rule, cached once per batch step."""
    _owner, _step = None, -1

    def __init__(self, carried):
        self.carried = carried

    def children(self):
        return (self.carried,)

    def reset(self, ids, frame):
        self._owner, self._step = frame.owner, -1

    def measure(self, frame):
        import torch
        if self._owner is frame.owner and self._step == frame.step:
            return self._signals
        p = torch.stack([frame.position[name] for name in OBJECTS], 1)
        q = torch.stack([frame.orientation[name] for name in OBJECTS], 1)
        v = torch.stack([frame.velocity[name] for name in OBJECTS], 1)
        held, ever_grasped, ever_lifted = self.carried.update(frame)
        contact = torch.stack([frame.contact[name] for name in OBJECTS], 1)
        paired, upright, stable = nesting_metrics(p, q, v, SETTINGS)
        released = contact.max(1).values < .02
        warmed = frame.episode_step >= SETTINGS['warmup_steps']
        stacked = paired.all(1) & upright.all(1) & stable.all(1) & released & warmed
        count = ever_lifted.sum(1)
        distance = torch.linalg.vector_norm(frame.gripper_point[:, None] - p, dim=-1).min(1).values
        eligible = stacked & (count >= 2) & (distance >= SETTINGS['gripper_clearance_m'])
        self._signals = dict(reached=distance < .075, grasped=ever_grasped.any(1), lifted=count >= 1,
            first_bowl_stacked=any_nested_pair(p, SETTINGS, q) & upright.all(1) & stable.all(1) & released & (count >= 1),
            second_bowl_lifted=count >= 2, all_bowls_stacked=eligible, holding=held.any(1),
            reach_distance=distance, carried_bowls=count, nested_pair_count=paired.sum(1))
        self._owner, self._step = frame.owner, frame.step
        return self._signals


nesting = Nesting(carried)


class Stage(Check):
    def __init__(self, name, source):
        self.name, self.source = name, source

    def children(self):
        return (self.source,)

    def evaluate(self, frame):
        return self.source.measure(frame)[self.name]

    def signals(self, frame, name):
        signals = self.source.measure(frame)
        return {key: signals[key] for key in (name, 'holding', 'reach_distance', 'carried_bowls', 'nested_pair_count')}


stages = {name: Stage(name, nesting) for name in (
    'reached', 'grasped', 'lifted', 'first_bowl_stacked', 'second_bowl_lifted', 'all_bowls_stacked')}
stages['success'] = hold(stages['all_bowls_stacked'], SETTINGS['confirm_steps'], reset_all=True)

rim = Rim(radius=.058, height=.012, tilt_degrees=21., correction_degrees=8., width=.004,
          tolerance=.005, precise=.006, travel_tolerance=.012, close_error='Bowl rim is too close to the arm base')
expert_options = {'move_seconds': 2.2, 'waypoint_timeout_seconds': 7.}


def expert(ctx):
    order = ctx.options.get('order', list(OBJECTS))
    if len(order) != 3 or set(order) != set(OBJECTS):
        raise ValueError('order must contain each of the three bowls exactly once')
    if ctx.arm_base is None:
        raise ValueError('Rim planning requires the actual robot base pose')
    moves = []
    for level, name in enumerate(order[1:], 1):
        moves += [approach(name, height=.035, rim=rim), pick(name, lift=.07, grasp_seconds=.8, rim=rim),
                  place(name, order[0], 'nested', level=level, pitch=.022, clearance=.012,
                        carry_height=.105, retreat=.04, release_seconds=.8, settle_seconds=1.4, rim=rim)]
    return moves


def sample_start(task, rng, sample):
    for _ in range(1000):
        positions = {}
        for name in task.spec.tracked_objects:
            for _ in range(100):
                xy = rng.uniform(task.spec.start['xy_min'], task.spec.start['xy_max'])
                if np.linalg.norm(xy) < task.spec.start.get('minimum_base_radius_m', 0.):
                    continue
                if all(np.linalg.norm(xy - np.asarray(p.position[:2])) >=
                       task.spec.start['minimum_separation_m'] for p in positions.values()):
                    positions[name] = Pose.from_lists((*xy, 0.))
                    break
            else:
                break
        if len(positions) == 3:
            return positions
    raise ValueError('Cannot fit three separated bowls in the start region')


def failure_mode(flags, state):
    if flags.get('all_bowls_stacked'):
        return 'stack_not_confirmed'
    if not flags.get('grasped'):
        return 'never_grasped'
    if not flags.get('second_bowl_lifted'):
        return 'one_bowl_only'
    return 'incomplete_stack'
