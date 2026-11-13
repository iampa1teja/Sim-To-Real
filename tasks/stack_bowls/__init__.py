"""Three original hollow bowls: any bottom-to-top color permutation is valid."""
import numpy as np
from core.tasks.base import Pose, Task, load_task_spec


class StackBowlsTask(Task):
    failure_modes = ('never_grasped', 'one_bowl_only', 'incomplete_stack', 'stack_not_confirmed')

    def sample_start(self, rng):
        for _ in range(1000):
            positions = {}
            for name in self.spec.tracked_objects:
                for _ in range(100):
                    xy = rng.uniform(self.spec.start['xy_min'], self.spec.start['xy_max'])
                    if np.linalg.norm(xy) < self.spec.start.get('minimum_base_radius_m', 0.):
                        continue
                    if all(np.linalg.norm(xy-np.asarray(p.position[:2])) >=
                           self.spec.start['minimum_separation_m'] for p in positions.values()):
                        positions[name] = Pose.from_lists((*xy, 0.))
                        break
                else:
                    break
            if len(positions) == 3:
                return positions
        raise ValueError('Cannot fit three separated bowls in the start region')

    def signals(self, env):
        from .mdp import stage_signals
        return stage_signals(env)

    def observe(self, state):
        return {name: bool(state.signals.get(name, False)) for name in self.stages}

    def failure_mode(self, flags, state):
        if self.is_success(flags):
            return None
        if flags.get('all_bowls_stacked'):
            return 'stack_not_confirmed'
        if not flags.get('grasped'):
            return 'never_grasped'
        if not flags.get('second_bowl_lifted'):
            return 'one_bowl_only'
        return 'incomplete_stack'


TASK = StackBowlsTask(load_task_spec('stack_bowls'))
