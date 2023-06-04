from collections import defaultdict
import numpy as np

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.core.eval_util import diagnostics_from_paths_statistics


def get_env_diagnostics(env):
    if isinstance(env, PickAndPlaceEnv):
        # return env.goal_conditioned_diagnostics
        return PnpDiagnostics(env)
    else:
        raise NotImplementedError
        # return env.goal_conditioned_diagnostics


def distance_between(path, obj1_idx, obj2_idx):
    states = path['observations']
    obj1_pos = states[2*obj1_idx:2*obj1_idx+2]
    obj2_pos = states[2*obj2_idx:2*obj2_idx+2]
    distance = np.linalg.norm(obj1_pos - obj2_pos)
    return distance


class PnpDiagnostics(object):
    def __init__(self, env):
        self._all_objects = env._all_objects
        self.success_threshold = env.success_threshold

    def __call__(self, paths, goals):
        stat_to_lists = defaultdict(list)
        for path, goal in zip(paths, goals):
            difference = path['observations'] - goal
            for i in range(len(self._all_objects)):
                distance = np.linalg.norm(
                    difference[:, 2 * i:2 * i + 2], axis=-1
                )
                distance_key = 'distance_to_target_obj_{}'.format(i)
                stat_to_lists[distance_key].append(distance)
                success_key = 'success_obj_{}'.format(i)
                stat_to_lists[success_key].append(
                    distance < self.success_threshold
                )
        return diagnostics_from_paths_statistics(stat_to_lists)
