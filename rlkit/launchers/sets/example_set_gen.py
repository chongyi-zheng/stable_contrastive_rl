from rlkit.launchers.rl_exp_launcher_util import get_envs

import numpy as np
import os
import time
from tqdm import tqdm

def gen_example_sets_full_experiment(variant):
    env = get_envs(variant)
    example_set_variant = variant['example_set_variant']
    gen_example_sets(env, example_set_variant)

def gen_example_sets(env, example_set_variant):
    subtask_codes = example_set_variant['subtask_codes']
    n = example_set_variant['n']
    other_dims_random = example_set_variant['other_dims_random']
    use_cache = example_set_variant.get('use_cache', False)

    if use_cache:
        cache_path = example_set_variant['cache_path']
        dataset = np.load(cache_path)[()]
        data_idxs = np.arange(dataset['list_of_waypoints'].shape[1])
        np.random.shuffle(data_idxs)
        data_idxs = data_idxs[:n]
        list_of_waypoints = dataset['list_of_waypoints'][:, data_idxs]
        goals = dataset['goals'][data_idxs]
        return create_and_save_dict(list_of_waypoints, goals)

    num_subtasks = len(subtask_codes)
    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size
    goals = np.zeros((n, goal_dim))
    list_of_waypoints = np.zeros((num_subtasks, n, goal_dim))

    t1 = time.time()
    print("Generating dataset...")

    # data collection
    for i in tqdm(range(n)):
        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal']
        goal = obs_dict['state_desired_goal']
        for (subtask_id, idx_dict) in enumerate(subtask_codes):
            wp = obs.copy()
            dims = []
            for (k, v) in idx_dict.items():
                if v >= 0:
                    wp[k] = goal[v]
                    dims.append(k)
            for (k, v) in idx_dict.items():
                if v < 0:
                    wp[k] = wp[-v - 10]
                    dims.append(k)
                    dims.append(-v - 10)
            other_dims = [d for d in np.arange(len(wp)) if d not in dims]
            if other_dims_random:
                wp[other_dims] = env.observation_space.spaces['state_achieved_goal'].sample()[other_dims]
            list_of_waypoints[subtask_id][i] = wp

        goals[i] = goal

    print("Done. Time:", time.time() - t1)

    return create_and_save_dict(list_of_waypoints, goals)

def create_and_save_dict(list_of_waypoints, goals):
    dataset = {
        'list_of_waypoints': list_of_waypoints,
        'goals': goals,
    }
    from rlkit.core import logger
    logdir = logger.get_snapshot_dir()
    np.save(
        os.path.join(logdir, 'example_dataset.npy'),
        dataset
    )
    return dataset