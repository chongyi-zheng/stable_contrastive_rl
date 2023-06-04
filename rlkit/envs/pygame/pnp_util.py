import time

import numpy as np
from rlkit.envs.images import EnvRenderer
import rlkit.torch.sets.set_projection as sp
from rlkit.torch.sets.vae_launcher import (
    sample_point_set_projector,
    sample_axis_set_projector,
)

from multiworld.envs.pygame import PickAndPlaceEnv

from scipy import linalg


def get_cond_distr_params(mu, sigma, x_dim, max_cond_num):
    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_mat = sigma_xy @ sigma_yy_inv
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx

    w, v = np.linalg.eig(sigma_xgy)
    l, h = np.min(w), np.max(w)
    target = 1 / max_cond_num
    if (l / h) < target:
        eps = (h * target - l) / (1 - target)
    else:
        eps = 0
    sigma_xgy_inv = linalg.inv(
        sigma_xgy + eps * np.identity(sigma_xgy.shape[0]))

    return mu_x, mu_y, mu_mat, sigma_xgy_inv


def print_matrix(matrix, format="raw", threshold=0.1, normalize=True):
    if normalize:
        matrix = matrix.copy() / np.max(np.abs(matrix))

    assert format in ["signed", "raw"]

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if format == "raw":
                value = matrix[i][j]
            elif format == "signed":
                if np.abs(matrix[i][j]) > threshold:
                    value = 1 * np.sign(matrix[i][j])
                else:
                    value = 0
            if format == "signed":
                print(int(value), end=", ")
            else:
                if value > 0:
                    print("", end=" ")
                print("{:.2f}".format(value), end=" ")
        print()
    print()


def gen_data(env, idx_masks, n, other_dims_random=True):
    from tqdm import tqdm
    num_masks = len(idx_masks)

    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size

    goals = np.zeros((n, goal_dim))
    list_of_waypoints = np.zeros((num_masks, n, goal_dim))

    t1 = time.time()
    print("Generating dataset...")

    # data collection
    for i in tqdm(range(n)):
        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal']
        goal = obs_dict['state_desired_goal']

        for (mask_id, idx_dict) in enumerate(idx_masks):
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
                wp[other_dims] = \
                    env.observation_space.spaces[
                        'state_achieved_goal'].sample()[
                        other_dims]
            list_of_waypoints[mask_id][i] = wp

        goals[i] = goal

    print("Done. Time:", time.time() - t1)

    return list_of_waypoints, goals


def sample_sets(env, set_descriptions, n):
    """
    :param env:  PickAndPlace env
    :param set_descriptions: list of dicts. dicts map from index to value.
    :param n:
    :return:
    """

    num_sets = len(set_descriptions)

    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size

    sets = np.zeros((num_sets, n, goal_dim))

    for set_i, set_projection in enumerate(set_descriptions):
        for sample_i in range(n):
            obs_dict = env.reset()
            obs = obs_dict['state_achieved_goal']
            sets[set_i, sample_i, :] = set_projection(obs)
    return sets


def sample_examples_with_images(
        env: PickAndPlaceEnv,
        renderer: EnvRenderer,
        set_projection: sp.SetProjection,
        num_samples: int,
        state_key,
        image_key,
):
    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size

    states = np.zeros((num_samples, goal_dim))
    imgs = np.zeros((num_samples, *renderer.image_shape))
    for i in range(num_samples):
        obs_dict = env.reset()
        orig_state = obs_dict['state_achieved_goal']
        new_state = set_projection(orig_state)
        env._set_positions(new_state)
        img = renderer(env)
        states[i, :] = new_state
        imgs[i, ...] = img
    data_dict = {
        state_key: states,
        image_key: imgs,
    }
    return data_dict


def generate_goals(env, n):
    """
    :param env:  PickAndPlace env
    :param set_descriptions: list of dicts. dicts map from index to value.
    :param n:
    :return:
    """

    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size
    goals = np.zeros((n, goal_dim))

    for sample_i in range(n):
        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal']
        goals[sample_i, :] = obs
    return goals


def infer_masks(env, idx_masks, mask_inference_variant):
    n = int(mask_inference_variant.get('n', 50))
    obs_noise = mask_inference_variant['noise']
    max_cond_num = mask_inference_variant['max_cond_num']
    normalize_sigma_inv = mask_inference_variant.get('normalize_sigma_inv',
                                                     True)
    other_dims_random = mask_inference_variant.get('other_dims_random', True)

    list_of_waypoints, goals = gen_data(env, idx_masks, n, other_dims_random)

    # add noise to all of the data
    list_of_waypoints += np.random.normal(0, obs_noise, list_of_waypoints.shape)
    goals += np.random.normal(0, obs_noise, goals.shape)

    masks = {
        'mask_mu_w': [],
        'mask_mu_g': [],
        'mask_mu_mat': [],
        'mask_sigma_inv': [],
    }
    for (mask_id, waypoints) in enumerate(list_of_waypoints):
        mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
        sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)
        mu_w, mu_g, mu_mat, sigma_inv = get_cond_distr_params(
            mu, sigma,
            x_dim=goals.shape[1],
            max_cond_num=max_cond_num
        )
        if normalize_sigma_inv:
            sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
        masks['mask_mu_w'].append(mu_w)
        masks['mask_mu_g'].append(mu_g)
        masks['mask_mu_mat'].append(mu_mat)
        masks['mask_sigma_inv'].append(sigma_inv)

    for k in masks.keys():
        masks[k] = np.array(masks[k])

    for mask_id in range(len(idx_masks)):
        # print('mask_mu_mat')
        # print_matrix(masks['mask_mu_mat'][mask_id])
        print('mask_sigma_inv for mask_id={}'.format(mask_id))
        print_matrix(masks['mask_sigma_inv'][mask_id])
    # exit()

    return masks


def generate_set_images(
        env,
        env_renderer,
        num_sets=1,
        num_samples_per_set=32,
):

    set_projections = sample_set_projections(env, num_sets)
    sets = sample_sets(env, set_projections, n=num_samples_per_set)

    def create_images(states):
        for state in states:
            env._set_positions(state)
            img = env_renderer(env)
            yield img

    for states in sets:
        yield list(create_images(states))


def sample_set_projections(env, num_sets):
    set_descriptions = []
    for i in range(num_sets // 3):
        set_descriptions.append(sample_point_set_projector(
            (env.num_objects + 1) * 2,
            index=(i % (env.num_objects + 1))
        ))
        set_descriptions.append(sample_axis_set_projector(
            (env.num_objects + 1) * 2,
            index=(i % (env.num_objects + 1)) * 2
        ))
        set_descriptions.append(sample_axis_set_projector(
            (env.num_objects + 1) * 2,
            index=(i % (env.num_objects + 1)) * 2 + 1
        ))
    # if num_sets % 2 == 1:
    #     set_descriptions.append(
    #         sample_point_set_projector((env.num_objects + 1) * 2)
    #     )
    return set_descriptions


def create_set_projection(
        version='point',
        axis_idx_to_value=None,
        a_axis_to_b_axis=None,
):
    if version == 'project_onto_axis':
        for k in axis_idx_to_value:
            if axis_idx_to_value[k] is None:
                axis_idx_to_value[k] = np.random.uniform(-4, 4, 1)
        return sp.ProjectOntoAxis(axis_idx_to_value)
    elif version == 'move_a_to_b':
        return sp.MoveAtoB(a_axis_to_b_axis)
    else:
        raise ValueError(version)
