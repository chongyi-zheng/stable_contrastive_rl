import numpy as np
import os
from scipy import linalg

from rlkit.launchers.sets.example_set_gen import gen_example_sets

def get_mask_params(
        env,
        example_set_variant,
        param_variant,
):
    masks = {}
    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size
    mask_format = param_variant['mask_format']
    if mask_format == 'vector':
        mask_keys = ['mask']
        mask_dims = [(goal_dim,)]
    elif mask_format == 'matrix':
        mask_keys = ['mask']
        mask_dims = [(goal_dim, goal_dim)]
    elif mask_format == 'distribution':
        mask_keys = ['mask_mu', 'mask_sigma_inv']
        mask_dims = [(goal_dim,), (goal_dim, goal_dim)]
    elif mask_format == 'cond_distribution':
        mask_keys = ['mask_mu_w', 'mask_mu_g', 'mask_mu_mat', 'mask_sigma_inv']
        mask_dims = [(goal_dim,), (goal_dim,), (goal_dim, goal_dim), (goal_dim, goal_dim)]
    else:
        raise TypeError

    subtask_codes = example_set_variant['subtask_codes']
    num_masks = len(subtask_codes)
    for mask_key, mask_dim in zip(mask_keys, mask_dims):
        masks[mask_key] = np.zeros([num_masks] + list(mask_dim))

    dataset = gen_example_sets(env, example_set_variant)
    noise = param_variant['noise']
    dataset = get_noisy_dataset(dataset, noise)

    ### set the means, if applicable ###
    if mask_format == 'distribution':
        for mask_id in range(num_masks):
            waypoints = dataset['list_of_waypoints'][mask_id]
            masks['mask_mu'][mask_id] = np.mean(waypoints, axis=0)
    elif mask_format == 'cond_distribution':
        for mask_id in range(num_masks):
            waypoints = dataset['list_of_waypoints'][mask_id]
            goals = dataset['goals']
            mu_w, mu_g, mu_mat, _ = get_cond_distr_params(
                mu=np.mean(np.concatenate((waypoints, goals), axis=1), axis=0),
                sigma=np.cov(np.concatenate((waypoints, goals), axis=1).T),
                x_dim=goals.shape[1],
            )
            masks['mask_mu_w'][mask_id] = mu_w
            masks['mask_mu_g'][mask_id] = mu_g
            masks['mask_mu_mat'][mask_id] = mu_mat

    ### set the variances ###
    if mask_format in ['vector', 'matrix']:
        var_key = 'mask'
    elif mask_format in ['distribution', 'cond_distribution']:
        var_key = 'mask_sigma_inv'
    else:
        raise TypeError

    infer_masks = param_variant['infer_masks']
    if infer_masks:
        for mask_id in range(num_masks):
            waypoints = dataset['list_of_waypoints'][mask_id]
            if mask_format == 'cond_distribution':
                goals = dataset['goals']
                _, _, _, sigma = get_cond_distr_params(
                    mu=np.mean(np.concatenate((waypoints, goals), axis=1), axis=0),
                    sigma=np.cov(np.concatenate((waypoints, goals), axis=1).T),
                    x_dim=goals.shape[1],
                )
            elif mask_format == 'distribution':
                sigma = np.cov(waypoints.T)
            else:
                raise TypeError
            masks[var_key][mask_id] = invert_matrix(sigma, param_variant['max_cond_num'])
    else:
        for (mask_id, idx_dict) in enumerate(subtask_codes):
            for (k, v) in idx_dict.items():
                if mask_format == 'vector':
                    assert k == v
                    masks['mask'][mask_id][k] = 1
                elif mask_format in ['matrix', 'distribution', 'cond_distribution']:
                    if v >= 0:
                        assert k == v
                        masks[var_key][mask_id][k, k] = 1
                    else:
                        src_idx = k
                        targ_idx = -(v + 10)
                        masks[var_key][mask_id][src_idx, src_idx] = 1
                        masks[var_key][mask_id][targ_idx, targ_idx] = 1
                        masks[var_key][mask_id][src_idx, targ_idx] = -1
                        masks[var_key][mask_id][targ_idx, src_idx] = -1

    normalize_mask = param_variant['normalize_mask']
    mask_threshold = param_variant['mask_threshold']
    for mask_id in range(num_masks):
        mask = masks[var_key][mask_id]
        if normalize_mask:
            mask /= np.max(np.abs(mask))

        if mask_threshold is not None:
            mask[np.abs(mask) <= mask_threshold * np.max(np.abs(mask))] = 0.0

    from rlkit.core import logger
    logdir = logger.get_snapshot_dir()
    np.save(
        os.path.join(logdir, 'masks.npy'),
        masks
    )

    for mask_id in range(num_masks):
        print('mask_sigma_inv for mask_id={}'.format(mask_id))
        print_matrix(masks[var_key][mask_id], precision=5) #precision=5
        # print(masks[var_key][mask_id].diagonal())

    return masks

def get_noisy_dataset(dataset, noise):
    return {
        k: dataset[k] + np.random.normal(0, noise, dataset[k].shape)
        for k in dataset.keys()
    }

def invert_matrix(matrix, max_cond_num=None):
    if max_cond_num is not None:
        w, v = np.linalg.eig(matrix)
        l, h = np.min(w), np.max(w)
        target = 1 / max_cond_num
        if (l / h) < target:
            eps = (h * target - l) / (1 - target)
        else:
            eps = 0
        matrix_inv = linalg.inv(matrix + eps * np.identity(matrix.shape[0]))
    else:
        matrix_inv = linalg.inv(matrix)
    return matrix_inv

def get_cond_distr_params(mu, sigma, x_dim):
    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_mat = sigma_xy @ sigma_yy_inv
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx

    return mu_x, mu_y, mu_mat, sigma_xgy

def print_matrix(matrix, format="raw", threshold=0.1, normalize=True, precision=5):
    if normalize:
        matrix = matrix.copy() / np.max(np.abs(matrix))

    matrix = matrix.reshape((matrix.shape[0], -1))

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
                if precision == 2:
                    print("{:.2f}".format(value), end=" ")
                elif precision == 5:
                    print("{:.5f}".format(value), end=" ")
        print()
    print()

def plot_Gaussian(
        mu,
        sigma=None,
        sigma_inv=None,
        bounds=None,
        list_of_dims=[[0, 1], [2, 3], [0, 2], [1, 3]],
        pt1=None,
        pt2=None,
        add_title=True,
):
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    num_subplots = len(list_of_dims)
    if num_subplots == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig, axs = plt.subplots(2, num_subplots // 2, figsize=(10, 10))
    lb, ub = bounds
    gran = (ub - lb) / 50
    x, y = np.mgrid[lb:ub:gran, lb:ub:gran]
    pos = np.dstack((x, y))

    assert (sigma is not None) ^ (sigma_inv is not None)
    if sigma is None:
        sigma = linalg.inv(sigma_inv + np.eye(len(mu)) * 1e-6)

    for i in range(len(list_of_dims)):
        dims = list_of_dims[i]
        rv = multivariate_normal(mu[dims], sigma[dims][:,dims], allow_singular=True)

        if num_subplots == 1:
            axs_obj = axs
        else:
            plt_idx1 = i // 2
            plt_idx2 = i % 2
            axs_obj = axs[plt_idx1, plt_idx2]

        axs_obj.contourf(x, y, rv.pdf(pos), cmap="Blues")
        if add_title:
            axs_obj.set_title(str(dims))

        if pt1 is not None:
            axs_obj.scatter([pt1[dims][0]], [pt1[dims][1]])

        if pt2 is not None:
            axs_obj.scatter([pt2[dims][0]], [pt2[dims][1]])

    return fig, axs
    # plt.show()
    # plt.axis('off')
    # axs.get_xaxis().set_visible(False)
    # axs.get_yaxis().set_visible(False)
    # plt.savefig("/tmp/test.png", bbox_inches='tight', pad_inches=0)