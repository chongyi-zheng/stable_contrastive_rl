"""
Have a separate function so that if other code needs to merge/unmerge obs,
goals, and whatnot, they do it in the same way.
"""
import torch
import numpy as np


def merge_into_flat_obs(obs, goals, num_steps_left):
    # Have a separate function so that if other code needs to merge obs,
    # goals, and whatnot, it does it in the same way.
    if isinstance(obs, np.ndarray):
        return np.hstack((obs, goals, num_steps_left))
    else:
        return torch.cat((obs, goals, num_steps_left), dim=1)

def extract_goals(flat_obs, ob_dim, goal_dim):
    return flat_obs[:, ob_dim:ob_dim+goal_dim]


def split_tau(flat_obs):
    return flat_obs[:, :-1], flat_obs[:, -1:]


def split_flat_obs(flat_obs, ob_dim, goal_dim):
    return (
        flat_obs[:, :ob_dim],
        flat_obs[:, ob_dim:ob_dim+goal_dim],
        flat_obs[:, ob_dim+goal_dim:],
    )
