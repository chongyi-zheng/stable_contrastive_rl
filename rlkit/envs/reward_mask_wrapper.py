import copy
import random
import warnings

import torch

# import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import time

class DiscreteDistribution:
    def __init__(self, items, weights=None):
        self.items = np.array(items)
        self.num_items = len(items)
        if weights is None: # assume uniform
            weights = np.ones((self.num_items, ))
        else:
            weights = np.array(weights)
        self.weights = weights / np.sum(weights)

    def sample(self, n=1):
        ind = np.random.choice(self.num_items, size=n, p=self.weights)
        return self.items[ind]

    @property
    def shape(self):
        return self.items[0].shape

    @property
    def dim(self):
        return len(self.items[0])

class RewardMaskWrapper(ProxyEnv, MultitaskEnv):
    """Base class for adding a reward mask"""
    def __init__(
        self,
        wrapped_env,
        mask_distribution,
        norm_ord=2,
    ):
        super().__init__(wrapped_env)
        self.mask_distribution = mask_distribution
        self.norm_ord = norm_ord

        mask_dim = mask_distribution.dim
        mask_low = -np.ones((mask_dim, ))
        mask_high = np.ones((mask_dim, ))

        spaces = self.wrapped_env.observation_space.spaces
        goal_space = spaces['desired_goal']
        new_goal_space = Box(
            np.hstack((goal_space.low, mask_low)),
            np.hstack((goal_space.high, mask_high)),
            dtype=np.float32
        )
        spaces['mask_desired_goal'] = new_goal_space
        spaces['mask_achieved_goal'] = new_goal_space
        spaces['mask'] = Box(mask_low, mask_high, dtype=np.float32)
        self.observation_space = Dict(spaces)

    def reset(self):
        obs = self._wrapped_env.reset()
        self.current_mask = self.sample_masks(batch_size=1)[0]
        self._update_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        self._update_obs(obs)
        self._update_info(info, obs)
        # reward = self.compute_reward(action, obs)
        return obs, reward, done, info

    def sample_masks(self, batch_size):
        return self.mask_distribution.sample(batch_size)

    def _update_obs(self, obs):
        obs["mask"] = self.current_mask.copy()
        obs["mask_desired_goal"] = np.hstack((
            obs["desired_goal"],
            self.current_mask
        )).copy()
        obs["mask_achieved_goal"] = np.hstack((
            obs["achieved_goal"],
            self.current_mask
        )).copy()

    def _update_info(self, info, obs):
        pass

    """
    Multitask functions
    """
    def sample_goals(self, batch_size):
        goals = self.wrapped_env.sample_goals(batch_size)
        masks = self.sample_masks(batch_size)
        goals["mask_desired_goal"] = np.hstack((
            goals["desired_goal"],
            masks
        )).copy()
        return goals

    def get_goal(self):
        return self.wrapped_env.get_goal()

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        reward = self.compute_rewards(actions, next_obs)
        return reward[0]


    def compute_rewards(self, actions, obs):
        achieved_goals = obs['mask_achieved_goal'][:, :4]
        desired_goals = obs['mask_desired_goal'][:, :4]
        mask = obs['mask_desired_goal'][:, 4:]

        return -np.linalg.norm(
            (achieved_goals - desired_goals) * mask,
            ord=self.norm_ord,
            axis=1
        )

    @property
    def goal_dim(self):
        return self.representation_size

    # def set_goal(self, goal):
    #     """
    #     Assume goal contains both image_desired_goal and any goals required for wrapped envs

    #     :param goal:
    #     :return:
    #     """
    #     self.desired_goal = goal
    #     # TODO: fix this hack / document this
    #     if self._goal_sampling_mode in {'presampled', 'env'}:
    #         self.wrapped_env.set_goal(goal)

    # def get_diagnostics(self, paths, **kwargs):
        # statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        # return statistics

    # def __getstate__(self):
    #     state = super().__getstate__()
    #     return state

    # def __setstate__(self, state):
    #     super().__setstate__(state)
