import numpy as np
from collections import OrderedDict
import gym
from gym.spaces import Dict

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable

import envs
from lexa import wrappers


class DmcEnv(gym.Env, Serializable):
    def __init__(self,
                 task,
                 size=(48, 48),
                 action_repeat=2,
                 use_goal_idx=False,
                 log_per_goal=False,
                 # time_limit=75,
                 **kwargs,
                 ):
        del kwargs

        env = envs.DmcEnv(task, size, action_repeat, use_goal_idx, log_per_goal)
        # env = wrappers.NormalizeActions(env)

        # env = wrappers.TimeLimit(env, time_limit)
        # env = wrappers.CollectDataset(env)
        # env = wrappers.RewardObs(env)

        self._env = env
        self._physics = self._env._env.physics
        self._set_spaces()

    def _set_spaces(self):
        self.action_space = self._env.action_space

        observation_dim = self._physics.data.qpos.shape[0]
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        self.observation_space = Dict([
            ('observation', state_space),
            ('state_observation', state_space),
            ('desired_goal', state_space),
            ('state_desired_goal', state_space),
            ('achieved_goal', state_space),
            ('state_achieved_goal', state_space),
        ])

    def get_contextual_diagnostics(self, paths, contexts):
        diagnostics = OrderedDict()

    def reset(self):
        # raise NotImplementedError
        obs = self._env.reset()

        return obs

    def step(self, *action):
        # raise NotImplementedError
        obs, reward, done, info = self._env.step(action)

        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        del mode

        self._env.render()

    def save_state(self, *save_path):
        raise NotImplementedError

    def load_state(self, load_path):
        raise NotImplementedError

