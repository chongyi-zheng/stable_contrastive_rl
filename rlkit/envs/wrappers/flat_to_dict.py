import gym
import numpy as np
from gym.spaces import Dict

from rlkit.core.util import SimpleWrapper
from rlkit.policies.base import Policy


class FlatToDictEnv(gym.Wrapper):
    """Wrap an environment that returns a flat obs to return a dict."""
    def __init__(self, env, observation_key):
        super().__init__(env)
        self.observation_key = observation_key
        new_ob_space = {
            self.observation_key: self.observation_space
        }
        self.observation_space = Dict(new_ob_space)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return {self.observation_key: obs}, reward, done, info

    def reset(self):
        obs = super().reset()
        return {self.observation_key: obs}


class FlatToDictPolicy(SimpleWrapper, Policy):
    """Wrap a policy that expect a flat obs so expects a dict obs."""

    def __init__(self, policy, observation_key):
        super().__init__(policy)
        self.policy = policy
        self.observation_key = observation_key

    def get_action(self, observation, *args, **kwargs):
        flat_ob = observation[self.observation_key]
        return self.policy.get_action(flat_ob, *args, **kwargs)
