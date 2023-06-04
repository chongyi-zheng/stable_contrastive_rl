import numpy as np
from gym.spaces import Box

from rlkit.envs.proxy_env import ProxyEnv


class TimeLimitedEnv(ProxyEnv):
    def __init__(self, wrapped_env, horizon):
        self._wrapped_env = wrapped_env
        self._horizon = horizon

        self._observation_space = Box(
            np.hstack((self._wrapped_env.observation_space.low, [0])),
            np.hstack((self._wrapped_env.observation_space.high, [1])),
        )
        self._t = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def horizon(self):
        return self._horizon

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        self._t += 1
        done = done or self._t == self.horizon
        new_obs = np.hstack((obs, float(self._t) / self.horizon))
        return new_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        self._t = 0
        new_obs = np.hstack((obs, self._t))
        return new_obs
