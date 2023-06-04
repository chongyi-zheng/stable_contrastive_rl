from gym import Env
from gym.spaces import Box
import numpy as np
import robosuite as suite
from rlkit.core.serializeable import Serializable


class RobosuiteStateWrapperEnv(Serializable, Env):
    def __init__(self, wrapped_env_id, observation_keys=('robot-state', 'object-state'), **wrapped_env_kwargs):
        Serializable.quick_init(self, locals())
        self._wrapped_env = suite.make(
            wrapped_env_id,
            **wrapped_env_kwargs
        )
        self.action_space = Box(self._wrapped_env.action_spec[0], self._wrapped_env.action_spec[1], dtype=np.float32)
        observation_dim = self._wrapped_env.observation_spec()['robot-state'].shape[0] \
                          + self._wrapped_env.observation_spec()['object-state'].shape[0]
        self.observation_space = Box(
            -np.inf * np.ones(observation_dim),
            np.inf * np.ones(observation_dim),
            dtype=np.float32,
        )
        self._observation_keys = observation_keys

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)
        obs = self.flatten_dict_obs(obs)
        return obs, reward, done, info

    def flatten_dict_obs(self, obs):
        obs = np.concatenate(tuple(obs[k] for k in self._observation_keys))
        return obs

    def reset(self):
        obs = self._wrapped_env.reset()
        obs = self.flatten_dict_obs(obs)
        return obs

    def render(self):
        self._wrapped_env.render()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

