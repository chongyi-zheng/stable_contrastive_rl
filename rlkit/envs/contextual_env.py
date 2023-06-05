import abc
from collections import OrderedDict

import gym
import gym.spaces
import numpy as np
from typing import Union, Callable, Any, Dict, List

from rlkit.core.distribution import DictDistribution
from rlkit import pythonplusplus as ppp


Path = Dict
Diagnostics = Dict
Context = Any
ContextualDiagnosticsFn = Callable[
    [List[Path], List[Context]],
    Diagnostics,
]


def batchify(x):
    return ppp.treemap(lambda x: x[None], x, atomic_type=np.ndarray)


def insert_reward(contexutal_env, info, obs, reward, done):
    info['ContextualEnv/old_reward'] = reward
    return info


def delete_info(contexutal_env, info, obs, reward, done):
    return {}


def maybe_flatten_obs(self, obs):
    if len(obs.shape) == 1:
        return obs.reshape(1, -1)
    return obs


class ContextualRewardFn(object, metaclass=abc.ABCMeta):
    """You can also just pass in a function."""

    @abc.abstractmethod
    def __call__(
            self,
            states: dict,
            actions,
            next_states: dict,
            contexts: dict
    ):
        pass


class UnbatchRewardFn(object):
    def __init__(self, reward_fn: ContextualRewardFn):
        self._reward_fn = reward_fn

    def __call__(
            self,
            state: dict,
            action,
            next_state: dict,
            context: dict
    ):
        states = batchify(state)
        actions = batchify(action)
        next_states = batchify(next_state)
        reward, terminal = self._reward_fn(
            states,
            actions,
            next_states,
            context,
            # debug=True,
        )
        return reward[0]


class ContextualEnv(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            context_distribution: DictDistribution,
            reward_fn: ContextualRewardFn,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            update_env_info_fn=None,
            contextual_diagnostics_fns: Union[
                None, List[ContextualDiagnosticsFn]] = None,
            unbatched_reward_fn=None,
    ):
        super().__init__(env)

        if contextual_diagnostics_fns is None:
            contextual_diagnostics_fns = []

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("ContextualEnvs require wrapping Dict spaces.")

        spaces = env.observation_space.spaces
        for key, space in context_distribution.spaces.items():
            spaces[key] = space

        self.observation_space = gym.spaces.Dict(spaces)
        self.reward_fn = reward_fn
        self._last_obs = None
        self._update_env_info = update_env_info_fn or insert_reward

        self._curr_context = None

        self._observation_key = observation_key
        del observation_keys

        self._context_distribution = context_distribution
        self._context_keys = list(context_distribution.spaces.keys())

        self._contextual_diagnostics_fns = contextual_diagnostics_fns

        if unbatched_reward_fn is None:
            unbatched_reward_fn = UnbatchRewardFn(reward_fn)

        self.unbatched_reward_fn = unbatched_reward_fn

    def reset(self):
        obs = self.env.reset()
        self._curr_context = self._context_distribution(
            context=obs).sample(1)
        self._add_context_to_obs(obs)
        self._last_obs = obs
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._add_context_to_obs(obs)
        new_reward = self._compute_reward(self._last_obs, action, obs, reward)
        self._last_obs = obs
        info = self._update_env_info(self, info, obs, reward, done)
        return obs, new_reward, done, info

    def _compute_reward(self, state, action, next_state, env_reward=None):
        """Do reshaping for reward_fn, which is implemented for batches."""
        if not self.reward_fn:
            return env_reward
        else:
            return self.unbatched_reward_fn(
                state, action, next_state, self._curr_context)

    def _add_context_to_obs(self, obs):
        for key in self._context_keys:
            obs[key] = self._curr_context[key][0]

    def get_diagnostics(self, paths):
        stats = OrderedDict()
        contexts = [self._get_context(p) for p in paths]
        for fn in self._contextual_diagnostics_fns:
            stats.update(fn(paths, contexts))
        return stats

    def _get_context(self, path):
        first_observation = path['observations'][0]
        return {
            key: first_observation[key] for key in self._context_keys
        }
