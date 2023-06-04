import abc
from typing import Callable, Any, Dict, List

import gym.spaces

Path = Dict
Diagnostics = Dict
Context = Any
ContextualDiagnosticsFn = Callable[
    [List[Path], List[Context]],
    Diagnostics,
]


class RewardFn(object, metaclass=abc.ABCMeta):
    """Some reward function"""
    @abc.abstractmethod
    def __call__(
            self,
            action,
            next_state: dict,
    ):
        pass


class ModularEnv(gym.Wrapper):
    """An env where you can separately specify the reward and transition."""
    def __init__(
            self,
            env: gym.Env,
            reward_fn: RewardFn,
    ):
        super().__init__(env)
        self.reward_fn = reward_fn

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.reward_fn:
            reward = self.reward_fn(action, obs)
        return obs, reward, done, info
