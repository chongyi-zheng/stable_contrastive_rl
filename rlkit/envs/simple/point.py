import os
from gym import spaces
import numpy as np
import gym


class Point(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, n=2, action_scale=0.2, fixed_goal=None):
        self.fixed_goal = fixed_goal
        self.n = n
        self.action_scale = action_scale
        self.goal = np.zeros((n,))
        self.state = np.zeros((n,))

    @property
    def action_space(self):
        return spaces.Box(
            low=-1*np.ones((self.n,)),
            high=1*np.ones((self.n,))
        )

    @property
    def observation_space(self):
        return spaces.Box(
            low=-5*np.ones((2 * self.n,)),
            high=5*np.ones((2 * self.n,))
        )

    def reset(self):
        self.state = np.zeros((self.n,))
        if self.fixed_goal is None:
            self.goal = np.random.uniform(-5, 5, size=(self.n,))
        else:
            self.goal = np.array(self.fixed_goal)
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1, 1) * self.action_scale
        new_state = self.state + action
        new_state = np.clip(new_state, -5, 5)
        self.state = new_state
        reward = -np.linalg.norm(new_state - self.goal)

        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        return np.concatenate([self.state, self.goal])
