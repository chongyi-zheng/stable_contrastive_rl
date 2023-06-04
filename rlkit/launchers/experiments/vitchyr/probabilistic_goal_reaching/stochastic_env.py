import numpy as np
import gym


class NoisyAction(gym.Wrapper):
    def __init__(self, env, noise_fraction):
        super().__init__(env)
        action_space = env.action_space
        self._action_noise_scale = noise_fraction * (
            action_space.high - action_space.low)
        self._action_shape = action_space.high.shape

    def step(self, action):
        noise = self._action_noise_scale * np.random.randn(*self._action_shape)
        noisy_action = action + noise
        return self.env.step(noisy_action)
