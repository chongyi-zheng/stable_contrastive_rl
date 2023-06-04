import typing

import gym
import numpy as np
from gym.spaces import Box, Dict

from rlkit.envs.images.env_renderer import EnvRenderer

# getting Product


def prod(val):
    res = 1
    for ele in val:
        res *= ele
    return res


class InsertImagesEnv(gym.Wrapper):
    """
    Add an image to the observation. Usage:

    ```
    obs = env.reset()
    print(obs.keys())  # ['observations']

    new_env = InsertImageEnv(
        env,
        {
            'image_observation': renderer_one,
            'debugging_img': renderer_two,
        },
    )
    obs = new_env.reset()
    print(obs.keys())  # ['observations', 'image_observation', 'debugging_img']
    ```
    """

    def __init__(
            self,
            wrapped_env: gym.Env,
            renderers: typing.Dict[str, EnvRenderer],
    ):
        super().__init__(wrapped_env)
        spaces = self.env.observation_space.spaces.copy()
        for image_key, renderer in renderers.items():
            if renderer.image_is_normalized:
                img_space = Box(
                    0, 1, (prod(renderer.image_shape), ), dtype=np.float32)
            else:
                img_space = Box(
                    0, 255, (prod(renderer.image_shape), ), dtype=np.uint8)
            spaces[image_key] = img_space
        self.renderers = renderers
        self.observation_space = Dict(spaces)
        self.action_space = self.env.action_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._update_obs(obs)
        return obs

    def get_observation(self):
        obs = self.env.get_observation()
        self._update_obs(obs)
        return obs

    def _update_obs(self, obs):
        for image_key, renderer in self.renderers.items():
            obs[image_key] = renderer(self.env)


class InsertImageEnv(InsertImagesEnv):
    """
    Add an image to the observation. Usage:
    ```
    obs = env.reset()
    print(obs.keys())  # ['observations']

    new_env = InsertImageEnv(env, renderer, image_key='pretty_picture')
    obs = new_env.reset()
    print(obs.keys())  # ['observations', 'pretty_picture']
    ```
    """

    def __init__(
            self,
            wrapped_env: gym.Env,
            renderer: EnvRenderer,
            image_key='image_observation',
    ):
        super().__init__(wrapped_env, {image_key: renderer})
