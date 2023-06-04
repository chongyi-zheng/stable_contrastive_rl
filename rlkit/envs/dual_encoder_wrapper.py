import abc
import numpy as np
import rlkit.torch.pytorch_util as ptu
from gym.spaces import Box, Dict
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.envs.wrappers import ProxyEnv

from rlkit.envs.encoder_wrappers import Encoder

class DualEncoderWrappedEnv(ProxyEnv):
    def __init__(self,
        wrapped_env,
        model: Encoder,
        input_model,
        step_keys_map=None,
        reset_keys_map=None,
        conditional_input_model=False,
    ):
        super().__init__(wrapped_env)
        self.model = model
        self.input_model = input_model
        self.conditional_input_model = conditional_input_model
        self.representation_size = self.model.representation_size
        self.input_representation_size = self.input_model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )
        input_latent_space = Box(
            -10 * np.ones(self.input_representation_size),
            10 * np.ones(self.input_representation_size),
            dtype=np.float32,
        )

        if step_keys_map is None:
            step_keys_map = {}
        if reset_keys_map is None:
            reset_keys_map = {}
        self.step_keys_map = step_keys_map
        self.reset_keys_map = reset_keys_map
        spaces = self.wrapped_env.observation_space.spaces
        for value in self.step_keys_map.values():
            spaces[value] = latent_space
        for value in self.reset_keys_map.values():
            spaces[value] = latent_space

        spaces['input_latent'] = input_latent_space
        self.observation_space = Dict(spaces)
        self.reset_obs = {}

    def step(self, action):
        self.model.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.model.eval()
        for key in self.step_keys_map:
            value = self.step_keys_map[key]
            obs[value] = self.model.encode_one_np(obs[key])
        obs = {**obs, **self.reset_obs}

        if self.conditional_input_model:
            obs['input_latent'] = self.input_model.encode_one_np(obs['image_observation'], self._initial_img)
        else:
            obs['input_latent'] = self.input_model.encode_one_np(obs['image_observation'])

        return obs

    def reset(self):
        self.model.eval()
        obs = self.wrapped_env.reset()
        self._initial_img = obs["image_observation"]
        for key in self.reset_keys_map:
            value = self.reset_keys_map[key]
            self.reset_obs[value] = self.model.encode_one_np(obs[key])
        obs = self._update_obs(obs)
        return obs
