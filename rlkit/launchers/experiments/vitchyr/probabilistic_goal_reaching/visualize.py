import numpy as np
import torch
from torch.distributions.kl import kl_divergence
from torch.distributions import Bernoulli

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.images import InsertImagesEnv, Renderer
from rlkit.envs.images.plot_renderer import (
    ScrollingPlotRenderer,
)
from rlkit.torch.disentanglement.networks import DisentangledMlpQf


class DynamicsModelEnvRenderer(Renderer):
    def __init__(
            self,
            model: DisentangledMlpQf,
            states_to_eval,
            show_prob=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        """Render an image."""
        self.model = model
        self.states_to_eval = ptu.from_numpy(states_to_eval)
        self.show_prob = show_prob

    def _create_image(self, env, obs, action):
        if action is None:
            return np.zeros(self.image_shape)
        obs = obs['state_observation']
        obs_torch = ptu.from_numpy(obs)[None]
        action_torch = ptu.from_numpy(action)[None]
        dist = self.model(obs_torch, action_torch)
        log_probs = dist.log_prob(self.states_to_eval)
        value_image = ptu.get_numpy(log_probs.reshape(self.image_chw[1:]))
        if self.show_prob:
            value_image = np.exp(value_image)

        # TODO: fix hardcoding of CHW
        value_img_rgb = np.repeat(
            value_image[None, :, :],
            3,
            axis=0
        )
        value_img_rgb = (
                (value_img_rgb - value_img_rgb.min()) /
                (value_img_rgb.max() - value_img_rgb.min() + 1e-9)
        )
        return value_img_rgb


class DiscountModelRenderer(Renderer):
    def __init__(
            self,
            model,
            states_to_eval,
            show_prob=False,
            discount_for_kl=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        """Render an image."""
        self.model = model
        self.states_to_eval = ptu.from_numpy(states_to_eval)
        self.show_prob = show_prob
        self.discount_for_kl = discount_for_kl

    def _create_image(self, env, obs, action):
        if action is None:
            return np.zeros(self.image_shape)
        obs = obs['state_observation']
        obs_torch = ptu.from_numpy(obs)[None]
        action_torch = ptu.from_numpy(action)[None]
        batch_size = self.states_to_eval.shape[0]
        combined_obs = torch.cat([
            obs_torch.repeat(batch_size, 1),
            self.states_to_eval,
        ], dim=1)

        action_repeated = action_torch.repeat(batch_size, 1)
        predictions = self.model(combined_obs, action_repeated)
        if self.discount_for_kl:
            predictions = kl_divergence(
                Bernoulli(predictions),
                Bernoulli(self.discount_for_kl),
            )

        value_image = ptu.get_numpy(predictions).reshape(
            self.image_chw[1:],
        )

        value_img_rgb = np.repeat(
            value_image[None, :, :],
            3,
            axis=0
        )
        value_img_rgb = (
                (value_img_rgb - value_img_rgb.min()) /
                (value_img_rgb.max() - value_img_rgb.min() + 1e-9)
        )
        return value_img_rgb


class ValueRenderer(Renderer):
    def __init__(
            self,
            value_fn,
            only_get_image_once_per_episode=False,
            max_out_walls=False,
            **kwargs):
        super().__init__(**kwargs)
        """Render an some (flat) values returned by value_fn.

        It's the responsibility of the caller to make sure that the output of
        value_fn can be reshaped into the correct image shape.
        """
        self.value_fn = value_fn
        self.only_get_image_once_per_episode = only_get_image_once_per_episode
        self.max_out_walls = max_out_walls
        self._image = None

    def reset(self):
        if self.only_get_image_once_per_episode:
            self._image = None

    def _create_image(self, env, obs, action):
        if action is None:
            return np.zeros(self.image_shape)
        if self.only_get_image_once_per_episode and self._image is not None:
            return self._image
        predictions = self.value_fn(obs, action)
        value_image = ptu.get_numpy(predictions).reshape(
            self.image_chw[1:],
        )

        value_img_rgb = np.repeat(
            value_image[None, :, :],
            3,
            axis=0
        )
        # import ipdb; ipdb.set_trace()
        if self.max_out_walls:
            # bottom wall
            value_img_rgb[:, 64:88, 24:104] = value_img_rgb.max()
            # left wall
            value_img_rgb[:, 40:88, 24:48] = value_img_rgb.max()
            # right wall
            value_img_rgb[:, 40:88, 128-48:128-24] = value_img_rgb.max()
        value_img_rgb = (
                (value_img_rgb - value_img_rgb.min()) /
                (value_img_rgb.max() - value_img_rgb.min() + 1e-9)
        )
        self._image = value_img_rgb
        return value_img_rgb


class ProductRenderer(Renderer):
    """Show the product of two images."""
    def __init__(
            self,
            discount_renderer,
            log_prob_renderer,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.discount_renderer = discount_renderer
        self.log_prob_renderer = log_prob_renderer

    def _create_image(self, *args, **kwargs):
        img1 = self.discount_renderer(*args, **kwargs)
        img2 = self.log_prob_renderer(*args, **kwargs)
        return (1-img1) * img2


class DynamicNumberEnvRenderer(ScrollingPlotRenderer):
    def __init__(self, *args, dynamic_number_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_number_fn = dynamic_number_fn

    def _create_image(self, env, obs, action, next_obs):
        if obs is None:
            return np.zeros(self._create_image_shape)
        number = self.dynamic_number_fn(obs, action, next_obs)
        img = super()._create_image(number)
        return img


class DynamicNumbersEnvRenderer(ScrollingPlotRenderer):
    def __init__(self, *args, dynamic_number_fns, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_number_fns = dynamic_number_fns

    def _create_image(self, env, obs, action, next_obs):
        if obs is None:
            return np.zeros(self._create_image_shape)
        number = self.dynamic_number_fn(obs, action, next_obs)
        img = super()._create_image(number)
        return img


class InsertDebugImagesEnv(InsertImagesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_obs = None

    def reset(self):
        self._last_obs = None
        obs = super().reset()
        for image_key, renderer in self.renderers.items():
            if isinstance(renderer, DynamicNumberEnvRenderer):
                renderer.reset()
            elif isinstance(renderer, ValueRenderer):
                renderer.reset()
        self._last_obs = obs
        return obs

    def _update_obs(self, obs, action=None):
        for image_key, renderer in self.renderers.items():
            if isinstance(renderer, DynamicsModelEnvRenderer):
                obs[image_key] = renderer(self.env, obs, action)
            elif isinstance(renderer, DynamicNumberEnvRenderer):
                obs[image_key] = renderer(self.env, self._last_obs, action, obs)
            elif isinstance(renderer, DiscountModelRenderer):
                obs[image_key] = renderer(self.env, obs, action)
            elif isinstance(renderer, ProductRenderer):
                obs[image_key] = renderer(self.env, obs, action)
            elif isinstance(renderer, ValueRenderer):
                obs[image_key] = renderer(self.env, obs, action)
            else:
                obs[image_key] = renderer(self.env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._update_obs(obs, action)
        self._last_obs = obs
        return obs, reward, done, info
