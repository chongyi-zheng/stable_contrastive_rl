import copy
import random
import warnings

import torch

# import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import time

from os import path as osp
from torchvision.utils import save_image
import rlkit.data_management.external.epic_kitchens_data_stub as epic

eps = 1e-5

class EncoderWrappedEnv(ProxyEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        reward_params=None,
        config_params=None,
        imsize=84,
        obs_size=None,
        vae_input_observation_key="image_observation",
        small_image_step=6,
    ):
        if config_params is None:
            config_params = dict
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.imsize = imsize
        self.config_params = config_params
        self.t = 0
        self.episode_num = 0
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.zT = self.reward_params["goal_latent"]
        self.z0 = self.reward_params["initial_latent"]
        self.dT = self.zT - self.z0

        self.small_image_step = small_image_step
        # if self.config_params["use_initial"]:
        #     self.dT = self.zT - self.z0
        # else:
        #     self.dT = self.zT

        self.vae_input_observation_key = vae_input_observation_key

        latent_size = obs_size or self.representation_size
        latent_space = Box(
            -10 * np.ones(latent_size),
            10 * np.ones(latent_size),
            dtype=np.float32,
        )
        goal_space = Box(
            np.zeros((0, )),
            np.zeros((0, )),
            dtype=np.float32,
        )
        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = goal_space
        spaces['achieved_goal'] = goal_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = goal_space
        spaces['latent_achieved_goal'] = goal_space

        concat_size = latent_size + spaces["state_observation"].low.size
        concat_space = Box(
            -10 * np.ones(concat_size),
            10 * np.ones(concat_size),
            dtype=np.float32,
        )
        spaces['concat_observation'] = concat_space
        small_image_size = 288 // self.small_image_step
        small_image_imglength = small_image_size * small_image_size * 3
        small_image_space = Box(
            0 * np.ones(small_image_imglength),
            1 * np.ones(small_image_imglength),
            dtype=np.float32,
        )
        spaces['small_image_observation'] = small_image_space
        small_image_observation_with_state_size = small_image_imglength + spaces["state_observation"].low.size
        small_image_observation_with_state_space = Box(
            0 * np.ones(small_image_observation_with_state_size),
            1 * np.ones(small_image_observation_with_state_size),
            dtype=np.float32,
        )
        spaces['small_image_observation_with_state'] = small_image_observation_with_state_space

        self.observation_space = Dict(spaces)

    def reset(self):
        self.vae.eval()
        self.t = 0
        self.episode_num += 1
        obs = self.wrapped_env.reset()
        # start_rollout(obs)
        self.x0 = obs["image_observation"]
        # self.save_image_util(self.x0, "demos/reset_initial")
        goal = self.sample_goal()
        self.set_goal(goal)
        obs = self.update_obs(obs)
        self.z0 = obs["latent_observation"]
        # if self.config_params["use_initial"]:
        #     print(self.dT)
        #     self.dT = self.zT - self.z0
        # else:
        #     self.dT = self.zT
        return obs

    def initialize(self, zs):
        if self.config_params["initial_type"] == "use_initial_from_trajectory":
            self.z0 = zs[0]
        if self.config_params["goal_type"] == "use_goal_from_trajectory":
            self.zT = zs[-1]

        # if self.config_params["use_initial"]:
        #     self.dT = self.zT - self.z0
        # else:
        #     self.dT = self.zT

    def step(self, action):
        self.vae.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self.update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            new_obs,
            # {'latent_achieved_goal': new_obs['latent_achieved_goal'],
            #  'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        return new_obs, reward, done, info

    def update_obs(self, obs):
        self.vae.eval()
        self.zt = self._encode_one(obs[self.vae_input_observation_key])
        # if self.config_params["use_initial"]:
        #     latent_obs = self.zt - self.z0
        # else:
        #     latent_obs = self.zt
        latent_obs = self.zt
        obs['z0'] = self.z0.copy()
        obs['zt'] = self.zt.copy()
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = np.array([])
        obs['latent_desired_goal'] = np.array([])
        obs['observation'] = latent_obs
        obs['achieved_goal'] = np.array([])
        obs['desired_goal'] = np.array([])
        # obs = {**obs, **self.desired_goal}

        state_obs = obs['state_observation'].copy()
        concat_obs = np.concatenate((latent_obs, state_obs))
        obs['concat_observation'] = concat_obs

        S = self.small_image_step
        small_image_obs = obs['image_observation'].reshape(3, 500, 300)[:, 106:394:S, 6:294:S] / 255.0
        obs['small_image_observation'] = small_image_obs.flatten()
        obs['small_image_observation_with_state'] = np.concatenate((obs['small_image_observation'], state_obs))

        return obs

    def _update_obs_latent(self, obs, z):
        self.vae.eval()
        self.zt = z
        # if self.config_params["use_initial"]:
        #     latent_obs = self.zt - self.z0
        # else:
        #     latent_obs = self.zt
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = np.array([])
        obs['latent_desired_goal'] = np.array([])
        obs['observation'] = latent_obs
        obs['achieved_goal'] = np.array([])
        obs['desired_goal'] = np.array([])
        # obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        pass
        # self.vae.eval()
        # latent_distribution_params = self.vae.encode(
        #     ptu.from_numpy(obs[self.vae_input_observation_key].reshape(1,-1))
        # )
        # latent_obs, logvar = ptu.get_numpy(latent_distribution_params[0])[0], ptu.get_numpy(latent_distribution_params[1])[0]
        # # assert (latent_obs == obs['latent_observation']).all()
        # latent_goal = self.desired_goal['latent_desired_goal']
        # dist = latent_goal - latent_obs
        # var = np.exp(logvar.flatten())
        # var = np.maximum(var, self.reward_min_variance)
        # err = dist * dist / 2 / var
        # mdist = np.sum(err)  # mahalanobis distance
        # info["vae_mdist"] = mdist
        # info["vae_success"] = 1 if mdist < self.epsilon else 0
        # info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        # info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        # info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def compute_reward(self, action, obs, info=None):
        self.vae.eval()

        # dt = obs["latent_observation"]
        # dT = self.dT

        zt = obs["latent_observation"]
        z0 = self.z0
        zg = self.zT
        dt = zt - z0
        dT = zg - z0

        # import ipdb; ipdb.set_trace()

        reward_type = self.reward_params.get("type", "regression_distance")
        if reward_type == "regression_distance":
            regression_pred_yt = (dt * dT).sum() / ((dT ** 2).sum() + eps)
            r = -np.abs(1-regression_pred_yt)
        if reward_type == "latent_distance":
            r = -np.linalg.norm(dt - dT)

        return r

        # next_obs = {
        #     k: v[None] for k, v in obs.items()
        # }
        # reward = self.compute_rewards(actions, next_obs)
        # return reward[0]

    def compute_rewards(self, actions, obs, info=None):
        self.vae.eval()
        print("\n\n\n obs keys \n\n\n", obs.keys())
        dt = obs["latent_observation"]
        dT = self.dT

        regression_pred_yt = (dt * dT).sum(axis=1) / ((dT ** 2).sum() + eps)

        return -np.abs(1-regression_pred_yt)

        # TODO: implement log_prob/mdist
        # if self.reward_type == 'latent_distance':
        #     achieved_goals = obs['latent_achieved_goal']
        #     desired_goals = obs['latent_desired_goal']
        #     dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        #     return -dist
        # elif self.reward_type == 'wrapped_env':
        #     return self.wrapped_env.compute_rewards(actions, obs)
        # else:
        #     raise NotImplementedError

    def get_diagnostics(self, paths, **kwargs):
        statistics = dict()
        # statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        # for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
        #     stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
        #     statistics.update(create_stats_ordered_dict(
        #         stat_name_in_paths,
        #         stats,
        #         always_show_all_stats=True,
        #     ))
        #     final_stats = [s[-1] for s in stats]
        #     statistics.update(create_stats_ordered_dict(
        #         "Final " + stat_name_in_paths,
        #         final_stats,
        #         always_show_all_stats=True,
        #     ))
        return statistics

    def _encode_one(self, img, name=None):
        im = img.reshape(1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
        im = im[:, :, 60:, 60:500]
        return self._encode(im, name)[0]

    def _encode_batch(self, imgs):
        im = imgs.reshape(-1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
        im = im[:, :, 60:, 60:500]
        return self._encode(im)

    def save_image_util(self, img, name):
        im = img.reshape(-1, 3, 500, 300).transpose([0, 1, 3, 2]) / 255.0
        im = im[:, :, 60:, 60:500]
        pt_img = ptu.from_numpy(im).view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH)
        save_image(pt_img.data.cpu(), '%s_%d.png'% (name, self.episode_num), nrow=1)
        # save_image(pt_img[:1, :, :, :].data.cpu(), 'forward.png', nrow=1)

    def _encode(self, imgs, name=None):
        self.vae.eval()
        pt_img = ptu.from_numpy(imgs).view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH)

        # save_image(pt_img.data.cpu(), 'demos/forward_%d.png' % self.t, nrow=1)
        # save_image(pt_img[:1, :, :, :].data.cpu(), 'forward.png', nrow=1)

        latent_distribution_params = self.vae.encode(pt_img)
        return ptu.get_numpy(latent_distribution_params)

    def _image_and_proprio_from_decoded(self, decoded):
        if decoded is None:
            return None, None
        if self.vae_input_key_prefix == 'image_proprio':
            images = decoded[:, :self.image_length]
            proprio = decoded[:, self.image_length:]
            return images, proprio
        elif self.vae_input_key_prefix == 'image':
            return decoded, None
        else:
            raise AssertionError("Bad prefix for the vae input key.")

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn('VAEWrapperEnv.custom_goal_sampler is not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn('VAEWrapperEnv.custom_goal_sampler was not loaded.')
        super().__setstate__(state)
