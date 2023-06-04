import copy
import random
import warnings

import torch

# import cv2
import numpy as np
import pickle
from gym import Env
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.proxy_env import ProxyEnv
from rlkit.util.io import load_local_or_remote_file
import time

class VAEWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """
    def __init__(
        self,
        wrapped_env,
        vae,
        vae_input_key_prefix='image',
        sample_from_true_prior=False,
        decode_goals=False,
        decode_goals_on_reset=True,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        goal_sampling_mode="vae_prior",
        presampled_goals_path=None,
        num_goals_to_presample=0,
        imsize=84,
        obs_size=None,
        norm_order=2,
        epsilon=20,
        presampled_goals=None,
    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        if type(vae) is str:
            self.vae = load_local_or_remote_file(vae)
        else:
            self.vae = vae
        self.representation_size = self.vae.representation_size
        self.input_channels = self.vae.input_channels
        self.sample_from_true_prior = sample_from_true_prior
        self._decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs=dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        self.decode_goals_on_reset = decode_goals_on_reset

        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )

        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self.observation_space = Dict(spaces)

        self._presampled_goals = presampled_goals

        if num_goals_to_presample > 0:
            self._presampled_goals = self.wrapped_env.sample_goals(num_goals_to_presample)
        if presampled_goals_path is not None:
            self._presampled_goals = load_local_or_remote_file(presampled_goals_path)

        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = self._presampled_goals[random.choice(list(self._presampled_goals))].shape[0]

        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in {'image', 'image_proprio'}
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = {'latent_desired_goal': latent_space.sample()}
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode


    def reset(self):

        self.vae.eval()
        obs = self.wrapped_env.reset()
        self._initial_obs = obs
        goal = self.sample_goal()
        self.set_goal(goal)

        if self.decode_goals_on_reset:
            imgs = self._decode_one(goal["latent_desired_goal"]).reshape(1, -1)
            self.decoded_goal_image = imgs.reshape(
                self.input_channels, self.imsize, self.imsize
            ).flatten()

        obs = self._update_obs(obs)
        return obs

    def step(self, action):
        self.vae.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            new_obs,
            # {'latent_achieved_goal': new_obs['latent_achieved_goal'],
            #  'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.vae.eval()
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])
        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs
        if self.decode_goals_on_reset:
            obs['decoded_goal_image'] = self.decoded_goal_image
        obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        self.vae.eval()
        latent_distribution_params = self.vae.encode(
            ptu.from_numpy(obs[self.vae_input_observation_key].reshape(1,-1))
        )
        latent_obs, logvar = ptu.get_numpy(latent_distribution_params[0])[0], ptu.get_numpy(latent_distribution_params[1])[0]
        # assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self.desired_goal['latent_desired_goal']
        dist = latent_goal - latent_obs
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    """
    Multitask functions
    """
    def sample_goals(self, batch_size):
        self.vae.eval()
        # print(self._goal_sampling_mode)
        # if self._goal_sampling_mode == "reset_of_env":
        # TODO: make mode a parameter you pass in

        if self._goal_sampling_mode == 'custom_goal_sampler':
            return self.custom_goal_sampler(batch_size)
        elif self._goal_sampling_mode == 'presampled':
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            # ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(sampled_goals['image_desired_goal'])
            return sampled_goals
        elif self._goal_sampling_mode == 'env':
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key])
        elif self._goal_sampling_mode == 'reset_of_env':
            assert batch_size == 1
            goal = self.wrapped_env.get_goal()
            goals = {k: v[None] for k, v in goal.items()}
            latent_goals = self._encode(
                goals[self.vae_input_desired_goal_key]
            )
        elif self._goal_sampling_mode == 'vae_prior':
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        # I don't think this should ever be true (decoding goals while sampling)? - Ashvin
        # if self._decode_goals:
        #     decoded_goals = self._decode(latent_goals)
        # else:
        #     decoded_goals = None
        decoded_goals = None

        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals
        )
        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        if proprio_goals is not None:
            goals['proprio_desired_goal'] = proprio_goals
        if image_goals is not None:
            goals['image_desired_goal'] = image_goals
        if decoded_goals is not None:
            goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        reward = self.compute_rewards(actions, next_obs)
        return reward[0]


    def compute_rewards(self, actions, obs):
        self.vae.eval()
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            success = dist < self.epsilon
            reward = success - 1
            return reward
        elif self.reward_type == 'latent_clamp':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            clamped_dist = np.minimum(dist, self.epsilon) / self.epsilon #Scale Distance
            reward = - clamped_dist
            return reward
        elif self.reward_type == 'success_prob':
            desired_goals = self._decode(obs['latent_desired_goal'])
            achieved_goals = self.vae.decode(ptu.from_numpy(obs['latent_achieved_goal']))
            prob = self.vae.logprob(desired_goals, achieved_goals).exp()
            reward = prob
            return 1/0 #not sure about this anymore, number will be too low
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'state_sparse':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)
            max_dist = np.linalg.norm(distance, axis=1, ord=np.inf)
            success = max_dist < self.epsilon
            reward = success - 1
            return reward
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @property
    def goal_dim(self):
        return self.representation_size

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
        # TODO: fix this hack / document this
        if self._goal_sampling_mode in {'presampled', 'env'}:
            self.wrapped_env.set_goal(goal)

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'custom_goal_sampler',
            'presampled',
            'vae_prior',
            'env',
            'reset_of_env'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode
        if mode == 'custom_goal_sampler':
            test_goals = self.custom_goal_sampler(1)
            if test_goals is None:
                self._goal_sampling_mode = 'vae_prior'
                warnings.warn(
                    "self.goal_sampler returned None. " + \
                    "Defaulting to vae_prior goal sampling mode"
                )

    @property
    def custom_goal_sampler(self):
        return self._custom_goal_sampler

    @custom_goal_sampler.setter
    def custom_goal_sampler(self, new_custom_goal_sampler):
        assert self.custom_goal_sampler is None, (
            "Cannot override custom goal setter"
        )
        self._custom_goal_sampler = new_custom_goal_sampler

    @property
    def decode_goals(self):
        return self._decode_goals

    @decode_goals.setter
    def decode_goals(self, _decode_goals):
        self._decode_goals = _decode_goals

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            gpu_info=dict(
                use_gpu=ptu._use_gpu,
                gpu_id=ptu._gpu_id,
            ),
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state, gpu_info):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)
        gpu_id = gpu_info['gpu_id']
        use_gpu = gpu_info['use_gpu']
        ptu.device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
        self.vae.to(ptu.device)

    def enable_render(self):
        self._decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self._decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def try_render(self, obs):
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(obs['image_observation']).transpose()
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
            init_img = self._initial_obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('initial_state', init_img)
            cv2.waitKey(1)
            init_reconstruction = self._reconstruct_img(
                self._initial_obs['image_observation']
            ).transpose()
            cv2.imshow('init_reconstruction', init_reconstruction)
            cv2.waitKey(1)

        if self.render_goals:
            goal = obs['image_desired_goal'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)

    def _sample_vae_prior(self, batch_size):
        self.vae.eval()
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        return sigma * n + mu

    def _decode(self, latents):
        self.vae.eval()
        reconstructions, _ = self.vae.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def _encode_one(self, img):
        return self._encode(img[None])[0]

    def _decode_one(self, latent):
        return self._decode(latent[None])[0]

    def _encode(self, imgs):
        self.vae.eval()
        latent_distribution_params = self.vae.encode(ptu.from_numpy(imgs))
        return ptu.get_numpy(latent_distribution_params[0])

    def _reconstruct_img(self, flat_img):
        self.vae.eval()
        latent_distribution_params = self.vae.encode(ptu.from_numpy(flat_img.reshape(1,-1)))
        reconstructions, _ = self.vae.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

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


class ConditionalVAEWrappedEnv(VAEWrappedEnv):
    def _update_info(self, info, obs):
        self.vae.eval()
        x0 = ptu.from_numpy(self._initial_obs["image_observation"][None])
        x = ptu.from_numpy(obs[self.vae_input_observation_key]).reshape(1,-1)
        latent_distribution_params = self.vae.encode(x, x0)
        mu, logvar, cond = ptu.get_numpy(latent_distribution_params[0])[0], \
                    ptu.get_numpy(latent_distribution_params[1])[0], \
                    ptu.get_numpy(latent_distribution_params[2])[0]
        # assert (latent_obs == obs['latent_observation']).all()
        latent_obs = np.concatenate([mu, cond], axis=0)
        latent_goal = self.desired_goal['latent_desired_goal']
        z_goal = latent_goal[:self.vae.latent_sizes[0]]
        dist = z_goal - mu
        var = np.exp(logvar.flatten())
        var = np.maximum(var, self.reward_min_variance)
        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        info["vae_mdist"] = mdist
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def compute_rewards(self, actions, obs):
        self.vae.eval()
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            #distance = np.multiply(1 / self.prior_var, desired_goals - achieved_goals)
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            success = dist < self.epsilon
            reward = success - 1
            return reward

        #WARNING: BELOW ARE HARD CODED FOR SIM PUSHER ENV (IN DIMENSION SIZES)
        elif self.reward_type == 'latent_score':
            dist = np.abs(obs['latent_achieved_goal'][:, :self.vae.latent_sizes[0]] - obs['latent_desired_goal'][:, :self.vae.latent_sizes[0]])
            score = np.sum(np.where(dist < self.epsilon, 1, 0), axis=1)
            return np.square(score)

            return reward
        elif self.reward_type == 'latent_bound':
            dist = np.abs(obs['latent_achieved_goal'] - obs['latent_desired_goal'])
            reward = - 1 * np.amax(dist, axis=1)
            return reward
        elif self.reward_type == 'success_prob':
            desired_goals = ptu.from_numpy(self._decode(obs['latent_desired_goal']))
            achieved_goals = ptu.from_numpy(self._decode(obs['latent_achieved_goal']))
            reward = ptu.get_numpy(self.vae.logprob(desired_goals, achieved_goals, mean=False).exp())
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return reward
        elif self.reward_type == 'log_success_prob':
            desired_goals = ptu.from_numpy(self._decode(obs['latent_desired_goal']))
            achieved_goals = ptu.from_numpy(self._decode(obs['latent_achieved_goal']))
            reward = - 1 * np.log(ptu.get_numpy(1e-5 + -1 * self.vae.logprob(desired_goals, achieved_goals, mean=False)))
            return reward

        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal'].reshape(-1, 4)
            desired_goals = obs['state_desired_goal'].reshape(-1, 4)
            return - np.linalg.norm(desired_goals - achieved_goals, axis=1)
        elif self.reward_type == 'state_sparse':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)
            max_dist = np.linalg.norm(distance, axis=1, ord=np.inf)
            success = max_dist < self.epsilon
            reward = success - 1
            return reward
        elif self.reward_type == 'state_hand_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, :1]
            return - distance
        elif self.reward_type == 'state_puck_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, 1:]
            return - distance
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    def _decode(self, latents):
        self.vae.eval()
        z = ptu.from_numpy(latents)
        reconstructions, _ = self.vae.decode(z)
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def _encode(self, imgs):
        self.vae.eval()
        x0 = ptu.from_numpy(self._initial_obs["image_observation"][None])
        latent = self.vae.encode(ptu.from_numpy(imgs), x0, distrib=False)
        return ptu.get_numpy(latent)

    def _reconstruct_img(self, flat_img, x_0=None):
        self.vae.eval()

        if x_0 is None:
            x_0 = ptu.from_numpy(self._initial_obs["image_observation"][None])
        else:
            x_0 = ptu.from_numpy(x_0.reshape(1, -1))

        latent = self.vae.encode(ptu.from_numpy(flat_img.reshape(1,-1)), x_0, distrib=False)
        reconstructions, _ = self.vae.decode(latent)
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _sample_vae_prior(self, batch_size, x0_latent=None):
        self.vae.eval()
        if x0_latent is None:
            x0 = ptu.from_numpy(self._initial_obs["image_observation"][None])
            if not self.sample_from_true_prior:
                return ptu.get_numpy(self.vae.sample_prior(batch_size, x0, true_prior=False))
            return ptu.get_numpy(self.vae.sample_prior(batch_size, x0))
        else:
            mu, sigma = 0, 1  # sample from prior
            n = np.random.randn(batch_size, self.vae.latent_sizes)
            z = sigma * n + mu
            return np.concatenate((z, x0_latent), axis=1)

def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
