from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.exploration_strategies.count_based.count_based import CountExploration
from torchvision.utils import save_image

from rlkit.core import logger
from rlkit.data_management.obs_dict_replay_buffer import (
    combine_dicts,
    ObsDictRelabelingBuffer
)
from rlkit.data_management.shared_obs_dict_replay_buffer import \
    SharedObsDictRelabelingBuffer
import rlkit.torch.pytorch_util as ptu
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from torch.distributions import Normal

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.networks import Mlp
from rlkit.util.ml_util import ConstantSchedule
from rlkit.util.ml_util import PiecewiseLinearSchedule
import os.path as osp

from multiworld.core.multitask_env import MultitaskEnv
from rlkit.data_management.replay_buffer import ReplayBuffer
import rlkit.data_management.images as image_np
from rlkit.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
from rlkit.torch.vae.conditional_conv_vae import DeltaCVAE

class OnlineConditionalVaeRelabelingBuffer(OnlineVaeRelabelingBuffer):
    def random_vae_training_data(self, batch_size, epoch):
        # epoch no longer needed. Using self.skew in sample_weighted_indices
        # instead.
        weighted_idxs = self.sample_weighted_indices(
            batch_size,
        )

        next_image_obs = self._next_obs[self.decoded_obs_key][weighted_idxs]
        observations = ptu.from_numpy(next_image_obs)

        x_0_indices = (weighted_idxs // 100) * 100
        x_0 = self._next_obs[self.decoded_obs_key][x_0_indices]
        x_0 = ptu.from_numpy(x_0)

        return dict(
            observations=observations,
            x_t=observations,
            x_0=x_0,
        )

    def refresh_latents(self, epoch):
        self.epoch = epoch
        self.skew = (self.epoch > self.start_skew_epoch)
        batch_size = 512
        next_idx = min(batch_size, self._size)

        if self.exploration_rewards_type == 'hash_count':
            # you have to count everything then compute exploration rewards
            cur_idx = 0
            next_idx = min(batch_size, self._size)
            while cur_idx < self._size:
                idxs = np.arange(cur_idx, next_idx)
                normalized_imgs = self._next_obs[self.decoded_obs_key][idxs]
                self.update_hash_count(normalized_imgs)
                cur_idx = next_idx
                next_idx += batch_size
                next_idx = min(next_idx, self._size)

        vae = self.env.vae
        vae.eval()

        def encode(imgs, x_0):
            latent = vae.encode(ptu.from_numpy(imgs), x_0, distrib=False)
            return ptu.get_numpy(latent)

        cur_idx = 0
        obs_sum = np.zeros(np.array([self.vae.latent_sizes, ]).sum())
        obs_square_sum = np.zeros(np.array([self.vae.latent_sizes, ]).sum())
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            x_0_idxs = (idxs // 100) * 100

            x_0 = ptu.from_numpy(self._obs[self.decoded_obs_key][x_0_idxs])

            self._obs[self.observation_key][idxs] = encode(self._obs[self.decoded_obs_key][idxs], x_0)
            self._next_obs[self.observation_key][idxs] = encode(self._next_obs[self.decoded_obs_key][idxs], x_0)
            # WARNING: we only refresh the desired/achieved latents for
            # "next_obs". This means that obs[desired/achieve] will be invalid,
            # so make sure there's no code that references this.
            # TODO: enforce this with code and not a comment
            self._next_obs[self.desired_goal_key][idxs] = encode(self._next_obs[self.decoded_desired_goal_key][idxs], x_0)
            self._next_obs[self.achieved_goal_key][idxs] = encode(self._next_obs[self.decoded_achieved_goal_key][idxs], x_0)
            normalized_imgs = self._next_obs[self.decoded_obs_key][idxs]
            if self._give_explr_reward_bonus:
                rewards = self.exploration_reward_func(
                    normalized_imgs,
                    idxs,
                    **self.priority_function_kwargs
                )
                self._exploration_rewards[idxs] = rewards.reshape(-1, 1)
            if self._prioritize_vae_samples:
                if (
                        self.exploration_rewards_type == self.vae_priority_type
                        and self._give_explr_reward_bonus
                ):
                    self._vae_sample_priorities[idxs] = (
                        self._exploration_rewards[idxs]
                    )
                else:
                    self._vae_sample_priorities[idxs] = (
                        self.vae_prioritization_func(
                            normalized_imgs,
                            idxs,
                            **self.priority_function_kwargs
                        ).reshape(-1, 1)
                    )
            obs_sum += self._obs[self.observation_key][idxs].sum(axis=0)
            obs_square_sum += np.power(self._obs[self.observation_key][idxs], 2).sum(axis=0)

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, self._size)
        self.vae.dist_mu = obs_sum/self._size
        self.vae.dist_std = np.sqrt(obs_square_sum/self._size - np.power(self.vae.dist_mu, 2))

        if self._prioritize_vae_samples:
            """
            priority^power is calculated in the priority function
            for image_bernoulli_prob or image_gaussian_inv_prob and
            directly here if not.
            """
            if self.vae_priority_type == 'vae_prob':
                self._vae_sample_priorities[:self._size] = relative_probs_from_log_probs(
                    self._vae_sample_priorities[:self._size]
                )
                self._vae_sample_probs = self._vae_sample_priorities[:self._size]
            else:
                self._vae_sample_probs = self._vae_sample_priorities[:self._size] ** self.power
            p_sum = np.sum(self._vae_sample_probs)
            assert p_sum > 0, "Unnormalized p sum is {}".format(p_sum)
            self._vae_sample_probs /= np.sum(self._vae_sample_probs)
            self._vae_sample_probs = self._vae_sample_probs.flatten()


    def vae_prob(self, next_vae_obs, indices, **kwargs):
        """TODO: add x0"""
        x_0 = self._next_obs[self.decoded_obs_key][indices]
        batch = dict(x_0=x_0, x_t=next_vae_obs)
        return compute_p_x_np_to_np(
            self.vae,
            batch,
            power=self.power,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            if isinstance(self.env.vae, DeltaCVAE):
                r1, r2 = self.env.vae.latent_sizes
            else:
                r1 = self.env.representation_size

            env_goals = np.random.randn(num_env_goals, r1) # self._sample_goals_from_env(num_env_goals)
            #env_goal = self._sample_goals_from_env(num_env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals

            resampled_goals[num_rollout_goals:last_env_goal_idx, :r1] = (
                env_goals
            )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx, :r1] = \
                    env_goals
                new_next_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx, :r1] = \
                    env_goals
        if num_future_goals > 0:
            future_obs_idxs = self._get_future_obs_indices(indices[-num_future_goals:])
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]

        if isinstance(self.env, MultitaskEnv):
            new_rewards = self.env.compute_rewards(
                new_actions,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        if not self.vectorized:
            new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }

        exploration_rewards_scale = float(self.explr_reward_scale_schedule.get_value(self.epoch))
        if self._give_explr_reward_bonus:
            batch_idxs = batch['indices'].flatten()
            batch['exploration_rewards'] = self._exploration_rewards[batch_idxs]
            batch['rewards'] += exploration_rewards_scale * batch['exploration_rewards']
        return batch


def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    probs = np.exp(log_probs - log_probs.mean())
    assert not np.any(probs <= 0), 'choose a smaller power'
    return probs

def compute_log_p_log_q_log_d(
    model,
    batch,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    x_0 = ptu.from_numpy(batch["x_0"])
    data = batch["x_t"]
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs, x_0)
    r1 = model.latent_sizes[0]
    batch_size = data.shape[0]
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, r1)),
                        ptu.ones((batch_size, r1)))
    mus, logvars = latent_distribution_params[:2]
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params[:2])
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params[:2])
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)

        if len(latent_distribution_params) == 3: # add conditioning for CVAEs
            latents = torch.cat((latents, latent_distribution_params[2]), dim=1)

        if decoder_distribution == 'bernoulli':
            decoded = model.decode(latents)[0]
            log_d_x_given_z = torch.log(imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=1)
        elif decoder_distribution == 'gaussian_identity_variance':
            _, obs_distribution_params = model.decode(latents)
            dec_mu, dec_logvar = obs_distribution_params
            dec_var = dec_logvar.exp()
            decoder_dist = Normal(dec_mu, dec_var.pow(.5))
            log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=1)
        else:
            raise EnvironmentError('Invalid Decoder Distribution Provided')

        log_p[:, i] = log_p_z
        log_q[:, i] = log_q_z_given_x
        log_d[:, i] = log_d_x_given_z
    return log_p, log_q, log_d

def compute_p_x_np_to_np(
    model,
    batch,
    power,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    # data = batch["observations"]
    # assert data.dtype != np.uint8, 'images should be normalized'
    assert power >= -1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        batch,
        decoder_distribution,
        num_latents_to_sample,
        sampling_method
    )

    if sampling_method == 'importance_sampling':
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
    elif sampling_method == 'biased_sampling' or sampling_method == 'true_prior_sampling':
        log_p_x = log_d.mean(dim=1)
    else:
        raise EnvironmentError('Invalid Sampling Method Provided')
    log_p_x_skewed = power * log_p_x
    return ptu.get_numpy(log_p_x_skewed)
