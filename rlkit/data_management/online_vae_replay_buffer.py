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

class OnlineVaeRelabelingBuffer(ObsDictRelabelingBuffer):

    def __init__(
            self,
            vae,
            *args,
            decoded_obs_key='image_observation',
            decoded_achieved_goal_key='image_achieved_goal',
            decoded_desired_goal_key='image_desired_goal',
            exploration_rewards_type='None',
            exploration_rewards_scale=1.0,
            vae_priority_type='None',
            start_skew_epoch=0,
            power=1.0,
            internal_keys=None,
            exploration_schedule_kwargs=None,
            priority_function_kwargs=None,
            exploration_counter_kwargs=None,
            relabeling_goal_sampling_mode='vae_prior',
            decode_vae_goals=False,
            **kwargs
    ):
        if internal_keys is None:
            internal_keys = []

        for key in [
            decoded_obs_key,
            decoded_achieved_goal_key,
            decoded_desired_goal_key
        ]:
            if key not in internal_keys:
                internal_keys.append(key)
        super().__init__(internal_keys=internal_keys, *args, **kwargs)
        # assert isinstance(self.env, VAEWrappedEnv)
        self.vae = vae
        self.decoded_obs_key = decoded_obs_key
        self.decoded_desired_goal_key = decoded_desired_goal_key
        self.decoded_achieved_goal_key = decoded_achieved_goal_key
        self.exploration_rewards_type = exploration_rewards_type
        self.exploration_rewards_scale = exploration_rewards_scale
        self.start_skew_epoch = start_skew_epoch
        self.vae_priority_type = vae_priority_type
        self.power = power
        self._relabeling_goal_sampling_mode = relabeling_goal_sampling_mode
        self.decode_vae_goals = decode_vae_goals

        if exploration_schedule_kwargs is None:
            self.explr_reward_scale_schedule = \
                ConstantSchedule(self.exploration_rewards_scale)
        else:
            self.explr_reward_scale_schedule = \
                PiecewiseLinearSchedule(**exploration_schedule_kwargs)

        self._give_explr_reward_bonus = (
                exploration_rewards_type != 'None'
                and exploration_rewards_scale != 0.
        )
        self._exploration_rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self._prioritize_vae_samples = (
                vae_priority_type != 'None'
                and power != 0.
        )
        self._vae_sample_priorities = np.zeros((self.max_size, 1), dtype=np.float32)
        self._vae_sample_probs = None

        self.use_dynamics_model = (
                self.exploration_rewards_type == 'forward_model_error'
        )
        if self.use_dynamics_model:
            self.initialize_dynamics_model()

        type_to_function = {
            'reconstruction_error': self.reconstruction_mse,
            'bce': self.binary_cross_entropy,
            'latent_distance': self.latent_novelty,
            'latent_distance_true_prior': self.latent_novelty_true_prior,
            'forward_model_error': self.forward_model_error,
            'gaussian_inv_prob': self.gaussian_inv_prob,
            'bernoulli_inv_prob': self.bernoulli_inv_prob,
            'vae_prob': self.vae_prob,
            'hash_count': self.hash_count_reward,
            'None': self.no_reward,
        }

        self.exploration_reward_func = (
            type_to_function[self.exploration_rewards_type]
        )
        self.vae_prioritization_func = (
            type_to_function[self.vae_priority_type]
        )

        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        if self.exploration_rewards_type == 'hash_count':
            if exploration_counter_kwargs is None:
                exploration_counter_kwargs = dict()
            self.exploration_counter = CountExploration(env=self.env, **exploration_counter_kwargs)
        self.epoch = 0

    def add_path(self, path):
        if self.decode_vae_goals:
            self.add_decoded_vae_goals_to_path(path)
        super().add_path(path)

    def add_decoded_vae_goals_to_path(self, path):
        # decoding the self-sampled vae images should be done in batch (here)
        # rather than in the env for efficiency
        desired_goals = combine_dicts(
            path['observations'],
            [self.desired_goal_key]
        )[self.desired_goal_key]
        desired_decoded_goals = self.env._decode(desired_goals)
        desired_decoded_goals = desired_decoded_goals.reshape(
            len(desired_decoded_goals),
            -1
        )
        for idx, next_obs in enumerate(path['observations']):
            path['observations'][idx][self.decoded_desired_goal_key] = \
                desired_decoded_goals[idx]
            path['next_observations'][idx][self.decoded_desired_goal_key] = \
                desired_decoded_goals[idx]

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        exploration_rewards_scale = float(self.explr_reward_scale_schedule.get_value(self.epoch))
        if self._give_explr_reward_bonus:
            batch_idxs = batch['indices'].flatten()
            batch['exploration_rewards'] = self._exploration_rewards[batch_idxs]
            batch['rewards'] += exploration_rewards_scale * batch['exploration_rewards']
        return batch

    def get_diagnostics(self):
        if self._vae_sample_probs is None or self._vae_sample_priorities is None:
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                np.zeros(self._size),
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                np.zeros(self._size),
            ))
        else:
            vae_sample_priorities = self._vae_sample_priorities[:self._size]
            vae_sample_probs = self._vae_sample_probs[:self._size]
            stats = create_stats_ordered_dict(
                'VAE Sample Weights',
                vae_sample_priorities,
            )
            stats.update(create_stats_ordered_dict(
                'VAE Sample Probs',
                vae_sample_probs,
            ))
        return stats

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

        cur_idx = 0
        obs_sum = np.zeros(self.vae.representation_size)
        obs_square_sum = np.zeros(self.vae.representation_size)
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.observation_key][idxs] = \
                self.env._encode(self._obs[self.decoded_obs_key][idxs])
            self._next_obs[self.observation_key][idxs] = \
                self.env._encode(self._next_obs[self.decoded_obs_key][idxs])
            # WARNING: we only refresh the desired/achieved latents for
            # "next_obs". This means that obs[desired/achieve] will be invalid,
            # so make sure there's no code that references this.
            # TODO: enforce this with code and not a comment
            self._next_obs[self.desired_goal_key][idxs] = \
                self.env._encode(self._next_obs[self.decoded_desired_goal_key][idxs])
            self._next_obs[self.achieved_goal_key][idxs] = \
                self.env._encode(self._next_obs[self.decoded_achieved_goal_key][idxs])
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
            obs_sum+= self._obs[self.observation_key][idxs].sum(axis=0)
            obs_square_sum+= np.power(self._obs[self.observation_key][idxs], 2).sum(axis=0)

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

    def sample_weighted_indices(self, batch_size):
        if (
            self._prioritize_vae_samples and
            self._vae_sample_probs is not None and
            self.skew
        ):
            indices = np.random.choice(
                len(self._vae_sample_probs),
                batch_size,
                p=self._vae_sample_probs,
            )
            assert (
                np.max(self._vae_sample_probs) <= 1 and
                np.min(self._vae_sample_probs) >= 0
            )
        else:
            indices = self._sample_indices(batch_size)
        return indices

    def _sample_goals_from_env(self, batch_size):
        self.env.goal_sampling_mode = self._relabeling_goal_sampling_mode
        return self.env.sample_goals(batch_size)

    def sample_buffer_goals(self, batch_size):
        """
        Samples goals from weighted replay buffer for relabeling or exploration.
        Returns None if replay buffer is empty.

        Example of what might be returned:
        dict(
            image_desired_goals: image_achieved_goals[weighted_indices],
            latent_desired_goals: latent_desired_goals[weighted_indices],
        )
        """
        if self._size == 0:
            return None
        weighted_idxs = self.sample_weighted_indices(
            batch_size,
        )
        next_image_obs = self._next_obs[self.decoded_obs_key][weighted_idxs]
        next_latent_obs = self._next_obs[self.achieved_goal_key][weighted_idxs]
        return {
            self.decoded_desired_goal_key:  next_image_obs,
            self.desired_goal_key:          next_latent_obs
        }

    def random_vae_training_data(self, batch_size, epoch):
        # epoch no longer needed. Using self.skew in sample_weighted_indices
        # instead.
        weighted_idxs = self.sample_weighted_indices(
            batch_size,
        )

        next_image_obs = self._next_obs[self.decoded_obs_key][weighted_idxs]
        observations = ptu.from_numpy(next_image_obs)
        return dict(
            observations=observations,
        )

    def reconstruction_mse(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)

        error = torch_input - recon_next_vae_obs
        mse = torch.sum(error ** 2, dim=1)
        return ptu.get_numpy(mse)

    def gaussian_inv_prob(self, next_vae_obs, indices):
        return np.exp(self.reconstruction_mse(next_vae_obs, indices))

    def binary_cross_entropy(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)

        error = - torch_input * torch.log(
            torch.clamp(
                recon_next_vae_obs,
                min=1e-30,  # corresponds to about -70
            )
        )
        bce = torch.sum(error, dim=1)
        return ptu.get_numpy(bce)

    def bernoulli_inv_prob(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)
        prob = (
                torch_input * recon_next_vae_obs
                + (1 - torch_input) * (1 - recon_next_vae_obs)
        ).prod(dim=1)
        return ptu.get_numpy(1 / prob)

    def vae_prob(self, next_vae_obs, indices, **kwargs):
        return compute_p_x_np_to_np(
            self.vae,
            next_vae_obs,
            power=self.power,
            **kwargs
        )

    def forward_model_error(self, next_vae_obs, indices):
        obs = self._obs[self.observation_key][indices]
        next_obs = self._next_obs[self.observation_key][indices]
        actions = self._actions[indices]

        state_action_pair = ptu.from_numpy(np.c_[obs, actions])
        prediction = self.dynamics_model(state_action_pair)
        mse = self.dynamics_loss(prediction, ptu.from_numpy(next_obs))
        return ptu.get_numpy(mse)

    def latent_novelty(self, next_vae_obs, indices):
        distances = ((self.env._encode(next_vae_obs) - self.vae.dist_mu) /
                     self.vae.dist_std) ** 2
        return distances.sum(axis=1)

    def latent_novelty_true_prior(self, next_vae_obs, indices):
        distances = self.env._encode(next_vae_obs) ** 2
        return distances.sum(axis=1)

    def _kl_np_to_np(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        mu, log_var = self.vae.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def update_hash_count(self, next_vae_obs):
        torch_input = ptu.from_numpy(next_vae_obs)
        mus, log_vars = self.vae.encode(torch_input)
        mus = ptu.get_numpy(mus)
        self.exploration_counter.increment_counts(mus)
        return None

    def hash_count_reward(self, next_vae_obs, indices):
        obs = self.env._encode(next_vae_obs)
        return self.exploration_counter.compute_count_based_reward(obs)

    def no_reward(self, next_vae_obs, indices):
        return np.zeros((len(next_vae_obs), 1))

    def initialize_dynamics_model(self):
        obs_dim = self._obs[self.observation_key].shape[1]
        self.dynamics_model = Mlp(
            hidden_sizes=[128, 128],
            output_size=obs_dim,
            input_size=obs_dim + self._action_dim,
        )
        self.dynamics_model.to(ptu.device)
        self.dynamics_optimizer = Adam(self.dynamics_model.parameters())
        self.dynamics_loss = MSELoss()

    def train_dynamics_model(self, batches=50, batch_size=100):
        if not self.use_dynamics_model:
            return
        for _ in range(batches):
            indices = self._sample_indices(batch_size)
            self.dynamics_optimizer.zero_grad()
            obs = self._obs[self.observation_key][indices]
            next_obs = self._next_obs[self.observation_key][indices]
            actions = self._actions[indices]
            if self.exploration_rewards_type == 'inverse_model_error':
                obs, next_obs = next_obs, obs

            state_action_pair = ptu.from_numpy(np.c_[obs, actions])
            prediction = self.dynamics_model(state_action_pair)
            mse = self.dynamics_loss(prediction, ptu.from_numpy(next_obs))

            mse.backward()
            self.dynamics_optimizer.step()

    def log_loss_under_uniform(self, model, data, batch_size, rl_logger, priority_function_kwargs):
        import torch.nn.functional as F
        log_probs_prior = []
        log_probs_biased = []
        log_probs_importance = []
        kles = []
        mses = []
        for i in range(0, data.shape[0], batch_size):
            img = data[i:min(data.shape[0], i + batch_size), :]
            torch_img = ptu.from_numpy(img)
            reconstructions, obs_distribution_params, latent_distribution_params = self.vae(torch_img)

            priority_function_kwargs['sampling_method'] = 'true_prior_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_prior = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'biased_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_biased = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'importance_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(model, img, **priority_function_kwargs)
            log_prob_importance = (log_p - log_q + log_d).mean()

            kle = model.kl_divergence(latent_distribution_params)
            mse = F.mse_loss(torch_img, reconstructions, reduction='elementwise_mean')
            mses.append(mse.item())
            kles.append(kle.item())
            log_probs_prior.append(log_prob_prior.item())
            log_probs_biased.append(log_prob_biased.item())
            log_probs_importance.append(log_prob_importance.item())

        rl_logger["Uniform Data Log Prob (Prior)"] = np.mean(log_probs_prior)
        rl_logger["Uniform Data Log Prob (Biased)"] = np.mean(log_probs_biased)
        rl_logger["Uniform Data Log Prob (Importance)"] = np.mean(log_probs_importance)
        rl_logger["Uniform Data KL"] = np.mean(kles)
        rl_logger["Uniform Data MSE"] = np.mean(mses)

    def _get_sorted_idx_and_train_weights(self):
        idx_and_weights = zip(range(len(self._vae_sample_probs)),
                              self._vae_sample_probs)
        return sorted(idx_and_weights, key=lambda x: x[1])

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
    data,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype != np.uint8, 'images should be normalized'
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    representation_size = model.representation_size
    batch_size = data.shape[0]
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, representation_size)),
                        ptu.ones((batch_size, representation_size)))
    mus, logvars = latent_distribution_params
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)

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
    data,
    power,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype != np.uint8, 'images should be normalized'
    assert power >= -1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
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
