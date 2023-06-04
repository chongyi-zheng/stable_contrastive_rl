import numpy as np
import torch  # NOQA

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.contextual import ContextualRewardFn
from rlkit import pythonplusplus as ppp

from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)

from rlkit.util.augment_util import create_aug_stack  # NOQA
from rlkit.experimental.kuanfang.utils.logging import logger as logging  # NOQA


def concat(*x):
    return np.concatenate(x, axis=0)


class AffordanceReplayBuffer(ContextualRelabelingReplayBuffer):
    """
    """

    def __init__(
            self,
            max_size,
            env,
            context_keys,
            observation_keys_to_save,
            sample_context_from_obs_dict_fn,
            reward_fn: ContextualRewardFn,
            context_distribution,
            fraction_future_context,
            fraction_next_context=0.,
            fraction_last_context=0.,
            fraction_distribution_context=0.,
            fraction_replay_buffer_context=0.,

            min_future_dt=None,
            max_future_dt=None,

            max_previous_dt=None,
            max_last_dt=None,

            # Affordance perturbed context.
            fraction_foresight_context=0.,
            fraction_perturbed_context=0.,
            vqvae=None,
            affordance=None,
            noise_level=None,
            post_process_batch_fn=None,

            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            observation_key_reward_fn=None,
            save_data_in_snapshot=False,
            internal_keys=None,
            context_keys_to_save=[],

            imsize=48,

            **kwargs
    ):
        super().__init__(
            max_size=max_size,
            env=env,
            context_keys=context_keys,
            observation_keys_to_save=observation_keys_to_save,
            sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
            reward_fn=reward_fn,
            context_distribution=context_distribution,
            fraction_future_context=fraction_future_context,
            fraction_distribution_context=fraction_distribution_context,
            fraction_next_context=fraction_next_context,
            fraction_replay_buffer_context=fraction_replay_buffer_context,
            post_process_batch_fn=post_process_batch_fn,
            observation_key=observation_key,
            observation_keys=observation_keys,
            observation_key_reward_fn=observation_key_reward_fn,
            save_data_in_snapshot=save_data_in_snapshot,
            internal_keys=internal_keys,
            context_keys_to_save=context_keys_to_save,
            **kwargs)

        if (
                fraction_distribution_context < 0
                or fraction_future_context < 0
                or fraction_next_context < 0
                or fraction_last_context < 0
                or fraction_replay_buffer_context < 0
                or fraction_foresight_context < 0
                or fraction_perturbed_context < 0
                or (fraction_future_context
                    + fraction_next_context
                    + fraction_last_context
                    + fraction_distribution_context
                    + fraction_replay_buffer_context
                    + fraction_foresight_context
                    + fraction_perturbed_context) > 1
        ):
            raise ValueError

        self._fraction_last_context = fraction_last_context

        self._fraction_perturbed_context = fraction_perturbed_context
        self._fraction_foresight_context = fraction_foresight_context

        self._affordance = affordance
        self._vqvae = vqvae
        self._min_future_dt = min_future_dt
        self._max_future_dt = max_future_dt
        self._max_previous_dt = max_previous_dt
        self._max_last_dt = max_last_dt
        self._noise_level = noise_level

    def _sample_indices(self,
                        batch_size,
                        min_future_dt=None,
                        max_previous_dt=None,
                        ):
        assert not ((
            min_future_dt is not None) and (max_previous_dt is not None))

        if min_future_dt is not None:
            all_indices = np.arange(self._size)
            lengths_to_end = (
                self._idx_to_future_obs_idx[:self._size, 1] -
                all_indices) % self.max_size
            candidates = all_indices[lengths_to_end >= min_future_dt]
            return np.random.choice(candidates, batch_size)

        elif max_previous_dt is None:
            all_indices = np.arange(self._size)
            idx_num_steps = self._idx_to_num_steps[:self._size, 1]
            candidates = all_indices[idx_num_steps <= max_previous_dt]
            return np.random.choice(candidates, batch_size)

        else:
            return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        num_distrib_contexts = int(
            batch_size * self._fraction_distribution_context)
        num_replay_buffer_contexts = int(
            batch_size * self._fraction_replay_buffer_context
        )
        num_future_contexts = int(
            batch_size * self._fraction_future_context)
        num_next_contexts = int(
            batch_size * self._fraction_next_context)
        num_last_contexts = int(
            batch_size * self._fraction_last_context)
        num_foresight_contexts = int(
            batch_size * self._fraction_foresight_context)
        num_perturbed_contexts = int(
            batch_size * self._fraction_perturbed_context)
        num_rollout_contexts = (
            batch_size
            - num_distrib_contexts
            - num_replay_buffer_contexts
            - num_future_contexts
            - num_next_contexts
            - num_last_contexts
            - num_foresight_contexts
            - num_perturbed_contexts
        )

        indices = self._sample_indices(batch_size,
                                       self._min_future_dt,
                                       self._max_previous_dt)

        obs_dict = self._batch_obs_dict(indices)
        next_obs_dict = self._batch_next_obs_dict(indices)

        context_source_dict = {
            'rollout': num_rollout_contexts,
            'distrib': num_distrib_contexts,
            'replay_buffer': num_replay_buffer_contexts,
            'future': num_future_contexts,
            'next': num_next_contexts,
            'last': num_last_contexts,
            'foresight': num_foresight_contexts,
            'perturbed': num_perturbed_contexts,
        }
        assert sum(context_source_dict.values()) == batch_size

        contexts = []
        contexts_to_save = []
        start_idx = 0
        end_idx = num_rollout_contexts
        for context_source, num_samples in context_source_dict.items():
            start_idx = end_idx
            end_idx = start_idx + num_samples

            if num_samples <= 0:
                continue

            if context_source == 'rollout':
                curr_obs = {
                    k: obs_dict[k][start_idx:end_idx]
                    for k in self.ob_keys_to_save  # self.observation_keys
                }
                context_obs_dict = curr_obs

            elif context_source == 'distrib':
                curr_obs = {
                    k: obs_dict[k][start_idx:end_idx]
                    for k in self.ob_keys_to_save  # self.observation_keys
                }
                context_obs_dict = self._context_distribution(
                    context=curr_obs).sample(num_distrib_contexts)

            elif context_source == 'replay_buffer':
                context_obs_dict = self._get_replay_buffer_contexts(
                    num_samples)

            elif context_source == 'next':
                start_state_indices = indices[start_idx:end_idx]
                context_obs_dict = self._get_next_contexts(
                    start_state_indices)

            elif context_source == 'future':
                start_state_indices = indices[start_idx:end_idx]
                context_obs_dict = self._get_future_contexts(
                    start_state_indices)

            elif context_source == 'last':
                start_state_indices = indices[start_idx:end_idx]
                context_obs_dict = self._get_last_contexts(
                    start_state_indices)

            elif context_source == 'foresight':
                start_state_indices = indices[start_idx:end_idx]
                context_obs_dict = self._get_foresight_contexts(
                    start_state_indices)

            elif context_source == 'perturbed':
                start_state_indices = indices[start_idx:end_idx]
                context_obs_dict = self._get_perturbed_contexts(
                    start_state_indices)

            else:
                raise ValueError

            sampled_contexts = {
                k: context_obs_dict[k]
                for k in self._context_keys}
            contexts.append(sampled_contexts)

            sampled_contexts_to_save = {
                k: context_obs_dict[k]
                for k in self._context_keys_to_save}
            contexts_to_save.append(sampled_contexts_to_save)

        actions = self._actions[indices]

        new_contexts = ppp.treemap(
            concat,
            *tuple(contexts),
            atomic_type=np.ndarray)

        new_contexts_to_save = ppp.treemap(
            concat,
            *tuple(contexts_to_save),
            atomic_type=np.ndarray)

        if len(self.observation_keys) == 1:
            obs = obs_dict[self.observation_keys[0]]
            next_obs = next_obs_dict[self.observation_keys[0]]
        else:
            obs = tuple(obs_dict[k] for k in self.observation_keys)
            next_obs = tuple(next_obs_dict[k] for k in self.observation_keys)

        batch = {
            'observations': obs,
            'actions': actions,
            'rewards': self._rewards[indices],
            'terminals': self._terminals[indices],
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
            **new_contexts,
            **new_contexts_to_save,
        }

        new_batch = self._post_process_batch_fn(
            batch,
            self,
            obs_dict,
            next_obs_dict,
            new_contexts,
            new_contexts_to_save
        )

        return new_batch

    def _get_future_obs_indices(
            self, start_state_indices, min_dt=None, max_dt=None):
        future_obs_idxs = []
        for i in start_state_indices:
            possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
            lb, ub = possible_future_obs_idxs
            assert ub != lb

            if ub < lb:
                ub = ub + self.max_size

            if min_dt is not None:
                assert max_dt >= 0
                lb = max(lb + min_dt - 1, lb)

            if max_dt is not None:
                assert max_dt >= 0
                ub = min(lb + max_dt, ub)

            assert ub > lb, 'i: %d, lb: %d, ub: %d' % (i, lb, ub)

            next_obs_i = int(np.random.randint(lb, ub)) % self.max_size

            future_obs_idxs.append(next_obs_i)

        future_obs_idxs = np.array(future_obs_idxs)
        return future_obs_idxs

    def _get_future_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(
            start_state_indices,
            min_dt=self._min_future_dt,
            max_dt=self._max_future_dt)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)

    def _get_last_obs_indices(self, start_state_indices, max_dt=None):
        last_obs_idxs = []
        for i in start_state_indices:
            possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
            lb, ub = possible_future_obs_idxs
            assert ub != lb

            if ub < lb:
                ub = ub + self.max_size

            if max_dt is not None:
                lb = max(ub - max_dt, lb)

            assert lb >= i, 'i: %d, lb: %d, ub: %d' % (i, lb, ub)
            assert ub > lb, 'i: %d, lb: %d, ub: %d' % (i, lb, ub)

            next_obs_i = int(np.random.randint(lb, ub)) % self.max_size

            last_obs_idxs.append(next_obs_i)

        last_obs_idxs = np.array(last_obs_idxs)
        return last_obs_idxs

    def _get_last_contexts(self, start_state_indices):
        future_obs_idxs = self._get_last_obs_indices(
            start_state_indices,
            max_dt=self._max_last_dt)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)

    def _get_foresight_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(
            start_state_indices, max_dt=self._max_future_dt)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)

        assert len(self.observation_keys) == 1
        observation_key = self.observation_keys[0]

        h0 = future_obs_dict[observation_key]
        h0 = ptu.from_numpy(h0)
        h0 = h0.view(
            -1,
            self._vqvae.embedding_dim,
            self._vqvae.root_len,
            self._vqvae.root_len)

        z = self._affordance.sample_prior(h0.shape[0])
        z = ptu.from_numpy(z)
        h1_pred = self._affordance.decode(z, cond=h0).detach()
        _, h1_pred = self._vqvae.vector_quantizer(h1_pred)

        # Debug.
        if 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 1, 1)
            s0 = self._vqvae.decode(h0)
            image = s0 + 0.5
            image = image[0]
            image = image.permute(1, 2, 0).contiguous()
            image = ptu.get_numpy(image)
            plt.imshow(image)

            plt.subplot(2, 1, 2)
            s1_pred = self._vqvae.decode(h1_pred)
            image = s1_pred + 0.5
            image = image[0]
            image = image.permute(1, 2, 0).contiguous()
            image = ptu.get_numpy(image)
            plt.imshow(image)

            plt.show()

        h1_pred = h1_pred.view(-1, self._vqvae.representation_size)
        h1_pred = ptu.get_numpy(h1_pred)

        foresight_obs_dict = {}
        for key, value in future_obs_dict.items():
            if key == observation_key:
                foresight_obs_dict[key] = h1_pred
            else:
                foresight_obs_dict[key] = value

        return self._sample_context_from_obs_dict_fn(foresight_obs_dict)

    def _get_perturbed_contexts(self, start_state_indices):
        obs_dict = self._batch_obs_dict(start_state_indices)
        future_obs_idxs = self._get_future_obs_indices(
            start_state_indices, max_dt=self._max_future_dt)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)

        assert len(self.observation_keys) == 1
        observation_key = self.observation_keys[0]

        h0 = obs_dict[observation_key]
        h0 = ptu.from_numpy(h0)
        h0 = h0.view(
            -1,
            self._vqvae.embedding_dim,
            self._vqvae.root_len,
            self._vqvae.root_len)

        h1 = future_obs_dict[observation_key]
        h1 = ptu.from_numpy(h1)
        h1 = h1.view(
            -1,
            self._vqvae.embedding_dim,
            self._vqvae.root_len,
            self._vqvae.root_len)

        z_mu, z_logvar = self._affordance.encode(h1, cond=h0)
        std = z_logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        z = eps.mul(std).add_(z_mu)
        noisy_z = z + self._noise_level * ptu.randn(
            1, self._affordance.representation_size)

        h1_pred = self._affordance.decode(noisy_z, cond=h0).detach()
        _, h1_pred = self._vqvae.vector_quantizer(h1_pred)

        # Debug.
        if 0:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 1, 1)
            s1 = self._vqvae.decode(h1)
            image = s1 + 0.5
            image = image[0]
            image = image.permute(1, 2, 0).contiguous()
            image = ptu.get_numpy(image)
            plt.imshow(image)

            plt.subplot(2, 1, 2)
            s1_pred = self._vqvae.decode(h1_pred)
            image = s1_pred + 0.5
            image = image[0]
            image = image.permute(1, 2, 0).contiguous()
            image = ptu.get_numpy(image)
            plt.imshow(image)

            plt.show()

        h1_pred = h1_pred.view(-1, self._vqvae.representation_size)
        h1_pred = ptu.get_numpy(h1_pred)

        perturbed_obs_dict = {}
        for key, value in future_obs_dict.items():
            if key == observation_key:
                perturbed_obs_dict[key] = h1_pred
            else:
                perturbed_obs_dict[key] = value

        return self._sample_context_from_obs_dict_fn(perturbed_obs_dict)
