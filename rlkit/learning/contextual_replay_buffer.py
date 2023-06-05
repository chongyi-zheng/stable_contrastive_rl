import numpy as np

from rlkit.envs.contextual import ContextualRewardFn
from rlkit import pythonplusplus as ppp
import rlkit.data_management.images as image_np

from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from rlkit.data_management.obs_dict_replay_buffer import (
    combine_dicts,
)

from rlkit.util.augment_util import create_aug_stack  # NOQA


class ContextualRelabelingReplayBuffer(ContextualRelabelingReplayBuffer):
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
            fraction_distribution_context,
            fraction_next_context=0.,
            fraction_replay_buffer_context=0.0,
            discount=0.99,

            post_process_batch_fn=None,

            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            observation_key_reward_fn=None,
            save_data_in_snapshot=False,
            internal_keys=None,
            context_keys_to_save=[],

            neg_from_the_same_traj=False,

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

        self._discount = discount
        self._neg_from_the_same_traj = neg_from_the_same_traj

        # need _ImageNumpyArr wrapper. It will slow down sampling.
        for key in self.ob_keys_to_save + self.internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            module = np
            arr_initializer = module.zeros
            self._obs[key] = arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=np.uint8 if key.startswith('image') else self.ob_spaces[key].dtype,
            )
            self._next_obs[key] = arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=np.uint8 if key.startswith('image') else self.ob_spaces[key].dtype,
            )

    def add_path(self, path, ob_dicts_already_combined=False):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        if not ob_dicts_already_combined:
            obs = combine_dicts(obs, self.ob_keys_to_save + self.internal_keys)
            next_obs = combine_dicts(
                next_obs, self.ob_keys_to_save + self.internal_keys)

        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = (
                np.s_[self._top:self._top + num_pre_wrap_steps, ...]
            )
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, ...]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    if key.startswith('image'):
                        self._obs[key][buffer_slice] = image_np.unnormalize_image(obs[key][path_slice])
                        self._next_obs[key][buffer_slice] = image_np.unnormalize_image(next_obs[key][path_slice])
                    else:
                        self._obs[key][buffer_slice] = obs[key][path_slice]
                        self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = [i, num_post_wrap_steps]
                self._idx_to_num_steps[i] = i - self._top
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = [i, num_post_wrap_steps]
                self._idx_to_num_steps[i] = (i - self._top) % self.max_size

        else:
            slc = np.s_[self._top:self._top + path_len, ...]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            self._rewards[slc] = rewards

            for key in self.ob_keys_to_save + self.internal_keys:
                if key.startswith('image'):
                    self._obs[key][slc] = image_np.unnormalize_image(obs[key])
                    self._next_obs[key][slc] = image_np.unnormalize_image(next_obs[key])
                else:
                    self._obs[key][slc] = obs[key]
                    self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = [i, self._top + path_len]
                self._idx_to_num_steps[i] = i - self._top

        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _batch_obs_dict(self, indices):
        pre_obs_batch = {
            key: self._obs[key]
            for key in self.ob_keys_to_save
        }
        obs_batch = ppp.treemap(
            lambda x: x.take(indices, axis=0),
            pre_obs_batch,
            atomic_type=np.ndarray)

        return obs_batch

    def _batch_next_obs_dict(self, indices):
        pre_next_obs_batch = {
            key: self._next_obs[key]
            for key in self.ob_keys_to_save
        }
        next_obs_batch = ppp.treemap(
            lambda x: x.take(indices, axis=0),
            pre_next_obs_batch,
            atomic_type=np.ndarray)

        return next_obs_batch

    def random_batch(self, batch_size):
        if self._neg_from_the_same_traj:
            indices = self._sample_indices_from_traj(batch_size)
        else:
            indices = self._sample_indices(batch_size)

        obs_dict = self._batch_obs_dict(indices)
        next_obs_dict = self._batch_next_obs_dict(indices)

        if batch_size > 0:
            future_contexts_base = self._get_future_contexts(
                indices)
            future_contexts = {
                k: future_contexts_base[k]
                for k in self._context_keys}

            future_contexts_to_save = {
                k: future_contexts_base[k]
                for k in self._context_keys_to_save}
        else:
            future_contexts = {
                k: obs_dict[k][:0] for k in self._context_keys}
            future_contexts_to_save = {
                k: obs_dict[k][:0] for k in self._context_keys_to_save}

        actions = self._actions.take(indices, axis=0)

        new_contexts = future_contexts
        new_contexts_to_save = future_contexts_to_save

        if len(self.observation_keys) == 1:
            obs = obs_dict[self.observation_keys[0]]
            next_obs = next_obs_dict[self.observation_keys[0]]
        else:
            obs = tuple(obs_dict[k] for k in self.observation_keys)
            next_obs = tuple(next_obs_dict[k] for k in self.observation_keys)

        observations_to_save = {
            k: obs_dict[k]
            for k in self.ob_keys_to_save
            if k not in self._context_keys + self._context_keys_to_save
        }
        next_observations_to_save = {
            'next_' + k: next_obs_dict[k]
            for k in self.ob_keys_to_save
            if k not in self._context_keys + self._context_keys_to_save
        }

        batch = {
            'observations': obs,
            'actions': actions,
            'rewards': self._rewards.take(indices, axis=0),
            'terminals': self._terminals.take(indices, axis=0),
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
            **observations_to_save,
            **next_observations_to_save,
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

    def _sample_indices_from_traj(self, batch_size, min_dt=None):
        """
        Sample as many transitions from the same trajectory as possible
        """
        if self._size == 0:
            return []

        # simply sample from the same trajectory now
        rand_idx = np.random.randint(0, self._size, 1)[0]
        traj_idx = self._idx_to_future_obs_idx[rand_idx][1]
        lb, ub = self._idx_to_future_obs_idx[traj_idx]
        if ub <= lb:
            ub = ub + self.max_size

        return np.random.randint(lb, ub, batch_size) % self.max_size

    def _get_future_obs_indices(self, start_state_indices, max_dt=None):
        possible_future_obs_idxs = self._idx_to_future_obs_idx[start_state_indices]
        lb, ub = possible_future_obs_idxs[:, 0], possible_future_obs_idxs[:, 1]
        if max_dt is not None:
            path_len = (ub - lb) % self.max_size
            ub = np.where(path_len >= max_dt, ((lb + max_dt) % max_dt).astype(int), ub)

        ub = np.where(ub <= lb, ub + self.max_size, ub)

        path_len = ub - lb
        # reference: https://github.com/google-research/google-research/blob/d39dfbb5d1256cc930d3946a933aa5dc5d416ad5/contrastive_rl/contrastive/builder.py#L133
        arange = np.arange(max(path_len))
        indices = np.repeat(
            arange[None],
            path_len.shape[0], axis=0)
        probs = self._discount ** indices.astype(float)
        is_future_mask = np.where(
            np.cumsum(np.ones_like(indices), axis=1) <= path_len[:, None],
            np.ones_like(indices), np.zeros_like(indices))
        probs = probs * is_future_mask
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # reference: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035#34190035
        probs_cumsum = probs.cumsum(axis=1)
        rand = np.random.rand(probs_cumsum.shape[0])[:, None]
        future_obs_idxs = (lb + arange[(probs_cumsum < rand).sum(axis=1)]) % self.max_size

        return future_obs_idxs

    def _get_future_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(start_state_indices)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)
