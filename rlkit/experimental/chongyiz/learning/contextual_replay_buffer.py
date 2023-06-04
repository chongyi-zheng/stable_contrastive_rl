# import abc
# from typing import Any, Dict

import numpy as np
import torch  # NOQA

import rlkit.torch.pytorch_util as ptu
# from rlkit.core.distribution import DictDistribution
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
# import pickle
# from PIL import Image
# import time


# def concat(*x):
#     return np.concatenate(x, axis=0)


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
            # max_future_dt=None,
            discount=0.99,

            # Affordance perturbed context.
            # fraction_foresight_context=0.0,
            # fraction_perturbed_context=0.0,
            # vqvae=None,
            # affordance=None,
            # noise_level=None,
            post_process_batch_fn=None,

            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            observation_key_reward_fn=None,
            save_data_in_snapshot=False,
            internal_keys=None,
            context_keys_to_save=[],

            # imsize=48,

            # augment_params=dict(),
            # augment_order=[],
            # augment_probability=0.0,
            # reencode_augmented_images=False,
            # image_to_latent_map=dict(),

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

        # (chongyiz): reinitialize obs and next_obs buffer cause we don't
        # need _ImageNumpyArr wrapper. It will slow down sampling.
        for key in self.ob_keys_to_save + self.internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            # DELETEM (chongyiz): module = image_np if key.startswith('image') else np
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

        # self._fraction_perturbed_context = fraction_perturbed_context
        # self._fraction_foresight_context = fraction_foresight_context

        # self._affordance = affordance
        # self._vqvae = vqvae
        # self._max_future_dt = max_future_dt
        # self._noise_level = noise_level

        # self.augment_stack = create_aug_stack(
        #     augment_order, augment_params, size=(imsize, imsize)
        # )
        # self.augment_probability = augment_probability
        # self.reencode_augmented_images = reencode_augmented_images
        # self.image_to_latent_map = image_to_latent_map
        # self._imsize = imsize

    # def set_augment_params(self, img):
    #     if torch.rand(1) < self.augment_probability:
    #         self.augment_stack.set_params(img)
    #     else:
    #         self.augment_stack.set_default_params(img)
    #
    # def augment(self, img):
    #     img = self.augment_stack(img)
    #     return img

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
        # return {
        #     key: self._next_obs[key][indices]
        #     for key in self.ob_keys_to_save
        # }

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
        # num_distrib_contexts = int(
        #     batch_size * self._fraction_distribution_context)
        # num_replay_buffer_contexts = int(
        #     batch_size * self._fraction_replay_buffer_context
        # )
        # num_next_contexts = int(batch_size * self._fraction_next_context)
        # num_future_contexts = int(batch_size * self._fraction_future_context)
        # num_foresight_contexts = int(
        #     batch_size * self._fraction_foresight_context)
        # num_perturbed_contexts = int(
        #     batch_size * self._fraction_perturbed_context)
        # num_rollout_contexts = (
        #     batch_size
        #     - num_future_contexts
        #     - num_distrib_contexts
        #     - num_next_contexts
        #     - num_replay_buffer_contexts
        #     # - num_foresight_contexts
        #     # - num_perturbed_contexts
        # )

        # torch.cuda.synchronize()
        # start_time = time.time()
        if self._neg_from_the_same_traj:
            indices = self._sample_indices_from_traj(batch_size)
        else:
            indices = self._sample_indices(batch_size)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time to sample indices: {} secs".format(end_time - start_time))

        # torch.cuda.synchronize()
        # start_time = time.time()
        obs_dict = self._batch_obs_dict(indices)
        next_obs_dict = self._batch_next_obs_dict(indices)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time to construct obs_dict: {} secs".format(end_time - start_time))

        # contexts = [{
        #     k: obs_dict[k][:num_rollout_contexts]
        #     for k in self._context_keys
        # }]
        # contexts_to_save = [{
        #     k: obs_dict[k][:num_rollout_contexts]
        #     for k in self._context_keys_to_save
        # }]

        # if num_distrib_contexts > 0:
        #     curr_obs = {
        #         k: obs_dict[k][
        #             num_rollout_contexts:
        #             num_rollout_contexts + num_distrib_contexts]
        #         for k in self.ob_keys_to_save  # self.observation_keys
        #     }
        #     sampled_contexts_base = self._context_distribution(
        #         context=curr_obs).sample(num_distrib_contexts, )
        #
        #     sampled_contexts = {
        #         k: sampled_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(sampled_contexts)
        #
        #     sampled_contexts_to_save = {
        #         k: sampled_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(sampled_contexts_to_save)
        #
        # if num_replay_buffer_contexts > 0:
        #     replay_buffer_contexts_base = self._get_replay_buffer_contexts(
        #         num_replay_buffer_contexts,
        #     )
        #
        #     replay_buffer_contexts = {
        #         k: replay_buffer_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(replay_buffer_contexts)
        #
        #     replay_buffer_contexts_to_save = {
        #         k: replay_buffer_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(replay_buffer_contexts_to_save)
        #
        # if num_next_contexts > 0:
        #     # start_idx = -(num_next_contexts +
        #     #               num_future_contexts +
        #     #               num_foresight_contexts +
        #     #               num_perturbed_contexts)
        #     start_idx = -(num_next_contexts +
        #                   num_future_contexts)
        #     end_idx = start_idx + num_next_contexts
        #     if end_idx == 0:
        #         start_state_indices = indices[start_idx:]
        #     else:
        #         start_state_indices = indices[start_idx:end_idx]
        #     next_contexts_base = self._get_next_contexts(start_state_indices)
        #
        #     next_contexts = {
        #         k: next_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(next_contexts)
        #
        #     next_contexts_to_save = {
        #         k: next_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(next_contexts_to_save)

        # torch.cuda.synchronize()
        # start_time = time.time()
        # # if num_future_contexts > 0:
        #     # start_idx = -(num_future_contexts +
        #     #               num_foresight_contexts +
        #     #               num_perturbed_contexts)
        #     # start_idx = -num_future_contexts
        #     # end_idx = start_idx + num_future_contexts
        #     # if end_idx == 0:
        #     #     start_state_indices = indices[start_idx:]
        #     # else:
        #     #     start_state_indices = indices[start_idx:end_idx]
        #     future_contexts_base = self._get_future_contexts(
        #         indices)
        #
        #     future_contexts = {
        #         k: future_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(future_contexts)
        #
        #     future_contexts_to_save = {
        #         k: future_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(future_contexts_to_save)

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

        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time to construct future contexts: {} secs".format(end_time - start_time))

        # if num_foresight_contexts > 0:
        #     start_idx = -(num_foresight_contexts +
        #                   num_perturbed_contexts)
        #     end_idx = start_idx + num_foresight_contexts
        #     if end_idx == 0:
        #         start_state_indices = indices[start_idx:]
        #     else:
        #         start_state_indices = indices[start_idx:end_idx]
        #     foresight_contexts_base = self._get_foresight_contexts(
        #         start_state_indices)
        #
        #     foresight_contexts = {
        #         k: foresight_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(foresight_contexts)
        #
        #     foresight_contexts_to_save = {
        #         k: foresight_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(foresight_contexts_to_save)
        #
        # if num_perturbed_contexts > 0:
        #     start_state_indices = indices[-num_perturbed_contexts:]
        #     perturbed_contexts_base = self._get_perturbed_contexts(
        #         start_state_indices)
        #
        #     perturbed_contexts = {
        #         k: perturbed_contexts_base[k]
        #         for k in self._context_keys}
        #     contexts.append(perturbed_contexts)
        #
        #     perturbed_contexts_to_save = {
        #         k: perturbed_contexts_base[k]
        #         for k in self._context_keys_to_save}
        #     contexts_to_save.append(perturbed_contexts_to_save)

        # torch.cuda.synchronize()
        # start_time = time.time()
        actions = self._actions.take(indices, axis=0)
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time to construct actions: {}".format(end_time - start_time))

        # start_time = time.time()
        # new_contexts = ppp.treemap(
        #     concat,
        #     *tuple(contexts),
        #     atomic_type=np.ndarray)
        #
        # new_contexts_to_save = ppp.treemap(
        #     concat,
        #     *tuple(contexts_to_save),
        #     atomic_type=np.ndarray)
        new_contexts = future_contexts
        new_contexts_to_save = future_contexts_to_save
        # end_time = time.time()
        # print("Time for ppp.treemap: {} secs".format(end_time - start_time))

        # For debugging
        # if len(self.observation_keys) == 1:
        #     obs = obs_dict[self.observation_keys[0]]
        #     next_obs = next_obs_dict[self.observation_keys[0]]
        # else:
        #     obs = tuple(obs_dict[k] for k in self.observation_keys)
        #     next_obs = tuple(next_obs_dict[k] for k in self.observation_keys)
        # if self.augment_probability > 0 and obs_dict['image_observation'].shape[0] > 0:  # NOQA
        #     import pickle
        #     pickle.dump(obs, open("orig_obs.pkl", "wb"))
        #     pickle.dump(next_obs, open("orig_next_obs.pkl", "wb"))
        #     pickle.dump(new_contexts, open("orig_new_contexts.pkl", "wb"))

        # if (self.augment_probability > 0 and
        #         obs_dict['image_observation'].shape[0] > 0):
        #
        #     self.set_augment_params(ptu.from_numpy(
        #         obs_dict['image_observation'].reshape(
        #             -1, 3, self._imsize, self._imsize)))
        #
        #     obs_dict['image_observation'] = ptu.from_numpy(
        #         obs_dict['image_observation'].reshape(
        #             -1, 3, self._imsize, self._imsize))
        #     next_obs_dict['image_observation'] = ptu.from_numpy(
        #         next_obs_dict['image_observation'].reshape(
        #             -1, 3, self._imsize, self._imsize))
        #     new_contexts['image_desired_goal'] = ptu.from_numpy(
        #         new_contexts['image_desired_goal'].reshape(
        #             -1, 3, self._imsize, self._imsize))
        #
        #     obs_dict['image_observation'] = self.augment(
        #         obs_dict['image_observation'])
        #     next_obs_dict['image_observation'] = self.augment(
        #         next_obs_dict['image_observation'])
        #     new_contexts['image_desired_goal'] = self.augment(
        #         new_contexts['image_desired_goal'])
        #
        #     obs_dict['image_observation'] = (
        #         obs_dict['image_observation'].reshape(
        #             -1, 3 * self._imsize * self._imsize)
        #     )
        #     next_obs_dict['image_observation'] = (
        #         next_obs_dict['image_observation'].reshape(
        #             -1, 3 * self._imsize * self._imsize)
        #     )
        #     new_contexts['image_desired_goal'] = (
        #         new_contexts['image_desired_goal'].reshape(
        #             -1, 3 * self._imsize * self._imsize)
        #     )
        #
        #     # for image_k, latent_k in self.image_to_latent_map.items():
        #     #     obs_dict[image_k] = ptu.from_numpy(
        #     #         obs_dict[image_k].reshape(
        #     #             -1, 3, self._imsize, self._imsize))
        #     #     next_obs_dict[image_k] = ptu.from_numpy(
        #     #         next_obs_dict[image_k].reshape(
        #     #             -1, 3, self._imsize, self._imsize))
        #     #     if image_k in new_contexts.keys():
        #     #         new_contexts[image_k] = ptu.from_numpy(
        #     #             new_contexts[image_k].reshape(
        #     #                 -1, 3, self._imsize, self._imsize))
        #     #
        #     #     obs_dict[image_k] = self.augment(obs_dict[image_k])
        #     #     next_obs_dict[image_k] = self.augment(
        #     #         next_obs_dict[image_k])
        #     #     if image_k in new_contexts.keys():
        #     #         new_contexts[image_k] = self.augment(
        #     #             new_contexts[image_k])
        #     #
        #     #     obs_dict[image_k] = ptu.get_numpy(
        #     #         obs_dict[image_k]).reshape(
        #     #             -1, 3 * self._imsize * self._imsize)
        #     #     next_obs_dict[image_k] = ptu.get_numpy(
        #     #         next_obs_dict[image_k]).reshape(
        #     #             -1, 3 * self._imsize * self._imsize)
        #     #     if image_k in new_contexts.keys():
        #     #         new_contexts[image_k] = ptu.get_numpy(
        #     #             new_contexts[image_k]).reshape(
        #     #                 -1, 3 * self._imsize * self._imsize)
        #
        #     # Re-encode latents with augmented images
        #     # if self.reencode_augmented_images:
        #     #     for image_k, latent_k in self.image_to_latent_map.items():
        #     #         obs_dict[latent_k] = self._vqvae.encode_np(
        #     #             obs_dict[image_k])
        #     #         next_obs_dict[latent_k] = self._vqvae.encode_np(
        #     #             next_obs_dict[image_k])
        #     #         if latent_k in new_contexts.keys():
        #     #             new_contexts[latent_k] = self._vqvae.encode_np(
        #     #                 new_contexts[image_k])

        # torch.cuda.synchronize()
        # start_time = time.time()
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

        # For debugging
        # if self.augment_probability > 0 and obs_dict['image_observation'].shape[0] > 0:  # NOQA
        #     import pickle
        #     pickle.dump(obs, open("obs.pkl", "wb"))
        #     pickle.dump(next_obs, open("next_obs.pkl", "wb"))
        #     pickle.dump(new_contexts, open("new_contexts.pkl", "wb"))
        #     assert False

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
        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time for post processing: {} secs".format(end_time - start_time))

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
        # import time
        # start_time = time.time()
        # future_obs_idxs = []
        # for i in start_state_indices:
        #     possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
        #     lb, ub = possible_future_obs_idxs
        #
        #     if max_dt is not None:
        #         path_len = (ub - lb) % self.max_size
        #         if path_len >= max_dt:
        #             ub = int((lb + max_dt) % max_dt)
        #
        #     # (chongyiz): use truncated geometric distribution instead of uniform distribution
        #     # reference: https://github.com/google-research/google-research/blob/483af463945c75fa42229b0bb817302482348831/c_learning/c_learning_utils.py#L26
        #     if ub <= lb:
        #         ub = ub + self.max_size  # TODO (chongyiz): check this line of code
        #
        #     path_len = ub - lb
        #     indices = np.arange(path_len)
        #     probs = self._discount ** indices.astype(float)
        #     probs = probs / np.sum(probs)
        #     next_obs_i = (lb + np.random.choice(indices, p=probs)) % self.max_size
        #     # next_obs_i = lb + np.random.choice(indices, p=probs)
        #
        #     future_obs_idxs.append(next_obs_i)
        # future_obs_idxs = np.array(future_obs_idxs)
        # end_time = time.time()
        # print("Time of for loop _get_future_obs_indices: {} secs".format(end_time - start_time))

        # start_time = time.time()
        possible_future_obs_idxs = self._idx_to_future_obs_idx[start_state_indices]
        lb, ub = possible_future_obs_idxs[:, 0], possible_future_obs_idxs[:, 1]
        if max_dt is not None:
            # TODO (chongyiz): check the following code
            path_len = (ub - lb) % self.max_size
            ub = np.where(path_len >= max_dt, ((lb + max_dt) % max_dt).astype(int), ub)

        ub = np.where(ub <= lb, ub + self.max_size, ub)

        path_len = ub - lb
        # (chongyiz): use truncated geometric distribution instead of uniform distribution
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
        # print("Time of batch _get_future_obs_indices: {} secs".format(end_time - start_time))

        return future_obs_idxs

    def _get_future_contexts(self, start_state_indices):
        future_obs_idxs = self._get_future_obs_indices(start_state_indices)
        # future_obs_idxs = self._get_future_obs_indices(
        #     start_state_indices, max_dt=self._max_future_dt)
        future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
        return self._sample_context_from_obs_dict_fn(future_obs_dict)

    # def _get_foresight_contexts(self, start_state_indices):
    #     future_obs_idxs = self._get_future_obs_indices(
    #         start_state_indices, max_dt=self._max_future_dt)
    #     future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
    #
    #     assert len(self.observation_keys) == 1
    #     observation_key = self.observation_keys[0]
    #
    #     h0 = future_obs_dict[observation_key]
    #     h0 = ptu.from_numpy(h0)
    #     h0 = h0.view(
    #         -1,
    #         self._vqvae.embedding_dim,
    #         self._vqvae.root_len,
    #         self._vqvae.root_len)
    #
    #     z = self._affordance.sample_prior(h0.shape[0])
    #     z = ptu.from_numpy(z)
    #     h1_pred = self._affordance.decode(z, cond=h0).detach()
    #     _, h1_pred = self._vqvae.vector_quantizer(h1_pred)
    #
    #     # Debug.
    #     if 0:
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         plt.subplot(2, 1, 1)
    #         s0 = self._vqvae.decode(h0)
    #         image = s0 + 0.5
    #         image = image[0]
    #         image = image.permute(1, 2, 0).contiguous()
    #         image = ptu.get_numpy(image)
    #         plt.imshow(image)
    #
    #         plt.subplot(2, 1, 2)
    #         s1_pred = self._vqvae.decode(h1_pred)
    #         image = s1_pred + 0.5
    #         image = image[0]
    #         image = image.permute(1, 2, 0).contiguous()
    #         image = ptu.get_numpy(image)
    #         plt.imshow(image)
    #
    #         plt.show()
    #
    #     h1_pred = h1_pred.view(-1, self._vqvae.representation_size)
    #     h1_pred = ptu.get_numpy(h1_pred)
    #
    #     foresight_obs_dict = {}
    #     for key, value in future_obs_dict.items():
    #         if key == observation_key:
    #             foresight_obs_dict[key] = h1_pred
    #         else:
    #             foresight_obs_dict[key] = value
    #
    #     return self._sample_context_from_obs_dict_fn(foresight_obs_dict)

    # def _get_perturbed_contexts(self, start_state_indices):
    #     obs_dict = self._batch_obs_dict(start_state_indices)
    #     future_obs_idxs = self._get_future_obs_indices(
    #         start_state_indices, max_dt=self._max_future_dt)
    #     future_obs_dict = self._batch_next_obs_dict(future_obs_idxs)
    #
    #     assert len(self.observation_keys) == 1
    #     observation_key = self.observation_keys[0]
    #
    #     h0 = obs_dict[observation_key]
    #     h0 = ptu.from_numpy(h0)
    #     h0 = h0.view(
    #         -1,
    #         self._vqvae.embedding_dim,
    #         self._vqvae.root_len,
    #         self._vqvae.root_len)
    #
    #     h1 = future_obs_dict[observation_key]
    #     h1 = ptu.from_numpy(h1)
    #     h1 = h1.view(
    #         -1,
    #         self._vqvae.embedding_dim,
    #         self._vqvae.root_len,
    #         self._vqvae.root_len)
    #
    #     z_mu, z_logvar = self._affordance.encode(h1, cond=h0)
    #     std = z_logvar.mul(0.5).exp_()
    #     eps = std.data.new(std.size()).normal_()
    #     z = eps.mul(std).add_(z_mu)
    #     noisy_z = z + self._noise_level * ptu.randn(
    #         1, self._affordance.representation_size)
    #
    #     h1_pred = self._affordance.decode(noisy_z, cond=h0).detach()
    #     _, h1_pred = self._vqvae.vector_quantizer(h1_pred)
    #
    #     # Debug.
    #     if 0:
    #         import matplotlib.pyplot as plt
    #         plt.figure()
    #         plt.subplot(2, 1, 1)
    #         s1 = self._vqvae.decode(h1)
    #         image = s1 + 0.5
    #         image = image[0]
    #         image = image.permute(1, 2, 0).contiguous()
    #         image = ptu.get_numpy(image)
    #         plt.imshow(image)
    #
    #         plt.subplot(2, 1, 2)
    #         s1_pred = self._vqvae.decode(h1_pred)
    #         image = s1_pred + 0.5
    #         image = image[0]
    #         image = image.permute(1, 2, 0).contiguous()
    #         image = ptu.get_numpy(image)
    #         plt.imshow(image)
    #
    #         plt.show()
    #
    #     h1_pred = h1_pred.view(-1, self._vqvae.representation_size)
    #     h1_pred = ptu.get_numpy(h1_pred)
    #
    #     perturbed_obs_dict = {}
    #     for key, value in future_obs_dict.items():
    #         if key == observation_key:
    #             perturbed_obs_dict[key] = h1_pred
    #         else:
    #             perturbed_obs_dict[key] = value
    #
    #     return self._sample_context_from_obs_dict_fn(perturbed_obs_dict)
