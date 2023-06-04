import copy

import torch
import numpy as np

from roboverse.bullet.misc import quat_to_deg

import rlkit.torch.pytorch_util as ptu
from rlkit.util.augment_util import create_aug_stack
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader


class AugmentedEncoderDictToMDPPathLoader(EncoderDictToMDPPathLoader):
    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model=None,
            model_path=None,
            reward_fn=None,
            compare_reward_fn=None,
            env=None,
            demo_paths=[],  # list of dicts
            normalize=False,
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            condition_encoding=False,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            object_list=None,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            # Return true to add path, false to ignore it
            data_filter_fn=lambda x: True,
            split_max_steps=None,
            augment_params=dict(),
            augment_order=[],
            augment_probability=0.0,
            **kwargs
    ):
        super().__init__(trainer=trainer,
                         replay_buffer=replay_buffer,
                         demo_train_buffer=demo_train_buffer,
                         demo_test_buffer=demo_test_buffer,
                         demo_paths=demo_paths,
                         model=model,
                         model_path=model_path,
                         reward_fn=reward_fn,
                         compare_reward_fn=compare_reward_fn,
                         env=env,
                         normalize=normalize,
                         demo_train_split=demo_train_split,
                         demo_data_split=demo_data_split,
                         add_demos_to_replay_buffer=add_demos_to_replay_buffer,
                         condition_encoding=condition_encoding,
                         bc_num_pretrain_steps=bc_num_pretrain_steps,
                         bc_batch_size=bc_batch_size,
                         bc_weight=bc_weight,
                         rl_weight=rl_weight,
                         q_num_pretrain_steps=q_num_pretrain_steps,
                         weight_decay=weight_decay,
                         eval_policy=eval_policy,
                         recompute_reward=recompute_reward,
                         object_list=object_list,
                         env_info_key=env_info_key,
                         obs_key=obs_key,
                         load_terminals=load_terminals,
                         delete_after_loading=delete_after_loading,
                         data_filter_fn=data_filter_fn,
                         split_max_steps=split_max_steps,
                         **kwargs)

        self.augment_params = augment_params
        self.augment_order = augment_order
        self.augment_probability = augment_probability

        # Image augmentation
        if augment_probability > 0:
            width = self.model['vqvae'].imsize
            height = self.model['vqvae'].imsize
            self.augment_stack = create_aug_stack(
                augment_order, augment_params, size=(width, height)
            )
        else:
            self.augment_stack = None

    def set_augment_params(self, img):
        if torch.rand(1) < self.augment_probability:
            self.augment_stack.set_params(img)
        else:
            self.augment_stack.set_default_params(img)

    def augment(self, images):
        # augmented_batch = dict()
        # for key, value in batch.items():
        #     augmented_batch[key] = value
        augmented_images = None

        if (self.augment_probability > 0 and
                images.shape[0] > 0):
            width = self.model['vqvae'].imsize
            height = self.model['vqvae'].imsize
            imlength = self.model['vqvae'].imlength

            # obs = batch['observations'].reshape(
            #     -1, 3, width, height)
            # next_obs = batch['next_observations'].reshape(
            #     -1, 3, width, height)
            # context = batch['contexts'].reshape(
            #     -1, 3, width, height)
            images = images.reshape(
                -1, 3, width, height)

            self.set_augment_params(images)
            # augmented_batch['observations'] = self.augment_stack(obs)

            # augmented_batch['next_observations'] = self.augment_stack(next_obs)

            # augmented_batch['contexts'] = self.augment_stack(context)

            augmented_images = self.augment_stack(images)

            augmented_images = augmented_images.reshape(
                -1, imlength)

        return augmented_images

    def preprocess(self,
                   observation,
                   use_latents=True,
                   use_gripper_obs=False):
        observation = copy.deepcopy(observation[:-1])
        # import ipdb; ipdb.set_trace()
        images = np.stack([observation[i]['image_observation']
                           for i in range(len(observation))])
        if use_gripper_obs:
            gripper_states = np.stack([
                np.concatenate(
                    (observation[i]['state_observation'][:3],
                     quat_to_deg(observation[i]['state_observation'][3:7])
                     / 360.,
                     observation[i]['state_observation'][7:8],
                     ),
                    axis=0)
                for i in range(len(observation))
            ])

        if self.normalize:
            raise NotImplementedError  # TODO(kuanfang): This is buggy.
            images = images / 255.0

        augmented_images = ptu.get_numpy(
            self.augment(ptu.from_numpy(images))
        )

        if self.condition_encoding:
            cond = images[0].repeat(len(observation), axis=0)
            latents = self.model['vqvae'].encode_np(images, cond)

            # FIXME (chongyiz): implement conditional encoding for augmented images
            raise NotImplementedError
        else:
            latents = self.model['vqvae'].encode_np(images)
            augmented_latents = self.model['vqvae'].encode_np(augmented_images)

        if 'obs_encoder' in self.model:
            vibs = self.model['obs_encoder'].encode_np(latents)
        else:
            vibs = None

        # latents = np.stack([observation[i]['latent_observation']
        #                   for i in range(len(observation))])

        for i in range(len(observation)):
            observation[i]['initial_latent_state'] = latents[0]
            observation[i]['latent_observation'] = latents[i]
            observation[i]['latent_desired_goal'] = latents[-1]
            observation[i]['initial_augmented_latent_state'] = augmented_latents[0]
            observation[i]['augmented_latent_observation'] = augmented_latents[i]
            observation[i]['augmented_latent_desired_goal'] = augmented_latents[-1]

            if vibs is not None:
                observation[i]['initial_vib_state'] = vibs[0]
                observation[i]['vib_observation'] = vibs[i]
                observation[i]['vib_desired_goal'] = vibs[-1]

            # (chongyiz): do we need to delete the image observation?
            # if use_latents:
            #     del observation[i]['image_observation']
            # else:
            #     observation[i]['initial_image_observation'] = images[0]
            #     observation[i]['image_observation'] = images[i]
            #     observation[i]['image_desired_goal'] = images[-1]
            observation[i]['initial_image_observation'] = images[0]
            observation[i]['image_observation'] = images[i]
            observation[i]['image_desired_goal'] = images[-1]
            observation[i]['initial_augmented_image_observation'] = augmented_images[0]
            observation[i]['augmented_image_observation'] = augmented_images[i]
            observation[i]['augmented_image_desired_goal'] = augmented_images[-1]

            if use_gripper_obs:
                observation[i]['gripper_state_observation'] = gripper_states[i]
                observation[i]['gripper_state_desired_goal'] = (
                    gripper_states[-1])

        return observation
