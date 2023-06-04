import copy

import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.util.io import load_local_or_remote_file
from rlkit.data_management.path_builder import PathBuilder

from roboverse.bullet.misc import quat_to_deg


class EncoderDictToMDPPathLoader(DictToMDPPathLoader):

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
            **kwargs
    ):
        super().__init__(trainer=trainer,
                         replay_buffer=replay_buffer,
                         demo_train_buffer=demo_train_buffer,
                         demo_test_buffer=demo_test_buffer,
                         demo_paths=demo_paths,
                         demo_train_split=demo_train_split,
                         demo_data_split=demo_data_split,
                         add_demos_to_replay_buffer=add_demos_to_replay_buffer,
                         bc_num_pretrain_steps=bc_num_pretrain_steps,
                         bc_batch_size=bc_batch_size,
                         bc_weight=bc_weight,
                         rl_weight=rl_weight,
                         q_num_pretrain_steps=q_num_pretrain_steps,
                         weight_decay=weight_decay,
                         eval_policy=eval_policy,
                         recompute_reward=recompute_reward,
                         env_info_key=env_info_key,
                         obs_key=obs_key,
                         load_terminals=load_terminals,
                         delete_after_loading=delete_after_loading,
                         data_filter_fn=data_filter_fn,
                         split_max_steps=split_max_steps,
                         **kwargs)

        if model is None:
            assert model_path is not None
            self.model = load_local_or_remote_file(
                model_path, delete_after_loading=delete_after_loading)
        else:
            assert model_path is None
            self.model = model

        self.condition_encoding = condition_encoding
        self.reward_fn = reward_fn
        self.compare_reward_fn = compare_reward_fn
        self.normalize = normalize
        self.object_list = object_list
        self.env = env

    def preprocess(self, observation, use_latents=True, use_gripper_obs=False):
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

        if self.condition_encoding:
            cond = images[0].repeat(len(observation), axis=0)
            latents = self.model['vqvae'].encode_np(images, cond)
        else:
            latents = self.model['vqvae'].encode_np(images)

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

            if vibs is not None:
                observation[i]['initial_vib_state'] = vibs[0]
                observation[i]['vib_observation'] = vibs[i]
                observation[i]['vib_desired_goal'] = vibs[-1]

            if use_latents:
                del observation[i]['image_observation']
            else:
                observation[i]['initial_image_observation'] = images[0]
                observation[i]['image_observation'] = images[i]
                observation[i]['image_desired_goal'] = images[-1]

            if use_gripper_obs:
                observation[i]['gripper_state_observation'] = gripper_states[i]
                observation[i]['gripper_state_desired_goal'] = (
                    gripper_states[-1])

        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(
                self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))

    def load_path(self,
                  path,
                  replay_buffer,
                  obs_dict=None,
                  use_latents=True,
                  use_gripper_obs=False):
        # Filter data #
        if not self.data_filter_fn(path):
            return

        rewards = []
        compare_rewards = []
        path_builder = PathBuilder()

        H = min(len(path['observations']), len(path['actions'])) - 1
        if obs_dict:
            traj_obs = self.preprocess(
                path['observations'],
                use_latents=use_latents,
                use_gripper_obs=use_gripper_obs)
            next_traj_obs = self.preprocess(
                path['next_observations'],
                use_latents=use_latents,
                use_gripper_obs=use_gripper_obs)
        else:
            traj_obs = self.preprocess_array_obs(
                path['observations'])
            next_traj_obs = self.preprocess_array_obs(
                path['next_observations'])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path['actions'][i]
            reward = path['rewards'][i]
            terminal = path['terminals'][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path['agent_infos'][i]
            env_info = path['env_infos'][i]

            terminal = np.array([terminal]).reshape((1,))

            if self.recompute_reward:
                reward, terminal = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)

            if self.recompute_reward and self.compare_reward_fn:
                compare_reward, _ = self.compare_reward_fn(
                    ob, action, next_ob, next_ob)
                compare_rewards.append(compare_reward)

            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print('loading path, length', len(
            path['observations']), len(path['actions']))
        print('actions', np.min(path['actions']), np.max(path['actions']))
        print('rewards', np.min(rewards), np.max(rewards))
        if self.compare_reward_fn:
            print('- rewards',
                  np.min(compare_rewards), np.max(compare_rewards))
        print('path sum rewards', sum(rewards), len(rewards))
        if self.compare_reward_fn:
            print('- path sum rewards',
                  sum(compare_rewards), len(compare_rewards))
