import glob
import numpy as np
import copy
import os
import pickle as pkl

from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.util.io import (
    load_local_or_remote_file,
    sync_down_folder,
    get_absolute_path
)
from rlkit.experimental.kuanfang.utils.real_utils import filter_step_fn

from rlkit.experimental.chongyiz.utils.path_builder import PathBuilder


def split_demo(demo, max_steps):
    if max_steps is None:
        return [demo]

    else:
        new_demo = []

        key, value = next(iter(demo.items()))
        horizon = len(value)

        t = np.random.randint(0, min(max_steps, horizon - max_steps))

        while True:
            new_demo_t = {}
            new_t = t + max_steps
            if new_t >= horizon:
                break

            for key, value in demo.items():
                if key in ['object_name', 'skill_id']:
                    new_demo_t[key] = value
                else:
                    new_demo_t[key] = value[t:new_t]

            t = new_t

            new_demo.append(new_demo_t)

        return new_demo


class DictToMDPPathLoader:
    """
    Path loader for that loads obs-dict demonstrations
    into a Trainer with EnvReplayBuffer
    """

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            env=None,
            reward_fn=None,
            demo_paths=None,  # list of dicts
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            demos_saving_path=None,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            # Return true to add path, false to ignore it
            data_filter_fn=lambda x: True,
            split_max_steps=None,
            filter_step_fn=filter_step_fn,
            min_action_value=None,
            action_round_thresh=None,
            min_path_length=None,
            **kwargs
    ):
        self.trainer = trainer
        self.delete_after_loading = delete_after_loading
        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demos_saving_path = demos_saving_path
        self.demo_train_split = demo_train_split
        self.demo_data_split = demo_data_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer
        self.data_filter_fn = data_filter_fn

        self.env = env
        self.reward_fn = reward_fn

        self.demo_paths = [] if demo_paths is None else demo_paths

        self.paths_to_save = []

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward
        self.load_terminals = load_terminals

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

        self.split_max_steps = split_max_steps

        self.filter_step_fn = filter_step_fn
        self.min_action_value = min_action_value
        self.action_round_thresh = action_round_thresh

        self.min_path_length = min_path_length

    def preprocess(self, observation):
        observation = copy.deepcopy(observation[:-1])
        # import ipdb; ipdb.set_trace()
        images = np.stack([observation[i]['image_observation']
                          for i in range(len(observation))])

        # if self.normalize:
        #     raise NotImplementedError  # TODO(kuanfang): This is buggy.
        #     images = images / 255.0

        for i in range(len(observation)):
            observation[i]['initial_image_observation'] = images[0]
            observation[i]['image_observation'] = images[i]
            observation[i]['image_achieved_goal'] = images[i]
            observation[i]['image_desired_goal'] = images[-1]

        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def load_path(self, path, replay_buffer, obs_dict=None, use_latents=True, use_gripper_obs=False):  # NOQA
        del use_latents
        del use_gripper_obs

        # Filter data #
        if not self.data_filter_fn(path):
            return

        rewards = []
        path_builder = PathBuilder()

        H = min(len(path['observations']), len(path['actions'])) - 1

        if self.min_path_length is not None:
            if H < self.min_path_length:
                return

        if obs_dict:
            traj_obs = self.preprocess(
                path['observations'])
            next_traj_obs = self.preprocess(
                path['next_observations'])
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

            if (self.filter_step_fn is not None and
                    self.min_action_value is not None):
                if self.filter_step_fn(ob, action, next_ob,
                                       self.min_action_value):
                    continue

            if self.action_round_thresh is not None:
                for dim in [0, 1, 2, 4]:
                    if action[dim] >= self.action_round_thresh:
                        action[dim] = 1.0
                    elif action[dim] <= -self.action_round_thresh:
                        action[dim] = -1.0
                    else:
                        pass

            terminal = np.array([terminal]).reshape((1, ))

            if self.recompute_reward:
                reward, terminal = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
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
        print('path sum rewards', sum(rewards), len(rewards))

        if self.filter_step_fn is not None:
            print("unfiltered states: %d / %d"
                  % (len(path["observations"]), H))

    def save_path(self, path, obs_dict=None):
        # Filter data #
        if not self.data_filter_fn(path):
            return

        rewards = []
        path_builder = PathBuilder()

        H = min(len(path['observations']), len(path['actions'])) - 1

        if self.min_path_length is not None:
            if H < self.min_path_length:
                return

        if obs_dict:
            traj_obs = self.preprocess(
                path['observations'])
            next_traj_obs = self.preprocess(
                path['next_observations'])
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

            if (self.filter_step_fn is not None and
                    self.min_action_value is not None):
                if self.filter_step_fn(ob, action, next_ob,
                                       self.min_action_value):
                    continue

            if self.action_round_thresh is not None:
                for dim in [0, 1, 2, 4]:
                    if action[dim] >= self.action_round_thresh:
                        action[dim] = 1.0
                    elif action[dim] <= -self.action_round_thresh:
                        action[dim] = -1.0
                    else:
                        pass

            terminal = np.array([terminal]).reshape((1,))

            if self.recompute_reward:
                reward, terminal = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
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
        # replay_buffer.add_path(path)
        self.paths_to_save.append(path)

        print('saving path, length', len(
            path['observations']), len(path['actions']))
        print('actions', np.min(path['actions']), np.max(path['actions']))
        print('rewards', np.min(rewards), np.max(rewards))
        print('path sum rewards', sum(rewards), len(rewards))

        if self.filter_step_fn is not None:
            print("unfiltered states: %d / %d"
                  % (len(path["observations"]), H))

    def dump_paths(self):
        assert os.path.exists(os.path.dirname(self.demos_saving_path))
        demos_saving_path = os.path.abspath(self.demos_saving_path)
        with open(demos_saving_path, "wb+") as f:
            pkl.dump(self.paths_to_save, f)
        print("Paths save to: {}".format(demos_saving_path))

    def load_demos(self):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    # Parameterize which demo is being tested (and all jitter variants)
    # If is_demo is False, we only add the demos to the
    # replay buffer, and not to the demo_test or demo_train buffers
    def load_demo_path(self,  # NOQA
                       path,
                       is_demo,
                       obs_dict,
                       train_split=None,
                       data_split=None,
                       sync_dir=None,
                       use_latents=True,
                       use_gripper_obs=False):
        print('loading off-policy path', path)

        if sync_dir is not None:
            sync_down_folder(sync_dir)
            paths = glob.glob(get_absolute_path(path))
        else:
            paths = [path]

        # input('Prepare to load demos...')

        data = []
        for filename in paths:
            data_i = load_local_or_remote_file(
                filename,
                delete_after_loading=self.delete_after_loading)
            data_i = list(data_i)

            if self.split_max_steps:
                new_data_i = []
                for j in range(len(data_i)):
                    data_i_j = split_demo(data_i[j],
                                          max_steps=self.split_max_steps)
                    new_data_i.extend(data_i_j)
                data_i = new_data_i

            data.extend(data_i)

        # if not is_demo:
        # data = [data]
        # random.shuffle(data)

        if train_split is None:
            train_split = self.demo_train_split

        if data_split is None:
            data_split = self.demo_data_split

        M = int(len(data) * train_split * data_split)
        N = int(len(data) * data_split)
        print('using', M, 'paths for training')

        if self.add_demos_to_replay_buffer:
            for path in data[:M]:  # TODO
                self.load_path(path,
                               self.replay_buffer,
                               obs_dict=obs_dict,
                               use_latents=use_latents,
                               use_gripper_obs=use_gripper_obs)
        else:
            for path in data[:M]:  # TODO
                self.save_path(path,
                               obs_dict=obs_dict)

        if is_demo:
            if self.demo_train_buffer:
                for path in data[:M]:
                    self.load_path(path,
                                   self.demo_train_buffer,
                                   obs_dict=obs_dict,
                                   use_latents=use_latents,
                                   use_gripper_obs=use_gripper_obs)

            if self.demo_test_buffer:
                for path in data[M:N]:
                    self.load_path(path,
                                   self.demo_test_buffer,
                                   obs_dict=obs_dict,
                                   use_latents=use_latents,
                                   use_gripper_obs=use_gripper_obs)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch
