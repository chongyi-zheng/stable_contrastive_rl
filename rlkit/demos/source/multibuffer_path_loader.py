import numpy as np
import copy
import glob
from rlkit.util.io import (
    load_local_or_remote_file, get_absolute_path
)

from rlkit.data_management.path_builder import PathBuilder
from rlkit.experimental.kuanfang.utils.real_utils import filter_step_fn


class MultibufferPathLoader:
    def __init__(
            self,
            replay_buffers,
            paths,
            default_path_loading_kwargs=None,
            model=None,
            model_path=None,

            # TODO refactor below operations outside the path loader
            reward_fn=None,
            filter_step_fn=filter_step_fn,
            min_action_value=None,
            **kwargs
    ):
        if kwargs:
            print("warning, unused kwargs in MultibufferPathLoader:", kwargs)
        if default_path_loading_kwargs is None:
            default_path_loading_kwargs = {}
        self.default_path_loading_kwargs = default_path_loading_kwargs

        self.paths = paths
        self.replay_buffers = replay_buffers

        if model is None:
            assert model_path is not None
            self.model = load_local_or_remote_file(
                model_path, delete_after_loading=True)
        else:
            assert model_path is None
            self.model = model

        self.reward_fn = reward_fn

        self.filter_step_fn = filter_step_fn
        self.min_action_value = min_action_value

        self.demo_trajectory_rewards = []

    def load_paths(self):
        for path_kwargs in self.paths:
            kwargs = copy.deepcopy(self.default_path_loading_kwargs)
            kwargs.update(path_kwargs)
            self.load_path(**kwargs)

    def load_path(self,
                  filepath,
                  buffer_names,
                  split_slice=None,
                  sync_dir=None,
                  repeat=1,
                  **kwargs):
        """Load a set of paths found in filepath (may be a glob)"""
        if split_slice is None:
            split_slice = (0, 1)

        print("loading off-policy path", filepath)

        if "*" in filepath:  # is not None:
            # sync_down_folder(sync_dir)
            filenames = glob.glob(get_absolute_path(filepath))
        else:
            filenames = [filepath]

        # data = []
        for filename in filenames:
            # data.extend(list(load_local_or_remote_file(filename)))
            # , delete_after_loading=self.delete_after_loading)))
            data = list(load_local_or_remote_file(filename))

            start = int(len(data) * split_slice[0])
            end = int(len(data) * split_slice[1])
            print("using", end - start, "paths for training")

            for path in data[start:end]:
                new_path = self.get_path_to_add(path, **kwargs)
                for buffer_name in buffer_names:
                    replay_buffer = self.replay_buffers[buffer_name]
                    for _ in range(repeat):
                        replay_buffer.add_path(new_path)

    def get_path_to_add(self, path, obs_dict=None, recompute_reward=True):
        rewards = []
        path_builder = PathBuilder()

        H = min(len(path["observations"]), len(path["actions"])) - 1
        if obs_dict:
            # traj_obs = self.preprocess(path["observations"])
            # next_traj_obs = self.preprocess(path["next_observations"])
            traj_obs = self.preprocess(path["observations"])
            next_traj_obs = self.preprocess(path["observations"][1:])
        else:
            traj_obs = self.preprocess_array_obs(path["observations"])
            next_traj_obs = self.preprocess_array_obs(
                path["next_observations"])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]

            if (self.filter_step_fn is not None and
                    self.min_action_value is not None):
                if self.filter_step_fn(ob, action, next_ob,
                                       self.min_action_value):
                    continue

            # if self.action_round_thresh is not None:
            #     for dim in [0, 1, 2, 4]:
            #         if action[dim] >= self.action_round_thresh:
            #             action[dim] = 1.0
            #         elif action[dim] <= -self.action_round_thresh:
            #             action[dim] = -1.0
            #         else:
            #             pass

            if recompute_reward:
                reward = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1,))
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
        print("rewards", np.min(rewards), np.max(rewards))
        print("loading path, length", len(
            path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))
        print("path sum rewards", sum(rewards), len(rewards))

        if self.filter_step_fn is not None:
            print("unfiltered states: %d / %d"
                  % (len(path["observations"]), H))

        return path

    def preprocess(self, observation):
        observation = copy.deepcopy(observation)
        images = np.stack([observation[i]['image_observation']
                          for i in range(len(observation))])

        latents = self.model.encode_np(images)

        for i in range(len(observation)):
            observation[i]['initial_latent_state'] = latents[0]
            observation[i]['latent_observation'] = latents[i]
            observation[i]['latent_achieved_goal'] = latents[i]
            observation[i]['latent_desired_goal'] = latents[-1]

        return observation
