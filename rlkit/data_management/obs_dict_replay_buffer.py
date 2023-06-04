import logging
import numpy as np
from gym.spaces import Dict, Discrete

from rlkit.data_management.replay_buffer import ReplayBuffer
import rlkit.data_management.images as image_np


class ObsDictReplayBuffer(ReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            ob_keys_to_save=None,
            internal_keys=None,
            observation_key=None,  # for backwards compatibility
            observation_keys=None,
            save_data_in_snapshot=False,
            reward_dim=1,
            preallocate_arrays=False,
    ):
        """

        :param max_size:
        :param env:
        :param ob_keys_to_save: List of keys to save
        """
        if observation_key is not None and observation_keys is not None:
            raise ValueError(
                'Only specify observation_key or observation_keys')
        if observation_key is None and observation_keys is None:
            raise ValueError(
                'Specify either observation_key or observation_keys'
            )
        if observation_keys is None:
            observation_keys = [observation_key]
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        assert isinstance(env.observation_space, Dict)
        self.max_size = max_size
        self.env = env
        self.observation_keys = observation_keys
        self.save_data_in_snapshot = save_data_in_snapshot

        self._action_dim = env.action_space.low.size
        self._actions = np.ones(
            (max_size, *env.action_space.shape),
            dtype=env.action_space.dtype,
        )
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.ones((max_size, 1), dtype='uint8')
        self.vectorized = reward_dim > 1
        self._rewards = np.ones((max_size, reward_dim))
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in observation_keys:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)
        for key in ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            module = image_np if key.startswith('image') else np
            arr_initializer = (
                module.ones if preallocate_arrays else module.zeros)
            self._obs[key] = arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=self.ob_spaces[key].dtype,
            )
            self._next_obs[key] = arr_initializer(
                (max_size, *self.ob_spaces[key].shape),
                dtype=self.ob_spaces[key].dtype,
            )

        self.ob_keys_to_save = ob_keys_to_save
        self._top = 0
        self._size = 0

        self._idx_to_future_obs_idx = np.ones((max_size, 2), dtype=np.int)
        self._idx_to_num_steps = np.ones((max_size, ), dtype=np.int)

        if isinstance(self.env.action_space, Discrete):
            raise NotImplementedError("TODO")

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

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
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]  # NOQA
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
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = [i, self._top + path_len]
                self._idx_to_num_steps[i] = i - self._top

        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size, min_dt=None):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        if len(self.observation_keys) == 1:
            obs = self._obs[self.observation_keys[0]][indices]
            next_obs = self._next_obs[self.observation_keys[0]][indices]
        else:
            obs = tuple(self._obs[k][indices] for k in self.observation_keys)
            next_obs = tuple(self._next_obs[k][indices]
                             for k in self.observation_keys)
        terminals = self._terminals[indices]
        batch = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

    def _sample_goals_from_env(self, batch_size):
        return self.env.sample_goals(batch_size)

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        if self.save_data_in_snapshot:
            snapshot.update({
                'observations': self.get_slice(self._obs, slice(0, self._top)),
                'next_observations': self.get_slice(
                    self._next_obs, slice(0, self._top)
                ),
                'actions': self._actions[:self._top],
                'terminals': self._terminals[:self._top],
                'rewards': self._rewards[:self._top],
                'idx_to_future_obs_idx': (
                    self._idx_to_future_obs_idx[:self._top]
                ),
                'idx_to_num_steps': (
                    self._idx_to_num_steps[:self._top]
                ),
            })
        return snapshot

    def get_slice(self, obs_dict, slc):
        new_dict = {}
        for key in self.ob_keys_to_save + self.internal_keys:
            new_dict[key] = obs_dict[key][slc]
        return new_dict

    def _get_future_obs_indices(self, start_state_indices):
        future_obs_idxs = []
        for i in start_state_indices:
            possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
            lb, ub = possible_future_obs_idxs
            if ub > lb:
                next_obs_i = int(np.random.randint(lb, ub))
            else:
                next_obs_i = int(np.random.randint(
                    lb, ub + self.max_size) % self.max_size)
            future_obs_idxs.append(next_obs_i)
        future_obs_idxs = np.array(future_obs_idxs)
        return future_obs_idxs


class ObsDictRelabelingBuffer(ObsDictReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            goal_keys=None,
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            ob_keys_to_save=None,
            use_multitask_rewards=True,
            recompute_rewards=True,
            **kwargs
    ):
        """

        :param max_size:
        :param env:
        :param fraction_goals_rollout_goals: Default, no her
        :param fraction_resampled_goals_env_goals:  of the resampled
            goals, what fraction are sampled from the env.
            Reset are sampled from future.
        :param ob_keys_to_save: List of keys to save
        """
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        if desired_goal_key not in ob_keys_to_save:
            ob_keys_to_save.append(desired_goal_key)
        if achieved_goal_key not in ob_keys_to_save:
            ob_keys_to_save.append(achieved_goal_key)
        if goal_keys is not None:
            for goal_key in goal_keys:
                if goal_key not in ob_keys_to_save:
                    ob_keys_to_save.append(goal_key)
                    # TODO: fix hack. Necessary for future-style relabeling
                    ob_keys_to_save.append(
                        goal_key.replace('desired', 'achieved')
                    )
        super().__init__(
            max_size,
            env,
            ob_keys_to_save=ob_keys_to_save,
            **kwargs
        )
        if goal_keys is None:
            self.goal_keys = [k for k in ob_keys_to_save if 'desired' in k]
        else:
            logging.warning("""
            Are you sure you want to set the goal keys manually?
            You're less likely to get bugs by setting it automatically.
            In particular, relabeling will ONLY relabel the keys in goal_keys,
            which may break if you have wrapped environments.
            For details, ask @vitchyr.
            """)
            self.goal_keys = list(goal_keys)
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.recompute_rewards = recompute_rewards
        self.use_multitask_rewards = use_multitask_rewards

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self._sample_goals_from_env(num_env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals

            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = env_goals[goal_key]  # NOQA
                new_next_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = env_goals[goal_key]  # NOQA

        if num_future_goals > 0:
            future_obs_idxs = self._get_future_obs_indices(
                indices[-num_future_goals:])
            for goal_key in self.goal_keys:
                achieved_k = goal_key.replace('desired', 'achieved')
                new_obs_dict[goal_key][-num_future_goals:] = (
                    self._next_obs[achieved_k][future_obs_idxs]
                )
                new_next_obs_dict[goal_key][-num_future_goals:] = (
                    self._next_obs[achieved_k][future_obs_idxs]
                )

        new_actions = self._actions[indices]

        if not self.recompute_rewards:
            new_rewards = self._rewards[indices]
        elif self.use_multitask_rewards:
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

        if len(self.observation_keys) == 1:
            new_obs = new_obs_dict[self.observation_keys[0]]
            new_next_obs = new_next_obs_dict[self.observation_keys[0]]
        else:
            new_obs = tuple(new_obs_dict[k] for k in self.observation_keys)
            new_next_obs = tuple(new_next_obs_dict[k]
                                 for k in self.observation_keys)
        resampled_goals = new_next_obs_dict[self.desired_goal_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch


def combine_dicts(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: np.array([d[key] for d in dicts])
        for key in keys
    }
