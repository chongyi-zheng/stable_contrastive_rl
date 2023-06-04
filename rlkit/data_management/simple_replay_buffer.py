from collections import OrderedDict

import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        obs_dtype=np.float64,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim), dtype=obs_dtype)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim), dtype=obs_dtype)
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self.env_info_sizes = env_info_sizes
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])

    """saving / copying buffers"""
    def copy_data(
            self,
            other_buffer: 'SimpleReplayBuffer',
            start_idx=0,
            end_idx=None,
    ):
        """
        Copy data from [start:end] from the other buffer to this buffer.
        :param other_buffer: A SimpleReplayBuffer (either this one, or from PEARL)
        :param start_idx: start index for copying.
        :param end_idx: end index for copying.
        :return: None

        :raise: ValueError if there is nothing to copy.
        """
        if end_idx is None:
            end_idx = other_buffer._top
            if other_buffer._top == start_idx:
                raise ValueError("nothing to copy!")
        if end_idx < 0 or end_idx <= start_idx:
            raise ValueError("end_idx must be larger than start_idx")
        num_new_steps = end_idx - start_idx
        end_i = self._top + num_new_steps
        this_slc = slice(self._top, end_i)
        other_slc = slice(start_idx, end_idx)
        if end_i > self._max_replay_buffer_size:
            raise NotImplementedError()
        self._observations[this_slc] = (
            other_buffer._observations[other_slc].copy()
        )
        self._actions[this_slc] = (
            other_buffer._actions[other_slc].copy()
        )
        self._rewards[this_slc] = (
            other_buffer._rewards[other_slc].copy()
        )
        self._terminals[this_slc] = (
            other_buffer._terminals[other_slc].copy()
        )
        self._next_obs[this_slc] = (
            other_buffer._next_obs[other_slc].copy()
        )
        from rlkit.data_management.multitask_replay_buffer import (
            SimpleReplayBuffer as OldPearlSimpleReplayBuffer
        )
        for key in self._env_info_keys:
            # TODO: remove this special case
            if key == 'sparse_reward' and isinstance(other_buffer, OldPearlSimpleReplayBuffer):
                    self._env_infos['sparse_reward'][this_slc] = (
                        other_buffer._sparse_rewards[other_slc].copy()
                    )
            else:
                self._env_infos[key][this_slc] = (
                    other_buffer._env_infos[key][other_slc]
                )
        self._top += num_new_steps
        self._size += num_new_steps

    def __getstate__(self):
        # Do not save self.replay_buffer since it's a duplicate and seems to
        # cause joblib recursion issues.
        return dict(
            observations = self._observations[:self._top],
            actions = self._actions[:self._top],
            rewards = self._rewards[:self._top],
            terminals = self._terminals[:self._top],
            next_obs = self._next_obs[:self._top],
            observation_dim = self._observation_dim,
            action_dim = self._action_dim,
            max_replay_buffer_size = self._max_replay_buffer_size,
            env_info_sizes = self.env_info_sizes,
            top = self._top,
            size = self._size,
        )

    def __setstate__(self, d):
        observation_dim = d["observation_dim"]
        max_replay_buffer_size = d["max_replay_buffer_size"]
        action_dim = d["action_dim"]

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')

        env_info_sizes = d["env_info_sizes"]
        self.env_info_sizes = env_info_sizes
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        self._top = d["top"]
        self._size = d["size"]

        # restore data
        self._observations[:self._top] = d["observations"]
        self._actions[:self._top] = d["actions"]
        self._rewards[:self._top] = d["rewards"]
        self._terminals[:self._top] = d["terminals"]
        self._next_obs[:self._top] = d["next_obs"]
