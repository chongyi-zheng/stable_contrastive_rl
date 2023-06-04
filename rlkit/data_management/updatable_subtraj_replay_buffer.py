import numpy as np
from cached_property import cached_property

from rlkit.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from rlkit.util.np_util import subsequences, assign_subsequences
from rlkit.data_management.subtraj_replay_buffer import SubtrajReplayBuffer


class UpdatableSubtrajReplayBuffer(SubtrajReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env: ContinuousMemoryAugmented,
            subtraj_length,
            memory_dim,
            keep_old_fraction=0.,
            **kwargs
    ):
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            subtraj_length=subtraj_length,
            **kwargs
        )
        self._action = None
        self.observations = None

        self.memory_dim = memory_dim
        # Note that this must be computed from the next time step.
        """
        self._dloss_dwrite[t] = dL/dw_t     (zero-indexed)
        """
        self._dloss_dmemories = np.zeros((self._max_replay_buffer_size,
                                          self.memory_dim))
        self._env_obs_dim = env.wrapped_env.observation_space.flat_dim
        self._env_action_dim = env.wrapped_env.action_space.flat_dim
        self._env_obs = np.zeros((max_replay_buffer_size, self._env_obs_dim))
        self._env_actions = np.zeros((max_replay_buffer_size, self._env_action_dim))

        self._memory_dim = env.memory_dim
        self._memories = np.zeros((max_replay_buffer_size, self._memory_dim))
        self.keep_old_fraction = keep_old_fraction

    def random_subtrajectories(self, batch_size, replace=False,
                               _fixed_start_indices=None):
        if _fixed_start_indices is None:
            start_indices = np.random.choice(
                self._valid_start_indices(),
                batch_size,
                replace=replace,
            )
        else:
            start_indices = _fixed_start_indices
        return self.get_trajectories(start_indices), start_indices

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, **kwargs):
        env_action, write = action  # write should be saved as next memory
        env_obs, memory = observation
        self._env_obs[self._top] = env_obs
        self._env_actions[self._top] = env_action
        self._memories[self._top] = memory
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self._episode_start_indices[self._top] = self._starting_episode
        self._starting_episode = False

        self.advance()

    def _get_trajectories(self, start_indices):
        next_memories = subsequences(self._memories, start_indices,
                                     self._subtraj_length, start_offset=1)
        return dict(
            env_obs=subsequences(self._env_obs, start_indices,
                                 self._subtraj_length),
            env_actions=subsequences(self._env_actions, start_indices,
                                     self._subtraj_length),
            next_env_obs=subsequences(self._env_obs, start_indices,
                                      self._subtraj_length,
                                      start_offset=1),
            memories=subsequences(self._memories, start_indices,
                                  self._subtraj_length),
            writes=next_memories,
            next_memories=next_memories,
            rewards=subsequences(self._rewards, start_indices,
                                 self._subtraj_length),
            terminals=subsequences(self._terminals, start_indices,
                                   self._subtraj_length),
            dloss_dwrites=subsequences(self._dloss_dmemories, start_indices,
                                         self._subtraj_length, start_offset=1),
        )

    def get_trajectory_minimal_covering_subsequences(
            self, trajectory_start_idxs, episode_length
    ):
        """
        A set of subsequences _covers_ a trajectory if for every sample in the
        trajectory, there exists at least one subsequence with that sample.

        This function returns the minimally sized set of subsequences that
        covers the trajectories starting at `trajectory_start_idxs`.

        Warning: only pass a starting index of a trajectory that's finished.

        :param trajectory_start_idxs: A list of trajectory start indices. A
        useful function: `self.get_all_valid_trajectory_start_indices()`
        :param episode_length: The length of the episode.
        :return: Tuple
            - list of subtrajectories (dictionaries)
            - list of the start indices
        """
        start_indices = []
        for trajectory_start_index in trajectory_start_idxs:
            last_subseq_start_idx = (
                trajectory_start_index + episode_length - self._subtraj_length
            )
            # Only do every `self._subtraj_length` to get the minimal covering
            # set of subsequences.
            for subsequence_start_idx in range(
                    trajectory_start_index,
                    last_subseq_start_idx + 1,
                    self._subtraj_length
            ):
                start_indices.append(subsequence_start_idx)
            # If the episode is length 10, but subtraj_length = 4, we would
            # only add [0, 4], so this adds `6` to start_indices to make sure
            # (e.g.) the last two time steps aren't left out.
            if last_subseq_start_idx not in start_indices:
                start_indices.append(last_subseq_start_idx)
        return self.get_trajectories(start_indices), start_indices

    def get_all_valid_trajectory_start_indices(self):
        return self._valid_start_episode_indices

    @cached_property
    def _stub_action(self):
        # Technically, the parent's method should work, but I think this is more
        # clear.
        return np.zeros(self._env_action_dim), np.zeros(self.memory_dim)

    def update_write_subtrajectories(self, updated_writes, start_indices):
        assign_subsequences(
            tensor=self._memories,
            new_values=updated_writes,
            start_indices=start_indices,
            length=self._subtraj_length,
            start_offset=1
        )

    def update_dloss_dmemories_subtrajectories(
            self,
            updated_dloss_dmemories,
            start_indices
    ):
        assign_subsequences(
            tensor=self._dloss_dmemories,
            new_values=updated_dloss_dmemories,
            start_indices=start_indices,
            length=self._subtraj_length,
            keep_old_fraction=self.keep_old_fraction,
        )

    def fraction_dloss_dmemories_zero(self):
        dloss_dmemories_loaded = self._dloss_dmemories[:self._size]
        return np.mean(dloss_dmemories_loaded == 0)

    def random_batch(self, batch_size, **kwargs):
        """
        Get flat random batch.
        :param batch_size:
        :param **kwargs:
        :return:
        """
        indices = np.random.choice(
            self._all_valid_start_indices,
            batch_size,
            replace=False
        )
        next_indices = (indices + 1) % self._size
        next_memories = self._memories[next_indices]
        return dict(
            env_obs=self._env_obs[indices],
            env_actions=self._env_actions[indices],
            next_env_obs=self._env_obs[next_indices],
            memories=self._memories[indices],
            writes=next_memories,
            next_memories=next_memories,
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
        )
