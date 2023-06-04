import numpy as np

from rlkit.data_management.subtraj_replay_buffer import SubtrajReplayBuffer
from rlkit.data_management.updatable_subtraj_replay_buffer import \
    UpdatableSubtrajReplayBuffer
from rlkit.util.np_util import subsequences


class OcmSubtrajReplayBuffer(UpdatableSubtrajReplayBuffer):
    """
    A replay buffer desired specifically for OneCharMem
    sub-trajectories
    """

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            subtraj_length,
            *args,
            **kwargs
    ):
        # TODO(vitchyr): Move this logic to environment
        self._target_numbers = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._times = np.zeros(max_replay_buffer_size, dtype='uint8')
        super().__init__(
            max_replay_buffer_size,
            env,
            subtraj_length,
            *args,
            only_sample_at_start_of_episode=False,
            **kwargs
        )

    def _add_sample(self, observation, action, reward, terminal,
                    final_state, agent_info=None, env_info=None):
        if env_info is not None:
            if 'target_number' in env_info:
                self._target_numbers[self._top] = env_info['target_number']
            if 'time' in env_info:
                self._times[self._top] = env_info['time']
        super()._add_sample(
            observation,
            action,
            reward,
            terminal,
            final_state
        )

    def _get_trajectories(self, start_indices):
        trajs = super()._get_trajectories(start_indices)
        trajs['target_numbers'] = subsequences(
            self._target_numbers,
            start_indices,
            self._subtraj_length,
        )
        trajs['times'] = subsequences(
            self._times,
            start_indices,
            self._subtraj_length,
        )
        return trajs
