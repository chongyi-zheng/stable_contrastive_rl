import numpy as np

from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.util.np_util import truncated_geometric


class TauReplayBuffer(HerReplayBuffer):
    def random_batch_random_tau(self, batch_size, max_tau):
        indices = np.random.randint(0, self._size, batch_size)
        next_obs_idxs = []
        for i in indices:
            possible_next_obs_idxs = self._idx_to_future_obs_idx[i]
            # This is generally faster than random.choice. Makes you wonder what
            # random.choice is doing
            num_options = len(possible_next_obs_idxs)
            tau = np.random.randint(0, min(max_tau+1, num_options))
            if num_options == 1:
                next_obs_i = 0
            else:
                if self.resampling_strategy == 'uniform':
                    next_obs_i = int(np.random.randint(0, tau+1))
                elif self.resampling_strategy == 'truncated_geometric':
                    next_obs_i = int(truncated_geometric(
                        p=self.truncated_geom_factor/tau,
                        truncate_threshold=num_options-1,
                        size=1,
                        new_value=0
                    ))
                else:
                    raise ValueError("Invalid resampling strategy: {}".format(
                        self.resampling_strategy
                    ))
            next_obs_idxs.append(possible_next_obs_idxs[next_obs_i])
        next_obs_idxs = np.array(next_obs_idxs)
        training_goals = self.env.convert_obs_to_goals(
            self._next_obs[next_obs_idxs]
        )
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            training_goals=training_goals,
            num_steps_left=self._num_steps_left[indices],
        )

