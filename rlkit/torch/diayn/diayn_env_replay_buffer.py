from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np

class DIAYNEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            skill_dim,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self._skill = np.zeros((max_replay_buffer_size, skill_dim))

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, agent_info, **kwargs):
        self._skill[self._top] = agent_info["skill"]

        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            **kwargs
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            skills=self._skill[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch