import random
import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer

class OnlineOfflineSplitReplayBuffer(ReplayBuffer):
    """
    Split online and offline data into two separate replay buffers and sample proportionally.
    """
    def __init__(
            self,
            offline_replay_buffer: ObsDictReplayBuffer,
            online_replay_buffer: ObsDictReplayBuffer,
            sample_online_fraction=None,
            online_mode=False,
            **kwargs
    ):
        self.offline_replay_buffer = offline_replay_buffer
        self.online_replay_buffer = online_replay_buffer
        self.sample_online_fraction = sample_online_fraction
        if not self.sample_online_fraction:
            self.sample_online_fraction = self.online_replay_buffer.max_size / (self.online_replay_buffer.max_size + self.offline_replay_buffer.max_size)
        self.online_mode = online_mode

        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_path(self, path):
        if self.online_mode:
            self.online_replay_buffer.add_path(path)
        else:
            self.offline_replay_buffer.add_path(path)
        self._size = self.num_steps_can_sample()

    def num_steps_can_sample(self):
        return self.online_replay_buffer.num_steps_can_sample() + self.offline_replay_buffer.num_steps_can_sample()

    def terminate_episode(self, *args, **kwargs):
        pass

    def get_replay_buffer(self):
        if self.online_mode:
            return self.online_replay_buffer
        else:
            return self.offline_replay_buffer

    def random_batch(self, batch_size):
        online_batch_size = min(int(self.sample_online_fraction*batch_size), self.online_replay_buffer.num_steps_can_sample())
        offline_batch_size = batch_size - online_batch_size
        online_batch = self.online_replay_buffer.random_batch(online_batch_size)
        offline_batch = self.offline_replay_buffer.random_batch(offline_batch_size)

        batch = dict()
        for (key, online_batch_value) in online_batch.items():
            assert key in offline_batch
            offline_batch_value = offline_batch[key]
            if key == 'indices':
                batch['online_indices'] = online_batch_value
                batch['offline_indicies'] = offline_batch_value
            else:
                batch[key] = np.concatenate((online_batch_value, offline_batch_value), axis=0)
        return batch
    
    def set_online_mode(self, online_mode):
        assert online_mode in [True, False]
        self.online_mode = online_mode

    def __getstate__(self):
        return dict(
            offline_replay_buffer=self.offline_replay_buffer,
            online_replay_buffer=self.online_replay_buffer,
            sample_online_fraction=self.sample_online_fraction,
            online_mode=self.online_mode,
        )

    def __setstate__(self, d):
        self.offline_replay_buffer = d['offline_replay_buffer']
        self.online_replay_buffer = d['online_replay_buffer']
        self.sample_online_fraction = d['sample_online_fraction']
        self.online_mode = d['online_mode']
