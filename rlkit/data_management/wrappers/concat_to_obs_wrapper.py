from rlkit.data_management.wrappers.proxy_buffer import ProxyBuffer
import numpy as np

class ConcatToObsWrapper(ProxyBuffer):
    def __init__(self, replay_buffer, keys_to_concat):
        """keys_to_concat: list of strings"""
        super().__init__(replay_buffer)
        self.keys_to_concat = keys_to_concat

    def random_batch(self, batch_size):
        batch = self._wrapped_buffer.random_batch(batch_size)
        obs = batch['observations']
        next_obs = batch['next_observations']
        to_concat = [batch[key] for key in self.keys_to_concat]
        batch['observations'] = np.concatenate([obs] + to_concat, axis=1)
        batch['next_observations'] = np.concatenate([next_obs] + to_concat, axis=1)
        return batch
