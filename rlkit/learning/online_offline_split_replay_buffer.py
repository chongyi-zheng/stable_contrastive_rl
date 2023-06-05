import random
import numpy as np
import torch

import rlkit.data_management.images as image_np
from rlkit import pythonplusplus as ppp

# import time

from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)


def concat(*x):
    return np.concatenate(x, axis=0)


class OnlineOfflineSplitReplayBuffer(OnlineOfflineSplitReplayBuffer):
    def __init__(
            self,
            offline_replay_buffer,
            online_replay_buffer,
            sample_online_fraction=None,
            online_mode=False,
            **kwargs
    ):
        super().__init__(offline_replay_buffer=offline_replay_buffer,
                         online_replay_buffer=online_replay_buffer,
                         sample_online_fraction=sample_online_fraction,
                         online_mode=online_mode,
                         **kwargs)

    def random_batch(self, batch_size):
        online_batch_size = min(int(self.sample_online_fraction * batch_size),
                                self.online_replay_buffer.num_steps_can_sample())
        offline_batch_size = batch_size - online_batch_size
        online_batch = self.online_replay_buffer.random_batch(online_batch_size)
        offline_batch = self.offline_replay_buffer.random_batch(offline_batch_size)

        # torch.cuda.synchronize()
        # start_time = time.time()
        # batch = dict()
        # for (key, online_batch_value) in online_batch.items():
        #     assert key in offline_batch
        #     offline_batch_value = offline_batch[key]
        #     if key == 'indices':
        #         batch['online_indices'] = online_batch_value
        #         batch['offline_indicies'] = offline_batch_value
        #     else:
        #         batch[key] = np.concatenate((online_batch_value, offline_batch_value), axis=0)
        #
        #         if batch[key].dtype == np.uint8:
        #             batch[key] = image_np.normalize_image(batch[key], dtype=np.float32)
        batch = ppp.treemap(
            concat,
            online_batch,
            offline_batch,
            atomic_type=np.ndarray)

        # torch.cuda.synchronize()
        # end_time = time.time()
        # print("Time to concatenate offline and online data: {} secs".format(end_time - start_time))

        return batch
