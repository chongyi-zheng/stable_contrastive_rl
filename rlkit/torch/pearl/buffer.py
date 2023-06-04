import numpy as np

from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.replay_buffer import ReplayBuffer


class PearlReplayBuffer(ReplayBuffer):
    """
    This replay replay buffer combines a normal replay buffer with
    an encoder replay buffer, so that samples contain the usual (s, a, r, s') tuple
    as well as a "context" variables used for sampling contexts.
    """

    def __init__(
            self,
            replay_buffer: MultiTaskReplayBuffer,
            encoder_replay_buffer: MultiTaskReplayBuffer,
            embedding_batch_size,
            train_task_indices,
            meta_batch_size,
    ):
        self.replay_buffer = replay_buffer
        self.encoder_replay_buffer = encoder_replay_buffer
        self.embedding_batch_size = embedding_batch_size
        self.train_task_indices = train_task_indices
        self.meta_batch_size = meta_batch_size

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        raise NotImplementedError()

    def terminate_episode(self):
        raise NotImplementedError()

    def num_steps_can_sample(self, **kwargs):
        raise NotImplementedError()
        pass

    def add_paths(self, paths, task_idx):
        self.replay_buffer.add_paths(task_idx, paths)
        self.encoder_replay_buffer.add_paths(task_idx, paths)

    def random_batch(self, batch_size):
        indices = np.random.choice(self.train_task_indices, self.meta_batch_size)

        context_batch = self.encoder_replay_buffer.sample_context(
            indices,
            self.embedding_batch_size
        )
        batch = self.replay_buffer.sample_batch(indices, batch_size)
        batch['context'] = context_batch
        batch['task_indices'] = indices
        return batch

    def sample_context(self, task_idx):
        return self.encoder_replay_buffer.sample_context(task_idx, self.embedding_batch_size)
