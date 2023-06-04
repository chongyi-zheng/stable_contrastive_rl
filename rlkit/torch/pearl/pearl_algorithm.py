from collections import OrderedDict

import numpy as np

from rlkit.core.timer import timer
from rlkit.torch.pearl.buffer import PearlReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


class PearlAlgorithm(TorchBatchRLAlgorithm):
    def __init__(
            self,
            train_task_indices,
            test_task_indices,
            replay_buffer: PearlReplayBuffer,
            *args,
            **kwargs
    ):
        super().__init__(*args, replay_buffer=replay_buffer, **kwargs)
        self.train_task_indices = train_task_indices
        self.test_task_indices = test_task_indices

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0:
            self._initialize_buffers()

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                timer.start_timer('exploration sampling', unique=False)
                task_idx = np.random.choice(self.train_task_indices)
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                    task_idx=task_idx,
                )
                timer.stop_timer('exploration sampling')

                timer.start_timer('replay buffer data storing', unique=False)
                self.replay_buffer.add_paths(new_expl_paths, task_idx)
                timer.stop_timer('replay buffer data storing')

                timer.start_timer('training', unique=False)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False

    def _initialize_buffers(self):
        for task_idx in self.train_task_indices:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                max_path_length=self.max_path_length,
                num_steps=self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                task_idx=task_idx,
            )
            self.replay_buffer.add_paths(init_expl_paths, task_idx)
        # TODO: how should I initialized these buffers?
        for task_idx in self.test_task_indices:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                max_path_length=self.max_path_length,
                num_steps=self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                task_idx=task_idx,
            )
            self.replay_buffer.add_paths(init_expl_paths, task_idx)
        self.expl_data_collector.end_epoch(-1)
