import abc
from collections import OrderedDict

from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core.timer import timer
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import (
    PathCollector,
    StepCollector,
)


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            num_epochs,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            num_epochs,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        self.training_mode(False)
        if self.min_num_steps_before_training > 0 and self.epoch == 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
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
                for _ in range(self.num_expl_steps_per_train_loop):
                    timer.start_timer('exploration sampling', unique=False)
                    self.expl_data_collector.collect_new_steps(
                        self.max_path_length,
                        1,  # num steps
                        discard_incomplete_paths=False,
                    )
                    timer.stop_timer('exploration sampling')

                    timer.start_timer('training', unique=False)
                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    timer.stop_timer('training')
                    self.training_mode(False)

            timer.start_timer('replay buffer data storing', unique=False)
            new_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(new_expl_paths)
            timer.stop_timer('replay buffer data storing')

        log_stats = self._get_diagnostics()

        return log_stats, False
