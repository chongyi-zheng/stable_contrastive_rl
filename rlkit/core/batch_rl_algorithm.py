from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger
# from rlkit.core import eval_util
# from rlkit.data_management.replay_buffer import ReplayBuffer
# from rlkit.samplers.data_collector.path_collector import PathCollector
from rlkit.core.rl_algorithm import BaseRLAlgorithm


class BatchRLAlgorithm(BaseRLAlgorithm):

    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            num_evals_per_train_loop=10,
            min_num_steps_before_training=0,
            # negative epochs are offline, positive epochs are online
            start_epoch=0,
            num_online_trains_per_train_loop=None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        if not num_online_trains_per_train_loop:
            self.num_online_trains_per_train_loop = num_trains_per_train_loop
        else:
            self.num_online_trains_per_train_loop = num_online_trains_per_train_loop  # NOQA
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_evals_per_train_loop = num_evals_per_train_loop
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.start_epoch = start_epoch
        self.epoch = self.start_epoch

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        timer.return_global_times = True
        self.offline_rl = True
        for epoch in range(self.start_epoch, self.num_epochs):
            self.offline_rl = epoch < 0
            self._begin_epoch()
            timer.start_timer('saving')
            logger.save_itr_params(self.epoch, self._get_snapshot())
            timer.stop_timer('saving')
            log_dict, _ = self._train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self._end_epoch()
        logger.save_itr_params(self.epoch, self._get_snapshot())

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            assert not self.offline_rl
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            if self.eval_data_collector is not None:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                if self.expl_data_collector is not None:
                    timer.start_timer('exploration sampling', unique=False)
                    if self.epoch >= 0 or self.epoch % self._offline_expl_epoch_freq == 0:  # NOQA
                        new_expl_paths = self.expl_data_collector.collect_new_paths(  # NOQA
                            self.max_path_length,
                            self.num_expl_steps_per_train_loop,
                            discard_incomplete_paths=False,
                        )
                    timer.stop_timer('exploration sampling')

                    timer.start_timer(
                        'replay buffer data storing', unique=False)
                    if not self.offline_rl:
                        self.replay_buffer.add_paths(new_expl_paths)
                    timer.stop_timer('replay buffer data storing')

                timer.start_timer('training', unique=False)
                num_trains = self.num_trains_per_train_loop
                if self.epoch >= 0:
                    num_trains = self.num_online_trains_per_train_loop
                for _ in range(num_trains):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                timer.stop_timer('training')

                # TODO
                if self.eval_replay_buffer is not None:
                    timer.start_timer('validation', unique=False)
                    num_evals = self.num_evals_per_train_loop
                    for _ in range(num_evals):
                        eval_data = self.eval_replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.eval(eval_data)
                    timer.stop_timer('validation')

        log_stats = self._get_diagnostics()
        return log_stats, False
