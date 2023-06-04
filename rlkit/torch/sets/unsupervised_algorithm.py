import typing
from collections import OrderedDict

from torch.utils import data

from rlkit.core import logger
from rlkit.core.logging import append_log
from rlkit.core.rl_algorithm import _get_epoch_timings
from rlkit.core.timer import timer
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class DictLoader(object):
    r"""Method for iterating over dictionaries."""

    def __init__(self, key_to_batch_sampler: typing.Dict[typing.Any, data.DataLoader]):
        if len(key_to_batch_sampler) == 0:
            raise ValueError("need at least one sampler")
        self.keys, self.samplers = zip(*key_to_batch_sampler.items())

    def __iter__(self):
        # values = []
        for values in zip(*self.samplers):
            yield dict(zip(self.keys, values))

    def __len__(self):
        return len(self.samplers[0])


class UnsupervisedTorchAlgorithm(object):
    def __init__(
            self,
            trainer: TorchTrainer,
            data_loader: DictLoader,
            num_iters: int,
            num_epochs_per_iter=1,
            progress_csv_file_name='progress.csv',
    ):
        self.trainer = trainer
        self.data_loader = data_loader
        self._start_epoch = 0
        self.epoch = self._start_epoch
        self.num_iters = num_iters
        self.num_epochs_per_iter = num_epochs_per_iter
        self.progress_csv_file_name = progress_csv_file_name

    def run(self):
        if self.progress_csv_file_name != 'progress.csv':
            logger.remove_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True
            )
            logger.add_tabular_output(
                self.progress_csv_file_name, relative_to_snapshot_dir=True,
            )
        timer.return_global_times = True
        for _ in range(self.num_iters):
            self._begin_epoch()
            timer.start_timer('saving')
            logger.save_itr_params(self.epoch, self._get_snapshot())
            timer.stop_timer('saving')
            log_dict, _ = self._train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self._end_epoch()
        logger.save_itr_params(self.epoch, self._get_snapshot())
        if self.progress_csv_file_name != 'progress.csv':
            logger.remove_tabular_output(
                self.progress_csv_file_name, relative_to_snapshot_dir=True,
            )
            logger.add_tabular_output(
                'progress.csv', relative_to_snapshot_dir=True,
            )

    def _train(self):
        done = (self.epoch == self.num_iters)
        if done:
            return OrderedDict(), done

        timer.start_timer('training', unique=False)
        for _ in range(self.num_epochs_per_iter):
            for batch in self.data_loader:
                self.trainer.train_from_torch(batch)
        timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        self.trainer.end_epoch(self.epoch)
        self.epoch += 1

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        return snapshot

    def _get_diagnostics(self):
        timer.start_timer('logging', unique=False)
        algo_log = OrderedDict()
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        append_log(algo_log, _get_epoch_timings())
        algo_log['epoch'] = self.epoch
        timer.stop_timer('logging')
        return algo_log

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
