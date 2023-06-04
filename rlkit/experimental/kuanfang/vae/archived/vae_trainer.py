import collections
import time
from collections import OrderedDict

import numpy as np
import torch  # NOQA
from torch import optim
from torch.nn import functional as F

from rlkit.core.loss import LossFunction
from rlkit.data_management.images import normalize_image
from rlkit.core import logger
from rlkit.util.ml_util import ConstantSchedule
from rlkit.torch import pytorch_util as ptu


class VaeTrainer(LossFunction):

    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=1e-3,
            normalize=False,
            background_subtract=False,
            weight_decay=0,
            num_epochs=None,
            tf_logger=None,
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        self.num_epochs = num_epochs

        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(self.beta)

        model.to(ptu.device)

        self.model = model

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=self.lr,
            weight_decay=weight_decay,
        )

        self.normalize = normalize
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self._extra_stats_to_log = None

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)
        self.num_batches = 0

        self.tf_logger = tf_logger

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def train_epoch(self, epoch, dataloader, batches=100):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.train_batch(epoch, batch)
            # self.train_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics['train/epoch_duration'].append(
            time.time() - start_time)

    def test_epoch(self, epoch, dataloader, batches=10):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            self.test_batch(epoch, batch)
            # self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics['test/epoch_duration'].append(
            time.time() - start_time)

    def train_batch(self, epoch, batch):
        self.num_batches += 1
        # self.model.train()
        self.optimizer.zero_grad()

        loss = self.compute_loss(batch, epoch, 'train')
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(batch, epoch, 'test')  # NOQA

    def compute_loss(self, batch, epoch, prefix):
        beta = float(self.beta_schedule.get_value(epoch))

        (mu, logvar), z, batch_recon = self.model(batch)

        recon_error = F.mse_loss(batch_recon, batch)

        kld = - 0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        loss = recon_error + beta * kld

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + 'losses'].append(loss.item())
        self.eval_statistics[prefix + 'recon_error'].append(recon_error.item())
        self.eval_statistics[prefix + 'klds'].append(kld.item())
        self.eval_statistics['num_train_batches'].append(self.num_batches)

        for i in range(batch.shape[-1]):
            self.tf_logger.log_histogram(
                'x/dim_%d' % (i), batch[..., i].detach(), epoch)
            self.tf_logger.log_histogram(
                'x_recon/dim_%d' % (i), batch_recon[..., i].detach(), epoch)

        self.tf_logger.log_histogram(prefix + 'z/z', z.detach(), epoch)
        self.tf_logger.log_histogram(prefix + 'z/mu', mu.detach(), epoch)
        self.tf_logger.log_histogram(prefix + 'z/logvar',
                                     logvar.detach(), epoch)

        self.tf_logger.log_images(prefix + 'image/original',
                                  batch[:4],
                                  epoch)
        self.tf_logger.log_images(prefix + 'image/recon',
                                  batch_recon[:4],
                                  epoch)

        return loss

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats
