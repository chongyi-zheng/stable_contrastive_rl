import time
import collections
from collections import OrderedDict

import numpy as np
import torch
from torch import optim

from rlkit.core.loss import LossFunction
from rlkit.core import logger
from rlkit.data_management.images import normalize_image
from rlkit.torch import pytorch_util as ptu

from rlkit.experimental.kuanfang.utils.timer import Timer
# from rlkit.experimental.kuanfang.utils import image_util
from rlkit.experimental.chongyiz.utils import image_util


class VqVaeTrainer(LossFunction):

    def __init__(
            self,
            vqvae,
            lr=3e-4,
            gradient_clip_value=1.0,
            normalize=False,
            background_subtract=False,
            prediction_mode='zq',

            train_cls_interval=10,
            classifier_noise_level=0.4,

            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            weight_decay=0,
            num_epochs=None,
            num_plots=4,
            num_plot_epochs=100,  # TODO
            tf_logger=None,

            augment_image=False,
    ):
        self.num_epochs = num_epochs
        self.num_plots = num_plots
        self.num_plot_epochs = num_plot_epochs
        vqvae.to(ptu.device)

        self.vqvae = vqvae

        self.imsize = vqvae.imsize
        self.representation_size = vqvae.representation_size
        self.input_channels = vqvae.input_channels
        self.imlength = vqvae.imlength
        self.embedding_dim = vqvae.embedding_dim
        self.root_len = vqvae.root_len

        self.lr = lr
        self.gradient_clip_value = gradient_clip_value

        self.vqvae_params = list(self.vqvae.parameters())
        self.vqvae_optimizer = optim.Adam(self.vqvae_params,
                                          lr=self.lr,
                                          weight_decay=weight_decay)

        self.prediction_mode = prediction_mode

        self.normalize = normalize
        self.background_subtract = background_subtract
        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )

        self.linearity_weight = linearity_weight
        self.distance_weight = distance_weight
        self.loss_weights = loss_weights

        # stateful tracking variables, reset every epoch
        self.eval_statistics = collections.defaultdict(list)
        self.eval_data = collections.defaultdict(list)
        self.num_batches = 0

        self.tf_logger = tf_logger
        self.timer = Timer(
            ['train', 'test', 'vqvae'])

        self.augment_image = augment_image
        if self.augment_image:
            self.image_augment = image_util.ImageAugment()

    @property
    def log_dir(self):
        return logger.get_snapshot_dir()

    def end_epoch(self, epoch):
        self.eval_statistics = collections.defaultdict(list)
        self.test_last_batch = None

    def get_diagnostics(self):
        stats = OrderedDict()
        for k in sorted(self.eval_statistics.keys()):
            stats[k] = np.mean(self.eval_statistics[k])
        return stats

    def batch_to_device(self, batch):
        if isinstance(batch, dict):
            _batch = batch
            batch = {}
            for key in _batch:
                batch[key] = _batch[key].to(ptu.device)
        else:
            batch = batch.to(ptu.device)

        return batch

    def train_epoch(self, epoch, dataloader, batches=100):
        start_time = time.time()
        self.timer.reset()
        for b in range(batches):
            batch = next(iter(dataloader))
            batch = self.batch_to_device(batch)
            should_log = (b == 0)
            self.train_batch(epoch, batch, should_log)
        self.eval_statistics['train/epoch_duration'].append(
            time.time() - start_time)

        print(
            '[Training Time] train: %.2f, test: %.2f, vqvae: %.2f' % (
                self.timer.time_acc['train'],
                self.timer.time_acc['test'],
                self.timer.time_acc['vqvae'],
            )
        )

    def test_epoch(self, epoch, dataloader, batches=10):
        start_time = time.time()
        for b in range(batches):
            batch = next(iter(dataloader))
            batch = self.batch_to_device(batch)
            should_log = (b == 0)
            self.test_batch(epoch, batch, should_log)
        self.eval_statistics['test/epoch_duration'].append(
            time.time() - start_time)

    def train_batch(self, epoch, batch, should_log):
        self.timer.tic('train')
        prefix = 'train'

        batch = batch['s']

        if self.augment_image:
            batch = self.image_augment(batch)

        self.num_batches += 1
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches

        self.timer.tic('vqvae')
        self.vqvae.train()
        self.vqvae_optimizer.zero_grad()
        vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
            batch, epoch, prefix, should_log)
        s_recon = vqvae_extra['s_recon'].detach()
        vqvae_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.vqvae_params, self.gradient_clip_value)
        self.vqvae_optimizer.step()
        self.timer.toc('vqvae')

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's_recon': s_recon,
            }

            self._plot_images(batch, plot_data, epoch, prefix)

        self.timer.toc('train')

    def test_batch(self, epoch, batch, should_log):
        self.timer.tic('test')
        prefix = 'test'

        batch = batch['s']

        if self.augment_image:
            batch = self.image_augment(batch)

        vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
            batch, epoch, prefix, should_log)
        s_recon = vqvae_extra['s_recon'].detach()

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's_recon': s_recon,
            }

            self._plot_images(batch, plot_data, epoch, prefix)

        self.timer.toc('test')

    def compute_loss(self, batch, epoch, prefix):
        return 0.0

    def _compute_vqvae_loss(self, batch, epoch, prefix, should_log):
        # if 's' in batch:
        #     batch = batch['s']

        vqvae_loss, vqvae_extra = self.vqvae.compute_loss(batch)
        loss = vqvae_loss

        h = vqvae_extra[self.prediction_mode]
        h = h.view(-1, self.embedding_dim, self.root_len, self.root_len)

        s_recon = vqvae_extra['recon']

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            vqvae_loss.item())

        for key in ['loss_vq', 'loss_recon', 'perplexity']:
            self.eval_statistics['%s/%s' % (prefix, key)].append(
                vqvae_extra[key].item())

        extra = {
            'h': h,
            's_recon': s_recon,
        }

        if should_log:
            for key in ['h']:
                self.tf_logger.log_histogram(
                    '%s/%s' % (prefix, key),
                    ptu.get_numpy(extra[key]),
                    epoch)

        return loss, extra

    def _plot_images(self, batch, plot_data, epoch, prefix):
        batch = batch[:self.num_plots]

        for key in plot_data.keys():
            if plot_data[key] is not None:
                plot_data[key] = plot_data[key][:self.num_plots]

        s_recon = plot_data['s_recon']

        image = torch.cat([batch, s_recon], dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_image' % (prefix),
            image[:self.num_plots],
            epoch)

    def dump_reconstructions(self, epoch):
        pass

    def dump_samples(self, epoch):
        return
