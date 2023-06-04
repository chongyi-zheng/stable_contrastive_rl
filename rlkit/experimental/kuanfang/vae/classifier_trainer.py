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


class ClassifierTrainer(LossFunction):

    def __init__(
            self,
            vqvae,
            affordance,
            classifier,
            use_pretrained_vqvae=False,
            lr=3e-4,
            gradient_clip_value=1.0,
            normalize=False,
            background_subtract=False,
            prediction_mode='zq',

            image_dist_thresh=None,

            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            weight_decay=0,
            num_epochs=None,
            num_plots=4,
            num_plot_epochs=100,
            tf_logger=None,
    ):
        self.num_epochs = num_epochs
        self.num_plots = num_plots
        self.num_plot_epochs = num_plot_epochs
        vqvae.to(ptu.device)

        self.vqvae = vqvae
        self.affordance = affordance
        self.classifier = classifier

        self.imsize = vqvae.imsize
        self.representation_size = vqvae.representation_size
        self.input_channels = vqvae.input_channels
        self.imlength = vqvae.imlength
        self.embedding_dim = vqvae.embedding_dim
        self.root_len = vqvae.root_len

        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction='none').to(ptu.device)

        self.lr = lr
        self.gradient_clip_value = gradient_clip_value

        self.cls_params = list(self.classifier.parameters())
        self.cls_optimizer = optim.Adam(self.cls_params,
                                        lr=self.lr,
                                        weight_decay=weight_decay)

        self.prediction_mode = prediction_mode
        self.image_dist_thresh = image_dist_thresh

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
            ['train', 'test', 'vqvae', 'affordance', 'gan', 'cls'])

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
            '[Training Time] train: %.2f, test: %.2f, '
            'vqvae: %.2f, affordance: %.2f, gan: %.2f, cls: %.2f' % (
                self.timer.time_acc['train'],
                self.timer.time_acc['test'],
                self.timer.time_acc['vqvae'],
                self.timer.time_acc['affordance'],
                self.timer.time_acc['gan'],
                self.timer.time_acc['cls'],
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

        self.num_batches += 1
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches

        h0 = batch['h'][:, 0]
        h1 = batch['h'][:, 1]
        h2 = batch['h'][:, 2]
        dt1 = batch['dt1']
        dt2 = batch['dt2']
        batch = batch['s']

        # weights = None
        weights = self._compute_weights(batch, epoch, prefix, should_log)

        self.timer.tic('cls')
        self.classifier.train()
        self.cls_optimizer.zero_grad()
        cls_loss = 0.0

        cls_loss, cls_extra = self._compute_classifier_loss(
            h0, h1, h2, weights,
            epoch, '%s/cls' % (prefix), should_log)

        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.cls_params, self.gradient_clip_value)
        self.cls_optimizer.step()
        self.timer.toc('cls')

        self.tf_logger.log_histogram(
            '%s/%s' % (prefix, 'dt1'),
            ptu.get_numpy(dt1),
            epoch)
        self.tf_logger.log_histogram(
            '%s/%s' % (prefix, 'dt2'),
            ptu.get_numpy(dt2),
            epoch)

        self.timer.toc('train')

    def test_batch(self, epoch, batch, should_log):
        self.timer.tic('test')
        prefix = 'test'

        h0 = batch['h'][:, 0]
        h1 = batch['h'][:, 1]
        h2 = batch['h'][:, 2]
        dt1 = batch['dt1']
        dt2 = batch['dt2']
        batch = batch['s']

        # weights = None
        weights = self._compute_weights(batch, epoch, prefix, should_log)

        cls_loss = 0.0

        cls_loss, cls_extra = self._compute_classifier_loss(
            h0, h1, h2, weights,
            epoch, '%s/cls' % (prefix), should_log)
        cls_loss += cls_loss

        self.timer.toc('test')

        self.tf_logger.log_histogram(
            '%s/%s' % (prefix, 'dt1'),
            ptu.get_numpy(dt1),
            epoch)
        self.tf_logger.log_histogram(
            '%s/%s' % (prefix, 'dt2'),
            ptu.get_numpy(dt2),
            epoch)

        if should_log and epoch % self.num_plot_epochs == 0:
            self._plot_images(batch, epoch, prefix)

    def compute_loss(self, batch, epoch, prefix):
        return 0.0

    def _compute_weights(self, batch, epoch, prefix, should_log):
        if self.image_dist_thresh is None:
            return None

        batch_size = batch.shape[0]
        image_dists_1 = torch.norm(
            (batch[:, 0].view(batch_size, -1) -
             batch[:, 1].view(batch_size, -1)),
            dim=-1)
        image_dists_2 = torch.norm(
            (batch[:, 0].view(batch_size, -1) -
             batch[:, 2].view(batch_size, -1)),
            dim=-1)

        weights = (
            (image_dists_1 >= self.image_dist_thresh) &
            (image_dists_2 >= self.image_dist_thresh)
        ).to(torch.float32)

        if should_log:
            # self.tf_logger.log_histogram(
            #     '%s/%s' % (prefix, 'image_dists'),
            #     ptu.get_numpy(image_dists),
            #     epoch)
            self.tf_logger.log_histogram(
                '%s/%s' % (prefix, 'weights'),
                ptu.get_numpy(weights),
                epoch)
            self.tf_logger.log_value(
                '%s/%s' % (prefix, 'weights_sum'),
                weights.mean().item(),
                epoch)

        return weights

    def _compute_classifier_loss(self, h0, h1, h2, weights,
                                 epoch, prefix, should_log):
        assert h0.shape[-1] == self.root_len
        assert h0.shape[-2] == self.root_len
        assert h1.shape[-1] == self.root_len
        assert h1.shape[-2] == self.root_len
        assert h2.shape[-1] == self.root_len
        assert h2.shape[-2] == self.root_len

        h0 = h0.detach()
        h1 = h1.detach()
        h2 = h2.detach()

        logits = self.classifier(
            h0=torch.cat([h0, h0, h1, h2], 0),
            h1=torch.cat([h1, h2, h0, h0], 0),
        )

        batch_size = h0.shape[0]
        targets = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.float32),
                torch.zeros((batch_size, 1), dtype=torch.float32),
                torch.ones((batch_size, 1), dtype=torch.float32),
                torch.zeros((batch_size, 1), dtype=torch.float32),
            ],
            0).to(ptu.device)

        loss = self.cls_loss_fn(logits, targets)

        if weights is not None:
            loss = torch.mean(loss * weights) / (torch.mean(weights) + 1e-8)
        else:
            loss = loss.mean()

        preds = (logits > 0).to(torch.float32)
        acc = torch.sum(
            (preds == targets).to(torch.float32)) / float(targets.shape[0])
        prc = (torch.sum(((preds == 1) & (targets == 1)).to(torch.float32)) /
               torch.sum((preds == 1).to(torch.float32)))
        rec = (torch.sum(((preds == 1) & (targets == 1)).to(torch.float32)) /
               torch.sum((targets == 1).to(torch.float32)))

        extra = {
            'acc': acc,
            'prc': prc,
            'rec': rec,
            'preds': preds,
            'targets': targets,
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in ['acc', 'prc', 'rec']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)

        return loss, extra

    def _plot_images(self, batch, epoch, prefix):
        batch = batch[:self.num_plots]

        image = torch.cat(
            [batch[:, 0], batch[:, 1], batch[:, 2]],
            dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_image' % (prefix),
            image,
            epoch)
