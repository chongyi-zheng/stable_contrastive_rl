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


class AffordanceTrainer(LossFunction):

    def __init__(
            self,
            affordance,
            classifier=None,
            lr=3e-4,
            gradient_clip_value=1.0,
            normalize=False,
            background_subtract=False,
            prediction_mode='zq',

            affordance_pred_weight=10000.,
            affordance_beta=0.5,
            wgan_gen_weight=100.,
            wgan_clip_value=0.01,

            image_dist_thresh=None,

            classifier_noise_level=0.4,

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
        affordance.to(ptu.device)

        self.affordance = affordance
        self.classifier = classifier

        # self.pred_loss_fn = F.smooth_l1_loss
        self.pred_loss_fn = torch.nn.SmoothL1Loss(
            reduction='none').to(ptu.device)

        self.lr = lr
        self.gradient_clip_value = gradient_clip_value

        self.affordance_params = list(self.affordance.parameters())
        self.affordance_optimizer = optim.Adam(self.affordance_params,
                                               lr=self.lr,
                                               weight_decay=weight_decay)

        self.prediction_mode = prediction_mode
        self.affordance_pred_weight = affordance_pred_weight
        self.affordance_beta = affordance_beta

        self.wgan_gen_weight = wgan_gen_weight
        self.wgan_clip_value = wgan_clip_value

        self.image_dist_thresh = image_dist_thresh
        self.classifier_noise_level = classifier_noise_level

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
            ['train', 'test', 'affordance'])

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
            'affordance: %.2f' % (
                self.timer.time_acc['train'],
                self.timer.time_acc['test'],
                self.timer.time_acc['affordance'],
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

        batch = batch['s']

        weights = self._compute_weights(batch, epoch, prefix, should_log)

        self.timer.tic('affordance')
        self.affordance.train()
        self.affordance_optimizer.zero_grad()
        aff_loss, aff_extra = self._compute_affordance_loss(
            batch, weights, epoch, '%s/affordance' % (prefix), should_log)
        s1_pred = aff_extra['s1_pred'].detach()
        aff_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.affordance_params, self.gradient_clip_value)
        self.affordance_optimizer.step()
        self.timer.toc('affordance')

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's0': batch[:, 0],
                's1': batch[:, 1],
                's1_pred': s1_pred,
            }

            self._plot_images(batch, plot_data, epoch, prefix)

        self.timer.toc('train')

    def test_batch(self, epoch, batch, should_log):
        self.timer.tic('test')
        prefix = 'test'

        batch = batch['s']

        weights = self._compute_weights(batch, epoch, prefix, should_log)

        aff_loss, aff_extra = self._compute_affordance_loss(
            batch, weights, epoch, '%s/affordance' % (prefix), should_log)
        s1_pred = aff_extra['s1_pred'].detach()

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's0': batch[:, 0],
                's1': batch[:, 1],
                's1_pred': s1_pred,
            }

            self._plot_images(batch, plot_data, epoch, prefix)

        self.timer.toc('test')

    def compute_loss(self, batch, epoch, prefix):
        return 0.0

    def _compute_weights(self, batch, epoch, prefix, should_log):
        if self.image_dist_thresh is None:
            return None

        batch_size = batch.shape[0]
        image_dists = torch.norm(
            (batch[:, 0].view(batch_size, -1) -
             batch[:, 1].view(batch_size, -1)),
            dim=-1)
        weights = (image_dists >= self.image_dist_thresh).to(torch.float32)

        if should_log:
            self.tf_logger.log_histogram(
                '%s/%s' % (prefix, 'image_dists'),
                ptu.get_numpy(weights),
                epoch)
            self.tf_logger.log_histogram(
                '%s/%s' % (prefix, 'weights'),
                ptu.get_numpy(image_dists),
                epoch)
            self.tf_logger.log_value(
                '%s/%s' % (prefix, 'weights_sum'),
                weights.mean().item(),
                epoch)

        return weights

    def _compute_affordance_loss(
            self, batch, weights, epoch, prefix, should_log):

        s0 = batch[:, 0]
        s1 = batch[:, 1]

        assert s0.shape[-1] == self.affordance.data_root_len
        assert s0.shape[-2] == self.affordance.data_root_len
        assert s1.shape[-1] == self.affordance.data_root_len
        assert s1.shape[-2] == self.affordance.data_root_len

        loss = 0.0

        (z_mu, z_logvar), z, s1_pred = self.affordance(s1, cond=s0)

        batch_size = s0.shape[0]
        loss_pred = self.pred_loss_fn(
            s1_pred.view(batch_size, -1),
            s1.view(batch_size, -1)).mean(-1)
        kld = - 0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)

        loss += (
            self.affordance_pred_weight * loss_pred +
            self.affordance_beta * kld
        )

        if weights is not None:
            loss = torch.mean(loss * weights) / (torch.mean(weights) + 1e-8)
            kld = torch.mean(kld * weights) / (torch.mean(weights) + 1e-8)
            loss_pred = torch.mean(
                loss_pred * weights) / (torch.mean(weights) + 1e-8)
        else:
            loss = loss.mean()
            kld = kld.mean()
            loss_pred = loss_pred.mean()

        extra = {
            'kld': kld,
            'loss_pred': loss_pred,

            's0': s0,
            's1': s1,
            's1_pred': s1_pred,

            'z_mu': z_mu,
            'z_logvar': z_logvar,

            'beta': ptu.from_numpy(np.array(self.affordance_beta)),
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in ['kld', 'loss_pred', 'beta']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)
            for key in ['s0', 's1', 's1_pred', 'z_mu', 'z_logvar']:
                self.tf_logger.log_histogram(
                    '%s/%s' % (prefix, key),
                    ptu.get_numpy(extra[key]),
                    epoch)

        return loss, extra

    def _compute_classifier_loss(
            self, batch, weights, epoch, prefix, should_log):
        loss = 0.0
        extra = {}

        return loss, extra

    def _plot_images(self, batch, plot_data, epoch, prefix):
        batch = batch[:self.num_plots]

        for key in plot_data.keys():
            if plot_data[key] is not None:
                plot_data[key] = plot_data[key][:self.num_plots]

        s1_pred = plot_data['s1_pred']
        image = torch.cat(
            [batch[:, 0], batch[:, 1], s1_pred],
            dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_pred_image' % (prefix),
            image[:self.num_plots],
            epoch)

        # Sample goals.
        s0 = plot_data['s0']
        goal_preds = self._sample_goals(s0)
        image = torch.cat(
            [batch[:, 0], batch[:, 1]] + goal_preds,
            dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_sampled_goals' % (prefix),
            image[:self.num_plots],
            epoch)

    def _sample_goals(self, s0):
        s0 = s0.view(
            -1,
            3,
            self.affordance.data_root_len,
            self.affordance.data_root_len)

        s1_preds = []
        for _ in range(self.num_plots):
            z = self.affordance.sample_prior(s0.shape[0])

            z = ptu.from_numpy(z)
            s1_pred = self.affordance.decode(z, cond=s0)
            s1_preds.append(s1_pred)

        return s1_preds
