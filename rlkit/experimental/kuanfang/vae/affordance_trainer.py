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
from rlkit.experimental.kuanfang.utils import image_util


class AffordanceTrainer(LossFunction):

    def __init__(
            self,
            vqvae,
            affordance,
            classifier=None,
            use_pretrained_vqvae=False,
            lr=3e-4,
            gradient_clip_value=1.0,
            normalize=False,
            background_subtract=False,
            prediction_mode='zq',

            affordance_pred_weight=10000.,
            affordance_beta=1.0,

            image_dist_thresh=None,

            num_vqvae_warmup_epochs=0,

            train_classifier_interval=10,
            classifier_noise_level=0.4,

            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            weight_decay=0,
            num_epochs=None,
            num_plots=4,
            num_plot_epochs=100,
            num_goal_samples=4,
            tf_logger=None,

            augment_image=False,
    ):
        self.num_epochs = num_epochs
        self.num_plots = num_plots
        self.num_plot_epochs = num_plot_epochs
        self.num_goal_samples = num_goal_samples
        vqvae.to(ptu.device)
        affordance.to(ptu.device)

        self.vqvae = vqvae
        self.affordance = affordance
        self.classifier = classifier

        self.use_pretrained_vqvae = use_pretrained_vqvae

        self.imsize = vqvae.imsize
        self.representation_size = vqvae.representation_size
        self.input_channels = vqvae.input_channels
        self.imlength = vqvae.imlength
        self.embedding_dim = vqvae.embedding_dim
        self.root_len = vqvae.root_len

        # self.pred_loss_fn = F.smooth_l1_loss
        self.pred_loss_fn = torch.nn.SmoothL1Loss(
            reduction='none').to(ptu.device)
        self.cls_loss_fn = torch.nn.BCEWithLogitsLoss(
            reduction='none').to(ptu.device)

        self.lr = lr
        self.gradient_clip_value = gradient_clip_value

        if not self.use_pretrained_vqvae:
            self.vqvae_params = list(self.vqvae.parameters())
            self.vqvae_optimizer = optim.Adam(self.vqvae_params,
                                              lr=self.lr,
                                              weight_decay=weight_decay)

        self.affordance_params = list(self.affordance.parameters())
        self.affordance_optimizer = optim.Adam(self.affordance_params,
                                               lr=self.lr,
                                               weight_decay=weight_decay)

        if self.classifier is not None:
            self.cls_params = list(self.classifier.parameters())
            self.cls_optimizer = optim.Adam(self.cls_params,
                                            lr=self.lr,
                                            weight_decay=weight_decay)

        self.prediction_mode = prediction_mode
        self.affordance_pred_weight = affordance_pred_weight
        self.affordance_beta = affordance_beta

        self.image_dist_thresh = image_dist_thresh

        self.num_vqvae_warmup_epochs = num_vqvae_warmup_epochs

        self.classifier_noise_level = classifier_noise_level
        self.train_classifier_interval = train_classifier_interval

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
            ['train', 'test', 'vqvae', 'affordance', 'cls'])

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
            should_train_classifier = (b % self.train_classifier_interval == 0)
            self.train_batch(epoch, batch, should_log, should_train_classifier)
        self.eval_statistics['train/epoch_duration'].append(
            time.time() - start_time)

        print(
            '[Training Time] train: %.2f, test: %.2f, '
            'vqvae: %.2f, affordance: %.2f, cls: %.2f' % (
                self.timer.time_acc['train'],
                self.timer.time_acc['test'],
                self.timer.time_acc['vqvae'],
                self.timer.time_acc['affordance'],
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

    def train_batch(self, epoch, batch, should_log, should_train_classifier):
        self.timer.tic('train')
        prefix = 'train'

        self.num_batches += 1
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches

        s = batch['s']
        if self.augment_image:
            s_shape = s.shape
            C, H, W = s_shape[-3:]
            s = s.reshape(-1, C, H, W)
            s = self.image_augment(s)
            s = s.reshape(s_shape)

        if not self.use_pretrained_vqvae:
            self.timer.tic('vqvae')
            self.vqvae.train()
            self.vqvae_optimizer.zero_grad()
            vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
                s, epoch, prefix, should_log)
            h = vqvae_extra['h'].detach()
            s_recon = vqvae_extra['s_recon'].detach()
            vqvae_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.vqvae_params, self.gradient_clip_value)
            self.vqvae_optimizer.step()
            self.timer.toc('vqvae')
        elif 'h' not in batch or self.augment_image:
            self.timer.tic('vqvae')
            vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
                s, epoch, prefix, should_log)
            h = vqvae_extra['h'].detach()
            s_recon = vqvae_extra['s_recon'].detach()
            self.timer.toc('vqvae')
        else:
            h = batch['h']
            s_recon = None

        self.timer.tic('affordance')
        self.affordance.train()
        self.affordance_optimizer.zero_grad()

        aff_loss, aff_extra = self.compute_affordance_loss(
            s, h, epoch, prefix, should_log)
        h_pred = aff_extra['h_pred'].detach()

        aff_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.affordance_params, self.gradient_clip_value)
        self.affordance_optimizer.step()
        self.timer.toc('affordance')

        if self.classifier is not None and should_train_classifier:
            self.timer.tic('cls')
            self.classifier.train()
            self.cls_optimizer.zero_grad()
            cls_loss = 0.0

            cls_loss, _ = self.compute_classifier_loss(
                h, h_pred, epoch, prefix, should_log)

            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.cls_params, self.gradient_clip_value)
            self.cls_optimizer.step()
            self.timer.toc('cls')

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's': s,
                'h': h,
                'h_pred': h_pred,
                's_recon': s_recon,
            }

            self._plot_images(plot_data, epoch, prefix)

        self.timer.toc('train')

    def test_batch(self, epoch, batch, should_log):
        self.timer.tic('test')
        prefix = 'test'

        s = batch['s']
        if self.augment_image:
            s_shape = s.shape
            C, H, W = s_shape[-3:]
            s = s.reshape(-1, C, H, W)
            s = self.image_augment(s)
            s = s.reshape(s_shape)

        if not self.use_pretrained_vqvae or 'h' not in batch or self.augment_image:
            vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
                s, epoch, prefix, should_log)
            h = vqvae_extra['h'].detach()
            s_recon = vqvae_extra['s_recon'].detach()
        else:
            h = batch['h']
            s_recon = None

        aff_loss, aff_extra = self.compute_affordance_loss(
            s, h, epoch, prefix, should_log)
        h_pred = aff_extra['h_pred'].detach()

        if self.classifier is not None:
            cls_loss, _ = self.compute_classifier_loss(
                h, h_pred, epoch, prefix, should_log)

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's': s,
                'h': h,
                'h_pred': h_pred,
                's_recon': s_recon,
            }

            self._plot_images(plot_data, epoch, prefix)

        self.timer.toc('test')

    def compute_loss(self, batch, epoch, prefix):
        return 0.0

    def _compute_weights(self, s0, sg, epoch, prefix, should_log):
        if self.image_dist_thresh is None:
            return None

        image_dists = torch.norm(
            (torch.flatten(s0, -3, -1) -
             torch.flatten(sg, -3, -1)),
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

    def compute_affordance_loss(self, s, h, epoch, prefix, should_log):

        assert h.shape[-1] == self.root_len
        assert h.shape[-2] == self.root_len

        loss = 0.0
        loss_pred = 0.0
        kld = 0.0
        extra = {}

        h_last = h[:, 0]
        h_pred = [h_last]
        num_steps = h.shape[1] - 1
        for t in range(num_steps):
            h_target = h[:, t + 1]

            prefix_t = '%s/affordance_t%d' % (prefix, t + 1)

            weights_t = self._compute_weights(
                s[:, t], s[:, t + 1], epoch, prefix, should_log=False)

            loss_t, extra_t = self._compute_affordance_loss(
                h_last,
                h_target,
                weights_t,
                epoch,
                prefix_t,
                should_log)

            loss += loss_t
            loss_pred += extra_t['loss_pred']
            kld += extra_t['kld']

            h_pred_t = extra_t['h1_pred'].detach()
            h_pred.append(h_pred_t)

            # If the transition is trivial, then skip this step and directly
            # copy the previous step as the context of the next step.
            if weights_t is None:
                h_last = h_pred_t
            else:
                h_last = torch.where(
                    weights_t[:, None, None, None] > 0,
                    h_pred_t,
                    h_last)

        loss /= num_steps
        loss_pred /= num_steps
        kld /= num_steps
        h_pred = torch.stack(h_pred, 1)

        extra = {
            'kld': kld,
            'loss_pred': loss_pred,
            'beta': ptu.from_numpy(np.array(self.affordance_beta)),

            'h_pred': h_pred,
        }

        if should_log:
            for key in ['kld', 'loss_pred', 'beta']:
                self.tf_logger.log_value(
                    '%s/affordance_all/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)

        return loss, extra

    def _compute_vqvae_loss(self, s, epoch, prefix, should_log):
        num_samples = s.shape[0]
        num_steps = s.shape[1]

        loss = 0.0
        loss_vq = 0.0
        loss_recon = 0.0

        h = []
        s_recon = []

        for t in range(num_steps):
            s_t = s[:, t]
            vqvae_loss_t, vqvae_extra_t = self.vqvae.compute_loss(s_t)
            loss += vqvae_loss_t
            loss_vq += vqvae_extra_t['loss_vq']
            loss_recon += vqvae_extra_t['loss_recon']

            h_t = vqvae_extra_t[self.prediction_mode]
            h.append(h_t)

            s_recon_t = vqvae_extra_t['recon']
            s_recon.append(s_recon_t)

            # self.eval_statistics[
            #     '%s/vqvae/t%d/%s' % (prefix, t, 'loss')].append(
            #         vqvae_loss_t.item())
            #
            # for key in ['loss_vq', 'loss_recon']:
            #     self.eval_statistics[
            #         '%s/vqvae/t%d/%s' % (prefix, t, key)].append(
            #             vqvae_extra_t[key].item())

        s_recon = torch.stack(s_recon, 1)
        h = torch.stack(h, 1)
        h = h.view(
            num_samples, num_steps,
            self.embedding_dim, self.root_len, self.root_len)

        extra = {
            'h': h,
            's_recon': s_recon,
        }

        if should_log:
            self.tf_logger.log_value(
                '%s/vqave/%s' % (prefix, 'loss'),
                loss.item(),
                epoch)
            # self.tf_logger.log_value(
            #     '%s/vqave/%s' % (prefix, 'loss_vq'),
            #     loss.item(),
            #     epoch)
            # self.tf_logger.log_value(
            #     '%s/vqave/%s' % (prefix, 'loss_recon'),
            #     loss.item(),
            #     epoch)

        return loss, extra

    def _compute_affordance_loss(
            self, h0, h1, weights, epoch, prefix, should_log):

        assert h0.shape[-1] == self.root_len
        assert h0.shape[-2] == self.root_len
        assert h1.shape[-1] == self.root_len
        assert h1.shape[-2] == self.root_len

        loss = 0.0

        zqs_t0 = h0
        zqs_t1 = h1

        (z_mu, z_logvar), z, zes_pred = self.affordance(zqs_t1, cond=zqs_t0)
        _, zqs_pred = self.vqvae.vector_quantizer(zes_pred)
        h1_pred = zqs_pred.detach()

        batch_size = h0.shape[0]
        loss_pred = self.pred_loss_fn(
            zes_pred.view(batch_size, -1),
            zqs_t1.view(batch_size, -1)).mean(-1)
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

            'h0': h0,
            'h1': h1,
            'h1_pred': h1_pred,

            'z_mu': z_mu,
            'z_logvar': z_logvar,

        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in ['kld', 'loss_pred']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)
            for key in [
                    'h0', 'h1', 'h1_pred',
                    'z_mu', 'z_logvar']:
                self.tf_logger.log_histogram(
                    '%s/%s' % (prefix, key),
                    ptu.get_numpy(extra[key]),
                    epoch)

        return loss, extra

    def compute_classifier_loss(self, h, h_pred, epoch, prefix, should_log):

        cls_loss_0, _ = self._compute_classifier_loss(
            h, h,
            None, epoch, '%s/cls_rr' % (prefix), should_log)
        cls_loss_1, _ = self._compute_classifier_loss(
            h, h_pred,
            None, epoch, '%s/cls_rf' % (prefix), should_log)
        cls_loss_2, _ = self._compute_classifier_loss(
            h_pred, h,
            None, epoch, '%s/cls_rf' % (prefix), should_log)
        cls_loss_3, _ = self._compute_classifier_loss(
            h_pred, h_pred,
            None, epoch, '%s/cls_ff' % (prefix), should_log)
        cls_loss = cls_loss_0 + cls_loss_1 + cls_loss_2 + cls_loss_3

        return cls_loss, {}

    def _compute_classifier_loss(
            self,
            h0,
            h1,
            weights,
            epoch,
            prefix,
            should_log):
        assert h0.shape[-1] == self.root_len
        assert h0.shape[-2] == self.root_len
        assert h1.shape[-1] == self.root_len
        assert h1.shape[-2] == self.root_len

        batch_size = h0.shape[0]
        num_steps = h0.shape[1]

        _h0 = torch.cat([
            h0[:, :-1],
            h0[:, :-3],
        ], 1).view(
            -1,
            h0.shape[-3],
            self.root_len,
            self.root_len)

        _h1 = torch.cat([
            h1[:, 1:],
            h1[:, 3:],
        ], 1).view(
            -1,
            h1.shape[-3],
            self.root_len,
            self.root_len)

        logits = self.classifier(
            h0=_h0,
            h1=_h1,
        )

        targets = torch.cat(
            [
                torch.ones((batch_size * (num_steps - 1), 1),
                           dtype=torch.float32),
                torch.zeros((batch_size * (num_steps - 3), 1),
                            dtype=torch.float32),
            ], 0).to(ptu.device)

        loss = self.cls_loss_fn(logits, targets)

        # if weights is not None:
        #     loss = torch.mean(loss * weights) / (torch.mean(weights) + 1e-8)
        # else:
        #     loss = loss.mean()

        loss = loss.mean()

        preds = (logits > 0).to(torch.float32)
        acc = torch.sum(
            (preds == targets).to(torch.float32)) / float(targets.shape[0])

        extra = {
            'acc': acc,
            'preds': preds,
            'targets': targets,
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in ['acc']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)

        return loss, extra

    def _plot_images(self, plot_data, epoch, prefix):
        for key in plot_data.keys():
            if plot_data[key] is not None:
                plot_data[key] = plot_data[key][:self.num_plots]

        s = plot_data['s']
        s_recon = plot_data['s_recon']
        h_pred = plot_data['h_pred']

        batch_size = h_pred.shape[0]
        num_steps = h_pred.shape[1]

        s = torch.unbind(s, 1)
        image = torch.cat(s, dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_original' % (prefix),
            image[:self.num_plots],
            epoch)

        if not self.use_pretrained_vqvae:
            image = torch.cat(torch.unbind(s_recon, 1), dim=-2) + 0.5
            image = image.permute(0, 2, 3, 1).contiguous()
            image = ptu.get_numpy(image)
            self.tf_logger.log_images(
                '%s_s_recon' % (prefix),
                image[:self.num_plots],
                epoch)

        h_pred = h_pred.view(batch_size * num_steps,
                             h_pred.shape[-3],
                             h_pred.shape[-2],
                             h_pred.shape[-1])
        s_pred = self.vqvae.decode(h_pred, self.prediction_mode)
        s_pred = s_pred.new(batch_size,
                            num_steps,
                            s_pred.shape[-3],
                            s_pred.shape[-2],
                            s_pred.shape[-1])

        s_pred = torch.unbind(s_pred, 1)
        image = torch.cat(s_pred, dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_pred' % (prefix),
            image[:self.num_plots],
            epoch)

        # Sample goals.
        s0 = plot_data['s'][:, 0]
        s0 = [s0] * self.num_goal_samples
        s0 = torch.stack(s0, 0)

        h0 = plot_data['h'][:, 0]
        h0 = [h0] * self.num_goal_samples
        h0 = torch.stack(h0, 0)
        h0 = h0.view(self.num_goal_samples * self.num_plots,
                     h0.shape[-3],
                     h0.shape[-2],
                     h0.shape[-1])

        goal_preds = [s0]
        for t in range(num_steps - 1):
            h1_pred, goal_pred = self._sample_goals(h0)

            goal_pred = goal_pred.view(
                self.num_goal_samples,
                self.num_plots,
                goal_pred.shape[-3],
                goal_pred.shape[-2],
                goal_pred.shape[-1])
            goal_preds.append(goal_pred)

            h0 = h1_pred

        goal_preds = torch.stack(goal_preds, 2)
        goal_preds = torch.unbind(goal_preds, 0)

        for i in range(self.num_goal_samples):
            goal_pred = goal_preds[i]
            goal_pred = torch.unbind(goal_pred, 1)
            image = torch.cat(goal_pred, dim=-2) + 0.5
            image = image.permute(0, 2, 3, 1).contiguous()
            image = ptu.get_numpy(image)
            self.tf_logger.log_images(
                '%s_sampled_goals_%d' % (prefix, i),
                image[:self.num_plots],
                epoch)

    def _sample_goals(self, h0):
        assert h0.shape[-3] == self.embedding_dim
        assert h0.shape[-2] == self.root_len
        assert h0.shape[-1] == self.root_len

        z = self.affordance.sample_prior(h0.shape[0])
        z = ptu.from_numpy(z)
        h1_pred = self.affordance.decode(z, cond=h0)

        goal_pred = self.vqvae.decode(h1_pred, self.prediction_mode)

        return h1_pred, goal_pred

    def dump_reconstructions(self, epoch):
        pass

    def dump_samples(self, epoch):
        return
