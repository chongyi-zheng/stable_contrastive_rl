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
            vqvae,
            affordance,
            classifier=None,
            discriminator=None,
            use_pretrained_vqvae=False,
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

            num_vqvae_warmup_epochs=0,
            num_wgan_warmup_epochs=0,

            train_cls_interval=10,
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
        vqvae.to(ptu.device)
        affordance.to(ptu.device)

        self.vqvae = vqvae
        self.affordance = affordance
        self.classifier = classifier
        self.discriminator = discriminator

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

        if self.discriminator is not None:
            self.dis_params = list(self.discriminator.parameters())
            self.dis_optimizer = optim.Adam(self.dis_params,
                                            lr=self.lr,
                                            weight_decay=weight_decay)

        self.prediction_mode = prediction_mode
        self.affordance_pred_weight = affordance_pred_weight
        self.affordance_beta = affordance_beta

        self.wgan_gen_weight = wgan_gen_weight
        self.wgan_clip_value = wgan_clip_value

        self.image_dist_thresh = image_dist_thresh
        self.classifier_noise_level = classifier_noise_level

        self.num_vqvae_warmup_epochs = num_vqvae_warmup_epochs
        self.num_wgan_warmup_epochs = num_wgan_warmup_epochs
        self.train_cls_interval = train_cls_interval

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
            should_train_cls = (b % self.train_cls_interval == 0)
            self.train_batch(epoch, batch, should_log, should_train_cls)
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

    def train_batch(self, epoch, batch, should_log, should_train_cls):
        self.timer.tic('train')
        prefix = 'train'

        self.num_batches += 1
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches

        if not self.use_pretrained_vqvae:
            batch = batch['s']
            self.timer.tic('vqvae')
            self.vqvae.train()
            self.vqvae_optimizer.zero_grad()
            vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
                batch, epoch, prefix, should_log)
            h0 = vqvae_extra['h0'].detach()
            h1 = vqvae_extra['h1'].detach()
            if self.classifier is not None:
                h2 = vqvae_extra['h2'].detach()
            s0_recon = vqvae_extra['s0_recon'].detach()
            s1_recon = vqvae_extra['s1_recon'].detach()
            vqvae_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.vqvae_params, self.gradient_clip_value)
            self.vqvae_optimizer.step()
            self.timer.toc('vqvae')

        else:
            h0 = batch['h'][:, 0]
            h1 = batch['h'][:, 1]
            if self.classifier is not None:
                h2 = batch['h'][:, 2]
            batch = batch['s']
            s0_recon = None
            s1_recon = None

        weights = self._compute_weights(batch, epoch, prefix, should_log)

        self.timer.tic('affordance')
        self.affordance.train()
        self.affordance_optimizer.zero_grad()
        aff_loss, aff_extra = self._compute_affordance_loss(
            h0, h1, weights, epoch, '%s/affordance' % (prefix), should_log)
        h1_pred = aff_extra['h1_pred'].detach()
        aff_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.affordance_params, self.gradient_clip_value)
        self.affordance_optimizer.step()
        self.timer.toc('affordance')

        if self.discriminator is not None:
            self.timer.tic('gan')
            self.discriminator.train()
            self.dis_optimizer.zero_grad()
            dis_loss, _ = self.compute_dis_loss(
                h0, h1, h1_pred, weights,
                epoch, '%s/gan' % (prefix), should_log)
            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.dis_params, self.gradient_clip_value)
            self.dis_optimizer.step()

            for p in self.dis_params:
                p.data.clamp_(-self.wgan_clip_value, self.wgan_clip_value)

            self.timer.toc('gan')

        if self.classifier is not None and should_train_cls:
            self.timer.tic('cls')
            self.classifier.train()
            self.cls_optimizer.zero_grad()
            cls_loss = 0.0

            cls_loss, cls_extra = self._compute_classifier_loss(
                h0, h1, h2, weights,
                epoch, '%s/cls1' % (prefix), should_log)

            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.cls_params, self.gradient_clip_value)
            self.cls_optimizer.step()
            self.timer.toc('cls')

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's0_recon': s0_recon,
                's1_recon': s1_recon,
                'h0': h0,
                'h1_pred': h1_pred,
            }

            if self.classifier:
                plot_data['h0_noisy'] = cls_extra['h0_noisy']
                plot_data['h1_noisy'] = cls_extra['h1_noisy']
                plot_data['h2_noisy'] = cls_extra['h2_noisy']

            self._plot_images(batch, plot_data, epoch, prefix)

        self.timer.toc('train')

    def test_batch(self, epoch, batch, should_log):
        self.timer.tic('test')
        prefix = 'test'

        if not self.use_pretrained_vqvae:
            batch = batch['s']
            vqvae_loss, vqvae_extra = self._compute_vqvae_loss(
                batch, epoch, prefix, should_log)
            h0 = vqvae_extra['h0'].detach()
            h1 = vqvae_extra['h1'].detach()
            if self.classifier is not None:
                h2 = vqvae_extra['h2'].detach()
            s0_recon = vqvae_extra['s0_recon'].detach()
            s1_recon = vqvae_extra['s1_recon'].detach()
        else:
            h0 = batch['h'][:, 0]
            h1 = batch['h'][:, 1]
            if self.classifier is not None:
                h2 = batch['h'][:, 2]
            batch = batch['s']
            s0_recon = None
            s1_recon = None

        weights = self._compute_weights(batch, epoch, prefix, should_log)

        aff_loss, aff_extra = self._compute_affordance_loss(
            h0, h1, weights, epoch, '%s/affordance' % (prefix), should_log)
        h1_pred = aff_extra['h1_pred'].detach()

        if self.discriminator is not None:
            dis_loss, _ = self.compute_dis_loss(
                h0, h1, h1_pred, weights,
                epoch, '%s/gan' % (prefix), should_log)

        if self.classifier is not None:
            cls_loss = 0.0

            cls_loss, cls_extra = self._compute_classifier_loss(
                h0, h1, h2, weights,
                epoch, '%s/cls1' % (prefix), should_log)
            cls_loss += cls_loss

        if should_log and epoch % self.num_plot_epochs == 0:
            plot_data = {
                's0_recon': s0_recon,
                's1_recon': s1_recon,
                'h0': h0,
                'h1_pred': h1_pred,
            }

            if self.classifier:
                plot_data['h0_noisy'] = cls_extra['h0_noisy']
                plot_data['h1_noisy'] = cls_extra['h1_noisy']
                plot_data['h2_noisy'] = cls_extra['h2_noisy']

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

    def _compute_vqvae_loss(self, batch, epoch, prefix, should_log):
        loss = 0.0

        h_ts = []
        recon_ts = []

        if self.classifier is not None:
            t_list = [0, 1, 2]
        else:
            t_list = [0, 1]

        for t in t_list:
            batch_t = batch[:, t]

            vqvae_loss_t, vqvae_extra_t = self.vqvae.compute_loss(batch_t)
            loss += vqvae_loss_t

            h_t = vqvae_extra_t[self.prediction_mode]
            h_ts.append(h_t)

            recon_t = vqvae_extra_t['recon']
            recon_ts.append(recon_t)

            self.eval_statistics[
                '%s/t%d/%s' % (prefix, t, 'loss')].append(
                    vqvae_loss_t.item())

            for key in ['loss_vq', 'loss_recon', 'perplexity']:
                self.eval_statistics['%s/t%d/%s' % (prefix, t, key)].append(
                    vqvae_extra_t[key].item())

        h0 = h_ts[0].view(
            -1, self.embedding_dim, self.root_len, self.root_len)
        h1 = h_ts[1].view(
            -1, self.embedding_dim, self.root_len, self.root_len)

        s0_recon = recon_ts[0]
        s1_recon = recon_ts[1]

        if self.classifier is not None:
            h2 = h_ts[2].view(
                -1, self.embedding_dim, self.root_len, self.root_len)
            s2_recon = recon_ts[2]
        else:
            h2 = torch.zeros_like(h1)
            s2_recon = torch.zeros_like(s1_recon)

        extra = {
            'h0': h0,
            'h1': h1,
            'h2': h2,
            's0_recon': s0_recon,
            's1_recon': s1_recon,
            's2_recon': s2_recon,
        }

        if should_log:
            for key in ['h0', 'h1', 'h2']:
                self.tf_logger.log_histogram(
                    '%s/%s' % (prefix, key),
                    ptu.get_numpy(extra[key]),
                    epoch)

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
        zqs_pred = zes_pred + (zqs_pred - zes_pred).detach()
        h1_pred = zqs_pred

        batch_size = h0.shape[0]
        loss_pred_h = self.pred_loss_fn(
            zes_pred.view(batch_size, -1),
            zqs_t1.view(batch_size, -1)).mean(-1)
        loss_pred_h_zq = self.pred_loss_fn(
            zqs_pred.view(batch_size, -1),
            zqs_t1.view(batch_size, -1)).mean(-1)
        kld = - 0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=-1)

        # VAE-GAN.
        # if self.discriminator is not None:
        #     if epoch <= self.num_wgan_warmup_epochs:
        #         affordance_pred_weight = self.affordance_pred_weight
        #     else:
        #         affordance_pred_weight = 0.0
        #
        #         h0 = h0.detach()
        #         logit_fake = self.discriminator(h0, zes_pred)
        #         gen_loss = -logit_fake.view(-1)
        #
        #         loss += self.wgan_gen_weight * gen_loss
        #
        #         if weights is not None:
        #             gen_loss = torch.mean(
        #                 gen_loss * weights) / (torch.mean(weights) + 1e-8)
        #         else:
        #             gen_loss = gen_loss.mean()
        #
        #         self.eval_statistics['%s/%s' % (prefix, 'gen_loss')].append(
        #             gen_loss.item())
        #
        #         if should_log:
        #             self.tf_logger.log_histogram(
        #                 '%s/%s' % (prefix, 'logit_fake'),
        #                 ptu.get_numpy(logit_fake),
        #                 epoch)

        loss += (
            self.affordance_pred_weight * loss_pred_h +
            self.affordance_beta * kld
        )

        if weights is not None:
            loss = torch.mean(loss * weights) / (torch.mean(weights) + 1e-8)
            kld = torch.mean(kld * weights) / (torch.mean(weights) + 1e-8)
            loss_pred_h = torch.mean(
                loss_pred_h * weights) / (torch.mean(weights) + 1e-8)
            loss_pred_h_zq = torch.mean(
                loss_pred_h_zq * weights) / (torch.mean(weights) + 1e-8)
        else:
            loss = loss.mean()
            kld = kld.mean()
            loss_pred_h = loss_pred_h.mean()
            loss_pred_h_zq = loss_pred_h_zq.mean()

        extra = {
            'kld': kld,
            'loss_pred_h': loss_pred_h,
            'loss_pred_h_zq': loss_pred_h_zq,

            'h0': h0,
            'h1': h1,
            'h1_pred': h1_pred,

            'z_mu': z_mu,
            'z_logvar': z_logvar,

            'beta': ptu.from_numpy(np.array(self.affordance_beta)),
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in [
                    'kld',
                    'loss_pred_h', 'loss_pred_h_zq',
                    'beta']:
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

    def _compute_classifier_loss(self, h0, h1, h2, weights,
                                 epoch, prefix, should_log):
        assert h0.shape[-1] == self.root_len
        assert h0.shape[-2] == self.root_len
        assert h1.shape[-1] == self.root_len
        assert h1.shape[-2] == self.root_len
        assert h2.shape[-1] == self.root_len
        assert h2.shape[-2] == self.root_len

        if self.classifier_noise_level:
            h0_noisy = self._noisy_affordance_reconstruct(
                h1, h0, noise_level=self.classifier_noise_level)
            h1_noisy = self._noisy_affordance_reconstruct(
                h0, h1, noise_level=self.classifier_noise_level)
            h2_noisy = self._noisy_affordance_reconstruct(
                h1, h2, noise_level=self.classifier_noise_level)

            h0 = h0_noisy
            h1 = h1_noisy
            h2 = h2_noisy

        h0 = h0.detach()
        h1 = h1.detach()
        h2 = h2.detach()

        logits = self.classifier(
            h0=torch.cat([h0, h0], 0),
            h1=torch.cat([h1, h2], 0),
        )

        batch_size = h0.shape[0]
        targets = torch.cat(
            [
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

        extra = {
            'acc': acc,
            'preds': preds,
            'targets': targets,

            'h0_noisy': h0_noisy,
            'h1_noisy': h1_noisy,
            'h2_noisy': h2_noisy,
        }

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        if should_log:
            for key in ['acc']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)

            # for key in ['preds', 'targets']:
            #     self.tf_logger.log_histogram(
            #         '%s/%s' % (prefix, key),
            #         cls_extra[key].detach(),
            #         epoch)

        return loss, extra

    def _noisy_affordance_reconstruct(self, h0, h1, noise_level):
        z, _ = self.affordance.encode(h1, cond=h0)
        noisy_z = z + noise_level * ptu.randn(
            1, self.affordance.representation_size)
        h1_recon = self.affordance.decode(noisy_z, cond=h0)
        return h1_recon

    def compute_dis_loss(self, h0, h1, h1_pred, weights,
                         epoch, prefix, should_log):
        logits_real = self.discriminator(h0, h1)
        logits_fake = self.discriminator(h0, h1_pred)

        if weights is not None:
            loss_real = -torch.mean(
                logits_real * weights) / (torch.mean(weights) + 1e-8)
            loss_fake = torch.mean(
                logits_fake * weights) / (torch.mean(weights) + 1e-8)
        else:
            loss_real = -torch.mean(logits_real)
            loss_fake = torch.mean(logits_fake)

        loss = loss_real + loss_fake

        self.eval_statistics['%s/%s' % (prefix, 'loss')].append(
            loss.item())

        extra = {
            'logits_real': logits_real,
            'logits_fake': logits_fake,
            'loss_real': loss_real,
            'loss_fake': loss_fake,
        }

        if should_log:
            for key in ['loss_real', 'loss_fake']:
                self.tf_logger.log_value(
                    '%s/%s' % (prefix, key),
                    extra[key].item(),
                    epoch)
            for key in ['logits_real', 'logits_fake']:
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

        s0_recon = plot_data['s0_recon']
        s1_recon = plot_data['s1_recon']
        h1_pred = plot_data['h1_pred']

        if not self.use_pretrained_vqvae:
            recons = [
                s0_recon,
                s1_recon,
            ]
            for t in [0, 1]:
                image = torch.cat(
                    [batch[:, t], recons[t]],
                    dim=-2) + 0.5
                image = image.permute(0, 2, 3, 1).contiguous()
                image = ptu.get_numpy(image)
                self.tf_logger.log_images(
                    '%s_t%d_image' % (prefix, t),
                    image[:self.num_plots],
                    epoch)

        s1_pred = self.vqvae.decode(h1_pred, self.prediction_mode)
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
        h0 = plot_data['h0']
        _, goal_preds = self._sample_goals(h0)
        image = torch.cat(
            [batch[:, 0], batch[:, 1]] + goal_preds,
            dim=-2) + 0.5
        image = image.permute(0, 2, 3, 1).contiguous()
        image = ptu.get_numpy(image)
        self.tf_logger.log_images(
            '%s_sampled_goals' % (prefix),
            image[:self.num_plots],
            epoch)

        if 'h0_noisy' in plot_data:
            h0_noisy = plot_data['h0_noisy']
            h1_noisy = plot_data['h1_noisy']
            h2_noisy = plot_data['h2_noisy']
            s0_noisy = self.vqvae.decode(h0_noisy, self.prediction_mode)
            s1_noisy = self.vqvae.decode(h1_noisy, self.prediction_mode)
            s2_noisy = self.vqvae.decode(h2_noisy, self.prediction_mode)
            image = torch.cat(
                [
                    s0_noisy,
                    s1_noisy,
                    s2_noisy,
                ],
                dim=-2) + 0.5
            image = image.permute(0, 2, 3, 1).contiguous()
            image = ptu.get_numpy(image)
            self.tf_logger.log_images(
                'z%s_noisy_image' % (prefix),
                image[:self.num_plots],
                epoch)

    def _sample_goals(self, h0):
        h0 = h0.view(
            -1, self.embedding_dim, self.root_len, self.root_len)

        h1_preds = []
        goal_preds = []
        for _ in range(self.num_plots):
            z = self.affordance.sample_prior(h0.shape[0])

            z = ptu.from_numpy(z)
            h1_pred = self.affordance.decode(z, cond=h0)
            h1_preds.append(h1_pred)

            goal_pred = self.vqvae.decode(h1_pred, self.prediction_mode)
            goal_preds.append(goal_pred)

        return h1_preds, goal_preds

    def dump_reconstructions(self, epoch):
        pass

    def dump_samples(self, epoch):
        return
