import time
import collections
from collections import OrderedDict
import itertools

import numpy as np
import torch
from torch import optim

from rlkit.core.loss import LossFunction
from rlkit.core import logger
from rlkit.data_management.images import normalize_image
from rlkit.torch import pytorch_util as ptu

from rlkit.experimental.kuanfang.utils.timer import Timer
from rlkit.experimental.kuanfang.vae.affordance_trainer import AffordanceTrainer


class GCPTrainer(AffordanceTrainer):

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
    ):
        super(GCPTrainer, self).__init__(
            vqvae,
            affordance,
            classifier=classifier,
            use_pretrained_vqvae=use_pretrained_vqvae,
            lr=lr,
            gradient_clip_value=gradient_clip_value,
            normalize=normalize,
            background_subtract=background_subtract,
            prediction_mode=prediction_mode,

            affordance_pred_weight=affordance_pred_weight,
            affordance_beta=affordance_beta,

            image_dist_thresh=image_dist_thresh,

            num_vqvae_warmup_epochs=num_vqvae_warmup_epochs,

            train_classifier_interval=train_classifier_interval,
            classifier_noise_level=classifier_noise_level,

            linearity_weight=linearity_weight,
            distance_weight=distance_weight,
            loss_weights=loss_weights,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            num_plots=num_plots,
            num_plot_epochs=num_plot_epochs,
            num_goal_samples=num_goal_samples,
            tf_logger=tf_logger
        )
        assert self.image_dist_thresh is None

    def compute_affordance_loss(self, s, h, epoch, prefix, should_log):

        assert h.shape[-1] == self.root_len
        assert h.shape[-2] == self.root_len

        loss = 0.0
        loss_pred = 0.0
        kld = 0.0
        extra = {}

        assert np.log2(h.shape[1] - 1).is_integer()

        num_steps = h.shape[1] - 1
        num_levels = int(np.log2(num_steps))
        ts = [0, num_steps]
        t_to_h = {
            0 : h[:, 0],
            num_steps: h[:, num_steps],
        }
        for _ in range(num_levels):
            ts_copy = ts.copy()
            for t_init, t_goal in pairwise(ts_copy):
                t_target = int((t_init + t_goal) // 2)

                h_last = torch.cat((t_to_h[t_init], t_to_h[t_goal]), dim=1) 
                h_target = h[:, t_target]

                prefix_t = '%s/affordance_t%d' % (prefix, t_target)

                loss_t, extra_t = self._compute_affordance_loss(
                    h_last,
                    h_target,
                    None,
                    epoch,
                    prefix_t,
                    should_log)
            
                loss += loss_t
                loss_pred += extra_t['loss_pred']
                kld += extra_t['kld']

                h_pred_t = extra_t['h1_pred'].detach()
                
                ts.append(t_target)
                t_to_h[t_target] = h_pred_t
            ts.sort()

        h_pred = [t_to_h[t] for t in ts]
            

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

        last_step = num_steps - 1
        ts = [0, last_step]
        num_levels = int(np.log2(last_step))

        sf = plot_data['s'][:, last_step]
        sf = [sf] * self.num_goal_samples
        sf = torch.stack(sf, 0)

        hf = plot_data['h'][:, last_step]
        hf = [hf] * self.num_goal_samples
        hf = torch.stack(hf, 0)
        hf = hf.view(self.num_goal_samples * self.num_plots,
                    hf.shape[-3],
                    hf.shape[-2],
                    hf.shape[-1])

        t_to_h = {
            0 : h0,
            last_step: hf,
        }
        t_to_s = {
            0 : s0,
            last_step: sf,
        }
        for _ in range(num_levels):
            ts_copy = ts.copy()
            for t_init, t_goal in pairwise(ts_copy):
                t_target = int((t_init + t_goal) // 2)
                h0 = torch.cat((t_to_h[t_init], t_to_h[t_goal]), dim=1)

                h1_pred, goal_pred = self._sample_goals(h0)
                goal_pred = goal_pred.view(
                    self.num_goal_samples,
                    self.num_plots,
                    goal_pred.shape[-3],
                    goal_pred.shape[-2],
                    goal_pred.shape[-1])
                t_to_h[t_target] = h1_pred
                t_to_s[t_target] = goal_pred
            ts.sort()
        
        goal_preds = [t_to_s[t] for t in ts]

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
        assert h0.shape[-3] == self.embedding_dim * 2
        assert h0.shape[-2] == self.root_len
        assert h0.shape[-1] == self.root_len

        z = self.affordance.sample_prior(h0.shape[0])
        z = ptu.from_numpy(z)
        h1_pred = self.affordance.decode(z, cond=h0)

        goal_pred = self.vqvae.decode(h1_pred, self.prediction_mode)

        return h1_pred, goal_pred

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)   