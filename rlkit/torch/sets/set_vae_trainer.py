import os.path as osp
from collections import OrderedDict
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.optim as optim
from rlkit.core.eval_util import create_stats_ordered_dict
from torch.distributions.kl import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.loss import LossFunction
from rlkit.core.timer import timer
from rlkit.util import ml_util
from rlkit.torch.distributions import MultivariateDiagonalNormal, Distribution
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_torch_trainer import VAE, compute_vae_terms
from rlkit.visualization.image import combine_images_into_grid

LossStatistics = OrderedDict
Loss = torch.Tensor


def compute_prior(q_z: Distribution):
    if not isinstance(q_z, MultivariateDiagonalNormal):
        raise NotImplementedError()
    second_moment = (q_z.variance + q_z.mean**2).mean(dim=0, keepdim=True)
    first_moment = q_z.mean.mean(dim=0, keepdim=True)
    variance = second_moment - first_moment**2
    stddev = torch.sqrt(variance)
    return MultivariateDiagonalNormal(loc=first_moment, scale_diag=stddev)


class SetVAETrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vae: VAE,
            vae_lr=1e-3,
            beta=1,
            beta_schedule: Optional[ml_util.ScalarSchedule] = None,
            set_loss_weight=1,
            loss_scale=1.0,
            vae_visualization_config=None,
            optimizer_class=optim.Adam,
            set_key='set',
            data_key='raw_next_observations',
            train_sets=None,
            eval_sets=None,
    ):
        super().__init__()
        if beta_schedule is None:
            beta_schedule = ml_util.ConstantSchedule(beta)
        self.vae = vae
        self.beta_schedule = beta_schedule
        self.set_loss_weight = set_loss_weight
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=vae_lr,
        )
        self._need_to_update_eval_statistics = True
        self.loss_scale = loss_scale
        self.eval_statistics = OrderedDict()
        self.set_key = set_key
        self.data_key = data_key
        self.train_sets = train_sets
        self.eval_sets = eval_sets

        self.vae_visualization_config = vae_visualization_config
        if not self.vae_visualization_config:
            self.vae_visualization_config = {}

        self.example_batch = {}
        self._iteration = 0
        self._num_train_batches = 0

    @property
    def _beta(self):
        return self.beta_schedule.get_value(self._iteration)

    def train_from_torch(self, batch):
        timer.start_timer('vae training', unique=False)
        self.vae.train()
        loss, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        self.vae_optimizer.zero_grad()
        loss.backward()
        self.vae_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False
            self.example_batch = batch
            self.eval_statistics['num_train_batches'] = self._num_train_batches
        self._num_train_batches += 1
        timer.stop_timer('vae training')

    def compute_loss(
            self,
            batch,
            skip_statistics=False
    ) -> Tuple[Loss, LossStatistics]:
        vae_terms = compute_vae_terms(self.vae, batch[self.data_key])
        kl = vae_terms.kl
        likelihood = vae_terms.likelihood
        set_loss = compute_set_loss(self.vae, batch[self.set_key])
        total_loss = (
                - likelihood + self._beta * kl + self.set_loss_weight * set_loss
        )

        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['log_prob'] = np.mean(ptu.get_numpy(
                likelihood
            ))
            eval_statistics['kl'] = np.mean(ptu.get_numpy(
                kl
            ))
            eval_statistics['set_loss'] = np.mean(ptu.get_numpy(
                set_loss
            ))
            eval_statistics['loss'] = np.mean(ptu.get_numpy(
                total_loss
            ))
            eval_statistics['beta'] = self._beta
            for k, v in vae_terms.p_x_given_z.get_diagnostics().items():
                eval_statistics['p_x_given_z/{}'.format(k)] = v
            for k, v in vae_terms.q_z.get_diagnostics().items():
                eval_statistics['q_z_given_x/{}'.format(k)] = v
            for name, set_list in [
                ('eval', self.eval_sets),
                ('train', self.train_sets),
            ]:
                for set_i, set in enumerate(set_list):
                    vae_terms = compute_vae_terms(self.vae, set)
                    kl = vae_terms.kl
                    likelihood = vae_terms.likelihood
                    set_loss = compute_set_loss(self.vae, set)
                    eval_statistics['{}/set{}/log_prob'.format(name, set_i)] = np.mean(
                        ptu.get_numpy(likelihood))
                    eval_statistics['{}/set{}/kl'.format(name, set_i)] = np.mean(
                        ptu.get_numpy(kl))
                    eval_statistics['{}/set{}/set_loss'.format(name, set_i)] = (
                        np.mean(ptu.get_numpy(set_loss)))
                    set_prior = compute_prior(self.vae.encoder(set))
                    eval_statistics.update(
                        create_stats_ordered_dict(
                            '{}/set{}/learned_prior/mean'.format(name, set_i),
                            ptu.get_numpy(set_prior.mean)
                        )
                    )
                    eval_statistics.update(
                        create_stats_ordered_dict(
                            '{}/set{}/learned_prior/stddev'.format(name, set_i),
                            ptu.get_numpy(set_prior.stddev)
                        )
                    )
                    for k, v in vae_terms.p_x_given_z.get_diagnostics().items():
                        eval_statistics['{}/set{}/p_x_given_z/{}'.format(
                            name, set_i, k)] = v
                    for k, v in vae_terms.q_z.get_diagnostics().items():
                        eval_statistics['{}/set{}/q_z_given_x/{}'.format(
                            name, set_i, k)] = v

        return total_loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.dump_debug_images(epoch, **self.vae_visualization_config)
        self._iteration = epoch  # TODO: rename to iteration?

    def dump_debug_images(
            self,
            epoch,
            dump_images=True,
            num_recons=10,
            num_samples=25,
            debug_period=10,
            unnormalize_images=True,
            image_format='CHW',
    ):
        """

        :param epoch:
        :param dump_images: Set to False to not dump any images.
        :param num_recons:
        :param num_samples:
        :param debug_period: How often do you dump debug images?
        :param unnormalize_images: Should your unnormalize images before
            dumping them? Set to True if images are floats in [0, 1].
        :return:
        """
        self.vae.eval()
        if not dump_images or epoch % debug_period != 0:
            return
        logdir = logger.get_snapshot_dir()

        def save_reconstruction(name, batch):
            example_batch = ptu.get_numpy(batch)
            recon_examples_np = ptu.get_numpy(self.vae.reconstruct(batch))

            top_row_example = example_batch[:num_recons]
            bottom_row_recon = recon_examples_np[:num_recons]

            save_imgs(
                imgs=list(top_row_example) + list(bottom_row_recon),
                file_path=osp.join(logdir, '{}_{}.png'.format(epoch, name)),
                unnormalize=unnormalize_images,
                max_num_cols=len(top_row_example),
                image_format=image_format,
            )

        batch = self.example_batch[self.data_key]
        save_reconstruction('recon', batch)

        raw_samples = ptu.get_numpy(self.vae.sample(num_samples))
        save_imgs(
            imgs=raw_samples,
            file_path=osp.join(logdir, '{}_vae_samples.png'.format(epoch)),
            unnormalize=unnormalize_images,
            image_format=image_format,
        )

        for name, list_of_sets in [
            ('train', self.train_sets),
            ('eval', self.eval_sets),
        ]:
            for i, set in enumerate(list_of_sets):
                save_reconstruction('recon_{}_set{}'.format(name, i), set)

                q_z = self.vae.encoder(set)
                set_prior = compute_prior(q_z)
                z = set_prior.sample([num_samples]).squeeze(1)
                p_x_given_z = self.vae.decoder(z)

                save_imgs(
                    imgs=ptu.get_numpy(p_x_given_z.mean),
                    file_path=osp.join(
                        logdir,
                        '{}_vae_samples_{}_set{}.png'.format(
                            epoch, name, i,
                        ),
                    ),
                    unnormalize=unnormalize_images,
                    image_format=image_format,
                )

    @property
    def networks(self):
        return [
            self.vae,
        ]

    @property
    def optimizers(self):
        return [
            self.vae_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            vae=self.vae,
        )


def compute_set_loss(vae, set):
    q_z = vae.encoder(set)
    set_prior = compute_prior(q_z)
    set_loss = kl_divergence(q_z, set_prior)
    return set_loss.mean()


def save_imgs(imgs, file_path, **kwargs):
    imwidth = imgs[0].shape[1]
    imheight = imgs[0].shape[2]
    imgs = np.clip(imgs, 0, 1)
    combined_img = combine_images_into_grid(
        imgs=list(imgs),
        imwidth=imwidth,
        imheight=imheight,
        **kwargs
    )
    cv2_img = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, cv2_img)

