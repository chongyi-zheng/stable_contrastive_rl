import os.path as osp
from collections import OrderedDict, namedtuple
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.loss import LossFunction
from rlkit.core.timer import timer
from rlkit.torch.core import PyTorchModule
from rlkit.torch.distributions import Distribution
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_base import VAEBase
from rlkit.visualization.image import combine_images_into_grid


class VAE(VAEBase):
    def __init__(
            self,
            encoder: DistributionGenerator,
            decoder: DistributionGenerator,
            latent_prior: Distribution,
    ):
        super().__init__(np.prod(latent_prior.mean.shape))
        self.encoder = encoder
        self.decoder = decoder
        self.latent_prior = latent_prior
        self.latent_shape = latent_prior.mean.shape[1:]
        if self.latent_prior.batch_shape != torch.Size([1]):
            raise ValueError('Use batch_shape of 1 for KL computation.')

    def reconstruct(self, x, use_latent_mean=True, use_generative_model_mean=True):
        q_z = self.encoder(x)
        if use_latent_mean:
            z = q_z.mean
        else:
            z = q_z.sample()

        p_x_given_z = self.decoder(z)
        if use_generative_model_mean:
            x_hat = p_x_given_z.mean
        else:
            x_hat = p_x_given_z.sample()
        return x_hat

    def sample(self, batch_size, use_generative_model_mean=True):
        # squeeze out extra batch dimension that was there for KL computation
        z = self.latent_prior.sample(torch.Size([batch_size])).squeeze(1)
        p_x_given_z = self.decoder(z)
        if use_generative_model_mean:
            x = p_x_given_z.mean
        else:
            x = p_x_given_z.sample()
        return x

    def encode(self, input):
        return self.encoder(input).mean

    def decode(self, latents):
        return self.decoder(latents).mean

    def rsample(self, latent_distribution_params):
        raise NotImplementedError()

    def reparameterize(self, latent_distribution_params):
        raise NotImplementedError()

    def logprob(self, inputs, obs_distribution_params):
        raise NotImplementedError()

    def kl_divergence(self, latent_distribution_params):
        raise NotImplementedError()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        raise NotImplementedError()

    # hack for now (I don't think this should be part of this interface)
    def encode_np(self, x):
        x_torch = ptu.from_numpy(x)
        z_torch = self.encoder(x_torch).mean
        return ptu.get_numpy(z_torch)

    def decode_np(self, latents):
        self.eval()
        reconstructions = self.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded


VAETerms = namedtuple(
    'VAETerms',
    'likelihood kl q_z p_x_given_z',
)
LossStatistics = OrderedDict
Loss = torch.Tensor


class VAETrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            vae: VAE,
            vae_lr=1e-3,
            beta=1,
            loss_scale=1.0,
            vae_visualization_config=None,
            optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.vae = vae
        self.beta = beta
        self.vae_optimizer = optimizer_class(
            self.vae.parameters(),
            lr=vae_lr,
        )
        self._need_to_update_eval_statistics = True
        self.loss_scale = loss_scale
        self.eval_statistics = OrderedDict()

        self.vae_visualization_config = vae_visualization_config
        if not self.vae_visualization_config:
            self.vae_visualization_config = {}

    def train_from_torch(self, batch):
        timer.start_timer('vae training', unique=False)
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        self.vae_optimizer.zero_grad()
        losses.vae_loss.backward()
        self.vae_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            self._need_to_update_eval_statistics = False
            self.example_obs_batch = batch['raw_next_observations']
        timer.stop_timer('vae training')

    def compute_loss(
        self,
        batch,
        skip_statistics=False
    ) -> Tuple[Loss, LossStatistics]:
        x = batch['raw_next_observations']
        vae_terms = compute_vae_terms(self.vae, x)
        vae_loss = - vae_terms.likelihood + self.beta * vae_terms.kl

        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['Log Prob'] = np.mean(ptu.get_numpy(
                vae_terms.likelihood
            ))
            eval_statistics['KL'] = np.mean(ptu.get_numpy(
                vae_terms.kl
            ))
            eval_statistics['loss'] = np.mean(ptu.get_numpy(
                vae_loss
            ))
            for k, v in vae_terms.p_x_given_z.get_diagnostics().items():
                eval_statistics['p_x_given_z/{}'.format(k)] = v
            for k, v in vae_terms.q_z.get_diagnostics().items():
                eval_statistics['q_z_given_x/{}'.format(k)] = v
        return vae_loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.dump_debug_images(epoch, **self.vae_visualization_config)

    def dump_debug_images(
        self,
        epoch,
        dump_images=True,
        num_recons=10,
        num_samples=25,
        debug_period=10,
        unnormalize_images=False,
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
        if not dump_images or epoch % debug_period != 0:
            return
        example_obs_batch_np = ptu.get_numpy(self.example_obs_batch)
        recon_examples_np = ptu.get_numpy(
            self.vae.reconstruct(self.example_obs_batch)
        )

        top_row_example = example_obs_batch_np[:num_recons]
        bottom_row_recon = np.clip(recon_examples_np, 0, 1)[:num_recons]

        recon_vis = combine_images_into_grid(
            imgs=list(top_row_example) + list(bottom_row_recon),
            imwidth=example_obs_batch_np.shape[2],
            imheight=example_obs_batch_np.shape[3],
            max_num_cols=len(top_row_example),
            image_format='CWH',
            unnormalize=unnormalize_images,
        )

        logdir = logger.get_snapshot_dir()
        cv2.imwrite(
            osp.join(logdir, '{}_recons.png'.format(epoch)),
            cv2.cvtColor(recon_vis, cv2.COLOR_RGB2BGR),
        )

        raw_samples = ptu.get_numpy(self.vae.sample(num_samples))
        vae_samples = np.clip(raw_samples, 0, 1)
        vae_sample_vis = combine_images_into_grid(
            imgs=list(vae_samples),
            imwidth=example_obs_batch_np.shape[2],
            imheight=example_obs_batch_np.shape[3],
            image_format='CWH',
            unnormalize=unnormalize_images,
        )
        cv2.imwrite(
            osp.join(logdir, '{}_vae_samples.png'.format(epoch)),
            cv2.cvtColor(vae_sample_vis, cv2.COLOR_RGB2BGR),
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


def compute_vae_terms(vae, x) -> VAETerms:
    q_z = vae.encoder(x)
    kl = kl_divergence(q_z, vae.latent_prior)
    z = q_z.rsample()
    p_x_given_z = vae.decoder(z)
    log_prob = p_x_given_z.log_prob(x)

    mean_log_prob = log_prob.mean()
    mean_kl = kl.mean()

    return VAETerms(
        likelihood=mean_log_prob,
        kl=mean_kl,
        q_z=q_z,
        p_x_given_z=p_x_given_z,
    )
