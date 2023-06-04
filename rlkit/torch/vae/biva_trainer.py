from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
import argparse
import json
import logging
import pickle
from rlkit.core.loss import LossFunction
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.utils import save_image
from rlkit.data_management.images import normalize_image
from rlkit.core import logger
import rlkit.core.util as util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.ml_util import ConstantSchedule
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset, InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from rlkit.torch.core import np_to_pytorch_batch
import collections
import time

from rlkit.torch.vae.biva.evaluation.freebits import FreeBits
from rlkit.torch.vae.biva.model import \
    DeepVae, get_deep_vae_mnist, get_deep_vae_cifar, VaeStage, LvaeStage, BivaStage
from rlkit.torch.vae.biva.utils import \
    LowerBoundedExponentialLR, training_step, test_step, summary2logger, save_model, load_model, \
    sample_model, DiscretizedMixtureLogits, batch_reduce, log_sum_exp, detach_to_device
#from booster.utils import EMA
from torch.distributions import Bernoulli


class BIVATrainer(ConvVAETrainer, LossFunction):
    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=1.0,
            beta_schedule=None,
            lr=2e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
            key_to_reconstruct='x_t',
            num_epochs=500,

            ema=0.9995,
            nr_mix=10,
            free_bits=2.0,
            
        ):
        super().__init__(
            model,
            batch_size,
            log_interval,
            beta,
            beta_schedule,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            linearity_weight,
            distance_weight,
            loss_weights,
            use_linear_dynamics,
            use_parallel_dataloading,
            train_data_workers,
            skew_dataset,
            skew_config,
            priority_function_kwargs,
            start_skew_epoch,
            weight_decay,
            key_to_reconstruct,
            num_epochs
        )
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=(0.9, 0.999,))
        self.scheduler = LowerBoundedExponentialLR(self.optimizer, 0.999999, 0.0001)
        self.likelihood = DiscretizedMixtureLogits(nr_mix)
        #self.likelihood = Bernoulli
        #self.evaluator = VariationalInference(self.likelihood, iw_samples=1)
        
        #IF ABOVE DOESNT WORK, TRY OTHER LIKELIHOOD
        #self.ema = EMA(model, ema)

        n_latents = len(self.model.latents)
        n_latents = 2 * n_latents - 1
        self.freebits = [free_bits] * n_latents
        self.kwargs = {'beta': beta, 'freebits': self.freebits}
        self.weights_initialized = False

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, epoch, False)

        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        #self.ema.update()

    def init_weights(self, batch):
        self.weights_initialized = True
        with torch.no_grad():
            self.model.train()
            obs = batch[self.key_to_reconstruct]
            self.model(obs)
            #loss, diagnostics, output = self.evaluator(self.model, obs, **self.kwargs)

    def compute_kls(self, kls, device):
        """compute kl and kl to be accounted in the loss"""

        # set kls and freebits as lists
        if not isinstance(kls, list):
            kls = [kls]

        if self.freebits is not None and not isinstance(self.freebits, list):
            self.freebits = [self.freebits for _ in kls]

        # apply freebits to each
        if self.freebits is not None:
            kls_loss = (FreeBits(fb)(kl) for fb, kl in zip(self.freebits, kls))
        else:
            kls_loss = kls

        # sum freebit kls
        kls_loss = [batch_reduce(kl)[:, None] for kl in kls_loss]
        kls_loss = batch_reduce(torch.cat(kls_loss, 1))

        # sum kls
        kls = [batch_reduce(kl)[:, None] for kl in kls]
        kls = batch_reduce(torch.cat(kls, 1))

        return kls.mean(), kls_loss.mean()


    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        self.kwargs['beta'] = float(self.beta_schedule.get_value(epoch))
        obs = batch[self.key_to_reconstruct]
        batch_size = obs.shape[0]
        if not self.weights_initialized:
            self.init_weights(batch)

        recon, kls = self.model(obs, **self.kwargs)

        nll = F.mse_loss(obs.view(batch_size, -1), recon.view(batch_size, -1), reduction='sum')
        kl, kls_loss = self.compute_kls(kls, obs.device)
        loss = nll + self.kwargs['beta'] * kls_loss

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics["beta"] = self.kwargs['beta']
        self.eval_statistics[prefix + "kles"].append(kl.item())
        self.eval_statistics[prefix + "log_prob"].append(nll.item())
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_data[prefix + "last_batch"] = (obs, recon.reshape(batch_size, -1).detach())

        return loss


    def dump_samples(self, epoch):
        save_dir = osp.join(self.log_dir, 'samples_%d.png' % epoch)
        n_samples = 64
        
        samples = self.model.sample_from_prior(n_samples)
        #samples = self.likelihood(logits=samples).sample()
        save_image(
            samples.data.view(n_samples, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )


class ConditionalBIVATrainer(ConvVAETrainer, LossFunction):
    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=1.0,
            beta_schedule=None,
            lr=2e-3,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
            key_to_reconstruct='x_t',
            num_epochs=500,

            ema=0.9995,
            nr_mix=10,
            free_bits=2.0,
            
        ):
        super().__init__(
            model,
            batch_size,
            log_interval,
            beta,
            beta_schedule,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            linearity_weight,
            distance_weight,
            loss_weights,
            use_linear_dynamics,
            use_parallel_dataloading,
            train_data_workers,
            skew_dataset,
            skew_config,
            priority_function_kwargs,
            start_skew_epoch,
            weight_decay,
            key_to_reconstruct,
            num_epochs
        )
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr, betas=(0.9, 0.999,))
        self.scheduler = LowerBoundedExponentialLR(self.optimizer, 0.999999, 0.0001)
        self.likelihood = DiscretizedMixtureLogits(nr_mix)
        #self.likelihood = Bernoulli
        #self.evaluator = VariationalInference(self.likelihood, iw_samples=1)
        
        #IF ABOVE DOESNT WORK, TRY OTHER LIKELIHOOD
        #self.ema = EMA(model, ema)

        n_latents = len(self.model.latents)
        n_latents = 2 * n_latents - 1
        self.freebits = [free_bits] * n_latents
        self.kwargs = {'beta': beta, 'freebits': self.freebits}
        self.weights_initialized = False

    def train_batch(self, epoch, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, epoch, False)

        loss.backward()
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        #self.ema.update()

    def init_weights(self, batch):
        self.weights_initialized = True
        with torch.no_grad():
            self.model.train()
            self.model(batch['x_t'], batch['env'], **self.kwargs)

    def compute_kls(self, kls):
        """compute kl and kl to be accounted in the loss"""

        # set kls and freebits as lists
        if not isinstance(kls, list):
            kls = [kls]

        if self.freebits is not None and not isinstance(self.freebits, list):
            self.freebits = [self.freebits for _ in kls]

        # apply freebits to each
        if self.freebits is not None:
            kls_loss = (FreeBits(fb)(kl) for fb, kl in zip(self.freebits, kls))
        else:
            kls_loss = kls

        # sum freebit kls
        kls_loss = [batch_reduce(kl)[:, None] for kl in kls_loss]
        kls_loss = batch_reduce(torch.cat(kls_loss, 1))

        # sum kls
        kls = [batch_reduce(kl)[:, None] for kl in kls]
        kls = batch_reduce(torch.cat(kls, 1))

        return kls.mean(), kls_loss.mean()

    def compute_nll(self, img, recon):
        return F.mse_loss(img.reshape(-1, self.imlength), \
                    recon.reshape(-1, self.imlength), reduction='sum')

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        self.kwargs['beta'] = float(self.beta_schedule.get_value(epoch))
        if not self.weights_initialized:
            self.init_weights(batch)

        delta_recon, cond_recon, delta_kls, cond_kls = \
                self.model(batch['x_t'], batch['env'], **self.kwargs)

        delta_nll = self.compute_nll(batch['x_t'], delta_recon)
        cond_nll = self.compute_nll(batch['env'], cond_recon)

        delta_kl, delta_kls_loss = self.compute_kls(delta_kls)
        cond_kl, cond_kls_loss = self.compute_kls(cond_kls)
        
        delta_loss = delta_nll + self.kwargs['beta'] * delta_kls_loss
        cond_loss = cond_nll + cond_kls_loss #Cond Beta = 1
        
        loss = delta_loss + cond_loss

        delta_recon = delta_recon.reshape(-1, self.imlength).detach()
        cond_recon = cond_recon.reshape(-1, self.imlength).detach()

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics["beta"] = self.kwargs['beta']
        self.eval_statistics[prefix + "kles"].append(delta_kl.item())
        self.eval_statistics[prefix + "log_prob"].append(delta_nll.item())
        self.eval_statistics[prefix + "cond_kles"].append(cond_kl.item())
        self.eval_statistics[prefix + "cond_log_prob"].append(cond_nll.item())
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_data[prefix + "last_batch"] = (batch, delta_recon, cond_recon)
        return loss


    def dump_reconstructions(self, epoch):
        self.dump_batch_reconstructions(epoch, 'train')
        self.dump_batch_reconstructions(epoch, 'test')


    def dump_samples(self, epoch):
        self.model.eval()
        self.dump_batch_samples(epoch, 'train')
        self.dump_batch_samples(epoch, 'test')


    def dump_batch_reconstructions(self, epoch, prefix):
        batch, reconstructions, env_reconstructions = self.eval_data["{0}/last_batch".format(prefix)]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3),
            env_reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, '{0}_recon{1}.png'.format(prefix, epoch))
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_batch_samples(self, epoch, prefix):
        batch, _, _ = self.eval_data["{0}/last_batch".format(prefix)]
        # self.dump_distances(batch, epoch)
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            samples = self.model.sample_images(n, env[:n])
            all_imgs.extend([
                samples.view(
                    n,
                    3,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, '{0}_sample{1}.png'.format(prefix, epoch))
        save_image(comparison.data.cpu(), save_dir, nrow=8)





