from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
from rlkit.core.loss import LossFunction
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn
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


class BiGANTrainer(ConvVAETrainer, LossFunction):
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
            generator_threshold=3.5,
            discriminator_noise=False,

            b_low=0.5,
            b_high=0.999,
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
        self.num_epochs = num_epochs
        self.generator_threshold = generator_threshold
        self.discriminator_noise=discriminator_noise

        self.criterion = nn.BCELoss()
        self.optimizerG = optim.Adam([{'params' : self.model.netE.parameters()},
                         {'params' : self.model.netG.parameters()}], lr=lr, betas=(b_low, b_high))
        self.optimizerD = optim.Adam(self. model.netD.parameters(), lr=lr, betas=(b_low, b_high))

    def noise(self, size, num_epochs, epoch):
        noise = ptu.randn(size)
        std = 0.1 * (num_epochs - epoch) / num_epochs
        return std * noise

    def fixed_noise(self, b_size):
        return ptu.randn(b_size, self.representation_size, 1, 1)

    def train_batch(self, epoch, batch):
        self.model.train()
        errD, errG = self.compute_loss(batch, epoch, False)

        if errG.item() < self.generator_threshold:
            self.optimizerD.zero_grad()
            errD.backward(retain_graph=True)
            self.optimizerD.step()

        self.optimizerG.zero_grad()
        errG.backward()
        self.optimizerG.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        errD, errG = self.compute_loss(batch, epoch, True)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        real_data = batch[self.key_to_reconstruct].reshape(-1, self.input_channels, self.imsize, self.imsize)
        batch_size = real_data.size(0)

        real_label = ptu.ones(batch_size)
        fake_label = ptu.zeros(batch_size)

        real_latent, _, _, _= self.model.netE(real_data)
        real_latent = real_latent.view(batch_size, self.representation_size, 1, 1)

        fake_latent = self.fixed_noise(batch_size)
        fake_data = self.model.netG(fake_latent)

        real_noise = 0
        fake_noise = 0

        if self.discriminator_noise:
            real_noise = self.noise(real_data.size(), self.num_epochs, epoch)
            fake_noise = self.noise(real_data.size(), self.num_epochs, epoch)

        real_pred, _ = self.model.netD(real_data + real_noise, real_latent)
        fake_pred, _ = self.model.netD(fake_data + fake_noise, fake_latent)

        errD = self.criterion(real_pred, real_label) + self.criterion(fake_pred, fake_label)
        errG = self.criterion(fake_pred, real_label) + self.criterion(real_pred, fake_label)

        recon = self.model.netG(real_latent)
        recon_error = F.mse_loss(recon, real_data)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "errD"].append(errD.item())
        self.eval_statistics[prefix + "errG"].append(errG.item())
        self.eval_statistics[prefix + "Recon Error"].append(recon_error.item())
        self.eval_data[prefix + "last_batch"] = (real_data.reshape(batch_size, -1), recon.reshape(batch_size, -1))

        return errD, errG

    def dump_samples(self, epoch):

        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        samples = ptu.randn(64, self.representation_size)
        samples = self.model.decode(samples)

        save_image(
            samples.data.view(64, self.input_channels, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )

class CVBiGANTrainer(BiGANTrainer, LossFunction):

    def fixed_noise(self, b_size, latent):
        z_cond = latent[:, self.model.latent_size:]
        z_delta = ptu.randn(b_size, self.model.latent_size, 1, 1)
        return torch.cat([z_delta, z_cond], dim=1)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        real_data = batch['x_t'].reshape(-1, self.input_channels, self.imsize, self.imsize)
        cond = batch['env'].reshape(-1, self.input_channels, self.imsize, self.imsize)
        batch_size = real_data.size(0)

        real_label = ptu.ones(batch_size)
        fake_label = ptu.zeros(batch_size)

        real_latent, _, _, _= self.model.netE(real_data, cond)
        fake_latent = self.fixed_noise(batch_size, real_latent)
        fake_data = self.model.netG(fake_latent)

        real_noise = 0
        fake_noise = 0
        cond_noise = 0

        if self.discriminator_noise:
            real_noise = self.noise(real_data.size(), self.num_epochs, epoch)
            fake_noise  = self.noise(real_data.size(), self.num_epochs, epoch)
            cond_noise = self.noise(real_data.size(), self.num_epochs, epoch)

        real_pred, _ = self.model.netD(real_data + real_noise, cond + cond_noise, real_latent)
        fake_pred, _ = self.model.netD(fake_data + fake_noise, cond + cond_noise, fake_latent)

        errD = self.criterion(real_pred, real_label) + self.criterion(fake_pred, fake_label)
        errG = self.criterion(fake_pred, real_label) + self.criterion(real_pred, fake_label)

        recon = self.model.netG(real_latent)
        recon_error = F.mse_loss(recon, real_data)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics[prefix + "errD"].append(errD.item())
        self.eval_statistics[prefix + "errG"].append(errG.item())
        self.eval_statistics[prefix + "Recon Error"].append(recon_error.item())
        self.eval_data[prefix + "last_batch"] = (batch, recon.reshape(batch_size, -1))

        return errD, errG

    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
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
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_samples(self, epoch):
        self.model.eval()
        batch, reconstructions = self.eval_data["test/last_batch"]
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                self.input_channels,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            latent = ptu.from_numpy(self.model.sample_prior(self.batch_size, ptu.get_numpy(env)))
            samples = self.model.netG(latent)
            all_imgs.extend([
                samples.view(
                    self.batch_size,
                    self.input_channels,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=8)
