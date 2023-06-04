import argparse
from collections import OrderedDict
import os
from os import path as osp
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rlkit.core.loss import LossFunction
from rlkit.core import logger
from torchvision.utils import save_image

class DCGANTrainer():

    def __init__(self, model, lr, beta, latent_size):
        self.model = model
        self.device = self.model.device

        self.img_list = []
        self.G_losses = {}
        self.D_losses = {}
        self.iters = 0
        self.criterion = nn.BCELoss()

        self.lr = lr
        self.beta = beta
        self.latent_size = latent_size

        self.fixed_noise = torch.randn(64, self.latent_size, 1, 1, device=self.device)

        self.optimizerD = optim.Adam(self.model.netD.parameters(), lr=lr, betas=(beta, 0.999))
        self.optimizerG = optim.Adam(self.model.netG.parameters(), lr=lr, betas=(beta, 0.999))
   
    @property
    def log_dir(self):
        return logger.get_snapshot_dir()


    def train_epoch(self, dataloader, epoch, num_epochs, get_data = lambda d: d):
        for i, data in enumerate(dataloader, 0):
            data = get_data(data)
            import ipdb; ipdb.set_trace()
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            real_cpu = data.to(self.device).float()
            b_size = real_cpu.size(0)

            real_label = torch.full((b_size,), 1, device=self.device)
            fake_label = torch.full((b_size,), 0, device=self.device)

            real_output = self.model.netD(real_cpu).view(-1)
            errD_real = self.criterion(real_output, real_label)
            D_x = real_output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(b_size, self.latent_size, 1, 1, device=self.device)
            fake = self.model.netG(noise)
            fake_output = self.model.netD(fake.detach()).view(-1)
            errD_fake = self.criterion(fake_output, fake_label)
            D_G_z1 = fake_output.mean().item()
            errD = errD_real + errD_fake

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.model.netG.zero_grad()
            output = self.model.netD(fake).view(-1)
            errG = self.criterion(output, real_label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()

            if errG.item() < 4:
                self.model.netD.zero_grad()
                errD_real.backward()
                errD_fake.backward()
                self.optimizerD.step()


            self.model.netD.zero_grad()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            self.G_losses.setdefault(epoch, []).append(errG.item())
            self.D_losses.setdefault(epoch, []).append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (self.iters % 200 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = self.model.netG(self.fixed_noise).detach().cpu()
                sample = vutils.make_grid(fake, padding=2, normalize=True)
                self.img_list.append(sample)
                self.dump_samples("sample " + str(epoch), self.iters, sample)
                self.dump_samples("real " + str(epoch), self.iters, vutils.make_grid(real_cpu.cpu().data[:64, ], padding=2, normalize=True))
            self.iters += 1

    def dump_samples(self, epoch, iters, sample):
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(sample,(1,2,0)))
        save_dir = osp.join(self.log_dir, str(epoch) + '-' + str(iters) + '.png')
        plt.savefig(save_dir)
        plt.close()

    def get_stats(self, epoch):
        stats = OrderedDict()
        stats["epoch"] = epoch
        stats["Generator Loss"] = np.mean(self.G_losses[epoch])
        stats["Discriminator Loss"] = np.mean(self.D_losses[epoch])
        return stats

    def get_G_losses(self):
        return self.G_losses

    def get_D_losses(self):
        return self.D_losses    

    def get_model(self):
        return self.model

    def get_img_list(self):
        return self.img_list
