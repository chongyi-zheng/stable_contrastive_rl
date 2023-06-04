import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
from rlkit.torch.core import PyTorchModule
import torchvision.utils as vutils
from rlkit.torch import pytorch_util as ptu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

class BiGAN(PyTorchModule):
    def __init__(self,
        representation_size=8,
        input_channels=3,
        dropout=0.2,
        imsize=48,
        architecture=None, # Not used
        decoder_output_activation=None,
        ):
        super().__init__()
        self.representation_size = representation_size
        self.imlength = input_channels * imsize * imsize
        self.input_channels = input_channels
        self.imsize = imsize

        self.netE = Encoder(representation_size, input_channels=input_channels, imsize=imsize, noise=True)
        self.netG = Generator(representation_size, input_channels=input_channels, imsize=imsize)
        self.netD = Discriminator(representation_size, input_channels=input_channels, imsize=imsize, dropout=dropout)

        self.netE.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def encode(self, input):
        return self.netE(input)[0].reshape(-1, self.representation_size)

    def decode(self, latent):
        return self.netG(latent)

    def encode_np(self, imgs):
        return ptu.get_numpy(self.encode(ptu.from_numpy(imgs)))

    def decode_np(self, latents):
        reconstructions = self.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def encode_one_np(self, img):
        return self.encode_np(img[None])[0]

    def decode_one_np(self, latent):
        return self.decode_np(latent[None])[0]

    def _reconstruct_img(self, flat_img):
        latent_distribution_params = self.vae.encode(ptu.from_numpy(flat_img.reshape(1,-1)))
        reconstructions = self.vae.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

class Generator(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48):
        super(Generator, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.output_bias = nn.Parameter(torch.zeros(self.input_channels, self.imsize, self.imsize), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.representation_size, 256, 8, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, self.input_channels, 1, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.representation_size, 1, 1))
        output = torch.sigmoid(output + self.output_bias)
        return output


class Encoder(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48, noise=False):
        super(Encoder, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.main1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

        self.main2 = nn.Sequential(
            nn.Conv2d(256, 512, 2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.main4 = nn.Sequential(
            nn.Conv2d(512, self.representation_size, 1, stride=1, bias=True)
        )

    def forward(self, input):
        input = input.view(-1, self.input_channels, self.imsize, self.imsize)

        batch_size = input.size()[0]
        x1 = self.main1(input)
        x2 = self.main2(x1)
        x3 = self.main3(x2)
        output = self.main4(x3).view(batch_size, self.representation_size, 1, 1)
        return output, x3, x2, x1

class Discriminator(nn.Module):

    def __init__(self, representation_size, input_channels=3, imsize=48, dropout=0.2):
        super(Discriminator, self).__init__()
        self.representation_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.dropout = dropout

        self.infer_x = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.representation_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def forward(self, x, z):
        x = x.view(-1, self.input_channels, self.imsize, self.imsize)

        output_x = self.infer_x(x)
        output_z = self.infer_z(z)

        output_features = self.infer_joint(torch.cat([output_x, output_z], dim=1))
        output = self.final(output_features)
        output = torch.sigmoid(output)
        return output.squeeze(), output_features.view(x.size()[0], -1)


class CVBiGAN(BiGAN):
    def __init__(
            self,
            representation_size=8,
            input_channels=3,
            dropout=0.2,
            imsize=48,
            architecture=None, # Not used
            decoder_output_activation=None,
            ):
        super().__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.imlength = input_channels * imsize * imsize
        self.input_channels = input_channels
        self.imsize = imsize

        self.netE = CVEncoder(representation_size, input_channels=self.input_channels, imsize=imsize, noise=True)
        self.netG = CVGenerator(representation_size, input_channels=self.input_channels, imsize=imsize)
        self.netD = CVDiscriminator(representation_size, input_channels=self.input_channels, imsize=imsize, dropout=dropout)

        self.netE.apply(weights_init)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

    def encode(self, input, cond):
        return self.netE(input, cond)[0].reshape(-1, self.representation_size)

    def decode(self, latent):
        return self.netG(latent).reshape(-1, self.input_channels, self.imsize, self.imsize)

    def encode_np(self, imgs, cond):
        return ptu.get_numpy(self.encode(ptu.from_numpy(imgs), ptu.from_numpy(cond)))

    def decode_np(self, latents):
        reconstructions = self.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def encode_one_np(self, img, cond):
        return self.encode_np(img[None], cond)[0]

    def decode_one_np(self, latent):
        return self.decode_np(latent[None])[0]

    def sample_prior(self, batch_size, cond=None, image_cond=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, axis=0)
        cond = ptu.from_numpy(cond)

        if image_cond:
            cond = cond.reshape(-1, self.input_channels, self.imsize, self.imsize)
            z_cond, _, _, _= self.netE.cond_encoder(cond)
            z_cond = z_cond.reshape(-1, self.latent_size)
        else:
            z_cond = cond[:, self.latent_size:]

        z_delta = ptu.randn(batch_size, self.latent_size)
        cond_sample = torch.cat([z_delta, z_cond], dim=1)
        
        return ptu.get_numpy(cond_sample)

class CVGenerator(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48):
        super(CVGenerator, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.output_bias = nn.Parameter(torch.zeros(self.input_channels, self.imsize, self.imsize), requires_grad=True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.representation_size, 256, 8, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 32, 1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, self.input_channels, 1, stride=1, bias=False)
        )

    def forward(self, input):
        output = self.main(input.view(-1, self.representation_size, 1, 1))
        output = torch.sigmoid(output + self.output_bias)
        return output

class CVEncoder(nn.Module):
    def __init__(self, representation_size, input_channels=3, imsize=48, noise=False):
        super(CVEncoder, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize

        self.encoder = Encoder(self.latent_size, input_channels=self.input_channels * 2, imsize=imsize, noise=True)
        self.cond_encoder = Encoder(self.latent_size, input_channels=self.input_channels, imsize=imsize, noise=True)

    def forward(self, x_delta, x_cond):
        x_delta = x_delta.view(-1, self.input_channels, self.imsize, self.imsize)
        x_cond = x_cond.view(-1, self.input_channels, self.imsize, self.imsize)
        batch_size = x_delta.size()[0]
        x_delta = torch.cat([x_delta, x_cond], dim=1)
        z_delta, _, _, _= self.encoder(x_delta)
        z_cond, _, _, _= self.cond_encoder(x_cond)
        output = torch.cat([z_delta, z_cond], dim=1)
        return output, None, None, None

class CVDiscriminator(nn.Module):

    def __init__(self, representation_size, input_channels=3, imsize=48, dropout=0.2):
        super(CVDiscriminator, self).__init__()
        self.representation_size = representation_size * 2
        self.latent_size = representation_size
        self.input_channels = input_channels
        self.imsize = imsize
        self.dropout = dropout

        self.infer_x = nn.Sequential(
            nn.Conv2d(self.input_channels * 2, 32, 3, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(32, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(128, 256, 4, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(256, 512, 4, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_z = nn.Sequential(
            nn.Conv2d(self.representation_size, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(512, 512, 1, stride=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.infer_joint = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout),

            nn.Conv2d(1024, 1024, 1, stride=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=self.dropout)
        )

        self.final = nn.Conv2d(1024, 1, 1, stride=1, bias=True)

    def forward(self, x, x_cond, z):
        obs = torch.cat([x, x_cond], dim=1)
        output_obs = self.infer_x(obs)
        output_z = self.infer_z(z)
        output_features = self.infer_joint(torch.cat([output_obs, output_z], dim=1))
        output = self.final(output_features)
        output = torch.sigmoid(output)
        return output.squeeze(), output_features.view(obs.size()[0], -1)
