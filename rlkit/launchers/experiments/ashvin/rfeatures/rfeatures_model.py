import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN, Mlp
from rlkit.torch.vae.vae_base import GaussianLatentVAE

import torchvision
import torchvision.transforms as transforms

import rlkit.data_management.external.epic_kitchens_data_stub as epic
from torch.utils.model_zoo import load_url as load_state_dict_from_url

MAX_BATCH_SIZE = 100

class TimestepPredictionModel(torch.nn.Module):
    def __init__(
            self,
            representation_size,
            architecture,
        
            normalize=True,
            output_classes=100,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=224,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,

            delta_features=False,
            pretrained_features=False,
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super().__init__()
#         super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize * self.imsize * self.input_channels
        self.representation_size = representation_size
        self.output_classes = output_classes

        self.normalize = normalize

        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_std = torch.tensor([0.229, 0.224, 0.225])
        self.img_mean = self.img_mean.repeat(epic.CROP_WIDTH, epic.CROP_HEIGHT, 1).transpose(0, 2).to(ptu.device)
        self.img_std = self.img_std.repeat(epic.CROP_WIDTH, epic.CROP_HEIGHT, 1).transpose(0, 2).to(ptu.device)
        # self.img_normalizer = torchvision.transforms.Normalize(self.img_mean, self.img_std)

        self.encoder = torchvision.models.resnet.ResNet(
            torchvision.models.resnet.BasicBlock, 
            [2, 2, 2, 2], 
            num_classes=representation_size,
        )
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.encoder = nn.DataParallel(self.encoder)

        if pretrained_features:
            exclude_names = ["fc"]
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet18-5c106cde.pth",
                progress=True,
            )
            new_state_dict = state_dict.copy()
            for key in state_dict:
                for name in exclude_names:
                    if name in key:
                        del new_state_dict[key]
                        break
            self.encoder.load_state_dict(new_state_dict, strict=False)

        self.delta_features = delta_features
        input_size = representation_size * 2 if delta_features else representation_size * 3

        self.predictor = Mlp(
            output_size=output_classes,
            input_size=input_size,
            **architecture
        )
        self.predictor = self.predictor.to("cuda:0")

        self.epoch = 0

    def get_latents(self, x0, xt, xT):
        bz = x0.shape[0]

        x = torch.cat([x0, xt, xT], dim=0).view(-1, 3, epic.CROP_HEIGHT, epic.CROP_WIDTH, )

        z = self.encode(x)
        # # import pdb; pdb.set_trace()
        # if self.normalize:
        #     x = x - self.img_mean
        #     x = x / self.img_std
        #     # x = self.img_normalizer(x)

        # zs = []
        # for i in range(0, 3 * bz, MAX_BATCH_SIZE):
        #     z = self.encoder(x[i:i+MAX_BATCH_SIZE, :, :, :])
        #     zs.append(z)

        # # z = self.encoder(x)
        # z = torch.cat(zs) # self.encoder(x) # .to("cuda:0")

        z0, zt, zT = z[:bz, :], z[bz:2*bz, :], z[2*bz:3*bz, :]

        return z0, zt, zT

    def forward(self, x0, xt, xT):
        z0, zt, zT = self.get_latents(x0, xt, xT)

        # z0 = self.encoder(x0.view(-1, 3, 456, 256)).to("cuda:0") #.view((-1, 3, 240, 240))[:, :, :224, :224])
        # zt = self.encoder(xt.view(-1, 3, 456, 256)).to("cuda:0") # .view((-1, 3, 240, 240))[:, :, :224, :224]) 
        # zT = self.encoder(xT.view(-1, 3, 456, 256)).to("cuda:0") # .view((-1, 3, 240, 240))[:, :, :224, :224])
        
        if self.delta_features:
            dt = zt - z0
            dT = zT - z0
            z = torch.cat([dt, dT], dim=1)
        else:
            z = torch.cat([z0, zt, zT], dim=1)
        
        out = self.predictor(z)
        
        return out

    def encode(self, x):
        bz = x.shape[0]

        if self.normalize:
            x = x - self.img_mean
            x = x / self.img_std

        zs = []
        for i in range(0, bz, MAX_BATCH_SIZE):
            z = self.encoder(x[i:i+MAX_BATCH_SIZE, :, :, :])
            zs.append(z)

        z = torch.cat(zs)

        return z