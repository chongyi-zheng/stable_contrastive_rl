# Adapted from pytorch examples

from __future__ import print_function
import copy
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from rlkit.core import logger
import os.path as osp
import numpy as np
from rlkit.util.ml_util import ConstantSchedule
from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, TwoHeadMlp
import rlkit.torch.pytorch_util as ptu

class VAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            input_size,
            hidden_sizes,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            output_scale=1,
            layer_norm=False,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_w = init_w
        hidden_sizes = list(hidden_sizes)
        self.encoder=TwoHeadMlp(hidden_sizes, representation_size, representation_size, input_size, layer_norm=layer_norm)
        hidden_sizes.reverse()
        self.decoder=Mlp(hidden_sizes, input_size, representation_size, layer_norm=layer_norm, output_activation=output_activation, output_bias=None)
        self.output_scale = output_scale

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z) * self.output_scale

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        # TODO: is the deepcopy necessary?
        self.__dict__.update(copy.deepcopy(d))


class AutoEncoder(VAE):
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar
