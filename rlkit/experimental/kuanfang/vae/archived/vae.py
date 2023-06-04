# Adapted from pytorch examples

# import copy

# import numpy as np
from torch import nn

from rlkit.pythonplusplus import identity
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, TwoHeadMlp
import rlkit.torch.pytorch_util as ptu


class AutoEncoder(PyTorchModule):

    def __init__(
            self,
            representation_size,
            input_size,
            hidden_sizes,
            output_activation=identity,
            layer_norm=False,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.output_activation = output_activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        hidden_sizes = list(hidden_sizes)
        self.encoder = Mlp(
            hidden_sizes,
            representation_size,
            input_size,
            layer_norm=layer_norm)

        hidden_sizes.reverse()
        self.decoder = Mlp(
            hidden_sizes,
            input_size,
            representation_size,
            layer_norm=layer_norm,
            output_activation=output_activation,
            output_bias=None)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar


class Vae(PyTorchModule):

    def __init__(
            self,
            representation_size,
            input_size,
            hidden_sizes,
            output_activation=identity,
            layer_norm=False,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.output_activation = output_activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        hidden_sizes = list(hidden_sizes)
        self.encoder = TwoHeadMlp(
            hidden_sizes,
            representation_size,
            representation_size,
            input_size,
            layer_norm=layer_norm)

        hidden_sizes.reverse()
        self.decoder = Mlp(
            hidden_sizes,
            input_size,
            representation_size,
            layer_norm=layer_norm,
            output_activation=output_activation)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return (mu, logvar), z, x_recon

    def sample_prior(self, batch_size):
        z_s = ptu.normal(mean=0., std=1., size=self.representation_size)
        return ptu.get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
