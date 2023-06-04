import torch
from rlkit.torch.core import PyTorchModule
import numpy as np
import abc
from torch.distributions import Normal
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.envs.encoder_wrappers import Encoder


class VAEBase(PyTorchModule, Encoder, metaclass=abc.ABCMeta):
    def __init__(
            self,
            representation_size,
    ):
        super().__init__()
        self._representation_size = representation_size

    @property
    def representation_size(self):
        return self._representation_size


    @abc.abstractmethod
    def encode(self, input):
        """
        :param input:
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rsample(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reparameterize(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def logprob(self, inputs, obs_distribution_params):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def kl_divergence(self, latent_distribution_params):
        """
        :param latent_distribution_params:
        :return: kl div between latent_distribution_params and prior on latent space
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: get latents from latent distribution params
        """
        raise NotImplementedError()

    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(input)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params

    ### Numpy functions

    def decode_np(self, latents):
        self.eval()
        reconstructions, _ = self.decode(ptu.from_numpy(latents))
        decoded = ptu.get_numpy(reconstructions)
        return decoded

    def encode_one_np(self, img):
        return self.encode_np(img[None])[0]

    def decode_one_np(self, latent):
        return self.decode_np(latent[None])[0]

    def encode_np(self, imgs):
        self.eval()
        latent_distribution_params = self.encode(ptu.from_numpy(imgs))
        return ptu.get_numpy(latent_distribution_params[0])

    def _reconstruct_img(self, flat_img):
        self.vae.eval()
        latent_distribution_params = self.vae.encode(ptu.from_numpy(flat_img.reshape(1,-1)))
        reconstructions, _ = self.vae.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]


class GaussianLatentVAE(VAEBase):
    def __init__(
            self,
            representation_size,
    ):
        super().__init__(representation_size)
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        stds = (0.5 * logvar).exp()
        epsilon = ptu.randn(*mu.size())
        latents = epsilon * stds + mu
        return latents

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        return latent_distribution_params[0].cpu()


def compute_bernoulli_log_prob(x, reconstruction_of_x):
    return -1 * F.binary_cross_entropy(
        reconstruction_of_x,
        x,
        reduction='elementwise_mean'
    )


def compute_gaussian_log_prob(input, dec_mu, dec_var):
    decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
    log_probs = decoder_dist.log_prob(input)
    vals = log_probs.sum(dim=1, keepdim=True)
    return vals.mean()
