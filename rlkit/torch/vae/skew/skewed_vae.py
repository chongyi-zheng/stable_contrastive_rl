"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
from collections import defaultdict

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import (
    BatchSampler, WeightedRandomSampler,
    RandomSampler,
)

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.core import PyTorchModule


class Encoder(nn.Sequential):
    def encode(self, x):
        return self.get_encoding_and_suff_stats(x)[0]

    def get_encoding_and_suff_stats(self, x):
        output = self(x)
        z_dim = output.shape[1] // 2
        means, log_var = (
            output[:, :z_dim], output[:, z_dim:]
        )
        stds = (0.5 * log_var).exp()
        epsilon = ptu.randn(means.shape)
        latents = epsilon * stds + means
        return latents, means, log_var, stds


class Decoder(nn.Sequential):
    def __init__(self, *args, output_var=1, output_offset=0):
        super().__init__(*args)
        self.output_var = output_var
        self.output_offset = output_offset

    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        return output + self.output_offset

    def decode(self, input):
        output = self(input)
        if self.output_var == 'learned':
            mu, logvar = torch.split(output, 2, dim=1)
            var = logvar.exp()
        else:
            mu = output
            var = self.output_var * ptu.ones_like(mu)
        return mu, var

    def reconstruct(self, input):
        mu, _ = self.decode(input)
        return mu

    def sample(self, latent):
        mu, var = self.decode(latent)
        return Normal(mu, var.pow(0.5)).sample()


class VAE(PyTorchModule):
    def __init__(
            self,
            encoder,
            decoder,
            z_dim,
            mode='importance_sampling',
            min_prob=1e-7,
            n_average=100,

            xy_range=((-1, 1), (-1, 1)),

            reset_vae_every_epoch=False,
            num_inner_vae_epochs=10,
            weight_loss=False,
            skew_sampling=False,
            batch_size=32,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.z_dim = z_dim
        self.mode = mode
        self.min_log_prob = np.log(min_prob)
        self.n_average = n_average

        self.xy_range = xy_range

        self.reset_vae_every_epoch = reset_vae_every_epoch
        self.num_inner_vae_epochs = num_inner_vae_epochs
        self.weight_loss = weight_loss
        self.skew_sampling = skew_sampling
        self.batch_size = batch_size
        self.encoder_opt = Adam(self.encoder.parameters())
        self.decoder_opt = Adam(self.decoder.parameters())

        self._epoch_stats = None

    def sample_given_z(self, latent):
        return self.decoder.sample(latent)

    def sample(self, num_samples):
        return ptu.get_numpy(self.sample_given_z(
            ptu.randn(num_samples, self.z_dim)
        ))

    def reconstruct(self, data):
        latents = self.encoder.encode(ptu.from_numpy(data))
        return ptu.get_numpy(self.decoder.reconstruct(latents))

    def compute_density(self, data):
        orig_data_length = len(data)
        data = np.vstack([
            data for _ in range(self.n_average)
        ])
        data = ptu.from_numpy(data)
        if self.mode == 'biased':
            latents, means, log_vars, stds = (
                self.encoder.get_encoding_and_suff_stats(data)
            )
            importance_weights = ptu.ones(data.shape[0])
        elif self.mode == 'prior':
            latents = ptu.randn(len(data), self.z_dim)
            importance_weights = ptu.ones(data.shape[0])
        elif self.mode == 'importance_sampling':
            latents, means, log_vars, stds = (
                self.encoder.get_encoding_and_suff_stats(data)
            )
            prior = Normal(ptu.zeros(1), ptu.ones(1))
            prior_log_prob = prior.log_prob(latents).sum(dim=1)

            encoder_distrib = Normal(means, stds)
            encoder_log_prob = encoder_distrib.log_prob(latents).sum(dim=1)

            importance_weights = (prior_log_prob - encoder_log_prob).exp()
        else:
            raise NotImplementedError()

        unweighted_data_log_prob = self.compute_log_prob(
            data, self.decoder, latents
        ).squeeze(1)
        unweighted_data_prob = unweighted_data_log_prob.exp()
        unnormalized_data_prob = unweighted_data_prob * importance_weights
        """
        Average over `n_average`
        """
        dp_split = torch.split(unnormalized_data_prob, orig_data_length, dim=0)
        # pre_avg.shape = ORIG_LEN x N_AVERAGE
        dp_stacked = torch.stack(dp_split, dim=1)
        # final.shape = ORIG_LEN
        unnormalized_dp = torch.sum(dp_stacked, dim=1, keepdim=False)

        """
        Compute the importance weight denomintors.
        This requires summing across the `n_average` dimension.
        """
        iw_split = torch.split(importance_weights, orig_data_length, dim=0)
        iw_stacked = torch.stack(iw_split, dim=1)
        iw_denominators = iw_stacked.sum(dim=1, keepdim=False)

        final = unnormalized_dp / iw_denominators
        return ptu.get_numpy(final)

    def fit(self, data, weights=None):
        if weights is None:
            weights = np.ones(len(data))
        sum_of_weights = weights.flatten().sum()
        weights = weights / sum_of_weights
        all_weights_pt = ptu.from_numpy(weights)

        indexed_train_data = IndexedData(data)
        if self.skew_sampling:
            base_sampler = WeightedRandomSampler(weights, len(weights))
        else:
            base_sampler = RandomSampler(indexed_train_data)

        train_dataloader = DataLoader(
            indexed_train_data,
            sampler=BatchSampler(
                base_sampler,
                batch_size=self.batch_size,
                drop_last=False,
            ),
        )
        if self.reset_vae_every_epoch:
            raise NotImplementedError()

        epoch_stats_list = defaultdict(list)
        for _ in range(self.num_inner_vae_epochs):
            for _, indexed_batch in enumerate(train_dataloader):
                idxs, batch = indexed_batch
                batch = batch[0].float().to(ptu.device)

                latents, means, log_vars, stds = (
                    self.encoder.get_encoding_and_suff_stats(
                        batch
                    )
                )
                beta = 1
                kl = self.kl_to_prior(means, log_vars, stds)
                reconstruction_log_prob = self.compute_log_prob(
                    batch, self.decoder, latents
                )

                elbo = - kl * beta + reconstruction_log_prob
                if self.weight_loss:
                    idxs = torch.cat(idxs)
                    batch_weights = all_weights_pt[idxs].unsqueeze(1)
                    loss = -(batch_weights * elbo).sum()
                else:
                    loss = - elbo.mean()
                self.encoder_opt.zero_grad()
                self.decoder_opt.zero_grad()
                loss.backward()
                self.encoder_opt.step()
                self.decoder_opt.step()

                epoch_stats_list['losses'].append(ptu.get_numpy(loss))
                epoch_stats_list['kls'].append(ptu.get_numpy(kl.mean()))
                epoch_stats_list['log_probs'].append(
                    ptu.get_numpy(reconstruction_log_prob.mean())
                )
                epoch_stats_list['latent-mean'].append(
                    ptu.get_numpy(latents.mean())
                )
                epoch_stats_list['latent-std'].append(
                    ptu.get_numpy(latents.std())
                )
                for k, v in create_stats_ordered_dict(
                    'weights',
                    ptu.get_numpy(all_weights_pt)
                ).items():
                    epoch_stats_list[k].append(v)

        self._epoch_stats = {
            'unnormalized weight sum': sum_of_weights,
        }
        for k in epoch_stats_list:
            self._epoch_stats[k] = np.mean(epoch_stats_list[k])

    def get_epoch_stats(self):
        return self._epoch_stats

    def get_plot_ranges(self):
        xrange, yrange = self.xy_range
        xdelta = (xrange[1] - xrange[0]) / 10.
        ydelta = (yrange[1] - yrange[0]) / 10.
        return (
            (xrange[0]-xdelta, xrange[1]+xdelta),
            (yrange[0]-ydelta, yrange[1]+ydelta)
        )

    @staticmethod
    def kl_to_prior(means, log_vars, stds):
        """
        KL between a Gaussian and a standard Gaussian.

        https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        """
        # Implement for one dimension. Broadcasting will take care of the constants.
        return 0.5 * (
                - log_vars
                - 1
                + (stds ** 2)
                + means ** 2
        ).sum(dim=1, keepdim=True)


    @staticmethod
    def compute_log_prob(batch, decoder, latents):
        mu, var = decoder.decode(latents)
        dist = Normal(mu, var.pow(0.5))
        vals = dist.log_prob(batch).sum(dim=1, keepdim=True)
        return vals




class IndexedData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]
