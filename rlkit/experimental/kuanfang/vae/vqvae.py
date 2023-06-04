from __future__ import print_function

from absl import logging

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from rlkit.torch import pytorch_util as ptu

from rlkit.experimental.kuanfang.vae import network_utils


def dbprint(*x):
    pass


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_residual_layers,
                 residual_hidden_dim,
                 ):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim // 2,
            kernel_size=4,
            stride=2,
            padding=1)
        self._conv_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1)
        self._conv_3 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1)
        self._residual_stack = network_utils.ResidualStack(
            in_channels=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x, inplace=True)

        x = self._conv_2(x)
        x = F.relu(x, inplace=True)

        x = self._conv_3(x)
        x = F.relu(x, inplace=True)

        x = self._residual_stack(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 num_residual_layers,
                 residual_hidden_dim,
                 out_channels=3,
                 out_activation='tanh',
                 ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1)

        self._residual_stack = network_utils.ResidualStack(
            in_channels=hidden_dim,
            num_residual_layers=num_residual_layers,
            residual_hidden_dim=residual_hidden_dim)

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=4,
            stride=2,
            padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=hidden_dim // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1)

        if out_activation is None:
            self._out_activation = None
        elif out_activation == 'tanh':
            self._out_activation = nn.Tanh()
        elif out_activation == 'sigmoid':
            self._out_activation = nn.Sigmoid()
        else:
            raise ValueError

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x, inplace=True)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x, inplace=True)

        x = self._conv_trans_2(x)

        if self._out_activation is not None:
            x = self._out_activation(x)

        return x


class VectorQuantizer(nn.Module):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 gaussion_prior=False,
                 # use_normalization=False,
                 ):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._codebook_embeddings = nn.Embedding(
            self._num_embeddings,
            self._embedding_dim)

        if gaussion_prior:
            self._codebook_embeddings.weight.data.normal_()

        else:
            self._codebook_embeddings.weight.data.uniform_(
                -1 / self._num_embeddings, 1 / self._num_embeddings)

    def forward(self, inputs, return_onehot=False):
        # See this link for the notations:
        # https://ml.berkeley.edu/blog/posts/vq-vae/
        # ze: The encoded continuous vector.
        # zi: The index of the closest embedding in the codebook.
        # zq: The closest embedding in the codebook.

        # Convert inputs from BCHW -> BHWC
        zes = inputs

        bhwc_zes = zes.permute(0, 2, 3, 1).contiguous()

        dbprint('bhwc_zes: ', bhwc_zes.shape)

        # Flatten input
        flat_zes = bhwc_zes.view(-1, self._embedding_dim)
        dbprint('flat_zes: ', flat_zes.shape)

        # Calculate distances
        codebook_embeddings = self._codebook_embeddings.weight

        distances = (
            torch.sum(flat_zes ** 2, dim=1, keepdim=True)
            + torch.sum(codebook_embeddings ** 2, dim=1)
            - 2 * torch.matmul(flat_zes, codebook_embeddings.t())
        )
        dbprint('distances: ', distances.shape)

        # Encoding
        zis = torch.argmin(distances, dim=1).unsqueeze(1)
        zis = zis.view(-1, bhwc_zes.shape[-3], bhwc_zes.shape[-2])
        dbprint('zis: ', zis.shape)

        # Quantize and unflatten
        bhwc_zqs, onehot_zis = self.convert_zi_to_zq(zis, return_onehot=True)
        bhwc_zqs = bhwc_zqs.view(bhwc_zes.shape)

        # Convert zqs from BHWC -> BCHW
        zqs = bhwc_zqs.permute(0, 3, 1, 2).contiguous()

        if return_onehot:
            return zis, zqs, onehot_zis
        else:
            return zis, zqs

    def convert_zi_to_onehot(self, zis):
        zis = zis.reshape(-1).unsqueeze(1)

        onehot_zis = torch.zeros(
            zis.shape[0],
            self._num_embeddings,
            device=zis.device)
        onehot_zis.scatter_(1, zis, 1)

        return onehot_zis

    def convert_zi_to_zq(self, zis, return_onehot=False):
        zq_shape = zis.shape + (self._embedding_dim,)
        onehot_zis = self.convert_zi_to_onehot(zis)
        zqs = torch.matmul(
            onehot_zis,
            self._codebook_embeddings.weight).view(zq_shape)

        if return_onehot:
            return zqs, onehot_zis
        else:
            return zqs


class VqVae(nn.Module):

    def __init__(
            self,
            embedding_dim=5,
            input_channels=3,
            hidden_dim=128,
            num_residual_layers=3,
            residual_hidden_dim=64,
            num_embeddings=512,
            commitment_cost=0.25,
            imsize=48,
            decay=0.0,
    ):
        super(VqVae, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.num_embeddings = num_embeddings
        self.imlength = imsize * imsize * input_channels

        self._encoder = Encoder(
            input_channels,
            hidden_dim,
            num_residual_layers,
            residual_hidden_dim)

        self._pre_vq_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        if decay > 0.0:
            raise NotImplementedError
        else:
            self._vector_quantizer = VectorQuantizer(
                num_embeddings,
                self.embedding_dim,
            )

        self._decoder = Decoder(
            self.embedding_dim,
            hidden_dim,
            num_residual_layers,
            residual_hidden_dim,
            3)

        # Calculate latent sizes
        if imsize not in [32, 36, 48, 84]:
            raise ValueError(imsize)
        else:
            self.root_len = int(imsize / 4)

        # Calculate latent sizes
        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim

        self._commitment_cost = commitment_cost

    @property
    def vector_quantizer(self):
        return self._vector_quantizer

    def encode(self, inputs, mode='zq', flatten=True):
        inputs = inputs.view(-1,
                             self.input_channels,
                             self.imsize,
                             self.imsize)

        image_feats = self._encoder(inputs)
        zes = self._pre_vq_conv(image_feats)
        zis, zqs = self._vector_quantizer(zes)

        if flatten:
            if mode == 'ze':
                return zes.reshape(-1, self.representation_size)
            elif mode == 'zi':
                return zis.reshape(-1, self.discrete_size)
            elif mode == 'zq':
                return zqs.reshape(-1, self.representation_size)
            else:
                raise ValueError('Unrecognized VQ-VAE mode: %s' % (mode))
        else:
            if mode == 'ze':
                return zes
            elif mode == 'zi':
                return zis
            elif mode == 'zq':
                return zqs
            else:
                raise ValueError('Unrecognized VQ-VAE mode: %s' % (mode))

    def decode(self, latents, mode='zq'):
        if mode == 'ze':
            zes = latents.view(
                -1, self.embedding_dim, self.root_len, self.root_len)
            zis, zqs = self._vector_quantizer(zes, return_onehot=False)
        elif mode == 'zi':
            zis = latents.view(-1, self.root_len, self.root_len)
            zqs = self._vector_quantizer.convert_zi_to_zq(zis)
            zqs = zqs.permute(0, 3, 1, 2).contiguous()
        elif mode == 'zq':
            zqs = latents.view(
                -1, self.embedding_dim, self.root_len, self.root_len)
        else:
            raise ValueError('Unrecognized VQ-VAE mode: %s' % (mode))

        return self._decoder(zqs)

    def compute_loss(self, inputs):
        assert inputs.shape[-1] == self.imsize
        assert inputs.shape[-2] == self.imsize

        inputs = inputs.view(-1,
                             self.input_channels,
                             self.imsize,
                             self.imsize)
        dbprint('- compute_loss -')
        dbprint('inputs: ', inputs.shape)

        image_feats = self._encoder(inputs)
        dbprint('image_feats: ', image_feats.shape)
        zes = self._pre_vq_conv(image_feats)
        dbprint('zes: ', zes.shape)
        zis, zqs, onehot_zis = self._vector_quantizer(zes, return_onehot=True)
        dbprint('zis: ', zis.shape)
        dbprint('zqs: ', zqs.shape)
        dbprint('onehot_zis: ', onehot_zis.shape)

        e_latent_loss = F.mse_loss(zqs.detach(), zes)
        q_latent_loss = F.mse_loss(zqs, zes.detach())
        loss_vq = q_latent_loss + self._commitment_cost * e_latent_loss

        avg_probs = torch.mean(onehot_zis, dim=0)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        zqs = zes + (zqs - zes).detach()
        recon = self.decode(zqs)
        dbprint('recon: ', recon.shape)
        loss_recon = F.mse_loss(recon, inputs)

        loss = loss_vq + loss_recon

        extra = {
            'loss_vq': loss_vq,
            'loss_recon': loss_recon,
            'perplexity': perplexity,

            'ze': zes,
            'zq': zqs,
            'zi': zis,

            'recon': recon.view(
                -1,
                self.input_channels,
                self.imsize,
                self.imsize),
        }

        return loss, extra

    def encode_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs - 0.5
        inputs = inputs.reshape(
            (-1,
             self.input_channels,
             self.imsize,
             self.imsize))
        inputs = np.transpose(inputs, [0, 1, 3, 2])
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs)))

    def decode_np(self, inputs, cont=True):
        assert cont is True
        outputs = ptu.get_numpy(self.decode(ptu.from_numpy(inputs)))
        outputs = outputs + 0.5
        outputs = np.clip(outputs, 0, 1)
        outputs = np.transpose(outputs, [0, 1, 3, 2])
        return outputs

    def encode_one_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs[None, ...]
        outputs = self.encode_np(inputs)
        return outputs[0]

    def decode_one_np(self, inputs, cont=True):
        inputs = inputs[None, ...]
        outputs = self.decode_np(inputs)
        return outputs[0]


def encode_dataset(vqvae, dataset, batch_size=1024, mode='zq'):
    data = dataset.data
    logging.info('data.shape: %r', data.shape)

    num_seqs = data.shape[0]
    num_steps = data.shape[1]
    num_samples = num_seqs * num_steps

    data = data.reshape([-1, data.shape[-3], data.shape[-2], data.shape[-1]])

    num_batches = int(np.ceil(float(num_samples) / float(batch_size)))
    encodings = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        end = min(end, num_samples)

        batch = data[start:end]

        if dataset.transform is not None:
            batch = dataset.transform(batch)

        batch = ptu.from_numpy(batch)
        batch = batch.view(
            -1, batch.shape[-3], batch.shape[-2], batch.shape[-1])

        encoding_i = vqvae.encode(batch, mode=mode, flatten=False)
        encoding_i = ptu.get_numpy(encoding_i)
        encodings.append(encoding_i)
        logging.info('Finished encoding the data %d / %d.'
                     % (end, num_samples))

    encodings = np.concatenate(encodings, axis=0)
    logging.info('encodings.shape: %r', encodings.shape)

    if mode == 'zq':
        encodings = np.reshape(
            encodings,
            (num_seqs,
             num_steps,
             encodings.shape[-3],
             encodings.shape[-2],
             encodings.shape[-1])
        )
    elif mode == 'zi':
        encodings = np.reshape(
            encodings,
            (num_seqs,
             num_steps,
             encodings.shape[-2],
             encodings.shape[-1])
        )
    else:
        raise NotImplementedError

    return encodings
