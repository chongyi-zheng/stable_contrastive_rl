from __future__ import print_function
import torch
import numpy as np
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from torch.autograd import Variable
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy, MlpPolicy
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN
from rlkit.torch.vae.vae_base import compute_bernoulli_log_prob, \
    compute_gaussian_log_prob, GaussianLatentVAE
from rlkit.torch.vae.conv_vae import ConvVAE
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(

            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens)
             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs, ):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers,
            num_residual_hiddens, out_channels=3):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,
            gaussion_prior=False):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)

        if gaussion_prior:
            self._embedding.weight.data.normal_()

        else:
            self._embedding.weight.data.uniform_(
                -1 / self._num_embeddings, 1 / self._num_embeddings)

        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings,
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        # q_latent_loss = F.mse_loss(quantized, inputs.detach(), reduction='sum')

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1,
            2).contiguous(), perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay,
            epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings,
            device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs, reduction='sum')
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1,
            2).contiguous(), perplexity, encoding_indices


class VQ_VAE(nn.Module):
    def __init__(
            self,
            embedding_dim=5,
            input_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            num_embeddings=512,
            commitment_cost=0.25,
            decoder_output_activation=None,  # IGNORED FOR NOW
            architecture=None,  # IGNORED FOR NOW
            imsize=48,
            decay=0.0):
        super(VQ_VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.num_embeddings = num_embeddings

        self._encoder = Encoder(input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings,
                self.embedding_dim,
                commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, self.embedding_dim,
                commitment_cost)

        self._decoder = Decoder(self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        # Calculate latent sizes
        if imsize == 32:
            self.root_len = 8
        elif imsize == 36:
            self.root_len = 9
        elif imsize == 48:
            self.root_len = 12
        elif imsize == 84:
            self.root_len = 21
        else:
            raise ValueError(imsize)

        self.discrete_size = self.root_len * self.root_len
        self.representation_size = self.discrete_size * self.embedding_dim
        # Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        vq_loss, quantized, perplexity, _ = self.quantize_image(inputs)
        recon = self.decode(quantized)

        recon_error = F.mse_loss(recon, inputs)
        return vq_loss, recon, perplexity, recon_error

    def quantize_image(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z = self._encoder(inputs)
        z = self._pre_vq_conv(z)
        return self._vq_vae(z)

    def encode(self, inputs, cont=True):
        _, quantized, _, encodings = self.quantize_image(inputs)

        if cont:
            return quantized.reshape(-1, self.representation_size)
        return encodings.reshape(-1, self.discrete_size)

    def latent_to_square(self, latents):
        return latents.reshape(-1, self.root_len, self.root_len)

    def discrete_to_cont(self, e_indices):
        e_indices = self.latent_to_square(e_indices)
        input_shape = e_indices.shape + (self.embedding_dim,)
        e_indices = e_indices.reshape(-1).unsqueeze(1)

        min_encodings = torch.zeros(e_indices.shape[0], self.num_embeddings,
            device=e_indices.device)
        min_encodings.scatter_(1, e_indices, 1)

        e_weights = self._vq_vae._embedding.weight
        quantized = torch.matmul(
            min_encodings, e_weights).view(input_shape)

        z_q = torch.matmul(min_encodings, e_weights).view(input_shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def set_pixel_cnn(self, pixel_cnn):
        self.pixel_cnn = pixel_cnn

    def sample_conditional_indices(self, batch_size, cond):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, axis=0)
        cond = ptu.from_numpy(cond)

        sampled_indices = self.pixel_cnn.generate(
            shape=(self.root_len, self.root_len),
            batch_size=batch_size,
            cond=cond)

        return sampled_indices

    def sample_prior(self, batch_size, cond=None):
        if self.pixel_cnn.is_conditional:
            sampled_indices = self.sample_conditional_indices(batch_size, cond)
        else:
            sampled_indices = self.pixel_cnn.generate(
                shape=(self.root_len, self.root_len),
                batch_size=batch_size)

        sampled_indices = sampled_indices.reshape(batch_size, self.discrete_size)
        z_q = self.discrete_to_cont(sampled_indices).reshape(-1, self.representation_size)
        return ptu.get_numpy(z_q)

    def decode(self, latents, cont=True):
        if cont:
            z_q = latents.reshape(-1, self.embedding_dim, self.root_len,
                self.root_len)
        else:
            z_q = self.discrete_to_cont(latents)

        return self._decoder(z_q)

    def encode_one_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))[0]

    def encode_np(self, inputs, cont=True):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs), cont=cont))

    def decode_one_np(self, inputs, cont=True):
        return np.clip(ptu.get_numpy(
            self.decode(ptu.from_numpy(inputs).reshape(1, -1), cont=cont))[0],
            0, 1)

    def decode_np(self, inputs, cont=True):
        return np.clip(
            ptu.get_numpy(self.decode(ptu.from_numpy(inputs), cont=cont)), 0, 1)


##### VAE'S IMPLEMENTED WITH THIS ARCHITECTURE #####
class VAE(nn.Module):
    def __init__(
            self,
            embedding_dim=1,
            input_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            decoder_output_activation=None,  # IGNORED FOR NOW
            architecture=None,  # IGNORED FOR NOW
            min_variance=1e-3,
            imsize=48,
            ):
        super(VAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.log_min_variance = float(np.log(min_variance))
        
        self._encoder = Encoder(input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)
        
        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)
        
        self._decoder = Decoder(self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        # Calculate latent sizes
        if imsize == 32:
            self.root_len = 8
        elif imsize == 36:
            self.root_len = 9
        elif imsize == 48:
            self.root_len = 12
        elif imsize == 84:
            self.root_len = 21
        else:
            raise ValueError(imsize)

        self.representation_size = self.root_len * self.root_len * self.embedding_dim
        # Calculate latent sizes

    def compute_loss(self, inputs):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z_s, kle = self.encode(inputs, computing_loss=True)
        recon = self.decode(z_s)

        recon_error = F.mse_loss(recon, inputs, reduction='sum')
        return recon, recon_error, kle

    def encode(self, inputs, computing_loss=False):
        inputs = inputs.view(-1,
            self.input_channels,
            self.imsize,
            self.imsize)

        z_conv = self._encoder(inputs)
        
        mu = self.f_mu(z_conv).reshape(-1, self.representation_size)
        unclipped_logvar = self.f_logvar(z_conv).reshape(-1, self.representation_size)
        logvar = self.log_min_variance + torch.abs(unclipped_logvar)
        
        if self.training:
            z_s = self.rsample(mu, logvar)
        else:
            z_s = mu

        if computing_loss:
            return z_s, self.kl_divergence(mu, logvar)
        return z_s

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

    def decode(self, latents):
        z_s = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
        return self._decoder(z_s)

    def encode_one_np(self, inputs):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs)))[0]

    def encode_np(self, inputs):
        return ptu.get_numpy(self.encode(ptu.from_numpy(inputs)))

    def decode_one_np(self, inputs):
        return np.clip(ptu.get_numpy(
            self.decode(ptu.from_numpy(inputs).reshape(1, -1)))[0],
            0, 1)

    def decode_np(self, inputs):
        return np.clip(
            ptu.get_numpy(self.decode(ptu.from_numpy(inputs))), 0, 1)


class CCVAE(nn.Module):
    def __init__(
            self,
            embedding_dim=1,
            input_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
            decoder_output_activation=None,
            architecture=None,
            min_variance=1e-3,
            imsize=48,
            ):
        super(CCVAE, self).__init__()
        self.imsize = imsize
        self.embedding_dim = embedding_dim
        self.pixel_cnn = None
        self.input_channels = input_channels
        self.imlength = imsize * imsize * input_channels
        self.log_min_variance = float(np.log(min_variance))
        
        self._encoder = Encoder(2 * input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        self._cond_encoder = Encoder(input_channels, num_hiddens,
            num_residual_layers,
            num_residual_hiddens)
        
        self.f_mu = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        self.f_logvar = nn.Conv2d(in_channels=num_hiddens,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1)

        self._conv_cond = nn.Conv2d(in_channels=num_hiddens,
                                    out_channels=self.embedding_dim,
                                    kernel_size=1,
                                    stride=1)
        
        self._decoder = Decoder(2 * self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        self._cond_decoder = Decoder(self.embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens)

        # Calculate latent sizes
        if imsize == 32:
            self.root_len = 8
        elif imsize == 36:
            self.root_len = 9
        elif imsize == 48:
            self.root_len = 12
        elif imsize == 84:
            self.root_len = 21
        else:
            raise ValueError(imsize)

        self.latent_size = self.root_len * self.root_len * self.embedding_dim
        self.representation_size = 2 * self.latent_size
        # Calculate latent sizes

    def compute_loss(self, x_delta, x_cond):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        z_cat, z_cond, kle = self.encode(x_delta, x_cond, computing_loss=True)
        
        delta_recon = self.decode(z_cat)
        cond_recon = self.decode(z_cond, cond=True)

        delta_recon_error = F.mse_loss(delta_recon, x_delta, reduction='sum')
        cond_recon_error = F.mse_loss(cond_recon, x_cond, reduction='sum')
        
        return delta_recon, delta_recon_error, cond_recon_error, kle

    def encode(self, x_delta, x_cond, computing_loss=False):
        x_delta = x_delta.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        x_cond = x_cond.view(-1,
          self.input_channels,
          self.imsize,
          self.imsize)

        obs = torch.cat([x_delta, x_cond], dim=1)

        z_conv = self._encoder(obs)
        cond_conv = self._cond_encoder(x_cond)
        
        mu = self.f_mu(z_conv).reshape(-1, self.latent_size)
        unclipped_logvar = self.f_logvar(z_conv).reshape(-1, self.latent_size)
        z_cond = self._conv_cond(cond_conv).reshape(-1, self.latent_size)
        logvar = self.log_min_variance + torch.abs(unclipped_logvar)
        
        if self.training:
            z_s = self.rsample(mu, logvar)
        else:
            z_s = mu

        z_cat = torch.cat([z_s, z_cond], dim=1)

        if computing_loss:
            return z_cat, z_cond, self.kl_divergence(mu, logvar)
        return z_cat

    def rsample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu, logvar):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def encode_cond(self, batch_size, cond):
        cond = cond.view(batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize)

        cond_conv = self._cond_encoder(cond)
        return self._conv_cond(cond_conv)

    def sample_prior(self, batch_size, cond=None, image_cond=True):
        if cond.shape[0] == 1:
            cond = cond.repeat(batch_size, axis=0)
        cond = ptu.from_numpy(cond)

        if image_cond:
            z_cond = self.encode_cond(batch_size, cond)
        else:
            z_cat = cond.reshape(batch_size, 2 * self.embedding_dim, self.root_len, self.root_len)
            z_cond = z_cat[:, self.embedding_dim:]

        z_delta = ptu.randn(batch_size, self.embedding_dim, self.root_len, self.root_len)
        z_cat = torch.cat([z_delta, z_cond], dim=1).view(-1, self.representation_size)
        
        return ptu.get_numpy(z_cat)

    def decode(self, latents, cond=False):
        if cond:
            z_cond = latents.reshape(-1, self.embedding_dim, self.root_len, self.root_len)
            return self._cond_decoder(z_cond)
        
        z_cat = latents.reshape(-1, 2 * self.embedding_dim, self.root_len, self.root_len)
        return self._decoder(z_cat)

    def encode_one_np(self, inputs, cond):
        inputs = ptu.from_numpy(inputs)
        cond = ptu.from_numpy(cond)
        return ptu.get_numpy(self.encode(inputs, cond))[0]

    def encode_np(self, inputs, cond):
        inputs = ptu.from_numpy(inputs)
        cond = ptu.from_numpy(cond)
        return ptu.get_numpy(self.encode(inputs, cond))

    def decode_one_np(self, inputs):
        recon = self.decode(ptu.from_numpy(inputs).reshape(1, -1))
        recon = ptu.get_numpy(recon)[0]
        return np.clip(recon, 0, 1)

    def decode_np(self, inputs):
        recon = ptu.get_numpy(self.decode(ptu.from_numpy(inputs)))
        return np.clip(recon, 0, 1)
