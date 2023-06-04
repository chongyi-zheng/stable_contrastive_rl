from typing import *

import torch
from torch import nn

from .architectures import get_conditional_deep_vae
from .stage import VaeStage, LvaeStage, BivaStage, ConditionalBivaStage
from .utils import DataCollector
from ..layers import PaddedNormedConv


class ConditionalDeepVae(nn.Module):
    """
    A Deep Hierarchical VAE.
    The model is a stack of N stages. Each stage features an inference and a generative path.
    Depending on the choice of the stage, multiple models can be implemented:
    - VAE: https://arxiv.org/abs/1312.6114
    - LVAE: https://arxiv.org/abs/1602.02282
    - BIVA: https://arxiv.org/abs/1902.02102
    """

    def __init__(self,
                 Stage: Any = BivaStage,
                 input_channels: Optional[int] = 3,
                 imsize: Optional[int] = 48,
                 padded_shp: Optional[Tuple] = None,
                 stages: List[List[Tuple]] = None,
                 latents: List = None,
                 nonlinearity: str = 'elu',
                 q_dropout: float = 0.5, #0.2
                 p_dropout: float = 0.5, #0.
                 features_out: Optional[int] = 3, #100
                 lambda_init: Optional[Callable] = None,
                 projection: Optional[nn.Module] = None,
                 **kwargs):

        """
        Initialize the Deep VAE model.
        :param Stage: stage constructor (VaeStage, LvaeStage, BivaStage)
        :param tensor_shp: Input tensor shape (batch_size, channels, *dimensions)
        :param padded_shp: pad input tensor to this shape
        :param stages: a list of list of tuple, each tuple describing a convolutional block (filters, stride, kernel_size)
        :param latents: a list describing the stochastic layers for each stage
        :param nonlinearity: activation function (gelu, elu, relu, tanh)
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param features_out: optional number of output features if different from the input
        :param lambda_init: lambda function applied to the input
        :param projection: projection layer with constructor __init__(output_shape)

        :param kwargs: additional arugments passed to each stage
        """
        super().__init__()
        stages, latents = self.get_default_architecture(stages, latents)
        self.latents = latents
        self.representation_size = 4 * sum([layer['N'] for layer in self.latents]) - 2 * self.latents[-1]['N']
        self.latent_sizes = [self.representation_size // 2, self.representation_size // 2]
        self.input_channels, self.imsize, self.imlength = input_channels, imsize, imsize * imsize * input_channels
        tensor_shp = (-1, self.input_channels, self.imsize, self.imsize)
        self.input_tensor_shape = tensor_shp
        self.lambda_init = lambda_init
        self.pad = None

        # Define arg dicts
        Act = {'elu': nn.ELU, 'relu': nn.ReLU, 'tanh': nn.Tanh()}[nonlinearity]
        block_args = {'act': Act, 'q_dropout': q_dropout, 'p_dropout': p_dropout}
        delta_input_shape = {'x': (-1, 2 * self.input_channels, self.imsize, self.imsize)}
        cond_input_shape = {'x': (-1, self.input_channels, self.imsize, self.imsize)}

        # Define both model stages
        self.delta_stages = self.create_biva_model(ConditionalBivaStage, block_args, delta_input_shape, stages, latents, kwargs)
        self.cond_stages = self.create_biva_model(BivaStage, block_args, cond_input_shape, stages, latents, kwargs)

        # Output convolution
        tensor_shp = self.delta_stages[0].p_output_shape['d']
        if features_out is None:
            features_out = self.input_tensor_shape[1]
        conv_obj = nn.Conv2d if len(tensor_shp) == 4 else nn.Conv1d
        conv_out = conv_obj(tensor_shp[1], features_out, 1)
        conv_out = PaddedNormedConv(tensor_shp, conv_out, weightnorm=True)
        self.projection = nn.Sequential(Act(), conv_out)

    def create_biva_model(self, stage_class, block_args, input_shape, stages, latents, kwargs):
        stages_ = []

        for i, (conv_data, z_data) in enumerate(zip(stages, latents)):
            top = i == len(stages) - 1
            bottom = i == 0

            stage = stage_class(input_shape, conv_data, z_data, top=top, bottom=bottom, **block_args, **kwargs)
            input_shape = stage.q_output_shape
            stages_ += [stage]

        return nn.ModuleList(stages_)

    def get_default_architecture(self, stages, latents):
        if stages is None:
            stages, _ = get_conditional_deep_vae()

        if latents is None:
            _, latents = get_conditional_deep_vae()

        return stages, latents

    def infer_cond(self, x_cond, **kwargs):
        cond_posteriors = []
        cond_data = {'x': x_cond}

        for stage in self.cond_stages:
            cond_data, cond_posterior = stage.infer(cond_data, **kwargs)
            cond_posteriors += [cond_posterior]

        return cond_posteriors

    def infer(self, x_delta, x_cond, **kwargs):
        """
        Forward pass through the inference network and return the posterior of each layer order from the top to the bottom.
        :param x: input tensor
        :param kwargs: additional arguments passed to each stage
        :return: a list that contains the data for each stage
        """
        delta_posteriors = []
        delta_data = {'x': torch.cat([x_delta, x_cond], dim=1)}
        
        for stage in self.delta_stages:
            delta_data, delta_posterior = stage.infer(delta_data, **kwargs)
            delta_posteriors += [delta_posterior]
        
        cond_posteriors = self.infer_cond(x_cond, **kwargs)
        return delta_posteriors, cond_posteriors

    def generate(self, delta_posteriors, cond_posteriors, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the generative model, compute KL and return reconstruction x_, KL and auxiliary data.
        If no posterior is provided, the prior is sampled.
        :param posteriors: a list containing the posterior for each stage
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': reconstruction logits, 'kl': kl for each stage, **auxiliary}
        """
        if delta_posteriors is None:
            delta_posteriors = [None for _ in self.delta_stages]

        delta_output_data, cond_output_data = DataCollector(), DataCollector()
        x_delta, x_cond = {}, {}
        conditioning = []

        #debugging: make sure this is done right
        for posterior, delta_posterior, stage in zip(cond_posteriors[::-1], delta_posteriors[::-1], self.cond_stages[::-1]):
            x_cond, data = stage(x_cond, posterior, **kwargs)
            conditioning.append(x_cond['z'])
            cond_output_data.extend(data)

        for posterior, cond, stage in zip(delta_posteriors[::-1], conditioning, self.delta_stages[::-1]):
            x_delta, data = stage(x_delta, posterior, cond, **kwargs)
            delta_output_data.extend(data)

        # output convolution
        x_delta = self.projection(x_delta['d'])
        x_cond = self.projection(x_cond['d'])

        # sort data: [z1, z2, ..., z_L]
        delta_output_data = delta_output_data.sort()

        return {'x_': x_delta, **delta_output_data}, {'x_': x_cond, **cond_output_data}

    def forward(self, x_delta, x_cond, **kwargs):
        """
        Forward pass through the inference model, the generative model and compute KL for each stage.
        x_ = p_\theta(x|z), z \sim q_\phi(z|x)
        kl_i = log q_\phi(z_i | h) - log p_\theta(z_i | h)

        :param x: input tensor
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': reconstruction logits, 'kl': kl for each stage, **auxiliary}
        """
        x_delta = x_delta.reshape(*self.input_tensor_shape)
        x_cond = x_cond.reshape(*self.input_tensor_shape)

        delta_posteriors, cond_posteriors = self.infer(x_delta, x_cond, **kwargs)
        
        delta_data, cond_data = self.generate(delta_posteriors, cond_posteriors, N=x_delta.size(0), **kwargs)

        return delta_data['x_'], cond_data['x_'], delta_data['kl'], cond_data['kl']

    def sample_images(self, batch_size, x_cond, **kwargs):
        """
        Sample the prior and pass through the generative model.
        x_ = p_\theta(x|z), z \sim p_\theta(z)

        :param N: number of samples (batch size)
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': sample logits}
        """
        x_cond = x_cond.reshape(-1, self.imlength)
        if x_cond.shape[0] == 1:
            x_cond = x_cond.repeat(batch_size, 1)

        x_cond = x_cond.reshape(*self.input_tensor_shape)
        cond_posteriors = self.infer_cond(x_cond, **kwargs)

        delta_data, cond_data = self.generate(None, cond_posteriors, N=batch_size, **kwargs)
        samples = delta_data['x_']

        return samples


class ConditionalBIVA(ConditionalDeepVae):
    def __init__(self, **kwargs):
        kwargs.pop('Stage', None)
        super().__init__(Stage=BivaStage, **kwargs)
