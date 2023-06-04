import os
import numpy as np
import torch
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.sac.policies.gaussian_policy import (
    LOG_SIG_MAX,
    LOG_SIG_MIN,
)
from rlkit.torch.distributions import MultivariateDiagonalNormal


from rlkit.experimental.chongyiz.networks.cnn import (
    CNN,
    TwoChannelCNN,
)

from vip import load_vip
from r3m import load_r3m
import mvp


class GaussianCNNPolicy(CNN, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            output_activation=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=output_activation,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)
        mean = self.last_fc(h)
        if self.output_activation is not None:
            mean = self.output_activation(mean)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)


class GoalConditionedGaussianFixedReprPolicy(TorchStochasticPolicy):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            output_activation=None,
            hidden_activation=nn.ReLU(),
            hidden_init=nn.init.xavier_uniform_,
            fc_normalization_type='none',
            encoder_type='vip',
            **kwargs
    ):
        super().__init__()
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert encoder_type in {'vip', 'r3m', 'mvp'}

        if encoder_type == 'vip':
            vip = load_vip("resnet50")
            self.encoder = vip
        elif encoder_type == 'r3m':
            r3m = load_r3m("resnet50")
            self.encoder = r3m
        elif encoder_type == 'mvp':
            mvp_model = mvp.load("vitb-mae-egosoup")
            mvp_model.freeze()
            self.encoder = mvp_model
        self.encoder.eval()
        # num_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        # self.encoder.device_ids = np.arange(num_devices).tolist()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.conv_input_length = self.input_width * self.input_height * self.input_channels

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.fc_normalization_type = fc_normalization_type

        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        if encoder_type == 'vip':
            fc_input_size = self.encoder.module.hidden_dim * 2
        elif encoder_type == 'r3m':
            fc_input_size = self.encoder.module.outdim * 2
        elif encoder_type == 'mvp':
            fc_input_size = self.encoder.embed_dim * 2
        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_input_size = hidden_size

            # fc_layer.weight.data.uniform_(-init_w, init_w)
            # fc_layer.bias.data.uniform_(-init_w, init_w)
            hidden_init(fc_layer.weight)
            fc_layer.bias.data.fill_(0)

            self.fc_layers.append(fc_layer)

            if self.fc_normalization_type == 'batch':
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                self.fc_norm_layers.append(nn.LayerNorm(hidden_size))

        self.last_fc = nn.Linear(fc_input_size, action_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def named_parameters(self, prefix='', recurse=True):
        # we want to skip parameters from pretrained encoders.
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if 'encoder' not in elem[0]:
                yield elem

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def forward(self, obs):
        conv_input = obs.narrow(start=0,
                                length=self.conv_input_length,
                                dim=1).contiguous()

        # reshape from batch of flattened images into (channels, w, h)
        conv_input = conv_input.view(conv_input.shape[0],
                                     self.input_channels,
                                     self.input_height,
                                     self.input_width)

        # vip expects image input to be [0-255]
        conv_input = conv_input * 255

        obs_h = self.encoder(conv_input[:, :3])
        g_h = self.encoder(conv_input[:, 3:])
        h = torch.cat([obs_h, g_h], dim=-1)

        h = self.apply_forward_fc(h)

        mean = self.last_fc(h)
        if self.output_activation is not None:
            mean = self.output_activation(mean)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)


class GaussianTwoChannelCNNPolicy(TwoChannelCNN, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            output_activation=None,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=output_activation,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)
        mean = self.last_fc(h)
        if self.output_activation is not None:
            mean = self.output_activation(mean)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)
