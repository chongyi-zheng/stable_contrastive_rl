import abc  # NOQA
import numpy as np
import torch
import torchvision  # NOQA
from torch import nn

from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy  # NOQA

from rlkit.torch.distributions import (
    MultivariateDiagonalNormal
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
)
# from rlkit.torch.sac.policies.base import (
#     DictTorchStochasticPolicy,
# )


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ObsEncoder(PyTorchModule):

    def encode(self, inputs):
        return self.forward(inputs)

    def encode_np(self, inputs, cont=True):
        assert cont is True
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))

    def encode_one_np(self, inputs, cont=True):
        inputs = inputs[None, ...]
        outputs = self.encode_np(inputs)
        return outputs[0]

    def decode_np(self, inputs, cont=True):
        raise NotImplementedError

    def decode_one_np(self, inputs, cont=True):
        raise NotImplementedError


class DeterministicObsEncoder(PyTorchModule):

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=128,
            use_normalization=False,

    ):
        super(DeterministicObsEncoder, self).__init__()

        self._layers = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                output_dim,
                bias=False,
            ),
        )

        self._use_normalization = use_normalization

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = output_dim

    def forward(self, inputs):
        encoding = self._layers(inputs)

        if self._use_normalization:
            encoding = encoding / (
                torch.norm(encoding, dim=-1, keepdim=True) + 1e-8)

        return encoding


class VariationalObsEncoder(ObsEncoder):

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=128,
            use_normalization=False,

    ):
        super(VariationalObsEncoder, self).__init__()

        assert not use_normalization

        self._layers = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),
        )

        self._mu_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_size = output_dim

    def forward(self, inputs, training=False):
        feat = self._layers(inputs)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            encoding = eps.mul(std).add_(mu)
            return encoding, (mu, logvar)
        else:
            encoding = mu
            return encoding


class CNNVariationalObsEncoder(ObsEncoder):

    def __init__(
            self,
            output_dim,

            input_width=48,
            input_height=48,
            input_channels=3,

            # CNN params
            kernel_sizes=[3, 3, 3],
            n_channels=[8, 16, 32],
            strides=[1, 1, 1],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            embedding_dim=5,  # TODO
            fc_hidden_sizes=[128],
            fc_normalization_type='none',

            # CNN params
            # kernel_sizes=[3, 3, 3, 3],
            # n_channels=[8, 8, 8, 8],
            # strides=[1, 1, 1, 1],
            # paddings=[1, 1, 1, 1],
            # pool_type='max2d',
            # pool_sizes=[2, 2, 2, 1],  # the one at the end means no pool
            # pool_strides=[2, 2, 2, 1],
            # pool_paddings=[0, 0, 0, 0],
            # embedding_dim=None,
            # fc_hidden_sizes=[128],
            # fc_normalization_type='none',

            # Network V2
            # kernel_sizes=[3, 3, 3, 3],
            # n_channels=[64, 64, 64, 64],
            # strides=[1, 1, 1, 1],
            # paddings=[1, 1, 1, 1],
            # pool_type='max2d',
            # pool_sizes=[2, 2, 2, 1],  # the one at the end means no pool
            # pool_strides=[2, 2, 2, 1],
            # pool_paddings=[0, 0, 0, 0],
            # embedding_dim=None,  # TODO
            # fc_hidden_sizes=[128],
            # fc_normalization_type='none',

            # Network V3
            # kernel_sizes=[3, 3, 3, 3],
            # n_channels=[16, 32, 64, 128],
            # strides=[1, 1, 1, 1],
            # paddings=[1, 1, 1, 1],
            # pool_type='max2d',
            # pool_sizes=[2, 2, 2, 1],  # the one at the end means no pool
            # pool_strides=[2, 2, 2, 1],
            # pool_paddings=[0, 0, 0, 0],
            # embedding_dim=None,  # TODO
            # fc_hidden_sizes=[256, 128],
            # fc_normalization_type='none',

            # Network V4
            # kernel_sizes=[3, 3, 3, 3, 3, 3],
            # n_channels=[64, 64, 64, 64, 64, 64],
            # strides=[1, 1, 1, 1, 1, 1],
            # paddings=[1, 1, 1, 1, 1, 1],
            # pool_type='max2d',
            # pool_sizes=[1, 2, 1, 2, 1, 1],
            # pool_strides=[1, 2, 1, 2, 1, 1],
            # pool_paddings=[0, 0, 0, 0, 0, 0],
            # embedding_dim=None,  # TODO
            # fc_hidden_sizes=[128, 128],
            # fc_normalization_type='none',

            use_normalization=False,

    ):
        super(CNNVariationalObsEncoder, self).__init__()

        assert not use_normalization

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.fc_normalization_type = fc_normalization_type
        self.input_dim = (
            self.input_channels * self.input_width * self.input_height)

        self._conv_layers = []
        num_conv_layers = len(n_channels)
        in_channels = input_channels
        for l in range(num_conv_layers):  # NOQA
            out_channels = n_channels[l]
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_sizes[l],
                stride=strides[l],
                padding=paddings[l])
            in_channels = out_channels
            self._conv_layers.append(layer)
            self._conv_layers.append(nn.ReLU(True))

            if pool_sizes[l] > 1:
                if pool_type == 'max2d':
                    layer = nn.MaxPool2d(
                        kernel_size=pool_sizes[l],
                        stride=pool_strides[l],
                        padding=pool_paddings[l],
                    )
                    self._conv_layers.append(layer)

        # Bottleneck conv layers.
        if embedding_dim is not None:
            layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embedding_dim,
                kernel_size=1,
                stride=1)
            self._conv_layers.append(layer)
            self._conv_layers.append(nn.ReLU(True))

        input_size = self.compute_conv_output_flat_size()

        # used only for injecting input directly into fc layers
        self._fc_layers = []
        for hidden_size in fc_hidden_sizes:
            output_size = hidden_size
            layer = nn.Linear(input_size, output_size)
            input_size = output_size
            self._fc_layers.append(layer)
            self._fc_layers.append(nn.ReLU(True))

        self._conv_layers = nn.Sequential(*self._conv_layers)
        self._fc_layers = nn.Sequential(*self._fc_layers)

        self._mu_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self.output_dim = output_dim
        self.representation_size = output_dim

    def forward(self, inputs, training=False):
        conv_input = inputs
        conv_input = conv_input.view(conv_input.shape[0],
                                     self.input_channels,
                                     self.input_height,
                                     self.input_width)

        feat = self._conv_layers(conv_input)
        feat = feat.view(feat.size(0), -1)
        feat = self._fc_layers(feat)

        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            encoding = eps.mul(std).add_(mu)
            return encoding, (mu, logvar)
        else:
            encoding = mu
            return encoding

    def compute_conv_output_flat_size(self):
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        for conv_layer in self._conv_layers:
            test_mat = conv_layer(test_mat)

        conv_output_flat_size = int(np.prod(test_mat.shape))

        return conv_output_flat_size

    def encode_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs.reshape(
            (-1,
             self.input_channels,
             self.input_width,
             self.input_height)
        )
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))


class ResNetVariationalObsEncoder(ObsEncoder):

    def __init__(
            self,
            output_dim,

            input_width=48,
            input_height=48,
            input_channels=3,

            bottleneck_dim=None,
            fc_hidden_sizes=[128],
            fixed=False,
    ):
        super(ResNetVariationalObsEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_dim = (
            self.input_channels * self.input_width * self.input_height)

        full_resnet = torchvision.models.resnet18(
            pretrained=True)
        resnet_conv = nn.Sequential(*list(full_resnet.children())[:-2])

        if fixed:
            for param in resnet_conv.parameters():
                param.requires_grad = False

        self._resnet = resnet_conv

        # Bottleneck conv layers.
        self._bottleneck_conv_layers = []
        if bottleneck_dim is not None:
            layer = nn.Conv2d(
                in_channels=512,
                out_channels=bottleneck_dim,
                kernel_size=1,
                stride=1)
            self._bottleneck_conv_layers.append(layer)
            self._bottleneck_conv_layers.append(nn.ReLU(True))
        self._bottleneck_conv_layers = nn.Sequential(
            *self._bottleneck_conv_layers)

        input_size = self.compute_conv_output_flat_size()

        # used only for injecting input directly into fc layers
        self._fc_layers = []
        for hidden_size in fc_hidden_sizes:
            output_size = hidden_size
            layer = nn.Linear(input_size, output_size)
            input_size = output_size
            self._fc_layers.append(layer)

            self._fc_layers.append(nn.ReLU(True))

        self._fc_layers = nn.Sequential(*self._fc_layers)

        self._mu_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self.output_dim = output_dim
        self.representation_size = output_dim

    def forward(self, inputs, training=False):
        conv_input = inputs
        conv_input = conv_input.view(conv_input.shape[0],
                                     self.input_channels,
                                     self.input_height,
                                     self.input_width)

        feat = self._resnet(conv_input)

        for conv_layer in self._bottleneck_conv_layers:
            feat = conv_layer(feat)

        feat = feat.view(feat.size(0), -1)
        feat = self._fc_layers(feat)

        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            encoding = eps.mul(std).add_(mu)
            return encoding, (mu, logvar)
        else:
            encoding = mu
            return encoding

    def compute_conv_output_flat_size(self):
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self._resnet(test_mat)

        for conv_layer in self._bottleneck_conv_layers:
            test_mat = conv_layer(test_mat)

        conv_output_flat_size = int(np.prod(test_mat.shape))

        return conv_output_flat_size

    def encode_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs.reshape(
            (-1,
             self.input_channels,
             self.input_width,
             self.input_height)
        )
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))


class VqvaeVariationalObsEncoder(ObsEncoder):

    def __init__(
            self,
            output_dim,

            input_width=48,
            input_height=48,
            input_channels=3,

            vqvae=None,
            fixed=True,

            # TODO: For Bridge Data, we might need to use conv layers here.
            # CNN params
            # kernel_sizes=[3, 3, 3],
            # n_channels=[8, 16, 32],
            # strides=[1, 1, 1],
            # paddings=[1, 1, 1],
            # pool_type='max2d',
            # pool_sizes=[2, 2, 1],  # the one at the end means no pool
            # pool_strides=[2, 2, 1],
            # pool_paddings=[0, 0, 0],
            # bottleneck_dim=None,

            # FC params
            fc_hidden_sizes=[128, 128],
    ):
        super(VqvaeVariationalObsEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_dim = (
            self.input_channels * self.input_width * self.input_height)

        self.fixed = fixed

        if self.fixed:
            for param in vqvae.parameters():
                param.requires_grad = False

        self._vqvae = vqvae

        input_size = self._vqvae.representation_size

        # self._conv_layers = []
        # num_conv_layers = len(n_channels)
        # in_channels = input_channels
        # for l in range(num_conv_layers):  # NOQA
        #     out_channels = n_channels[l]
        #     layer = nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         kernel_size=kernel_sizes[l],
        #         stride=strides[l],
        #         padding=paddings[l])
        #     in_channels = out_channels
        #     self._conv_layers.append(layer)
        #     self._conv_layers.append(nn.ReLU(True))
        #
        #     if pool_sizes[l] > 1:
        #         if pool_type == 'max2d':
        #             layer = nn.MaxPool2d(
        #                 kernel_size=pool_sizes[l],
        #                 stride=pool_strides[l],
        #                 padding=pool_paddings[l],
        #             )
        #             self._conv_layers.append(layer)
        #
        # if bottleneck_dim is not None:
        #     layer = nn.Conv2d(
        #         in_channels=512,
        #         out_channels=bottleneck_dim,
        #         kernel_size=1,
        #         stride=1)
        #     self._conv_layers.append(layer)
        #     self._conv_layers.append(nn.ReLU(True))
        #
        # self._conv_layers = nn.Sequential(
        #     *self._conv_layers)

        # used only for injecting input directly into fc layers
        self._fc_layers = []
        for hidden_size in fc_hidden_sizes:
            output_size = hidden_size
            layer = nn.Linear(input_size, output_size)
            input_size = output_size
            self._fc_layers.append(layer)

            self._fc_layers.append(nn.ReLU(True))

        self._fc_layers = nn.Sequential(*self._fc_layers)

        self._mu_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(input_size, output_dim),
        )

        self.output_dim = output_dim
        self.representation_size = output_dim
        self.input_channels = self._vqvae.input_channels
        self.imsize = self._vqvae.imsize

    def forward(self, inputs, training=False):

        precomputed_vqvae_encoding = (
            inputs.shape[-1] == self._vqvae.representation_size)

        if precomputed_vqvae_encoding:
            feat = inputs
        else:
            obs = inputs
            obs = obs - 0.5
            obs = obs.view(obs.shape[0],
                           self.input_channels,
                           self.input_height,
                           self.input_width)
            obs = obs.permute([0, 1, 3, 2])

            feat = self._vqvae.encode(obs, flatten=False)

        if self.fixed:
            feat = feat.detach()

        # for conv_layer in self._conv_layers:
        #     feat = conv_layer(feat)

        feat = feat.view(feat.size(0), -1)
        feat = self._fc_layers(feat)

        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            encoding = eps.mul(std).add_(mu)
            return encoding, (mu, logvar)
        else:
            encoding = mu
            return encoding

    def encode_np(self, inputs, cont=True):
        assert cont is True
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))


class Dynamics(PyTorchModule):

    def __init__(
            self,
            obs_encoding_dim,
            action_dim,
            hidden_dim=128,
            use_normalization=False,

    ):
        super(Dynamics, self).__init__()

        self._layers = nn.Sequential(
            nn.Linear(
                obs_encoding_dim + action_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            # nn.Linear(
            #     hidden_dim,
            #     hidden_dim,
            # ),
            # nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                obs_encoding_dim,
                bias=False,
            ),
        )

        self._use_normalization = use_normalization

    def forward(self, encoding, action):
        inputs = torch.cat([encoding, action], -1)
        delta_encoding = self._layers(inputs)
        pred_encoding = encoding + delta_encoding

        if self._use_normalization:
            pred_encoding = pred_encoding / (
                torch.norm(pred_encoding, dim=-1, keepdim=True) + 1e-8)

        return pred_encoding


class QFNetwork(PyTorchModule):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=128,
        encoding_dim=32,
        init_w=3e-3,
        output_activation=identity,
    ):
        super().__init__()

        self.obs_dim = obs_dim

        self._obs_encoder = ObsEncoder(
            input_dim=obs_dim,
            output_dim=encoding_dim,
            use_normalization=False,
        )

        self._layers = nn.Sequential(
            nn.Linear(
                encoding_dim * 2 + action_dim,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),
        )

        self._output_layer = nn.Linear(hidden_dim, 1)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)
        self._output_activation = output_activation

    def forward(self, input_obs, action):

        # obs = input_obs['obs']
        # obs_encoding = self._obs_encoder(obs)
        #
        # context = input_obs['context']
        # context_encoding = self._obs_encoder(context)

        obs = input_obs[..., :self.obs_dim]
        obs_encoding = self._obs_encoder(obs)

        context = input_obs[..., self.obs_dim:]
        context_encoding = self._obs_encoder(context)

        net = torch.cat(
            (obs_encoding, context_encoding - obs_encoding, action), dim=-1)
        net = self._layers(net)

        preactivation = self._output_layer(net)
        output = self._output_activation(preactivation)

        return output


class VFNetwork(PyTorchModule):
    def __init__(
        self,
        obs_dim,
        hidden_dim=128,
        encoding_dim=32,
        init_w=3e-3,
    ):
        super().__init__()

        self.obs_dim = obs_dim

        self._obs_encoder = ObsEncoder(
            input_dim=obs_dim,
            output_dim=encoding_dim,
            use_normalization=False,
        )

        self._layers = nn.Sequential(
            nn.Linear(
                encoding_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),
        )

        self._output_layer = nn.Linear(hidden_dim, 1)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)

    def forward(self, input_obs):

        # obs = input_obs['obs']
        # obs_encoding = self._obs_encoder(obs)
        #
        # context = input_obs['context']
        # context_encoding = self._obs_encoder(context)

        obs = input_obs[..., :self.obs_dim]
        obs_encoding = self._obs_encoder(obs)

        context = input_obs[..., self.obs_dim:]
        context_encoding = self._obs_encoder(context)

        net = torch.cat(
            (obs_encoding, context_encoding - obs_encoding), dim=-1)
        net = self._layers(net)

        output = self._output_layer(net)

        return output


class GaussianPolicy(TorchStochasticPolicy):
    # class GaussianPolicy(DictTorchStochasticPolicy):

    def __init__(
            self,
            obs_dim,
            action_dim,
            encoding_dim=32,
            hidden_dim=128,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture='shared',
            output_activation=None,
            **kwargs
    ):
        super().__init__(
        )

        self.obs_dim = obs_dim

        self._obs_encoder = ObsEncoder(
            input_dim=obs_dim,
            output_dim=encoding_dim,
            use_normalization=False,
        )

        self._layers = nn.Sequential(
            nn.Linear(
                encoding_dim * 2,
                hidden_dim,
            ),
            nn.ReLU(True),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(True),
        )

        self._output_layer = nn.Linear(hidden_dim, action_dim)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)
        self._output_activation = output_activation

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == 'values':
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, input_obs):

        # obs = input_obs['obs']
        # obs_encoding = self._obs_encoder(obs)
        #
        # context = input_obs['context']
        # context_encoding = self._obs_encoder(context)

        obs = input_obs[..., :self.obs_dim]
        obs_encoding = self._obs_encoder(obs)

        context = input_obs[..., self.obs_dim:]
        context_encoding = self._obs_encoder(context)

        h = torch.cat((obs_encoding, context_encoding), dim=-1)
        h = self._layers(h)

        mean = self._output_layer(h)
        if self._output_activation is not None:
            mean = self._output_activation(mean)

        if self.std is None:
            if self.std_architecture == 'values':
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


class EncodingGaussianPolicy(TorchStochasticPolicy):

    def __init__(
            self,
            obs_dim,
            action_dim,
            obs_encoder,
            hidden_sizes=[128, 128],
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture='shared',
            output_activation=None,
            goal_is_encoded=False,
            always_use_encoded_input=False,
            **kwargs
    ):
        super().__init__(
        )

        self._obs_dim = obs_dim
        self._obs_encoder = obs_encoder
        self._goal_is_encoded = goal_is_encoded

        layers = []

        input_dim = self._obs_encoder.output_dim * 2
        for output_dim in hidden_sizes:
            layer = nn.Linear(
                input_dim,
                output_dim,
            )
            layers.append(layer)

            layer = nn.ReLU(True)
            layers.append(layer)

            input_dim = output_dim

        self._layers = nn.Sequential(*layers)

        self._output_layer = nn.Linear(input_dim, action_dim)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)
        self._output_activation = output_activation

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == 'values':
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

        self._always_use_encoded_input = always_use_encoded_input

    def forward(self, obs_and_goals, encoded_input=False):
        if encoded_input or self._always_use_encoded_input:
            h = obs_and_goals
        else:
            obs = obs_and_goals[..., :self._obs_dim]
            obs_feat = self._obs_encoder(obs)

            goal = obs_and_goals[..., self._obs_dim:]

            if self._goal_is_encoded:
                assert goal.shape[-1] == self._obs_encoder.output_dim
                goal_feat = goal  # TODO
            else:
                if isinstance(self._obs_encoder, VqvaeVariationalObsEncoder):
                    assert (
                        goal.shape[-1] == self._obs_encoder.input_dim or
                        goal.shape[-1] == self._obs_encoder._vqvae.representation_size)  # NOQA
                else:
                    assert goal.shape[-1] == self._obs_encoder.input_dim
                goal_feat = self._obs_encoder(goal)

            h = torch.cat((obs_feat, goal_feat), dim=-1)
            # h = h.detach()  # TODO

        net = self._layers(h)

        mean = self._output_layer(net)
        if self._output_activation is not None:
            mean = self._output_activation(mean)

        if self.std is None:
            if self.std_architecture == 'values':
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


class EncodingGaussianPolicyV2(TorchStochasticPolicy):

    def __init__(
            self,
            obs_dim,
            action_dim,
            obs_encoder,
            hidden_sizes=[128, 128],
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture='shared',
            output_activation=None,
            goal_is_encoded=False,
            **kwargs
    ):
        super().__init__(
        )

        self._obs_dim = obs_dim
        self._goal_encoder = obs_encoder
        self._goal_is_encoded = goal_is_encoded

        obs_encoder_kwargs = dict()

        # if isinstance(self._goal_encoder, DeterministicObsEncoder):
        #     obs_encoder_class = DeterministicObsEncoder
        #     obs_encoder_kwargs['input_dim'] = obs_dim
        #     obs_encoder_kwargs['output_dim'] = 128
        #     # obs_encoder_kwargs['input_dim'] = self._goal_encoder.input_dim
        #     # obs_encoder_kwargs['output_dim'] = self._goal_encoder.output_dim  # NOQA
        # elif isinstance(self._goal_encoder, VariationalObsEncoder):
        #     pass
        # else:
        #     raise ValueError

        obs_encoder_class = DeterministicObsEncoder
        obs_encoder_kwargs['input_dim'] = obs_dim
        obs_encoder_kwargs['output_dim'] = 128

        self._obs_encoder = obs_encoder_class(
            **obs_encoder_kwargs,
        )

        # self._obs_encoder = CNNVariationalObsEncoder(
        #     input_width=self._goal_encoder.input_width,
        #     input_height=self._goal_encoder.input_height,
        #     input_channels=self._goal_encoder.input_channels,
        #     output_dim=self._goal_encoder.output_dim,
        # )

        layers = []

        input_dim = (self._obs_encoder.output_dim +
                     self._goal_encoder.output_dim)
        for output_dim in hidden_sizes:
            layer = nn.Linear(
                input_dim,
                output_dim,
            )
            layers.append(layer)

            layer = nn.ReLU(True)
            layers.append(layer)

            input_dim = output_dim

        self._layers = nn.Sequential(*layers)

        self._output_layer = nn.Linear(input_dim, action_dim)
        self._output_layer.weight.data.uniform_(-init_w, init_w)
        self._output_layer.bias.data.fill_(0)
        self._output_activation = output_activation

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == 'values':
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs_and_goals, encoded_input=False):
        obs = obs_and_goals[..., :self._obs_dim]
        goal = obs_and_goals[..., self._obs_dim:]

        obs_feat = self._obs_encoder(obs)

        if encoded_input:
            assert goal.shape[-1] == self._goal_encoder.output_dim
            goal_feat = goal
        else:
            if self._goal_is_encoded:
                assert goal.shape[-1] == self._goal_encoder.output_dim
                goal_feat = goal
            else:
                if isinstance(self._goal_encoder, VqvaeVariationalObsEncoder):
                    assert (
                        goal.shape[-1] == self._goal_encoder.input_dim or
                        goal.shape[-1] == self._goal_encoder._vqvae.representation_size)  # NOQA
                else:
                    assert goal.shape[-1] == self._goal_encoder.input_dim

                goal_feat = self._goal_encoder(goal)

        h = torch.cat((obs_feat, goal_feat), dim=-1)
        net = self._layers(h)

        mean = self._output_layer(net)
        if self._output_activation is not None:
            mean = self._output_activation(mean)

        if self.std is None:
            if self.std_architecture == 'values':
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
