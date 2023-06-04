import numpy as np
import torch
from torch import nn

from rlkit.experimental.kuanfang.networks.encoding_networks import ObsEncoder
from rlkit.experimental.kuanfang.vae.vqvae import Encoder
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.distributions import (
    MultivariateDiagonalNormal
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class ContrastiveCNNObsEncoder(ObsEncoder):
    def __init__(self,
                 output_dim=64,
                 input_width=48,
                 input_height=48,
                 input_channels=3):
        super(ContrastiveCNNObsEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_dim = input_width * input_height * input_channels
        self.output_dim = output_dim

        self._conv_encoder = nn.Sequential(
            nn.Conv2d(3, 32, (8, 8), 4, 3),  # CHW = (32, 12, 12)
            nn.ReLU(True),
            nn.Conv2d(32, 64, (4, 4), 2, 1),  # CHW = (64, 6, 6)
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), 2, 1),  # CHW = (64, 3, 3)
            nn.ReLU(True),
        )

        self._final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=self.output_dim,
            kernel_size=3,
            stride=1)

        self.representation_size = self.compute_conv_output_flat_size()

    def compute_conv_output_flat_size(self):
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self.forward(test_mat)

        conv_output_flat_size = int(np.prod(test_mat.shape))

        return conv_output_flat_size

    def forward(self, inputs):
        conv_input = inputs
        conv_input = conv_input.view(conv_input.shape[0],
                                     self.input_channels,
                                     self.input_height,
                                     self.input_width)
        feat = self._conv_encoder(conv_input)
        feat = self._final_conv(feat)
        feat = feat.view(feat.size(0), -1)

        return feat

    def encode_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs.reshape(
            (-1,
             self.input_channels,
             self.input_width,
             self.input_height)
        )
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))


class ContrastiveResCNNObsEncoder(ObsEncoder):
    def __init__(self,
                 output_dim=5,
                 input_width=48,
                 input_height=48,
                 input_channels=3,
                 hidden_dim=128,
                 num_residual_layers=3,
                 residual_hidden_dim=64,
                 ):
        super(ContrastiveResCNNObsEncoder, self).__init__()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.input_dim = input_width * input_height * input_channels
        self.output_dim = output_dim

        self._conv_encoder = Encoder(
            input_channels,
            hidden_dim,
            num_residual_layers,
            residual_hidden_dim)

        self._final_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1)

        self.representation_size = self.compute_conv_output_flat_size()

    def compute_conv_output_flat_size(self):
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )

        test_mat = self.forward(test_mat)

        conv_output_flat_size = int(np.prod(test_mat.shape))

        return conv_output_flat_size

    def forward(self, inputs):
        conv_input = inputs
        conv_input = conv_input.view(conv_input.shape[0],
                                     self.input_channels,
                                     self.input_height,
                                     self.input_width)
        feat = self._conv_encoder(conv_input)
        feat = self._final_conv(feat)
        feat = feat.view(feat.size(0), -1)

        return feat

    def encode_np(self, inputs, cont=True):
        assert cont is True
        inputs = inputs.reshape(
            (-1,
             self.input_channels,
             self.input_width,
             self.input_height)
        )
        return ptu.get_numpy(self.forward(ptu.from_numpy(inputs)))


class ContrastiveEncodingGaussianPolicy(TorchStochasticPolicy):
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
        # TODO (chongyiz): implement hidden_init
        super().__init__(
        )

        self._obs_dim = obs_dim
        self._obs_encoder = obs_encoder
        self._goal_is_encoded = goal_is_encoded

        layers = []

        input_dim = self._obs_encoder.representation_size * 2
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

        self._always_use_encoded_input = always_use_encoded_input

    def forward(self, obs_and_goals, encoded_input=False):
        if encoded_input or self._always_use_encoded_input:
            h = obs_and_goals
        else:
            obs = obs_and_goals[..., :self._obs_dim]
            obs_feat = self._obs_encoder(obs)

            goal = obs_and_goals[..., self._obs_dim:]

            if self._goal_is_encoded:
                assert goal.shape[-1] == self._obs_encoder.representation_size
                goal_feat = goal  # TODO
            else:
                # if isinstance(self._obs_encoder, VqvaeVariationalObsEncoder):
                #     assert (
                #         goal.shape[-1] == self._obs_encoder.input_dim or
                #         goal.shape[-1] == self._obs_encoder._vqvae.representation_size)  # NOQA
                # else:
                #     assert goal.shape[-1] == self._obs_encoder.input_dim
                assert goal.shape[-1] == self._obs_encoder.input_dim
                goal_feat = self._obs_encoder(goal)

            h = torch.cat((obs_feat, goal_feat), dim=-1)
            # h = h.detach()  # TODO

        net = self._layers(h)

        mean = self._output_layer(net)
        if self._output_activation is not None:
            mean = self._output_activation(mean)

        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(net))
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
