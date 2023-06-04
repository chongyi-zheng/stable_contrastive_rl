import numpy as np
from rlkit.torch.core import PyTorchModule
from rlkit.torch import pytorch_util as ptu
from torch import nn as nn
import torch


class ExpectableQF(PyTorchModule):
    """
    A Q-function whose expectation w.r.t. actions is computable in closed form.
    """
    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
    ):
        super().__init__()

        self.obs_fc = nn.Linear(obs_dim, hidden_size)
        self.action_fc = nn.Linear(action_dim, hidden_size)
        hidden_init(self.obs_fc.weight)
        hidden_init(self.action_fc.weight)
        self.obs_fc.bias.data.fill_(b_init_value)
        self.action_fc.bias.data.fill_(b_init_value)

        self.last_fc = nn.Linear(2*hidden_size, 1)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action, action_stds=None):
        h_obs = torch.tanh(self.obs_fc(obs))
        if action_stds is not None:
            """
            action_fc_weight has the same variance for every row in the batch
            action_stds has difference variances for everything
            Let
                A = action dimension size
                H = hidden size
                B = batch size
            Then
                action_fc.weight is  H x A
                action_stds is B x A
            and so variance is
                B X H
            where the summing happens across the action dimension, as needed.
            """
            weight_vars = self.action_fc.weight**2  # H x A
            action_vars = action_stds**2  # B x A
            variance = action_vars @ weight_vars.transpose(0, 1)  # B x H
            conv_factor_inv = torch.sqrt(1 + np.pi / 2 * variance)
            h_action = torch.tanh(self.action_fc(action) / conv_factor_inv)
        else:
            h_action = torch.tanh(self.action_fc(action))
        h = torch.cat((h_obs, h_action), dim=1)
        return self.last_fc(h)
