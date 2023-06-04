import torch
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from rlkit.experimental.kuanfang.vae import network_utils
from rlkit.experimental.kuanfang.vae.affordance_networks import CcVae

class CcVaeGCP(CcVae):

    def __init__(
            self,
            data_channels,
            data_root_len=12,
            z_dim=8,

            hidden_dim=64,
            unet_inner_dim=512,
            z_fc_dim=512,

            # hidden_dim=32,
            # unet_inner_dim=128,
            # z_fc_dim=128,

            # hidden_dim=16,
            # unet_inner_dim=64,
            # z_fc_dim=64,

            use_normalization=False,
            output_scaling_factor=None,
    ):
        super(CcVaeGCP, self).__init__(
            data_channels*2,
            decoder_output_layer_channels=data_channels,
            data_root_len=data_root_len,
            z_dim=z_dim,

            hidden_dim=hidden_dim,
            unet_inner_dim=unet_inner_dim,
            z_fc_dim=z_fc_dim,

            use_normalization=use_normalization,
            output_scaling_factor=output_scaling_factor,
        )

    def encode(self, data, cond):
        assert data.shape[-1] == self.data_root_len
        assert data.shape[-2] == self.data_root_len
        assert cond.shape[-1] == self.data_root_len
        assert cond.shape[-2] == self.data_root_len

        init_cond = cond[:,:data.shape[1]]
        goal_cond = cond[:,data.shape[1]:]

        init_diff = init_cond - data
        goal_diff = data - goal_cond
        img_diff = torch.cat((init_diff, goal_diff), dim=1)
        encoder_input = torch.cat((img_diff, cond), dim=1)

        feat = self._encoder(encoder_input)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar