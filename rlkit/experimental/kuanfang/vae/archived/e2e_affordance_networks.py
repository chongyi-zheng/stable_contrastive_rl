import torch
from torch import nn

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from rlkit.experimental.kuanfang.vae import network_utils  # NOQA


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 input_nc=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc,
                             inner_nc,
                             kernel_size=4,
                             stride=2,
                             padding=1,
                             bias=False)
        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)

        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv]

        elif outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downconv]
            up = [uprelu, upconv, nn.ReLU(True)]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2,
                                        outer_nc,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

        self.inner_nc = inner_nc
        self.outer_nc = outer_nc

    def forward(self, x, cond):
        # print('-------------------')
        # print('x: ', x.shape)
        # print(self.inner_nc, self.outer_nc)
        state = x
        state = self.down(state)
        # print('down: ', state.shape)
        state = self.submodule(state, cond)
        # print('submodule: ', state.shape)
        state = self.up(state)
        # print('up: ', state.shape)

        if self.outermost:
            return state
        else:
            # Add skip connections
            return torch.cat([x, state], 1)


class FlattenFCBlock(nn.Module):

    def __init__(self,
                 outer_nc,
                 inner_nc,
                 cond_nc,
                 input_w=3,
                 input_h=3,
                 input_nc=None,
                 use_fc=False,
                 use_dropout=False):
        super(FlattenFCBlock, self).__init__()
        if input_nc is None:
            input_nc = outer_nc

        self.outer_nc = outer_nc
        self.inner_nc = inner_nc
        self.input_nc = input_nc
        self.input_w = input_w
        self.input_h = input_h

        self.flatten = Flatten()

        self.down = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(
                input_w * input_h * input_nc + cond_nc,
                inner_nc),
        )
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(
                inner_nc,
                input_w * input_h * outer_nc),
        )

    def forward(self, x, cond):
        state = self.flatten(x)
        state = torch.cat([state, cond], 1)
        state = self.down(state)
        state = self.up(state)
        state = state.view(-1, self.outer_nc, self.input_w, self.input_h)
        return torch.cat([x, state], 1)


class CcVae(PyTorchModule):

    def __init__(
            self,
            data_channels,
            data_root_len=48,
            z_dim=4,

            hidden_dim=8,
            unet_inner_dim=64,
            z_fc_dim=64,

            use_normalization=True,  # TODO
            output_scaling_factor=None,
    ):
        super(CcVae, self).__init__()

        if data_root_len == 48:
            embedding_root_len = 3
        else:
            raise ValueError

        self.representation_size = z_dim
        self.data_root_len = data_root_len

        self._encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=data_channels * 2,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Conv2d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            nn.Conv2d(
                in_channels=hidden_dim * 4,
                out_channels=hidden_dim * 8,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),

            Flatten(),
            nn.Linear(
                hidden_dim * 8 * embedding_root_len * embedding_root_len,
                z_fc_dim,
                bias=False,
            ),
            nn.ReLU(True),
        )

        self._mu_layer = nn.Sequential(
            nn.Linear(z_fc_dim, z_dim),
        )

        self._logvar_layer = nn.Sequential(
            nn.Linear(z_fc_dim, z_dim),
        )

        self._z_encoder = nn.Sequential(
            nn.Linear(z_dim, z_fc_dim, bias=False),
            nn.ReLU(True),
        )

        # Decoder
        unet_block = FlattenFCBlock(
            outer_nc=8 * hidden_dim,
            inner_nc=unet_inner_dim,
            input_nc=None,
            cond_nc=z_fc_dim,
        )
        unet_block = UnetSkipConnectionBlock(
            outer_nc=4 * hidden_dim,
            inner_nc=8 * hidden_dim,
            input_nc=None,
            submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(
            outer_nc=2 * hidden_dim,
            inner_nc=4 * hidden_dim,
            input_nc=None,
            submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(
            outer_nc=hidden_dim,
            inner_nc=2 * hidden_dim,
            input_nc=None,
            submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(
            outer_nc=hidden_dim,
            inner_nc=hidden_dim,
            input_nc=data_channels,
            submodule=unet_block,
            outermost=True)

        self._decoder = unet_block

        self._output_layer = [
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=data_channels,
                kernel_size=1,
                stride=1)
        ]

        if use_normalization:
            self._output_layer += [nn.Tanh()]

        self._output_layer = nn.Sequential(*self._output_layer)
        self._output_scaling_factor = output_scaling_factor

    def encode(self, data, cond):
        assert data.shape[-1] == self.data_root_len
        assert data.shape[-2] == self.data_root_len
        assert cond.shape[-1] == self.data_root_len
        assert cond.shape[-2] == self.data_root_len

        img_diff = cond - data
        encoder_input = torch.cat((img_diff, cond), dim=1)

        feat = self._encoder(encoder_input)
        mu = self._mu_layer(feat)
        logvar = self._logvar_layer(feat)

        return mu, logvar

    def decode(self, z, cond):
        z_feat = self._z_encoder(z)

        state = self._decoder.forward(cond, cond=z_feat)
        recon = self._output_layer(state)

        if self._output_scaling_factor is not None:
            recon = recon * self._output_scaling_factor

        return recon

    def forward(self, data, cond):
        mu, logvar = self.encode(data, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return (mu, logvar), z, recon

    def sample_prior(self, batch_size):
        z_s = ptu.randn(batch_size, self.representation_size)
        return ptu.get_numpy(z_s)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
